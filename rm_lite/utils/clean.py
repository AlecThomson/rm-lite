"""RM-clean utils"""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm

from rm_lite.utils.fitting import (
    gaussian,
    gaussian_integrand,
    unit_centred_gaussian,
)
from rm_lite.utils.logging import TqdmToLogger, logger

TQDM_OUT = TqdmToLogger(logger, level=logging.INFO)


class RMCleanResults(NamedTuple):
    """Results of the RM-CLEAN calculation"""

    clean_fdf_arr: NDArray[np.complex128]
    """The cleaned Faraday dispersion function cube"""
    model_fdf_arr: NDArray[np.complex128]
    """The clean components cube"""
    clean_iter_arr: NDArray[np.int16]
    """The number of iterations for each pixel"""
    resid_fdf_arr: NDArray[np.complex128]
    """The residual Faraday dispersion function cube"""


class CleanLoopResults(NamedTuple):
    """Results of the RM-CLEAN loop"""

    clean_fdf_spectrum: NDArray[np.complex128]
    """The cleaned Faraday dispersion function cube"""
    resid_fdf_spectrum: NDArray[np.complex128]
    """The residual Faraday dispersion function cube"""
    model_fdf_spectrum: NDArray[np.complex128]
    """The clean components cube"""
    iter_count: int
    """The number of iterations"""


class MinorLoopResults(NamedTuple):
    """Results of the RM-CLEAN minor loop"""

    clean_fdf_spectrum: NDArray[np.complex128]
    """The cleaned Faraday dispersion function cube"""
    resid_fdf_spectrum: NDArray[np.complex128]
    """The residual Faraday dispersion function cube"""
    resid_fdf_spectrum_mask: np.ma.MaskedArray
    """The masked residual Faraday dispersion function cube"""
    model_fdf_spectrum: NDArray[np.complex128]
    """The clean components cube"""
    model_rmsf_spectrum: NDArray[np.complex128]
    """ Model * RMSF """
    iter_count: int
    """The number of iterations"""


def restore_fdf(
    model_fdf_spectrum: NDArray[np.complex128],
    phi_double_arr_radm2: NDArray[np.float64],
    fwhm_rmsf: float,
) -> NDArray[np.complex128]:
    clean_beam = unit_centred_gaussian(
        x=phi_double_arr_radm2,
        fwhm=fwhm_rmsf,
    ) / gaussian_integrand(amplitude=1, fwhm=fwhm_rmsf)
    restored_fdf = np.convolve(
        model_fdf_spectrum.real, clean_beam, mode="valid"
    ) + 1j * np.convolve(model_fdf_spectrum.imag, clean_beam, mode="valid")
    return restored_fdf[1:-1]


def rmclean(
    dirty_fdf_arr: NDArray[np.complex128],
    phi_arr_radm2: NDArray[np.float64],
    rmsf_arr: NDArray[np.complex128],
    phi_double_arr_radm2: NDArray[np.float64],
    fwhm_rmsf_arr: NDArray[np.float64],
    mask: float,
    threshold: float,
    max_iter: int = 1000,
    gain: float = 0.1,
    mask_arr: NDArray[np.bool_] | None = None,
) -> RMCleanResults:
    """Perform RM-CLEAN on a Faraday dispersion function array.

    Args:
        dirty_fdf_arr (NDArray[np.complex128]): Dirty Faraday dispersion function array
        phi_arr_radm2 (NDArray[np.float64]): Faraday depth array in rad/m^2
        rmsf_arr (NDArray[np.complex128]): RMSF array
        phi_double_arr_radm2 (NDArray[np.float64]): Double-length Faraday depth array in rad/m^2
        fwhm_rmsf_arr (NDArray[np.float64]): FWHM of the RMSF array
        mask (float): Masking threshold - pixels below this value are not cleaned
        threshold (float): Cleaning threshold - stop when all pixels are below this value
        max_iter (int, optional): Maximum clean iterations. Defaults to 1000.
        gain (float, optional): Glean loop gain. Defaults to 0.1.
        mask_arr (NDArray[np.bool_] | None, optional): Additional mask of pixels to avoid. Defaults to None.

    Returns:
        RMCleanResults: clean_fdf_arr, model_fdf_arr, clean_iter_arr, resid_fdf_arr
    """
    _bad_result = RMCleanResults(
        clean_fdf_arr=dirty_fdf_arr,
        model_fdf_arr=np.zeros_like(dirty_fdf_arr),
        clean_iter_arr=np.zeros_like(phi_arr_radm2, dtype=int),
        resid_fdf_arr=dirty_fdf_arr,
    )
    # Sanity checks on array sizes
    n_phi = phi_arr_radm2.shape[0]
    if n_phi != dirty_fdf_arr.shape[0]:
        logger.error("'phi_arr_radm2' and 'dirty_fdf_arr' are not the same length.")
        return _bad_result
    n_phi2 = phi_double_arr_radm2.shape[0]
    if n_phi2 != rmsf_arr.shape[0]:
        logger.error("mismatch in 'phi_double_arr_radm2' and 'rmsf_arr' length.")
        return _bad_result
    if not (n_phi2 >= 2 * n_phi):
        logger.error("the Faraday depth of the RMSF must be twice the FDF.")
        return _bad_result
    n_dimension = len(dirty_fdf_arr.shape)
    if not n_dimension <= 3:
        logger.error("FDF array dimensions must be <= 3.")
        return _bad_result
    if n_dimension != len(rmsf_arr.shape):
        logger.error("the input RMSF and FDF must have the same number of axes.")
        return _bad_result
    if rmsf_arr.shape[1:] != dirty_fdf_arr.shape[1:]:
        logger.error("the xy dimensions of the RMSF and FDF must match.")
        return _bad_result
    if mask_arr is not None:
        if mask_arr.shape != dirty_fdf_arr.shape[1:]:
            logger.error("pixel mask must match xy dimension of FDF cube.")
            return _bad_result
    else:
        mask_arr = np.ones(dirty_fdf_arr.shape[1:], dtype=bool)

    # Reshape the FDF & RMSF array to 3 dimensions and mask array to 2
    if n_dimension == 1:
        dirty_fdf_arr = np.reshape(dirty_fdf_arr, (dirty_fdf_arr.shape[0], 1, 1))
        rmsf_arr = np.reshape(rmsf_arr, (rmsf_arr.shape[0], 1, 1))
        mask_arr = np.reshape(mask_arr, (1, 1))
        fwhm_rmsf_arr = np.reshape(fwhm_rmsf_arr, (1, 1))
    elif n_dimension == 2:
        dirty_fdf_arr = np.reshape(dirty_fdf_arr, [*list(dirty_fdf_arr.shape[:2]), 1])
        rmsf_arr = np.reshape(rmsf_arr, [*list(rmsf_arr.shape[:2]), 1])
        mask_arr = np.reshape(mask_arr, (dirty_fdf_arr.shape[1], 1))
        fwhm_rmsf_arr = np.reshape(fwhm_rmsf_arr, (dirty_fdf_arr.shape[1], 1))

    iter_count_arr = np.zeros_like(mask_arr, dtype=int)

    # Determine which pixels have components above the cutoff
    abs_fdf_cube = np.abs(np.nan_to_num(dirty_fdf_arr))
    cutoff_mask = np.where(np.max(abs_fdf_cube, axis=0) >= mask, 1, 0)
    pixels_to_clean = np.rot90(np.where(cutoff_mask > 0))

    num_pixels = dirty_fdf_arr.shape[-1] * dirty_fdf_arr.shape[-2]
    num_pixels_clean = len(pixels_to_clean)
    logger.info(f"Cleaning {num_pixels_clean}/{num_pixels} spectra.")

    # Initialise arrays to hold the residual FDF, clean components, clean FDF
    # Residual is initially copies of dirty FDF, so that pixels that are not
    #  processed get correct values (but will be overridden when processed)
    clean_fdf_spectrum = np.zeros_like(dirty_fdf_arr)
    model_fdf_spectrum = np.zeros(dirty_fdf_arr.shape, dtype=complex)
    resid_fdf_arr = dirty_fdf_arr.copy()

    # Loop through the pixels containing a polarised signal
    for yi, xi in tqdm(pixels_to_clean):
        clean_loop_results = minor_cycle(
            phi_arr_radm2=phi_arr_radm2,
            phi_double_arr_radm2=phi_double_arr_radm2,
            dirty_fdf_spectrum=dirty_fdf_arr[:, yi, xi],
            rmsf_spectrum=rmsf_arr[:, yi, xi],
            rmsf_fwhm=fwhm_rmsf_arr[yi, xi],
            mask=mask,
            threshold=threshold,
            max_iter=max_iter,
            gain=gain,
        )
        clean_fdf_spectrum[:, yi, xi] = clean_loop_results.clean_fdf_spectrum
        resid_fdf_arr[:, yi, xi] = clean_loop_results.resid_fdf_spectrum
        model_fdf_spectrum[:, yi, xi] = clean_loop_results.model_fdf_spectrum
        iter_count_arr[yi, xi] = clean_loop_results.iter_count

    # Restore the residual to the cleaned FDF (moved outside of loop:
    # will now work for pixels/spectra without clean components)
    clean_fdf_spectrum += resid_fdf_arr

    # Remove redundant dimensions
    clean_fdf_spectrum = np.squeeze(clean_fdf_spectrum)
    model_fdf_spectrum = np.squeeze(model_fdf_spectrum)
    iter_count_arr = np.squeeze(iter_count_arr)
    resid_fdf_arr = np.squeeze(resid_fdf_arr)

    return RMCleanResults(
        clean_fdf_spectrum, model_fdf_spectrum, iter_count_arr, resid_fdf_arr
    )


def minor_loop(
    resid_fdf_spectrum_mask: np.ma.MaskedArray,
    phi_arr_radm2: NDArray[np.float64],
    phi_double_arr_radm2: NDArray[np.float64],
    rmsf_spectrum: NDArray[np.float64],
    rmsf_fwhm: float,
    max_iter: int,
    gain: float,
    mask: float,
    threshold: float,
    start_iter: int = 0,
    update_mask: bool = True,
    peak_find_arr: NDArray[np.float64] | None = None,
) -> MinorLoopResults:
    # Trust nothing
    resid_fdf_spectrum_mask = resid_fdf_spectrum_mask.copy()
    resid_fdf_spectrum = resid_fdf_spectrum_mask.data.copy()
    model_fdf_spectrum = np.zeros_like(resid_fdf_spectrum)
    clean_fdf_spectrum = np.zeros_like(resid_fdf_spectrum)
    model_rmsf_spectrum = np.zeros_like(resid_fdf_spectrum)
    rmsf_spectrum = rmsf_spectrum.copy()
    phi_arr_radm2 = phi_arr_radm2.copy()
    mask_arr = ~resid_fdf_spectrum_mask.mask.copy()
    mask_arr_original = mask_arr.copy()
    iter_count = start_iter

    if peak_find_arr is not None:
        peak_find_arr = peak_find_arr.copy()
        peak_find_arr_mask = np.ma.array(peak_find_arr, mask=~mask_arr)
    else:
        peak_find_arr_mask = None

    # Find the index of the peak of the RMSF
    max_rmsf_index = np.nanargmax(np.abs(rmsf_spectrum))

    # Calculate the padding in the sampled RMSF
    # Assumes only integer shifts and symmetric RMSF
    n_phi_pad = int((len(phi_double_arr_radm2) - len(phi_arr_radm2)) / 2)

    logger.info(f"Starting minor loop...cleaning {mask_arr.sum()} pixels")
    for iter_count in range(start_iter, max_iter + 1):
        if resid_fdf_spectrum_mask.mask.all():
            logger.warning(
                f"All channels masked. Exiting loop...performed {iter_count} iterations"
            )
            break
        if iter_count == max_iter:
            logger.warning(
                f"Max iterations reached. Exiting loop...performed {iter_count} iterations"
            )
            break
        if np.ma.max(np.ma.abs(resid_fdf_spectrum_mask)) < threshold:
            logger.info(
                f"Threshold reached. Exiting loop...performed {iter_count} iterations"
            )
            break
        # Get the absolute peak channel, values and Faraday depth
        if peak_find_arr_mask is not None:
            peak_fdf_index = np.ma.argmax(np.abs(peak_find_arr_mask))
        else:
            peak_fdf_index = np.ma.argmax(np.abs(resid_fdf_spectrum_mask))
        peak_fdf = resid_fdf_spectrum_mask[peak_fdf_index]
        peak_rm = phi_arr_radm2[peak_fdf_index]

        # A clean component is "loop-gain * peak_fdf
        clean_component = gain * peak_fdf
        model_fdf_spectrum[peak_fdf_index] += clean_component

        # At which channel is the clean_component located at in the RMSF?
        peak_rmsf_index = peak_fdf_index + n_phi_pad

        # Shift the RMSF & clip so that its peak is centred above this clean_component
        shifted_rmsf_spectrum = np.roll(
            rmsf_spectrum, peak_rmsf_index - max_rmsf_index
        )[n_phi_pad:-n_phi_pad]
        model_rmsf_spectrum += clean_component * shifted_rmsf_spectrum

        # Subtract the product of the clean_component shifted RMSF from the residual FDF
        resid_fdf_spectrum -= clean_component * shifted_rmsf_spectrum
        if peak_find_arr is not None:
            peak_find_arr -= np.abs(clean_component * shifted_rmsf_spectrum)

        # Restore the clean_component * a Gaussian to the cleaned FDF
        clean_fdf_spectrum += gaussian(
            x=phi_arr_radm2,
            amplitude=clean_component,
            mean=peak_rm,
            fwhm=rmsf_fwhm,
        )
        # Remake masked residual FDF
        if update_mask:
            mask_arr = np.abs(resid_fdf_spectrum) > mask
            # Mask anything that was previously masked
            mask_arr = mask_arr & mask_arr_original
        resid_fdf_spectrum_mask = np.ma.array(resid_fdf_spectrum, mask=~mask_arr)
        if peak_find_arr_mask is not None:
            peak_find_arr_mask = np.ma.array(peak_find_arr, mask=~mask_arr)

    return MinorLoopResults(
        clean_fdf_spectrum=clean_fdf_spectrum,
        resid_fdf_spectrum=resid_fdf_spectrum,
        resid_fdf_spectrum_mask=resid_fdf_spectrum_mask,
        model_fdf_spectrum=model_fdf_spectrum,
        model_rmsf_spectrum=model_rmsf_spectrum,
        iter_count=iter_count,
    )


def minor_cycle(
    phi_arr_radm2: NDArray[np.float64],
    phi_double_arr_radm2: NDArray[np.float64],
    dirty_fdf_spectrum: NDArray[np.complex128],
    rmsf_spectrum: NDArray[np.complex128],
    rmsf_fwhm: float,
    mask: float,
    threshold: float,
    max_iter: int,
    gain: float,
) -> CleanLoopResults:
    # Initialise arrays to hold the residual FDF, clean components, clean FDF
    resid_fdf_spectrum = dirty_fdf_spectrum.copy()

    mask_arr = np.abs(dirty_fdf_spectrum) > mask
    resid_fdf_spectrum_mask = np.ma.array(resid_fdf_spectrum, mask=~mask_arr)

    initial_loop_results = minor_loop(
        resid_fdf_spectrum_mask=resid_fdf_spectrum_mask,
        phi_arr_radm2=phi_arr_radm2,
        phi_double_arr_radm2=phi_double_arr_radm2,
        rmsf_spectrum=rmsf_spectrum,
        rmsf_fwhm=rmsf_fwhm,
        max_iter=max_iter,
        gain=gain,
        mask=mask,
        threshold=threshold,
        start_iter=0,
        update_mask=True,
    )

    # Deep clean
    # Mask where clean components have been added
    mask_arr = np.abs(initial_loop_results.model_fdf_spectrum) > 0
    resid_fdf_spectrum = initial_loop_results.resid_fdf_spectrum
    resid_fdf_spectrum_mask = np.ma.array(resid_fdf_spectrum, mask=~mask_arr)

    deep_loop_results = minor_loop(
        resid_fdf_spectrum_mask=resid_fdf_spectrum_mask,
        phi_arr_radm2=phi_arr_radm2,
        phi_double_arr_radm2=phi_double_arr_radm2,
        rmsf_spectrum=rmsf_spectrum,
        rmsf_fwhm=rmsf_fwhm,
        max_iter=max_iter,
        gain=gain,
        mask=mask,
        threshold=threshold,
        start_iter=initial_loop_results.iter_count,
        update_mask=False,
    )

    clean_fdf_spectrum = np.squeeze(
        deep_loop_results.clean_fdf_spectrum + initial_loop_results.clean_fdf_spectrum
    )
    resid_fdf_spectrum = np.squeeze(deep_loop_results.resid_fdf_spectrum)
    model_fdf_spectrum = np.squeeze(
        deep_loop_results.model_fdf_spectrum + initial_loop_results.model_fdf_spectrum
    )

    return CleanLoopResults(
        clean_fdf_spectrum=clean_fdf_spectrum,
        resid_fdf_spectrum=resid_fdf_spectrum,
        model_fdf_spectrum=model_fdf_spectrum,
        iter_count=deep_loop_results.iter_count,
    )
