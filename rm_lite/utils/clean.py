"""RM-clean utils"""

from __future__ import annotations

import logging
from typing import NamedTuple, TypeVar

import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm

from rm_lite.utils.arrays import nd_to_two_d, two_d_to_nd
from rm_lite.utils.fitting import (
    gaussian,
    gaussian_integrand,
    unit_centred_gaussian,
)
from rm_lite.utils.logging import TqdmToLogger, logger

TQDM_OUT = TqdmToLogger(logger, level=logging.INFO)

DType = TypeVar("DType", bound=np.generic)


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

    def with_options(self, **kwargs):
        as_dict = self._asdict()
        as_dict.update(kwargs)

        return RMCleanResults(**as_dict)


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
    return np.array(restored_fdf[1:-1])


class RMSynthArrays(NamedTuple):
    """Arrays for RM-synthesis"""

    dirty_fdf_arr: NDArray[np.complex128]
    """Dirty Faraday dispersion function array"""
    phi_arr_radm2: NDArray[np.float64]
    """Faraday depth array in rad/m^2"""
    phi_double_arr_radm2: NDArray[np.float64]
    """Double-length Faraday depth array in rad/m^2"""
    rmsf_arr: NDArray[np.complex128]
    """RMSF array"""
    fwhm_rmsf_arr: NDArray[np.float64]
    """FWHM of the RMSF array"""
    fdf_mask_arr: NDArray[np.bool_] | None = None
    """Mask of pixels to clean"""


class RMCleanOptions(NamedTuple):
    """Options for RM-CLEAN"""

    mask: float
    """Masking threshold - pixels below this value are not cleaned"""
    threshold: float
    """Cleaning threshold - stop when all pixels are below this value"""
    max_iter: int = 1000
    """Maximum clean iterations"""
    gain: float = 0.1
    """Clean loop gain"""
    mask_arr: NDArray[np.bool_] | None = None
    """Additional mask of pixels to avoid"""


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

    rm_synth_arrays = RMSynthArrays(
        dirty_fdf_arr=dirty_fdf_arr,
        phi_arr_radm2=phi_arr_radm2,
        rmsf_arr=rmsf_arr,
        phi_double_arr_radm2=phi_double_arr_radm2,
        fwhm_rmsf_arr=fwhm_rmsf_arr,
    )
    clean_options = RMCleanOptions(
        mask=mask,
        threshold=threshold,
        max_iter=max_iter,
        gain=gain,
        mask_arr=mask_arr,
    )

    return _rmclean_nd(rm_synth_arrays, clean_options)


def _rmclean_nd(
    rm_synth_arrays: RMSynthArrays,
    clean_options: RMCleanOptions,
) -> RMCleanResults:
    # Sanity checks on array sizes
    checks: list[tuple[bool, str]] = [
        (
            rm_synth_arrays.phi_arr_radm2.shape[0]
            == rm_synth_arrays.dirty_fdf_arr.shape[0],
            f"'phi_arr_radm2' (size {rm_synth_arrays.phi_arr_radm2.shape[0]}) and 'dirty_fdf_arr' (size {rm_synth_arrays.dirty_fdf_arr.shape[0]}) are not the same length.",
        ),
        (
            rm_synth_arrays.phi_double_arr_radm2.shape[0]
            == rm_synth_arrays.rmsf_arr.shape[0],
            f"Mismatch in 'phi_double_arr_radm2' (size {rm_synth_arrays.phi_double_arr_radm2.shape[0]}) and 'rmsf_arr' (size {rm_synth_arrays.rmsf_arr.shape[0]}) length.",
        ),
        (
            len(rm_synth_arrays.phi_double_arr_radm2)
            >= 2 * len(rm_synth_arrays.phi_arr_radm2),
            f"The Faraday depth of the RMSF (size {len(rm_synth_arrays.phi_double_arr_radm2)}) must be at least twice the FDF (size {len(rm_synth_arrays.phi_arr_radm2)}).",
        ),
        (
            rm_synth_arrays.dirty_fdf_arr.ndim <= 3,
            f"FDF array dimensions ({rm_synth_arrays.dirty_fdf_arr.ndim}) must be <= 3.",
        ),
        (
            rm_synth_arrays.dirty_fdf_arr.ndim == rm_synth_arrays.rmsf_arr.ndim,
            f"The input RMSF (ndim {rm_synth_arrays.rmsf_arr.ndim}) and FDF (ndim {rm_synth_arrays.dirty_fdf_arr.ndim}) must have the same number of axes.",
        ),
        (
            rm_synth_arrays.rmsf_arr.shape[1:]
            == rm_synth_arrays.dirty_fdf_arr.shape[1:],
            f"The xy dimensions of the RMSF {rm_synth_arrays.rmsf_arr.shape[1:]} and FDF {rm_synth_arrays.dirty_fdf_arr.shape[1:]} must match.",
        ),
    ]
    if clean_options.mask_arr is not None:
        checks.append(
            (
                clean_options.mask_arr.shape == rm_synth_arrays.dirty_fdf_arr.shape[1:],
                f"Mask array dimensions {clean_options.mask_arr.shape} must match the xy dimensions of the FDF cube {rm_synth_arrays.dirty_fdf_arr.shape[1:]}.",
            )
        )

    for condition, error_msg in checks:
        if not condition:
            raise ValueError(error_msg)

    # Reshape the arrays to 2D i.e. [phi, x, y] -> [phi, x*y]
    dirty_fdf_arr_2d = nd_to_two_d(rm_synth_arrays.dirty_fdf_arr)
    rmsf_arr_2d = nd_to_two_d(rm_synth_arrays.rmsf_arr)
    iter_count_arr_2d = np.zeros(dirty_fdf_arr_2d.shape[1:], dtype=int)
    fwhm_rmsf_arr_2d = nd_to_two_d(rm_synth_arrays.fwhm_rmsf_arr)

    # Initialise arrays to hold the residual FDF, clean components, clean FDF
    # Residual is initially copies of dirty FDF, so that pixels that are not
    #  processed get correct values (but will be overridden when processed)
    clean_fdf_spectrum_2d = np.zeros_like(dirty_fdf_arr_2d)
    model_fdf_spectrum_2d = np.zeros(dirty_fdf_arr_2d.shape, dtype=complex)
    resid_fdf_arr_2d = dirty_fdf_arr_2d.copy()

    # Loop through the pixels containing a polarised signal
    for pix_idx in tqdm(range(dirty_fdf_arr_2d.shape[1])):
        clean_loop_results = minor_cycle(
            rm_synth_1d_arrays=RMSynthArrays(
                dirty_fdf_arr=resid_fdf_arr_2d[:, pix_idx],
                phi_arr_radm2=rm_synth_arrays.phi_arr_radm2,
                rmsf_arr=rmsf_arr_2d[:, pix_idx],
                phi_double_arr_radm2=rm_synth_arrays.phi_double_arr_radm2,
                fwhm_rmsf_arr=fwhm_rmsf_arr_2d,
                fdf_mask_arr=nd_to_two_d(rm_synth_arrays.fdf_mask_arr)
                if rm_synth_arrays.fdf_mask_arr is not None
                else None,
            ),
            clean_options=clean_options,
        )
        clean_fdf_spectrum_2d[:, pix_idx] = clean_loop_results.clean_fdf_spectrum
        resid_fdf_arr_2d[:, pix_idx] = clean_loop_results.resid_fdf_spectrum
        model_fdf_spectrum_2d[:, pix_idx] = clean_loop_results.model_fdf_spectrum
        iter_count_arr_2d[pix_idx] = clean_loop_results.iter_count

    # Restore the residual to the cleaned FDF (moved outside of loop:
    # will now work for pixels/spectra without clean components)
    clean_fdf_spectrum_2d += resid_fdf_arr_2d

    # Reshape the arrays back to their original shape
    clean_fdf_spectrum = two_d_to_nd(
        clean_fdf_spectrum_2d, rm_synth_arrays.dirty_fdf_arr.shape
    )
    model_fdf_spectrum = two_d_to_nd(
        model_fdf_spectrum_2d, rm_synth_arrays.dirty_fdf_arr.shape
    )
    if rm_synth_arrays.dirty_fdf_arr.shape[1:] == ():
        iter_count_arr = two_d_to_nd(iter_count_arr_2d, (1,))
    else:
        iter_count_arr = two_d_to_nd(
            iter_count_arr_2d, rm_synth_arrays.dirty_fdf_arr.shape[1:]
        )
    resid_fdf_arr = two_d_to_nd(resid_fdf_arr_2d, rm_synth_arrays.dirty_fdf_arr.shape)

    return RMCleanResults(
        clean_fdf_spectrum, model_fdf_spectrum, iter_count_arr, resid_fdf_arr
    )


class MinorLoopArrays(NamedTuple):
    """Arrays for the RM-CLEAN minor loop"""

    resid_fdf_spectrum_mask: np.ma.MaskedArray
    """Residual Faraday dispersion function spectrum"""
    phi_arr_radm2: NDArray[np.float64]
    """Faraday depth array in rad/m^2"""
    phi_double_arr_radm2: NDArray[np.float64]
    """Double-length Faraday depth array in rad/m^2"""
    rmsf_spectrum: NDArray[np.complex128]
    """RMSF spectrum"""
    rmsf_fwhm: float
    """FWHM of the RMSF"""
    peak_find_arr: NDArray[np.float64] | None = None
    """Peak finding array"""

    def with_options(self, **kwargs):
        as_dict = self._asdict()
        as_dict.update(kwargs)

        return MinorLoopArrays(**as_dict)


class MinorLoopOptions(NamedTuple):
    """Options for the RM-CLEAN minor loop"""

    max_iter: int
    """Maximum number of iterations"""
    gain: float
    """Loop gain"""
    mask: float
    """Masking threshold"""
    threshold: float
    """Threshold for stopping the loop"""
    start_iter: int = 0
    """Starting iteration"""
    update_mask: bool = True
    """Update the mask after each iteration"""

    def with_options(self, **kwargs):
        as_dict = self._asdict()
        as_dict.update(kwargs)

        return MinorLoopOptions(**as_dict)


def minor_loop(
    minor_loop_arrays: MinorLoopArrays,
    minor_loop_options: MinorLoopOptions,
) -> MinorLoopResults:
    # Trust nothing
    resid_fdf_spectrum_mask = minor_loop_arrays.resid_fdf_spectrum_mask.copy()
    resid_fdf_spectrum = minor_loop_arrays.resid_fdf_spectrum_mask.data.copy()
    model_fdf_spectrum = np.zeros_like(resid_fdf_spectrum)
    clean_fdf_spectrum = np.zeros_like(resid_fdf_spectrum)
    model_rmsf_spectrum = np.zeros_like(resid_fdf_spectrum)
    rmsf_spectrum = minor_loop_arrays.rmsf_spectrum.copy()
    phi_arr_radm2 = minor_loop_arrays.phi_arr_radm2.copy()
    mask_arr = ~resid_fdf_spectrum_mask.mask.copy()
    mask_arr_original = mask_arr.copy()
    iter_count = int(minor_loop_options.start_iter)

    if minor_loop_arrays.peak_find_arr is not None:
        peak_find_arr = minor_loop_arrays.peak_find_arr.copy()
        peak_find_arr_mask = np.ma.array(
            minor_loop_arrays.peak_find_arr.copy(), mask=~mask_arr
        )
    else:
        peak_find_arr_mask = None

    # Find the index of the peak of the RMSF
    max_rmsf_index = np.nanargmax(np.abs(rmsf_spectrum))

    # Calculate the padding in the sampled RMSF
    # Assumes only integer shifts and symmetric RMSF
    n_phi_pad = int(
        (len(minor_loop_arrays.phi_double_arr_radm2) - len(phi_arr_radm2)) / 2
    )

    logger.info(f"Starting minor loop... {mask_arr.sum()} pixels in the mask")
    for iter_count in range(
        minor_loop_options.start_iter, minor_loop_options.max_iter + 1
    ):
        if resid_fdf_spectrum_mask.mask.all():
            logger.warning(
                f"All channels masked. Exiting loop...performed {iter_count} iterations"
            )
            break
        if iter_count == minor_loop_options.max_iter:
            logger.warning(
                f"Max iterations reached. Exiting loop...performed {iter_count} iterations"
            )
            break
        if np.ma.max(np.ma.abs(resid_fdf_spectrum_mask)) < minor_loop_options.threshold:
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
        clean_component = minor_loop_options.gain * peak_fdf
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
        if minor_loop_arrays.peak_find_arr is not None:
            peak_find_arr -= np.abs(clean_component * shifted_rmsf_spectrum)

        # Restore the clean_component * a Gaussian to the cleaned FDF
        clean_fdf_spectrum += gaussian(
            x=phi_arr_radm2,
            amplitude=clean_component,
            mean=float(peak_rm),
            fwhm=minor_loop_arrays.rmsf_fwhm,
        )
        # Remake masked residual FDF
        if minor_loop_options.update_mask:
            mask_arr = np.abs(resid_fdf_spectrum) > minor_loop_options.mask
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
    rm_synth_1d_arrays: RMSynthArrays,
    clean_options: RMCleanOptions,
) -> CleanLoopResults:
    for array in (
        rm_synth_1d_arrays.dirty_fdf_arr,
        rm_synth_1d_arrays.phi_arr_radm2,
        rm_synth_1d_arrays.phi_double_arr_radm2,
    ):
        if array.ndim != 1:
            msg = "Arrays in minor cycle must be 1D."
            raise ValueError(msg)

    # Initialise arrays to hold the residual FDF, clean components, clean FDF
    resid_fdf_spectrum = rm_synth_1d_arrays.dirty_fdf_arr.copy()

    mask_arr = np.abs(rm_synth_1d_arrays.dirty_fdf_arr) > clean_options.mask

    if rm_synth_1d_arrays.fdf_mask_arr is not None:
        assert rm_synth_1d_arrays.fdf_mask_arr.ndim == 1, (
            "Arrays in minor cycle must be 1D."
        )
        mask_arr = np.logical_and(
            mask_arr,
            rm_synth_1d_arrays.fdf_mask_arr.astype(bool),
        )

    resid_fdf_spectrum_mask = np.ma.array(resid_fdf_spectrum, mask=~mask_arr)

    minor_loop_arrays = MinorLoopArrays(
        resid_fdf_spectrum_mask=resid_fdf_spectrum_mask,
        phi_arr_radm2=rm_synth_1d_arrays.phi_arr_radm2,
        phi_double_arr_radm2=rm_synth_1d_arrays.phi_double_arr_radm2,
        rmsf_spectrum=rm_synth_1d_arrays.rmsf_arr,
        rmsf_fwhm=float(rm_synth_1d_arrays.fwhm_rmsf_arr.squeeze()),
    )

    minor_loop_options = MinorLoopOptions(
        max_iter=clean_options.max_iter,
        gain=clean_options.gain,
        mask=clean_options.mask,
        threshold=clean_options.threshold,
        start_iter=0,
        update_mask=True,
    )

    logger.info("Starting initial minor loop...")
    initial_loop_results = minor_loop(
        minor_loop_arrays=minor_loop_arrays,
        minor_loop_options=minor_loop_options,
    )

    # Deep clean
    # Mask where clean components have been added
    mask_arr = np.abs(initial_loop_results.model_fdf_spectrum) > 0
    resid_fdf_spectrum = initial_loop_results.resid_fdf_spectrum
    resid_fdf_spectrum_mask = np.ma.array(resid_fdf_spectrum, mask=~mask_arr)

    logger.info("Initial loop complete. Starting deep clean...")

    deep_loop_results = minor_loop(
        minor_loop_arrays=minor_loop_arrays.with_options(
            resid_fdf_spectrum_mask=resid_fdf_spectrum_mask
        ),
        minor_loop_options=minor_loop_options.with_options(
            start_iter=initial_loop_results.iter_count,
            update_mask=False,
        ),
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
