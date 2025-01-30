"""RM-clean utils"""

from __future__ import annotations

import logging
from functools import partial
from typing import Callable, Literal, NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import convolve
from tqdm.auto import tqdm

from rm_lite.utils.fitting import (
    fit_rmsf,
    fwhm_to_sigma,
    gaussian,
    gaussian_integrand,
    unit_centred_gaussian,
)
from rm_lite.utils.logging import TqdmToLogger, logger
from rm_lite.utils.synthesis import compute_rmsf_params

TQDM_OUT = TqdmToLogger(logger, level=logging.INFO)


class RMCleanResults(NamedTuple):
    """Results of the RM-CLEAN calculation"""

    clean_fdf_arr: NDArray[np.float64]
    """The cleaned Faraday dispersion function cube"""
    model_fdf_arr: NDArray[np.float64]
    """The clean components cube"""
    clean_iter_arr: NDArray[np.float64]
    """The number of iterations for each pixel"""
    resid_fdf_arr: NDArray[np.float64]
    """The residual Faraday dispersion function cube"""


class CleanLoopResults(NamedTuple):
    """Results of the RM-CLEAN loop"""

    clean_fdf_spectrum: NDArray[np.float64]
    """The cleaned Faraday dispersion function cube"""
    resid_fdf_spectrum: NDArray[np.float64]
    """The residual Faraday dispersion function cube"""
    model_fdf_spectrum: NDArray[np.float64]
    """The clean components cube"""
    iter_count: int
    """The number of iterations"""


class MinorLoopResults(NamedTuple):
    """Results of the RM-CLEAN minor loop"""

    clean_fdf_spectrum: NDArray[np.float64]
    """The cleaned Faraday dispersion function cube"""
    resid_fdf_spectrum: NDArray[np.float64]
    """The residual Faraday dispersion function cube"""
    resid_fdf_spectrum_mask: np.ma.MaskedArray
    """The masked residual Faraday dispersion function cube"""
    model_fdf_spectrum: NDArray[np.float64]
    """The clean components cube"""
    model_rmsf_spectrum: NDArray[np.float64]
    """ Model * RMSF """
    iter_count: int
    """The number of iterations"""


def restore_fdf(
    model_fdf_spectrum: NDArray[np.float64],
    phi_double_arr_radm2: NDArray[np.float64],
    fwhm_rmsf: float,
) -> NDArray[np.float64]:
    clean_beam = unit_centred_gaussian(
        x=phi_double_arr_radm2,
        fwhm=fwhm_rmsf,
    ) / gaussian_integrand(amplitude=1, fwhm=fwhm_rmsf)
    restored_fdf = np.convolve(
        model_fdf_spectrum.real, clean_beam, mode="valid"
    ) + 1j * np.convolve(model_fdf_spectrum.imag, clean_beam, mode="valid")
    return restored_fdf[1:-1]


def rmclean(
    dirty_fdf_arr: NDArray[np.float64],
    phi_arr_radm2: NDArray[np.float64],
    rmsf_arr: NDArray[np.float64],
    phi_double_arr_radm2: NDArray[np.float64],
    fwhm_rmsf_arr: NDArray[np.float64],
    mask: float,
    threshold: float,
    max_iter: int = 1000,
    gain: float = 0.1,
    mask_arr: NDArray[np.float64] | None = None,
) -> RMCleanResults:
    _bad_result = RMCleanResults(
        clean_fdf_arr=dirty_fdf_arr,
        model_fdf_arr=np.zeros_like(dirty_fdf_arr),
        clean_iter_arr=np.zeros_like(phi_arr_radm2),
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
    dirty_fdf_spectrum: NDArray[np.float64],
    rmsf_spectrum: NDArray[np.float64],
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


@np.vectorize
def _scale_bias_function(
    scale: float,
    scale_0: float,
    scale_bias: float,
) -> float:
    """Offringa et al. (2017) scale-bias function.

    Args:
        scale (float): Scale parameter (relative to PSF FWHM)
        scale_0 (float): The first non-zero scale parameter
        scale_bias (float): The scale-bias parameter

    Returns:
        float: Weighting factor per scale
    """
    if scale == 0:
        return 1.0
    return scale_bias ** -(-1 - np.log2(scale / scale_0))


def scale_bias_function(
    scales: NDArray[np.float64],
    scale_bias: float,
) -> NDArray[np.float64]:
    """Offringa et al. (2017) scale-bias function.

    Args:
        scales (NDArray[np.float64]): Scale parameters (relative to PSF FWHM)
        scale_bias (float): The scale-bias parameter

    Returns:
        NDArray[np.float64]: Weighting factors per scale
    """
    if len(scales) == 1:
        return np.ones_like(scales)
    return _scale_bias_function(scales, scale_0=scales[1], scale_bias=scale_bias)


def scale_bias_function_cornwell(scales: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1 - 0.6 * scales / scales.max()


def hanning(x_arr: NDArray[np.float64], length: float) -> NDArray[np.float64]:
    """Hanning window function.

    Args:
        x_arr (NDArray[np.float64]): Array of x values
        length (float): Length of the window

    Returns:
        NDArray[np.float64]: Hanning window function array
    """
    han = (1 / length) * np.cos(np.pi * x_arr / length) ** 2
    return np.where(np.abs(x_arr) < length / 2, han, 0)


def tapered_quad_kernel_function(
    phi_double_arr_radm2: NDArray[np.float64],
    scale: float,
    rmsf_fwhm: float,
    sum_normalised: bool = True,
) -> NDArray[np.float64]:
    """Tapered quadratic kernel function.

    Args:
        phi_double_arr_radm2 (NDArray[np.float64]): Phi array in rad/m^2
        scale (float): Scale (in FWHM units)
        rmsf_fwhm (float): RMSF FWHM in rad/m^2

    Returns:
        NDArray[np.float64]: Kernel function array (sum normalised)
    """
    scale_radm2 = scale * rmsf_fwhm
    kernel = hanning(phi_double_arr_radm2, scale_radm2) * (
        1 - (np.abs(phi_double_arr_radm2) / scale_radm2) ** 2
    )
    if sum_normalised:
        kernel /= kernel.sum()
    else:
        kernel /= kernel.max()
    return kernel


def gaussian_scale_kernel_function(
    phi_double_arr_radm2: NDArray[np.float64],
    scale: float,
    rmsf_fwhm: float,
    sum_normalised: bool = True,
) -> NDArray[np.float64]:
    """Gaussian scale kernel function.

    Args:
        phi_double_arr_radm2 (NDArray[np.float64]): Phi array in rad/m^2
        scale (float): Scale (in FWHM units)
        rmsf_fwhm (float): RMSF FWHM in rad/m^2

    Returns:
        NDArray[np.float64]: Kernel function array (sum normalised)
    """
    rmsf_sigma = fwhm_to_sigma(rmsf_fwhm)
    sigma = (3 / 16) * scale * rmsf_sigma
    kernel = unit_centred_gaussian(
        x=phi_double_arr_radm2,
        stddev=sigma,
    )
    if sum_normalised:
        kernel /= kernel.sum()
    else:
        kernel /= kernel.max()
    return kernel


KERNEL_FUNCS: dict[str, Callable] = {
    "tapered_quad": tapered_quad_kernel_function,
    "gaussian": gaussian_scale_kernel_function,
}


def convolve_fdf_scale(
    scale: float,
    fwhm: float,
    fdf_arr: NDArray[np.float64],
    phi_double_arr_radm2: NDArray[np.float64],
    kernel: Literal["tapered_quad", "gaussian"] = "gaussian",
    sum_normalised: bool = True,
) -> NDArray[np.float64]:
    """Convolve the FDF with a Gaussian kernel.

    Args:
        scale (float): Scale parameter (relative to PSF FWHM)
        fwhm (float): FWHM of the RMSF
        fdf_arr (NDArray[np.float64]): FDF array (complex)
        phi_double_arr_radm2 (NDArray[np.float64]): Double-length Faraday depth array (rad/m^2)
        kernel (Literal["tapered_quad", "gaussian"]): Kernel function

    Raises:
        ValueError: If an invalid normalization method is provided

    Returns:
        NDArray[np.float64]: Convolved FDF array
    """
    if scale == 0:
        return fdf_arr
    kernel_func = KERNEL_FUNCS.get(kernel, gaussian_scale_kernel_function)
    kernel_func_partial = partial(kernel_func, sum_normalised=sum_normalised)
    kernel_arr = kernel_func_partial(phi_double_arr_radm2, scale, fwhm)

    if np.iscomplexobj(fdf_arr):
        kernel_arr = kernel_arr * (1 + 1j) / np.sqrt(2)

    mode = "reflect"
    conv_spec = convolve(fdf_arr, kernel_arr, mode=mode)
    if mode == "valid":
        conv_spec = conv_spec[1:-1]

    assert len(conv_spec) == len(fdf_arr), "Convolved FDF has wrong length."

    return conv_spec


def find_significant_scale(
    scales: NDArray[np.float64],
    scale_bias: float,
    fdf_arr: NDArray[np.float64],
    phi_double_arr_radm2: NDArray[np.float64],
    fwhm: float,
    kernel: Literal["tapered_quad", "gaussian"] = "gaussian",
) -> tuple[float, float]:
    scale_parameters = scale_bias_function(scales, scale_bias)
    # scale_parameters = scale_bias_function_cornwell(scales)
    peaks = np.zeros_like(scales)
    for i, scale in enumerate(scales):
        fdf_conv = convolve_fdf_scale(
            scale=scale,
            fdf_arr=fdf_arr,
            fwhm=fwhm,
            phi_double_arr_radm2=phi_double_arr_radm2,
            kernel=kernel,
            sum_normalised=False,
        )
        peak = np.max(np.abs(fdf_conv))
        peaks[i] = peak
    scale_norm = scales.copy()
    scale_norm[scales == 0] = 1
    activated_index = np.argmax(peaks * scale_parameters / scale_norm)
    return scales[activated_index], scale_parameters[activated_index]


def multiscale_minor_loop(
    scales: NDArray[np.float64],
    scale_bias: float,
    resid_fdf_spectrum_mask: np.ma.MaskedArray,
    phi_arr_radm2: NDArray[np.float64],
    phi_double_arr_radm2: NDArray[np.float64],
    rmsf_spectrum: NDArray[np.float64],
    rmsf_fwhm: float,
    max_iter: int,
    max_iter_sub_minor: int,
    gain: float,
    mask: float,
    threshold: float,
    start_iter: int = 0,
    update_mask: bool = True,
    kernel: Literal["tapered_quad", "gaussian"] = "gaussian",
) -> MinorLoopResults:
    # Trust nothing
    resid_fdf_spectrum_mask = resid_fdf_spectrum_mask.copy()
    resid_fdf_spectrum = resid_fdf_spectrum_mask.data.copy()
    model_fdf_spectrum = np.zeros_like(resid_fdf_spectrum)
    clean_fdf_spectrum = np.zeros_like(resid_fdf_spectrum)
    rmsf_spectrum = rmsf_spectrum.copy()
    phi_arr_radm2 = phi_arr_radm2.copy()
    mask_arr = ~resid_fdf_spectrum_mask.mask.copy()
    iter_count = start_iter

    logger.info(f"Starting multiscale cycles...cleaning {mask_arr.sum()} pixels")
    for iter_count in range(start_iter, max_iter + 1):
        # Break conditions
        if resid_fdf_spectrum_mask.mask.all():
            logger.warning(
                f"All channels masked. Exiting loop...performed {iter_count} M-S iterations"
            )
            break
        if iter_count == max_iter:
            logger.warning(
                f"Max iterations reached. Exiting loop...performed {iter_count} M-S iterations"
            )
            break
        if np.ma.max(np.ma.abs(resid_fdf_spectrum_mask)) < threshold:
            logger.info(
                f"Threshold reached. Exiting loop...performed {iter_count} M-S iterations"
            )
            break

        activated_scale, scale_parameter = find_significant_scale(
            scales=scales,
            scale_bias=scale_bias,
            fdf_arr=np.abs(resid_fdf_spectrum),
            fwhm=rmsf_fwhm,
            phi_double_arr_radm2=phi_double_arr_radm2,
        )
        # activated_scale = 20
        logger.info(f"Cleaning activated scale: {activated_scale}")
        if activated_scale == 0:
            resid_fdf_spectrum_conv = resid_fdf_spectrum.copy()
            # peak_find_arr = np.abs(resid_fdf_spectrum)
        else:
            resid_fdf_spectrum_conv_abs = convolve_fdf_scale(
                scale=activated_scale,
                fdf_arr=np.abs(resid_fdf_spectrum),
                fwhm=rmsf_fwhm,
                phi_double_arr_radm2=phi_double_arr_radm2,
                kernel=kernel,
                sum_normalised=True,
            )
            resid_fdf_spectrum_conv = resid_fdf_spectrum_conv_abs * np.exp(
                1j * np.angle(resid_fdf_spectrum)
            )
            # peak_find_arr = convolve_fdf_scale(
            #     scale=activated_scale,
            #     fdf_arr=np.abs(resid_fdf_spectrum),
            #     fwhm=rmsf_fwhm,
            #     phi_double_arr_radm2=phi_double_arr_radm2,
            #     kernel=kernel,
            # )
        # resid_fdf_spectrum_conv *= scale_parameter

        if activated_scale == 0:
            rmsf_spectrum_conv = rmsf_spectrum.copy()
        else:
            rmsf_spectrum_conv_abs = convolve_fdf_scale(
                scale=activated_scale,
                fdf_arr=np.abs(rmsf_spectrum),
                fwhm=rmsf_fwhm,
                phi_double_arr_radm2=phi_double_arr_radm2,
                kernel=kernel,
                sum_normalised=True,
            )
            rmsf_spectrum_conv = rmsf_spectrum_conv_abs * np.exp(
                1j * np.angle(rmsf_spectrum)
            )

        scale_factor = np.nanmax(np.abs(rmsf_spectrum_conv))
        rmsf_spectrum_conv /= scale_factor
        resid_fdf_spectrum_conv /= scale_factor

        # Redo
        # rmsf_spectrum_conv = convolve_fdf_scale(
        #     scale=activated_scale,
        #     fdf_arr=rmsf_spectrum,
        #     fwhm=rmsf_fwhm,
        #     phi_double_arr_radm2=phi_double_arr_radm2,
        #     kernel=kernel,
        # )
        # rmsf_spectrum_conv /= np.nanmax(np.abs(rmsf_spectrum_conv))

        if update_mask:
            mask_arr = np.abs(resid_fdf_spectrum_conv) > mask
        resid_fdf_spectrum_mask_conv = np.ma.array(
            resid_fdf_spectrum_conv, mask=~mask_arr
        )

        conv_fwhm = fit_rmsf(
            np.abs(rmsf_spectrum_conv),
            phi_double_arr_radm2=phi_double_arr_radm2,
            fwhm_rmsf_radm2=rmsf_fwhm * activated_scale,
        )
        sub_minor_results = minor_loop(
            resid_fdf_spectrum_mask=resid_fdf_spectrum_mask_conv,
            phi_arr_radm2=phi_arr_radm2,
            phi_double_arr_radm2=phi_double_arr_radm2,
            rmsf_spectrum=rmsf_spectrum_conv,
            rmsf_fwhm=float(conv_fwhm),
            # rmsf_fwhm=rmsf_fwhm * activated_scale,
            max_iter=max_iter_sub_minor,
            gain=gain,
            mask=mask * activated_scale,
            threshold=threshold,
            start_iter=iter_count,
            update_mask=update_mask,
            # peak_find_arr=peak_find_arr,
        )

        # Convolve the clean components with the RMSF
        clean_deltas = sub_minor_results.model_fdf_spectrum
        iter_count += sub_minor_results.iter_count  # noqa: PLW2901
        if activated_scale == 0:
            clean_model = clean_deltas
        else:
            clean_model = convolve_fdf_scale(
                scale=activated_scale,
                fwhm=rmsf_fwhm,
                fdf_arr=clean_deltas,
                phi_double_arr_radm2=phi_double_arr_radm2,
                kernel=kernel,
                sum_normalised=True,
            )
        model_fdf_spectrum += clean_model

        clean_spectrum = restore_fdf(
            model_fdf_spectrum=model_fdf_spectrum,
            phi_double_arr_radm2=phi_double_arr_radm2,
            fwhm_rmsf=rmsf_fwhm,
        )
        shifted_rmsf = sub_minor_results.model_rmsf_spectrum
        # shifted_rmsf = np.convolve(
        #     clean_model,
        #     rmsf_spectrum / gaussian_integrand(1, fwhm=rmsf_fwhm),
        #     mode="valid",
        # )[1:-1]

        clean_fdf_spectrum += clean_spectrum
        resid_fdf_spectrum -= shifted_rmsf

        if update_mask:
            mask_arr = np.abs(resid_fdf_spectrum) > mask

        resid_fdf_spectrum_mask = np.ma.array(resid_fdf_spectrum, mask=~mask_arr)

    return MinorLoopResults(
        clean_fdf_spectrum=clean_fdf_spectrum,
        resid_fdf_spectrum=resid_fdf_spectrum,
        resid_fdf_spectrum_mask=resid_fdf_spectrum_mask,
        model_fdf_spectrum=model_fdf_spectrum,
        model_rmsf_spectrum=np.zeros_like(model_fdf_spectrum),
        iter_count=iter_count,
    )


def multiscale_cycle(
    scales: NDArray[np.float64],
    scale_bias: float,
    phi_arr_radm2: NDArray[np.float64],
    phi_double_arr_radm2: NDArray[np.float64],
    dirty_fdf_spectrum: NDArray[np.float64],
    rmsf_spectrum: NDArray[np.float64],
    rmsf_fwhm: float,
    mask: float,
    threshold: float,
    max_iter: int,
    max_iter_sub_minor: int,
    gain: float,
    kernel: Literal["tapered_quad", "gaussian"] = "gaussian",
) -> CleanLoopResults:
    # Initialise arrays to hold the residual FDF, clean components, clean FDF
    resid_fdf_spectrum = dirty_fdf_spectrum.copy()
    mask_arr = np.abs(resid_fdf_spectrum) > mask
    resid_fdf_spectrum_mask = np.ma.array(resid_fdf_spectrum, mask=~mask_arr)

    initial_loop_results = multiscale_minor_loop(
        scales=scales,
        scale_bias=scale_bias,
        resid_fdf_spectrum_mask=resid_fdf_spectrum_mask,
        phi_arr_radm2=phi_arr_radm2,
        phi_double_arr_radm2=phi_double_arr_radm2,
        rmsf_spectrum=rmsf_spectrum,
        rmsf_fwhm=rmsf_fwhm,
        max_iter_sub_minor=max_iter_sub_minor,
        max_iter=max_iter,
        gain=gain,
        mask=mask,
        threshold=threshold,
        start_iter=0,
        update_mask=True,
        kernel=kernel,
    )

    # Deep clean
    # Mask where clean components have been added
    mask_arr = np.abs(initial_loop_results.model_fdf_spectrum) > 0
    resid_fdf_spectrum = initial_loop_results.resid_fdf_spectrum
    resid_fdf_spectrum_mask = np.ma.array(resid_fdf_spectrum, mask=~mask_arr)

    logger.info(f"Starting deep clean...cleaning {mask_arr.sum()} pixels")
    deep_loop_results = multiscale_minor_loop(
        scales=scales,
        scale_bias=scale_bias,
        resid_fdf_spectrum_mask=resid_fdf_spectrum_mask,
        phi_arr_radm2=phi_arr_radm2,
        phi_double_arr_radm2=phi_double_arr_radm2,
        rmsf_spectrum=rmsf_spectrum,
        rmsf_fwhm=rmsf_fwhm,
        max_iter=max_iter,
        max_iter_sub_minor=max_iter_sub_minor,
        gain=gain,
        mask=mask,
        threshold=threshold,
        start_iter=0,
        update_mask=False,
        kernel=kernel,
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


def mutliscale_rmclean(
    freq_arr_hz: NDArray[np.float64],
    dirty_fdf_arr: NDArray[np.float64],
    phi_arr_radm2: NDArray[np.float64],
    rmsf_arr: NDArray[np.float64],
    phi_double_arr_radm2: NDArray[np.float64],
    fwhm_rmsf_arr: NDArray[np.float64],
    mask: float,
    threshold: float,
    max_iter: int = 1000,
    max_iter_sub_minor: int = 10_000,
    gain: float = 0.1,
    scale_bias: float = 0.9,
    scales: NDArray[np.float64] | None = None,
    mask_arr: NDArray[np.float64] | None = None,
    kernel: Literal["tapered_quad", "gaussian"] = "gaussian",
) -> RMCleanResults:
    _bad_result = RMCleanResults(
        clean_fdf_arr=dirty_fdf_arr,
        model_fdf_arr=np.zeros_like(dirty_fdf_arr),
        clean_iter_arr=np.zeros_like(phi_arr_radm2),
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

    # Compute the scale parameters
    rmsf_params = compute_rmsf_params(
        freq_arr_hz=freq_arr_hz,
        weight_arr=np.ones_like(freq_arr_hz),
        super_resolution=False,
    )
    max_scale = rmsf_params.phi_max_scale / rmsf_params.rmsf_fwhm_meas
    logger.info(
        f"Maximum Faraday scale {rmsf_params.phi_max_scale:0.2f} / (rad/m^2) -- {max_scale:0.2f} / RMSF FWHM."
    )
    if scales is None:
        scales = np.arange(0, max_scale, step=0.1)

    logger.info(f"Using scales: {scales}")

    if scales.max() > max_scale:
        logger.warning(
            f"Maximum scale parameter {scales.max()} is greater than the RMSF max scale {max_scale}."
        )
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
        clean_loop_results = multiscale_cycle(
            scales=scales,
            scale_bias=scale_bias,
            phi_arr_radm2=phi_arr_radm2,
            phi_double_arr_radm2=phi_double_arr_radm2,
            dirty_fdf_spectrum=dirty_fdf_arr[:, yi, xi],
            rmsf_spectrum=rmsf_arr[:, yi, xi],
            rmsf_fwhm=fwhm_rmsf_arr[yi, xi],
            mask=mask,
            threshold=threshold,
            max_iter=max_iter,
            max_iter_sub_minor=max_iter_sub_minor,
            gain=gain,
            kernel=kernel,
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
