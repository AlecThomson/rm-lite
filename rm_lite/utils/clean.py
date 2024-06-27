#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RM-clean utils"""

import logging
from functools import partial
from typing import Callable, Dict, Literal, NamedTuple, Optional, Tuple

import numpy as np
from tqdm.auto import tqdm

from rm_lite.utils.fitting import gaussian, gaussian_integrand, unit_centred_gaussian
from rm_lite.utils.logging import logger, TqdmToLogger
from rm_lite.utils.synthesis import compute_rmsf_params

TQDM_OUT = TqdmToLogger(logger, level=logging.INFO)


class RMCleanResults(NamedTuple):
    """Results of the RM-CLEAN calculation"""

    clean_fdf_array: np.ndarray
    """The cleaned Faraday dispersion function cube"""
    model_fdf_array: np.ndarray
    """The clean components cube"""
    clean_iter_array: np.ndarray
    """The number of iterations for each pixel"""
    resid_fdf_array: np.ndarray
    """The residual Faraday dispersion function cube"""


class CleanLoopResults(NamedTuple):
    """Results of the RM-CLEAN loop"""

    clean_fdf_spectrum: np.ndarray
    """The cleaned Faraday dispersion function cube"""
    resid_fdf_spectrum: np.ndarray
    """The residual Faraday dispersion function cube"""
    model_fdf_spectrum: np.ndarray
    """The clean components cube"""
    iter_count: int
    """The number of iterations"""


class MinorLoopResults(NamedTuple):
    """Results of the RM-CLEAN minor loop"""

    clean_fdf_spectrum: np.ndarray
    """The cleaned Faraday dispersion function cube"""
    resid_fdf_spectrum: np.ndarray
    """The residual Faraday dispersion function cube"""
    resid_fdf_spectrum_mask: np.ma.MaskedArray
    """The masked residual Faraday dispersion function cube"""
    model_fdf_spectrum: np.ndarray
    """The clean components cube"""
    model_rmsf_spectrum: np.ndarray
    """ Model * RMSF """
    iter_count: int
    """The number of iterations"""


def restore_fdf(
    model_fdf_spectrum: np.ndarray,
    phi_double_arr_radm2: np.ndarray,
    fwhm_rmsf: float,
) -> np.ndarray:
    clean_beam = unit_centred_gaussian(
        x=phi_double_arr_radm2,
        fwhm=fwhm_rmsf,
    ) / gaussian_integrand(amplitude=1, fwhm=fwhm_rmsf)
    restored_fdf = np.convolve(
        model_fdf_spectrum.real, clean_beam, mode="valid"
    ) + 1j * np.convolve(model_fdf_spectrum.imag, clean_beam, mode="valid")
    return restored_fdf[1:-1]


def rmclean(
    dirty_fdf_array: np.ndarray,
    phi_arr_radm2: np.ndarray,
    rmsf_array: np.ndarray,
    phi_double_arr_radm2: np.ndarray,
    fwhm_rmsf_arr: np.ndarray,
    mask: float,
    threshold: float,
    max_iter: int = 1000,
    gain: float = 0.1,
    mask_array: Optional[np.ndarray] = None,
) -> RMCleanResults:
    _bad_result = RMCleanResults(
        clean_fdf_array=dirty_fdf_array,
        model_fdf_array=np.zeros_like(dirty_fdf_array),
        clean_iter_array=np.zeros_like(phi_arr_radm2),
        resid_fdf_array=dirty_fdf_array,
    )
    # Sanity checks on array sizes
    n_phi = phi_arr_radm2.shape[0]
    if n_phi != dirty_fdf_array.shape[0]:
        logger.error("'phi_arr_radm2' and 'dirty_fdf_array' are not the same length.")
        return _bad_result
    n_phi2 = phi_double_arr_radm2.shape[0]
    if not n_phi2 == rmsf_array.shape[0]:
        logger.error("missmatch in 'phi_double_arr_radm2' and 'rmsf_array' length.")
        return _bad_result
    if not (n_phi2 >= 2 * n_phi):
        logger.error("the Faraday depth of the RMSF must be twice the FDF.")
        return _bad_result
    n_dimension = len(dirty_fdf_array.shape)
    if not n_dimension <= 3:
        logger.error("FDF array dimensions must be <= 3.")
        return _bad_result
    if not n_dimension == len(rmsf_array.shape):
        logger.error("the input RMSF and FDF must have the same number of axes.")
        return _bad_result
    if not rmsf_array.shape[1:] == dirty_fdf_array.shape[1:]:
        logger.error("the xy dimesions of the RMSF and FDF must match.")
        return _bad_result
    if mask_array is not None:
        if not mask_array.shape == dirty_fdf_array.shape[1:]:
            logger.error("pixel mask must match xy dimension of FDF cube.")
            return _bad_result
    else:
        mask_array = np.ones(dirty_fdf_array.shape[1:], dtype=bool)

    # Reshape the FDF & RMSF array to 3 dimensions and mask array to 2
    if n_dimension == 1:
        dirty_fdf_array = np.reshape(dirty_fdf_array, (dirty_fdf_array.shape[0], 1, 1))
        rmsf_array = np.reshape(rmsf_array, (rmsf_array.shape[0], 1, 1))
        mask_array = np.reshape(mask_array, (1, 1))
        fwhm_rmsf_arr = np.reshape(fwhm_rmsf_arr, (1, 1))
    elif n_dimension == 2:
        dirty_fdf_array = np.reshape(
            dirty_fdf_array, list(dirty_fdf_array.shape[:2]) + [1]
        )
        rmsf_array = np.reshape(rmsf_array, list(rmsf_array.shape[:2]) + [1])
        mask_array = np.reshape(mask_array, (dirty_fdf_array.shape[1], 1))
        fwhm_rmsf_arr = np.reshape(fwhm_rmsf_arr, (dirty_fdf_array.shape[1], 1))

    iter_count_array = np.zeros_like(mask_array, dtype=int)

    # Determine which pixels have components above the cutoff
    abs_fdf_cube = np.abs(np.nan_to_num(dirty_fdf_array))
    cutoff_mask = np.where(np.max(abs_fdf_cube, axis=0) >= mask, 1, 0)
    pixels_to_clean = np.rot90(np.where(cutoff_mask > 0))

    num_pixels = dirty_fdf_array.shape[-1] * dirty_fdf_array.shape[-2]
    num_pixels_clean = len(pixels_to_clean)
    logger.info("Cleaning {:}/{:} spectra.".format(num_pixels_clean, num_pixels))

    # Initialise arrays to hold the residual FDF, clean components, clean FDF
    # Residual is initially copies of dirty FDF, so that pixels that are not
    #  processed get correct values (but will be overridden when processed)
    clean_fdf_spectrum = np.zeros_like(dirty_fdf_array)
    model_fdf_spectrum = np.zeros(dirty_fdf_array.shape, dtype=complex)
    resid_fdf_array = dirty_fdf_array.copy()

    # Loop through the pixels containing a polarised signal
    for yi, xi in tqdm(pixels_to_clean):
        clean_loop_results = minor_cycle(
            phi_arr_radm2=phi_arr_radm2,
            phi_double_arr_radm2=phi_double_arr_radm2,
            dirty_fdf_spectrum=dirty_fdf_array[:, yi, xi],
            rmsf_spectrum=rmsf_array[:, yi, xi],
            rmsf_fwhm=fwhm_rmsf_arr[yi, xi],
            mask=mask,
            threshold=threshold,
            max_iter=max_iter,
            gain=gain,
        )
        clean_fdf_spectrum[:, yi, xi] = clean_loop_results.clean_fdf_spectrum
        resid_fdf_array[:, yi, xi] = clean_loop_results.resid_fdf_spectrum
        model_fdf_spectrum[:, yi, xi] = clean_loop_results.model_fdf_spectrum
        iter_count_array[yi, xi] = clean_loop_results.iter_count

    # Restore the residual to the CLEANed FDF (moved outside of loop:
    # will now work for pixels/spectra without clean components)
    clean_fdf_spectrum += resid_fdf_array

    # Remove redundant dimensions
    clean_fdf_spectrum = np.squeeze(clean_fdf_spectrum)
    model_fdf_spectrum = np.squeeze(model_fdf_spectrum)
    iter_count_array = np.squeeze(iter_count_array)
    resid_fdf_array = np.squeeze(resid_fdf_array)

    return RMCleanResults(
        clean_fdf_spectrum, model_fdf_spectrum, iter_count_array, resid_fdf_array
    )


def minor_loop(
    resid_fdf_spectrum_mask: np.ma.MaskedArray,
    phi_arr_radm2: np.ndarray,
    phi_double_arr_radm2: np.ndarray,
    rmsf_spectrum: np.ndarray,
    rmsf_fwhm: float,
    max_iter: int,
    gain: float,
    mask: float,
    threshold: float,
    start_iter: int = 0,
    update_mask: bool = True,
) -> MinorLoopResults:
    # Trust nothing
    resid_fdf_spectrum_mask = resid_fdf_spectrum_mask.copy()
    resid_fdf_spectrum = resid_fdf_spectrum_mask.data.copy()
    model_fdf_spectrum = np.zeros_like(resid_fdf_spectrum)
    clean_fdf_spectrum = np.zeros_like(resid_fdf_spectrum)
    model_rmsf_spectrum = np.zeros_like(resid_fdf_spectrum)
    rmsf_spectrum = rmsf_spectrum.copy()
    phi_arr_radm2 = phi_arr_radm2.copy()
    mask_array = ~resid_fdf_spectrum_mask.mask.copy()
    iter_count = start_iter

    # Find the index of the peak of the RMSF
    max_rmsf_index = np.nanargmax(np.abs(rmsf_spectrum))

    # Calculate the padding in the sampled RMSF
    # Assumes only integer shifts and symmetric RMSF
    n_phi_pad = int((len(phi_double_arr_radm2) - len(phi_arr_radm2)) / 2)

    logger.info(f"Starting minor loop...cleaning {mask_array.sum()} pixels")
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
                f"Thresold reached. Exiting loop...performed {iter_count} iterations"
            )
            break
        # Get the absolute peak channel, values and Faraday depth
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

        # Restore the clean_component * a Gaussian to the cleaned FDF
        clean_fdf_spectrum += gaussian(
            x=phi_arr_radm2,
            amplitude=clean_component,
            mean=peak_rm,
            fwhm=rmsf_fwhm,
        )
        # Remake masked residual FDF
        if update_mask:
            mask_array = np.abs(resid_fdf_spectrum) > mask
        resid_fdf_spectrum_mask = np.ma.array(resid_fdf_spectrum, mask=~mask_array)

    return MinorLoopResults(
        clean_fdf_spectrum=clean_fdf_spectrum,
        resid_fdf_spectrum=resid_fdf_spectrum,
        resid_fdf_spectrum_mask=resid_fdf_spectrum_mask,
        model_fdf_spectrum=model_fdf_spectrum,
        model_rmsf_spectrum=model_rmsf_spectrum,
        iter_count=iter_count,
    )


def minor_cycle(
    phi_arr_radm2: np.ndarray,
    phi_double_arr_radm2: np.ndarray,
    dirty_fdf_spectrum: np.ndarray,
    rmsf_spectrum: np.ndarray,
    rmsf_fwhm: float,
    mask: float,
    threshold: float,
    max_iter: int,
    gain: float,
) -> CleanLoopResults:
    # Initialise arrays to hold the residual FDF, clean components, clean FDF
    resid_fdf_spectrum = dirty_fdf_spectrum.copy()

    mask_array = np.abs(dirty_fdf_spectrum) > mask
    resid_fdf_spectrum_mask = np.ma.array(resid_fdf_spectrum, mask=~mask_array)

    inital_loop_results = minor_loop(
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
    mask_array = np.abs(inital_loop_results.model_fdf_spectrum) > 0
    resid_fdf_spectrum = inital_loop_results.resid_fdf_spectrum
    resid_fdf_spectrum_mask = np.ma.array(resid_fdf_spectrum, mask=~mask_array)

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
        start_iter=inital_loop_results.iter_count,
        update_mask=False,
    )

    clean_fdf_spectrum = np.squeeze(
        deep_loop_results.clean_fdf_spectrum + inital_loop_results.clean_fdf_spectrum
    )
    resid_fdf_spectrum = np.squeeze(deep_loop_results.resid_fdf_spectrum)
    model_fdf_spectrum = np.squeeze(
        deep_loop_results.model_fdf_spectrum + inital_loop_results.model_fdf_spectrum
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
    return scale_bias ** (-1 - np.log2(scale / scale_0))


def scale_bias_function(
    scales: np.ndarray,
    scale_bias: float,
) -> np.ndarray:
    """Offringa et al. (2017) scale-bias function.

    Args:
        scales (np.ndarray): Scale parameters (relative to PSF FWHM)
        scale_bias (float): The scale-bias parameter

    Returns:
        np.ndarray: Weighting factors per scale
    """
    if len(scales) == 1:
        return np.ones_like(scales)
    return _scale_bias_function(scales, scale_0=scales[1], scale_bias=scale_bias)


def hanning(x_array: np.ndarray, length: float) -> np.ndarray:
    """Hanning window function.

    Args:
        x_array (np.ndarray): Array of x values
        length (float): Length of the window

    Returns:
        np.ndarray: Hanning window function array
    """
    han = (1 / length) * np.cos(np.pi * x_array / length) ** 2
    han = np.where(np.abs(x_array) < length / 2, han, 0)
    return han


def tapered_quad_kernel_function(
    phi_double_arr_radm2: np.ndarray,
    scale: float,
    rmsf_fwhm: float,
    sum_normalised: bool = True,
) -> np.ndarray:
    """Tapered quadratic kernel function.

    Args:
        phi_double_arr_radm2 (np.ndarray): Phi array in rad/m^2
        scale (float): Scale (in FWHM units)
        rmsf_fwhm (float): RMSF FWHM in rad/m^2

    Returns:
        np.ndarray: Kernel function array (sum normalised)
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
    phi_double_arr_radm2: np.ndarray,
    scale: float,
    rmsf_fwhm: float,
    sum_normalised: bool = True,
) -> np.ndarray:
    """Gaussian scale kernel function.

    Args:
        phi_double_arr_radm2 (np.ndarray): Phi array in rad/m^2
        scale (float): Scale (in FWHM units)
        rmsf_fwhm (float): RMSF FWHM in rad/m^2

    Returns:
        np.ndarray: Kernel function array (sum normalised)
    """
    sigma = (3 / 16) * scale * rmsf_fwhm
    kernel = unit_centred_gaussian(
        x=phi_double_arr_radm2,
        stddev=sigma,
    )
    if sum_normalised:
        kernel /= kernel.sum()
    else:
        kernel /= kernel.max()
    return kernel


KERNEL_FUNCS: Dict[str, Callable] = {
    "tapered_quad": tapered_quad_kernel_function,
    "gaussian": gaussian_scale_kernel_function,
}


def convolve_fdf_scale(
    scale: float,
    fwhm: float,
    fdf_array: np.ndarray,
    phi_double_arr_radm2: np.ndarray,
    kernel: Literal["tapered_quad", "gaussian"] = "gaussian",
    sum_normalised: bool = True,
) -> np.ndarray:
    """Convolve the FDF with a Gaussian kernel.

    Args:
        scale (float): Scale parameter (relative to PSF FWHM)
        fwhm (float): FWHM of the RMSF
        fdf_array (np.ndarray): FDF array (complex)
        phi_double_arr_radm2 (np.ndarray): Double-length Faraday depth array (rad/m^2)
        kernel (Literal["tapered_quad", "gaussian"]): Kernel function

    Raises:
        ValueError: If an invalid normalization method is provided

    Returns:
        np.ndarray: Convolved FDF array
    """
    if scale == 0:
        return fdf_array
    kernel_func = KERNEL_FUNCS.get(kernel, gaussian_scale_kernel_function)
    kernel_func_partial = partial(kernel_func, sum_normalised=sum_normalised)
    kernel_array = kernel_func_partial(phi_double_arr_radm2, scale, fwhm)

    if len(fdf_array) == len(phi_double_arr_radm2):
        mode = "same"
    else:
        mode = "valid"
    conv_spec = np.convolve(fdf_array.real, kernel_array, mode=mode) + 1j * np.convolve(
        fdf_array.imag, kernel_array, mode=mode
    )
    if mode == "valid":
        conv_spec = conv_spec[1:-1]

    assert len(conv_spec) == len(fdf_array), "Convolved FDF has wrong length."

    return conv_spec


def find_significant_scale(
    scales: np.ndarray,
    scale_parameters: np.ndarray,
    fdf_array: np.ndarray,
    phi_double_arr_radm2: np.ndarray,
    fwhm: float,
    kernel: Literal["tapered_quad", "gaussian"] = "gaussian",
) -> Tuple[float, float]:
    peaks = np.zeros_like(scales)
    for i, scale in enumerate(scales):
        fdf_conv = convolve_fdf_scale(
            scale=scale,
            fdf_array=fdf_array,
            fwhm=fwhm,
            phi_double_arr_radm2=phi_double_arr_radm2,
            kernel=kernel,
        )
        peak = np.max(np.abs(fdf_conv))
        peaks[i] = peak
    activated_index = np.argmax(peaks * scale_parameters)
    return scales[activated_index], scale_parameters[activated_index]


def multiscale_minor_loop(
    scales: np.ndarray,
    scale_parameters: np.ndarray,
    resid_fdf_spectrum_mask: np.ma.MaskedArray,
    phi_arr_radm2: np.ndarray,
    phi_double_arr_radm2: np.ndarray,
    rmsf_spectrum: np.ndarray,
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
    mask_array = ~resid_fdf_spectrum_mask.mask.copy()
    iter_count = start_iter

    logger.info(f"Starting multiscale cycles...cleaning {mask_array.sum()} pixels")
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
                f"Thresold reached. Exiting loop...performed {iter_count} M-S iterations"
            )
            break

        activated_scale, scale_parameter = find_significant_scale(
            scales=scales,
            scale_parameters=scale_parameters,
            fdf_array=resid_fdf_spectrum,
            fwhm=rmsf_fwhm,
            phi_double_arr_radm2=phi_double_arr_radm2,
        )
        logger.info(f"Cleaning activated scale: {activated_scale}")

        if activated_scale == 0:
            resid_fdf_spectrum_conv = resid_fdf_spectrum.copy()
        else:
            resid_fdf_spectrum_conv = convolve_fdf_scale(
                scale=activated_scale,
                fdf_array=resid_fdf_spectrum,
                fwhm=rmsf_fwhm,
                phi_double_arr_radm2=phi_double_arr_radm2,
                kernel=kernel,
            )
        if activated_scale == 0:
            rmsf_spectrum_conv = rmsf_spectrum.copy()
        else:
            rmsf_spectrum_conv = convolve_fdf_scale(
                scale=activated_scale,
                fdf_array=rmsf_spectrum,
                fwhm=rmsf_fwhm,
                phi_double_arr_radm2=phi_double_arr_radm2,
                kernel=kernel,
            )
        scale_factor = np.nanmax(np.abs(rmsf_spectrum_conv))
        rmsf_spectrum_conv /= scale_factor

        if update_mask:
            mask_array = np.abs(resid_fdf_spectrum_conv) > mask * scale_factor
        resid_fdf_spectrum_mask_conv = np.ma.array(
            resid_fdf_spectrum_conv, mask=~mask_array
        )

        sub_minor_results = minor_loop(
            resid_fdf_spectrum_mask=resid_fdf_spectrum_mask_conv,
            phi_arr_radm2=phi_arr_radm2,
            phi_double_arr_radm2=phi_double_arr_radm2,
            rmsf_spectrum=rmsf_spectrum_conv,
            rmsf_fwhm=float(np.hypot(rmsf_fwhm * activated_scale, rmsf_fwhm)),
            max_iter=max_iter_sub_minor,
            gain=gain,
            mask=mask * scale_factor,
            threshold=threshold * scale_factor,
            start_iter=iter_count,
            update_mask=update_mask,
        )

        # Convolve the clean components with the RMSF
        clean_deltas = sub_minor_results.model_fdf_spectrum
        iter_count += sub_minor_results.iter_count
        if activated_scale == 0:
            clean_model = clean_deltas
        else:
            clean_model = convolve_fdf_scale(
                scale=activated_scale,
                fwhm=rmsf_fwhm,
                fdf_array=clean_deltas,
                phi_double_arr_radm2=phi_double_arr_radm2,
                kernel=kernel,
            )
        model_fdf_spectrum += clean_model

        clean_spectrum = restore_fdf(
            model_fdf_spectrum=model_fdf_spectrum,
            phi_double_arr_radm2=phi_double_arr_radm2,
            fwhm_rmsf=rmsf_fwhm,
        )
        shifted_rmsf = np.convolve(
            clean_model,
            rmsf_spectrum,
            mode="valid",
        )[1:-1]
        # shifted_rmsf = sub_minor_results.model_rmsf_spectrum
        clean_fdf_spectrum += clean_spectrum
        resid_fdf_spectrum -= shifted_rmsf

        if update_mask:
            mask_array = np.abs(resid_fdf_spectrum) > mask
        resid_fdf_spectrum_mask = np.ma.array(resid_fdf_spectrum, mask=~mask_array)

    return MinorLoopResults(
        clean_fdf_spectrum=clean_fdf_spectrum,
        resid_fdf_spectrum=resid_fdf_spectrum,
        resid_fdf_spectrum_mask=resid_fdf_spectrum_mask,
        model_fdf_spectrum=model_fdf_spectrum,
        model_rmsf_spectrum=np.zeros_like(model_fdf_spectrum),
        iter_count=iter_count,
    )


def multiscale_cycle(
    scales: np.ndarray,
    scale_parameters: np.ndarray,
    phi_arr_radm2: np.ndarray,
    phi_double_arr_radm2: np.ndarray,
    dirty_fdf_spectrum: np.ndarray,
    rmsf_spectrum: np.ndarray,
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
    mask_array = np.abs(resid_fdf_spectrum) > mask
    resid_fdf_spectrum_mask = np.ma.array(resid_fdf_spectrum, mask=~mask_array)

    initial_loop_results = multiscale_minor_loop(
        scales=scales,
        scale_parameters=scale_parameters,
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
    mask_array = np.abs(initial_loop_results.model_fdf_spectrum) > 0
    resid_fdf_spectrum = initial_loop_results.resid_fdf_spectrum
    resid_fdf_spectrum_mask = np.ma.array(resid_fdf_spectrum, mask=~mask_array)

    logger.info(f"Starting deep clean...cleaning {mask_array.sum()} pixels")
    deep_loop_results = multiscale_minor_loop(
        scales=scales,
        scale_parameters=scale_parameters,
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
    freq_array_hz: np.ndarray,
    dirty_fdf_array: np.ndarray,
    phi_arr_radm2: np.ndarray,
    rmsf_array: np.ndarray,
    phi_double_arr_radm2: np.ndarray,
    fwhm_rmsf_arr: np.ndarray,
    mask: float,
    threshold: float,
    max_iter: int = 1000,
    max_iter_sub_minor: int = 10_000,
    gain: float = 0.1,
    scale_bias: float = 0.9,
    scales: Optional[np.ndarray] = None,
    mask_array: Optional[np.ndarray] = None,
    kernel: Literal["tapered_quad", "gaussian"] = "gaussian",
) -> RMCleanResults:
    _bad_result = RMCleanResults(
        clean_fdf_array=dirty_fdf_array,
        model_fdf_array=np.zeros_like(dirty_fdf_array),
        clean_iter_array=np.zeros_like(phi_arr_radm2),
        resid_fdf_array=dirty_fdf_array,
    )
    # Sanity checks on array sizes
    n_phi = phi_arr_radm2.shape[0]
    if n_phi != dirty_fdf_array.shape[0]:
        logger.error("'phi_arr_radm2' and 'dirty_fdf_array' are not the same length.")
        return _bad_result
    n_phi2 = phi_double_arr_radm2.shape[0]
    if not n_phi2 == rmsf_array.shape[0]:
        logger.error("missmatch in 'phi_double_arr_radm2' and 'rmsf_array' length.")
        return _bad_result
    if not (n_phi2 >= 2 * n_phi):
        logger.error("the Faraday depth of the RMSF must be twice the FDF.")
        return _bad_result
    n_dimension = len(dirty_fdf_array.shape)
    if not n_dimension <= 3:
        logger.error("FDF array dimensions must be <= 3.")
        return _bad_result
    if not n_dimension == len(rmsf_array.shape):
        logger.error("the input RMSF and FDF must have the same number of axes.")
        return _bad_result
    if not rmsf_array.shape[1:] == dirty_fdf_array.shape[1:]:
        logger.error("the xy dimesions of the RMSF and FDF must match.")
        return _bad_result
    if mask_array is not None:
        if not mask_array.shape == dirty_fdf_array.shape[1:]:
            logger.error("pixel mask must match xy dimension of FDF cube.")
            return _bad_result
    else:
        mask_array = np.ones(dirty_fdf_array.shape[1:], dtype=bool)

    # Reshape the FDF & RMSF array to 3 dimensions and mask array to 2
    if n_dimension == 1:
        dirty_fdf_array = np.reshape(dirty_fdf_array, (dirty_fdf_array.shape[0], 1, 1))
        rmsf_array = np.reshape(rmsf_array, (rmsf_array.shape[0], 1, 1))
        mask_array = np.reshape(mask_array, (1, 1))
        fwhm_rmsf_arr = np.reshape(fwhm_rmsf_arr, (1, 1))
    elif n_dimension == 2:
        dirty_fdf_array = np.reshape(
            dirty_fdf_array, list(dirty_fdf_array.shape[:2]) + [1]
        )
        rmsf_array = np.reshape(rmsf_array, list(rmsf_array.shape[:2]) + [1])
        mask_array = np.reshape(mask_array, (dirty_fdf_array.shape[1], 1))
        fwhm_rmsf_arr = np.reshape(fwhm_rmsf_arr, (dirty_fdf_array.shape[1], 1))

    # Compute the scale parameters
    rmsf_params = compute_rmsf_params(
        freq_array_hz=freq_array_hz,
        weight_array=np.ones_like(freq_array_hz),
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
    iter_count_array = np.zeros_like(mask_array, dtype=int)

    # Determine which pixels have components above the cutoff
    abs_fdf_cube = np.abs(np.nan_to_num(dirty_fdf_array))
    cutoff_mask = np.where(np.max(abs_fdf_cube, axis=0) >= mask, 1, 0)
    pixels_to_clean = np.rot90(np.where(cutoff_mask > 0))

    num_pixels = dirty_fdf_array.shape[-1] * dirty_fdf_array.shape[-2]
    num_pixels_clean = len(pixels_to_clean)
    logger.info("Cleaning {:}/{:} spectra.".format(num_pixels_clean, num_pixels))

    # Initialise arrays to hold the residual FDF, clean components, clean FDF
    # Residual is initially copies of dirty FDF, so that pixels that are not
    #  processed get correct values (but will be overridden when processed)
    clean_fdf_spectrum = np.zeros_like(dirty_fdf_array)
    model_fdf_spectrum = np.zeros(dirty_fdf_array.shape, dtype=complex)
    resid_fdf_array = dirty_fdf_array.copy()

    # Loop through the pixels containing a polarised signal
    for yi, xi in tqdm(pixels_to_clean):
        clean_loop_results = multiscale_cycle(
            scales=scales,
            scale_parameters=scale_bias_function(scales, scale_bias),
            phi_arr_radm2=phi_arr_radm2,
            phi_double_arr_radm2=phi_double_arr_radm2,
            dirty_fdf_spectrum=dirty_fdf_array[:, yi, xi],
            rmsf_spectrum=rmsf_array[:, yi, xi],
            rmsf_fwhm=fwhm_rmsf_arr[yi, xi],
            mask=mask,
            threshold=threshold,
            max_iter=max_iter,
            max_iter_sub_minor=max_iter_sub_minor,
            gain=gain,
            kernel=kernel,
        )
        clean_fdf_spectrum[:, yi, xi] = clean_loop_results.clean_fdf_spectrum
        resid_fdf_array[:, yi, xi] = clean_loop_results.resid_fdf_spectrum
        model_fdf_spectrum[:, yi, xi] = clean_loop_results.model_fdf_spectrum
        iter_count_array[yi, xi] = clean_loop_results.iter_count

    # Restore the residual to the CLEANed FDF (moved outside of loop:
    # will now work for pixels/spectra without clean components)
    clean_fdf_spectrum += resid_fdf_array

    # Remove redundant dimensions
    clean_fdf_spectrum = np.squeeze(clean_fdf_spectrum)
    model_fdf_spectrum = np.squeeze(model_fdf_spectrum)
    iter_count_array = np.squeeze(iter_count_array)
    resid_fdf_array = np.squeeze(resid_fdf_array)

    return RMCleanResults(
        clean_fdf_spectrum, model_fdf_spectrum, iter_count_array, resid_fdf_array
    )
