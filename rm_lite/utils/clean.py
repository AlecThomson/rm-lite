#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RM-clean utils"""

import logging
from typing import Literal, NamedTuple, Optional

import numpy as np
from tqdm.auto import tqdm, trange

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
    resid_fdf_spectrum: np.ndarray,
    model_fdf_spectrum: np.ndarray,
    clean_fdf_spectrum: np.ndarray,
    model_rmsf_spectrum: np.ndarray,
    phi_arr_radm2: np.ndarray,
    rmsf_spectrum: np.ndarray,
    rmsf_fwhm: float,
    mask_array: np.ndarray,
    max_rmsf_index: int,
    n_phi_pad: int,
    max_iter: int,
    gain: float,
    mask: float,
    threshold: float,
    start_iter: int = 0,
    update_mask: bool = True,
    desc: str = "Minor loop",
) -> MinorLoopResults:
    # Trust nothing
    resid_fdf_spectrum_mask = resid_fdf_spectrum_mask.copy()
    resid_fdf_spectrum = resid_fdf_spectrum.copy()
    model_fdf_spectrum = model_fdf_spectrum.copy()
    clean_fdf_spectrum = clean_fdf_spectrum.copy()
    model_rmsf_spectrum = model_rmsf_spectrum.copy()
    rmsf_spectrum = rmsf_spectrum.copy()
    phi_arr_radm2 = phi_arr_radm2.copy()
    mask_array = mask_array.copy()
    iter_count = start_iter
    for iter_count in trange(start_iter, max_iter + 1, file=TQDM_OUT, desc=desc):
        if resid_fdf_spectrum_mask.mask.all():
            logger.warning("All channels masked. Exiting loop...")
            return MinorLoopResults(
                clean_fdf_spectrum=clean_fdf_spectrum,
                resid_fdf_spectrum=resid_fdf_spectrum,
                resid_fdf_spectrum_mask=resid_fdf_spectrum_mask,
                model_fdf_spectrum=model_fdf_spectrum,
                model_rmsf_spectrum=model_rmsf_spectrum,
                iter_count=iter_count,
            )
        if iter_count == max_iter:
            logger.warning("Max iterations reached. Exiting loop...")
            return MinorLoopResults(
                clean_fdf_spectrum=clean_fdf_spectrum,
                resid_fdf_spectrum=resid_fdf_spectrum,
                resid_fdf_spectrum_mask=resid_fdf_spectrum_mask,
                model_fdf_spectrum=model_fdf_spectrum,
                model_rmsf_spectrum=model_rmsf_spectrum,
                iter_count=iter_count,
            )
        if np.ma.max(np.ma.abs(resid_fdf_spectrum_mask)) < threshold:
            logger.info("Thresold reached. Exiting loop...")
            return MinorLoopResults(
                clean_fdf_spectrum=clean_fdf_spectrum,
                resid_fdf_spectrum=resid_fdf_spectrum,
                resid_fdf_spectrum_mask=resid_fdf_spectrum_mask,
                model_fdf_spectrum=model_fdf_spectrum,
                model_rmsf_spectrum=model_rmsf_spectrum,
                iter_count=iter_count,
            )
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
    model_fdf_spectrum = np.zeros_like(resid_fdf_spectrum)
    clean_fdf_spectrum = np.zeros_like(resid_fdf_spectrum)
    model_rmsf_spectrum = np.zeros_like(resid_fdf_spectrum)

    # Find the index of the peak of the RMSF
    max_rmsf_index = np.nanargmax(np.abs(rmsf_spectrum))

    # Calculate the padding in the sampled RMSF
    # Assumes only integer shifts and symmetric RMSF
    n_phi_pad = int((len(phi_double_arr_radm2) - len(phi_arr_radm2)) / 2)

    mask_array = np.abs(dirty_fdf_spectrum) > mask
    resid_fdf_spectrum_mask = np.ma.array(resid_fdf_spectrum, mask=~mask_array)

    (
        clean_fdf_spectrum,
        resid_fdf_spectrum,
        resid_fdf_spectrum_mask,
        model_fdf_spectrum,
        model_rmsf_spectrum,
        iter_count,
    ) = minor_loop(
        resid_fdf_spectrum_mask=resid_fdf_spectrum_mask,
        resid_fdf_spectrum=resid_fdf_spectrum,
        model_fdf_spectrum=model_fdf_spectrum,
        clean_fdf_spectrum=clean_fdf_spectrum,
        model_rmsf_spectrum=model_rmsf_spectrum,
        phi_arr_radm2=phi_arr_radm2,
        rmsf_spectrum=rmsf_spectrum,
        rmsf_fwhm=rmsf_fwhm,
        mask_array=mask_array,
        max_rmsf_index=max_rmsf_index,
        n_phi_pad=n_phi_pad,
        max_iter=max_iter,
        gain=gain,
        mask=mask,
        threshold=threshold,
        start_iter=0,
        update_mask=True,
        desc="Minor loop",
    )

    # Deep clean
    # Mask where clean components have been added
    mask_array = np.abs(model_fdf_spectrum) > 0
    resid_fdf_spectrum_mask = np.ma.array(resid_fdf_spectrum, mask=~mask_array)
    (
        clean_fdf_spectrum,
        resid_fdf_spectrum,
        resid_fdf_spectrum_mask,
        model_fdf_spectrum,
        model_rmsf_spectrum,
        iter_count,
    ) = minor_loop(
        resid_fdf_spectrum_mask=resid_fdf_spectrum_mask,
        resid_fdf_spectrum=resid_fdf_spectrum,
        model_fdf_spectrum=model_fdf_spectrum,
        model_rmsf_spectrum=model_rmsf_spectrum,
        clean_fdf_spectrum=clean_fdf_spectrum,
        phi_arr_radm2=phi_arr_radm2,
        rmsf_spectrum=rmsf_spectrum,
        rmsf_fwhm=rmsf_fwhm,
        mask_array=mask_array,
        max_rmsf_index=max_rmsf_index,
        n_phi_pad=n_phi_pad,
        max_iter=max_iter,
        gain=gain,
        mask=mask,
        threshold=threshold,
        start_iter=iter_count,
        update_mask=False,
        desc="Deep clean",
    )

    clean_fdf_spectrum = np.squeeze(clean_fdf_spectrum)
    resid_fdf_spectrum = np.squeeze(resid_fdf_spectrum)
    model_fdf_spectrum = np.squeeze(model_fdf_spectrum)

    return CleanLoopResults(
        clean_fdf_spectrum=clean_fdf_spectrum,
        resid_fdf_spectrum=resid_fdf_spectrum,
        model_fdf_spectrum=model_fdf_spectrum,
        iter_count=iter_count,
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


def convolve_fdf(
    scale: float,
    fwhm: float,
    fdf_array: np.ndarray,
    phi_arr_radm2: np.ndarray,
    normalize: Optional[Literal["peak", "sum"]] = None,
) -> np.ndarray:
    """Convolve the FDF with a Gaussian kernel.

    Args:
        scale (float): Scale parameter (relative to PSF FWHM)
        fwhm (float): FWHM of the RMSF
        fdf_array (np.ndarray): FDF array (complex)
        phi_arr_radm2 (np.ndarray): Faraday depth array (rad/m^2)
        normalize (Optional[Literal["peak", "sum"]], optional): Normalisation type. Defaults to None.

    Raises:
        ValueError: If an invalid normalization method is provided

    Returns:
        np.ndarray: Convolved FDF array
    """
    if scale == 0:
        return fdf_array
    kernel = unit_centred_gaussian(x=phi_arr_radm2, fwhm=scale * fwhm) * (1 + 0j)
    if normalize == "peak":
        kernel /= kernel.max()
    elif normalize == "sum":
        kernel /= gaussian_integrand(amplitude=1, fwhm=scale * fwhm)
    elif normalize is not None:
        raise ValueError(f"Invalid normalization method: {normalize}")
    return np.convolve(fdf_array, kernel, mode="same")


def find_significant_scale(
    scales: np.ndarray,
    scale_paramters: np.ndarray,
    abs_dirty_fdf: np.ndarray,
    phi_arr_radm2: np.ndarray,
    fwhm: float,
) -> float:
    peak_vals = np.zeros_like(scales)
    for i, (scale, scale_param) in enumerate(zip(scales, scale_paramters)):
        abs_fdf_conv = np.abs(
            convolve_fdf(
                scale=scale,
                fwhm=fwhm,
                fdf_array=abs_dirty_fdf,
                phi_arr_radm2=phi_arr_radm2,
                normalize="sum",
            )
        )
        abs_fdf_conv_scaled = abs_fdf_conv * scale_param
        peak_val = np.nanmax(abs_fdf_conv_scaled)
        peak_vals[i] = peak_val
    activated_scale = scales[np.argmax(peak_vals)]
    return activated_scale


def multiscale_minor_loop(
    scales: np.ndarray,
    scale_paramters: np.ndarray,
    resid_fdf_spectrum_mask: np.ma.MaskedArray,
    resid_fdf_spectrum: np.ndarray,
    model_fdf_spectrum: np.ndarray,
    clean_fdf_spectrum: np.ndarray,
    model_rmsf_spectrum: np.ndarray,
    phi_arr_radm2: np.ndarray,
    phi_double_arr_radm2: np.ndarray,
    rmsf_spectrum: np.ndarray,
    rmsf_fwhm: float,
    mask_array: np.ndarray,
    max_rmsf_index: int,
    n_phi_pad: int,
    max_iter: int,
    max_iter_sub_minor: int,
    gain: float,
    mask: float,
    threshold: float,
    start_iter: int = 0,
    update_mask: bool = True,
    desc: str = "Minor loop",
) -> MinorLoopResults:
    # Trust nothing
    resid_fdf_spectrum_mask = resid_fdf_spectrum_mask.copy()
    resid_fdf_spectrum = resid_fdf_spectrum.copy()
    model_fdf_spectrum = model_fdf_spectrum.copy()
    clean_fdf_spectrum = clean_fdf_spectrum.copy()
    model_rmsf_spectrum = model_rmsf_spectrum.copy()
    rmsf_spectrum = rmsf_spectrum.copy()
    phi_arr_radm2 = phi_arr_radm2.copy()
    mask_array = mask_array.copy()
    iter_count = start_iter
    while iter_count < max_iter:
        # Break conditions
        if resid_fdf_spectrum_mask.mask.all():
            logger.warning("All channels masked. Exiting loop...")
            # return MinorLoopResults(
            #     clean_fdf_spectrum=clean_fdf_spectrum,
            #     resid_fdf_spectrum=resid_fdf_spectrum,
            #     resid_fdf_spectrum_mask=resid_fdf_spectrum_mask,
            #     model_fdf_spectrum=model_fdf_spectrum,
            #     model_rmsf_spectrum=model_rmsf_spectrum,
            #     iter_count=iter_count,
            # )
            break
        if iter_count == max_iter:
            logger.warning("Max iterations reached. Exiting loop...")
            # return MinorLoopResults(
            #     clean_fdf_spectrum=clean_fdf_spectrum,
            #     resid_fdf_spectrum=resid_fdf_spectrum,
            #     resid_fdf_spectrum_mask=resid_fdf_spectrum_mask,
            #     model_fdf_spectrum=model_fdf_spectrum,
            #     model_rmsf_spectrum=model_rmsf_spectrum,
            #     iter_count=iter_count,
            # )
            break
        if np.ma.max(np.ma.abs(resid_fdf_spectrum_mask)) < threshold:
            logger.info("Thresold reached. Exiting loop...")
            # return MinorLoopResults(
            #     clean_fdf_spectrum=clean_fdf_spectrum,
            #     resid_fdf_spectrum=resid_fdf_spectrum,
            #     resid_fdf_spectrum_mask=resid_fdf_spectrum_mask,
            #     model_fdf_spectrum=model_fdf_spectrum,
            #     model_rmsf_spectrum=model_rmsf_spectrum,
            #     iter_count=iter_count,
            # )
            break

        activated_scale = find_significant_scale(
            scales=scales,
            scale_paramters=scale_paramters,
            abs_dirty_fdf=np.abs(resid_fdf_spectrum),
            fwhm=rmsf_fwhm,
            phi_arr_radm2=phi_arr_radm2,
        )
        logger.info(f"Cleaning activated scale: {activated_scale}")

        if activated_scale == 0:
            resid_fdf_spectrum_conv = resid_fdf_spectrum.copy()
        else:
            resid_fdf_spectrum_conv = convolve_fdf(
                scale=activated_scale,
                fdf_array=resid_fdf_spectrum,
                fwhm=rmsf_fwhm,
                phi_arr_radm2=phi_arr_radm2,
                normalize="sum",
            )
        if activated_scale == 0:
            rmsf_spectrum_conv = rmsf_spectrum.copy()
        else:
            rmsf_spectrum_conv = convolve_fdf(
                scale=activated_scale,
                fdf_array=rmsf_spectrum,
                fwhm=rmsf_fwhm,
                phi_arr_radm2=phi_double_arr_radm2,
                normalize="sum",
            )
        rmsf_spectrum_conv /= np.nanmax(np.abs(rmsf_spectrum_conv))

        # Find the index of the peak of the RMSF
        max_rmsf_index = int(np.nanargmax(np.abs(rmsf_spectrum_conv)))

        # Calculate the padding in the sampled RMSF
        # Assumes only integer shifts and symmetric RMSF
        n_phi_pad = int((len(phi_double_arr_radm2) - len(phi_arr_radm2)) / 2)
        mask_array = np.abs(resid_fdf_spectrum_conv) > mask
        resid_fdf_spectrum_mask = np.ma.array(resid_fdf_spectrum_conv, mask=~mask_array)

        sub_minor_results = minor_loop(
            resid_fdf_spectrum_mask=resid_fdf_spectrum_mask,
            resid_fdf_spectrum=resid_fdf_spectrum,
            model_fdf_spectrum=model_fdf_spectrum,
            clean_fdf_spectrum=clean_fdf_spectrum,
            model_rmsf_spectrum=model_rmsf_spectrum,
            phi_arr_radm2=phi_arr_radm2,
            rmsf_spectrum=rmsf_spectrum_conv,
            rmsf_fwhm=float(np.hypot(rmsf_fwhm * activated_scale, rmsf_fwhm)),
            mask_array=mask_array,
            max_rmsf_index=max_rmsf_index,
            n_phi_pad=n_phi_pad,
            max_iter=max_iter_sub_minor,
            gain=gain,
            mask=mask,
            threshold=threshold,
            start_iter=iter_count,
            update_mask=True,
            desc="Sub-minor loop",
        )

        # Convolve the clean components with the RMSF
        clean_deltas = sub_minor_results.model_fdf_spectrum
        iter_count += sub_minor_results.iter_count
        if activated_scale == 0:
            clean_model = clean_deltas
        else:
            clean_model = (
                convolve_fdf(
                    scale=activated_scale,
                    fwhm=rmsf_fwhm,
                    fdf_array=clean_deltas,
                    phi_arr_radm2=phi_arr_radm2,
                    normalize="sum",
                )
                * np.pi
            )
        model_fdf_spectrum += clean_model

        clean_spectrum = convolve_fdf(
            scale=1,
            fwhm=rmsf_fwhm,
            fdf_array=model_fdf_spectrum,
            phi_arr_radm2=phi_arr_radm2,
            normalize=None,
        )
        shifted_rmsf = np.convolve(
            model_fdf_spectrum,
            rmsf_spectrum,
            mode="valid",
        )[1:-1]
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
        model_rmsf_spectrum=model_rmsf_spectrum,
        iter_count=iter_count,
    )


def multiscale_cycle(
    scales: np.ndarray,
    scale_paramters: np.ndarray,
    phi_arr_radm2: np.ndarray,
    phi_double_arr_radm2: np.ndarray,
    dirty_fdf_spectrum: np.ndarray,
    rmsf_spectrum: np.ndarray,
    rmsf_fwhm: float,
    mask: float,
    threshold: float,
    max_iter: int,
    gain: float,
):
    # Initialise arrays to hold the residual FDF, clean components, clean FDF
    resid_fdf_spectrum = dirty_fdf_spectrum.copy()
    model_fdf_spectrum = np.zeros_like(resid_fdf_spectrum)
    clean_fdf_spectrum = np.zeros_like(resid_fdf_spectrum)
    model_rmsf_spectrum = np.zeros_like(resid_fdf_spectrum)
    mask_array = np.abs(resid_fdf_spectrum) > mask
    resid_fdf_spectrum_mask = np.ma.array(resid_fdf_spectrum, mask=~mask_array)

    # Find the index of the peak of the RMSF
    max_rmsf_index = np.nanargmax(np.abs(rmsf_spectrum))

    # Calculate the padding in the sampled RMSF
    # Assumes only integer shifts and symmetric RMSF
    n_phi_pad = int((len(phi_double_arr_radm2) - len(phi_arr_radm2)) / 2)

    (
        clean_fdf_spectrum,
        resid_fdf_spectrum,
        resid_fdf_spectrum_mask,
        model_fdf_spectrum,
        model_rmsf_spectrum,
        iter_count,
    ) = multiscale_minor_loop(
        scales=scales,
        scale_paramters=scale_paramters,
        resid_fdf_spectrum_mask=resid_fdf_spectrum_mask,
        resid_fdf_spectrum=resid_fdf_spectrum,
        model_fdf_spectrum=model_fdf_spectrum,
        clean_fdf_spectrum=clean_fdf_spectrum,
        model_rmsf_spectrum=model_rmsf_spectrum,
        phi_arr_radm2=phi_arr_radm2,
        phi_double_arr_radm2=phi_double_arr_radm2,
        rmsf_spectrum=rmsf_spectrum,
        rmsf_fwhm=rmsf_fwhm,
        mask_array=mask_array,
        max_rmsf_index=max_rmsf_index,
        n_phi_pad=n_phi_pad,
        max_iter=max_iter,
        gain=gain,
        mask=mask,
        threshold=threshold,
        start_iter=0,
        update_mask=True,
        desc="Multi-scale minor loop",
    )

    # Deep clean
    # Mask where clean components have been added
    mask_array = np.abs(model_fdf_spectrum) > 0
    resid_fdf_spectrum_mask = np.ma.array(resid_fdf_spectrum, mask=~mask_array)

    (
        clean_fdf_spectrum,
        resid_fdf_spectrum,
        resid_fdf_spectrum_mask,
        model_fdf_spectrum,
        model_rmsf_spectrum,
        iter_count,
    ) = multiscale_minor_loop(
        scales=scales,
        scale_paramters=scale_paramters,
        resid_fdf_spectrum_mask=resid_fdf_spectrum_mask,
        resid_fdf_spectrum=resid_fdf_spectrum,
        model_fdf_spectrum=model_fdf_spectrum,
        clean_fdf_spectrum=clean_fdf_spectrum,
        model_rmsf_spectrum=model_rmsf_spectrum,
        phi_arr_radm2=phi_arr_radm2,
        phi_double_arr_radm2=phi_double_arr_radm2,
        rmsf_spectrum=rmsf_spectrum,
        rmsf_fwhm=rmsf_fwhm,
        mask_array=mask_array,
        max_rmsf_index=max_rmsf_index,
        n_phi_pad=n_phi_pad,
        max_iter=max_iter,
        gain=gain,
        mask=mask,
        threshold=threshold,
        start_iter=iter_count,
        update_mask=False,
        desc="Multi-scale deep clean",
    )

    return CleanLoopResults(
        clean_fdf_spectrum=clean_fdf_spectrum,
        resid_fdf_spectrum=resid_fdf_spectrum,
        model_fdf_spectrum=model_fdf_spectrum,
        iter_count=iter_count,
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
    max_iter: int = 10_000,
    gain: float = 0.1,
    scale_bias: float = 0.9,
    scales: Optional[np.ndarray] = None,
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
        scales = np.arange(0, max_scale, step=1)

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
            scale_paramters=scale_bias_function(scales, scale_bias),
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
