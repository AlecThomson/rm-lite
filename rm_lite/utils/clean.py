#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RM-clean utils"""

import logging
from typing import NamedTuple, Optional

import numpy as np
from tqdm.auto import tqdm, trange

from rm_lite.utils.fitting import gaussian, GAUSSIAN_SIGMA_TO_FWHM
from rm_lite.utils.logging import logger, TqdmToLogger

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


def rmclean_window(
    dirty_fdf_array: np.ndarray,
    phi_arr_radm2: np.ndarray,
    rmsf_array: np.ndarray,
    phi_double_arr_radm2: np.ndarray,
    fwhm_rmsf_arr: np.ndarray,
    cutoff: float,
    max_iter: int = 1000,
    gain: float = 0.1,
    mask_array: Optional[np.ndarray] = None,
    window: float = 0,
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
        mask_array = np.ones(dirty_fdf_array.shape[1:], dtype="bool")

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
    cutoff_mask = np.where(np.max(abs_fdf_cube, axis=0) >= cutoff, 1, 0)
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
        clean_loop_results = minor_cycle_window(
            phi_arr_radm2=phi_arr_radm2,
            phi_double_arr_radm2=phi_double_arr_radm2,
            dirty_fdf_spectrum=dirty_fdf_array[:, yi, xi],
            rmsf_spectrum=rmsf_array[:, yi, xi],
            rmsf_fwhm=fwhm_rmsf_arr[yi, xi],
            cutoff=cutoff,
            window=window,
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
    window: float = 0,
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


def minor_cycle_window(
    phi_arr_radm2: np.ndarray,
    phi_double_arr_radm2: np.ndarray,
    dirty_fdf_spectrum: np.ndarray,
    rmsf_spectrum: np.ndarray,
    rmsf_fwhm: float,
    cutoff: float,
    window: float,
    max_iter: int,
    gain: float,
) -> CleanLoopResults:
    # Initialise arrays to hold the residual FDF, clean components, clean FDF
    resid_fdf_spectrum = dirty_fdf_spectrum.copy()
    model_fdf_spectrum = np.zeros_like(resid_fdf_spectrum)
    clean_fdf_spectrum = np.zeros_like(resid_fdf_spectrum)

    # Find the index of the peak of the RMSF
    max_rmsf_index = np.nanargmax(rmsf_spectrum)

    # Calculate the padding in the sampled RMSF
    # Assumes only integer shifts and symmetric RMSF
    n_phi_pad = int((len(phi_double_arr_radm2) - len(phi_arr_radm2)) / 2)

    for iter_count in trange(max_iter + 1, file=TQDM_OUT, desc="Minor cycle"):
        if iter_count == max_iter:
            logger.warning("Max iterations reached. Exiting loop.")
            break
        if np.max(np.abs(resid_fdf_spectrum)) < cutoff:
            logger.info("Residual below cutoff. Exiting loop.")
            break
        # while np.max(np.abs(resid_fdf_spectrum)) >= cutoff and iter_count < max_iter:
        # Get the absolute peak channel, values and Faraday depth
        peak_fdf_index = np.argmax(np.abs(resid_fdf_spectrum))
        peak_fdf = resid_fdf_spectrum[peak_fdf_index]
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

        # Subtract the product of the clean_component shifted RMSF from the residual FDF
        resid_fdf_spectrum -= clean_component * shifted_rmsf_spectrum

        # Restore the clean_component * a Gaussian to the cleaned FDF
        clean_fdf_spectrum += Gaussian1D(
            amplitude=clean_component,
            mean=peak_rm,
            stddev=rmsf_fwhm / GAUSSIAN_SIGMA_TO_FWHM,
        )(phi_arr_radm2)

    # Create a mask for the pixels that have been cleaned
    mask = np.abs(model_fdf_spectrum) > 0
    delta_phi = phi_arr_radm2[1] - phi_arr_radm2[0]
    fwhm_rmsf_arr_pix = rmsf_fwhm / delta_phi
    for i in np.where(mask)[0]:
        start = int(i - fwhm_rmsf_arr_pix / 2)
        end = int(i + fwhm_rmsf_arr_pix / 2)
        mask[start:end] = True
    resid_fdf_spectrum_mask = np.ma.array(resid_fdf_spectrum, mask=~mask)

    # Clean again within mask
    # while (
    #     np.ma.max(np.ma.abs(resid_fdf_spectrum_mask)) >= window
    #     and iter_count < max_iter
    # ):
    for iter_count in trange(
        iter_count, max_iter + 1, file=TQDM_OUT, desc="Window cycle"
    ):
        if resid_fdf_spectrum_mask.mask.all():
            logger.warning("All channels masked. Exiting loop.")
            break
        if iter_count == max_iter:
            logger.warning("Max iterations reached. Exiting loop.")
            break
        if np.ma.max(np.ma.abs(resid_fdf_spectrum_mask)) < window:
            logger.info("Residual below window. Exiting loop.")
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

        # Subtract the product of the clean_component shifted RMSF from the residual FDF
        resid_fdf_spectrum -= clean_component * shifted_rmsf_spectrum

        # Restore the clean_component * a Gaussian to the cleaned FDF
        clean_fdf_spectrum += Gaussian1D(
            amplitude=clean_component,
            mean=peak_rm,
            stddev=rmsf_fwhm / GAUSSIAN_SIGMA_TO_FWHM,
        )(phi_arr_radm2)

        # Remake masked residual FDF
        resid_fdf_spectrum_mask = np.ma.array(resid_fdf_spectrum, mask=~mask)

    clean_fdf_spectrum = np.squeeze(clean_fdf_spectrum)
    resid_fdf_spectrum = np.squeeze(resid_fdf_spectrum)
    model_fdf_spectrum = np.squeeze(model_fdf_spectrum)

    return CleanLoopResults(
        clean_fdf_spectrum=clean_fdf_spectrum,
        resid_fdf_spectrum=resid_fdf_spectrum,
        model_fdf_spectrum=model_fdf_spectrum,
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

    # Find the index of the peak of the RMSF
    max_rmsf_index = np.nanargmax(rmsf_spectrum)

    # Calculate the padding in the sampled RMSF
    # Assumes only integer shifts and symmetric RMSF
    n_phi_pad = int((len(phi_double_arr_radm2) - len(phi_arr_radm2)) / 2)

    mask_array = np.abs(dirty_fdf_spectrum) > mask
    # TODO: Masking based on FWHM of RMSF, or not...
    # delta_phi = phi_arr_radm2[1] - phi_arr_radm2[0]
    # fwhm_rmsf_arr_pix = rmsf_fwhm / delta_phi
    # for i in np.where(mask_array)[0]:
    #     start = int(i - fwhm_rmsf_arr_pix / 2)
    #     end = int(i + fwhm_rmsf_arr_pix / 2)
    #     mask_array[start:end] = True
    resid_fdf_spectrum_mask = np.ma.array(resid_fdf_spectrum, mask=~mask_array)

    for iter_count in trange(max_iter + 1, file=TQDM_OUT, desc="Minor cycle"):
        if resid_fdf_spectrum_mask.mask.all():
            logger.warning("All channels masked. Starting deep clean...")
            break
        if iter_count == max_iter:
            logger.warning("Max iterations reached. Exiting loop.")
            break
        if np.ma.max(np.ma.abs(resid_fdf_spectrum_mask)) < threshold:
            logger.info("Thresold reached. Exiting loop.")
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
        mask_array = np.abs(resid_fdf_spectrum) > mask
        resid_fdf_spectrum_mask = np.ma.array(resid_fdf_spectrum, mask=~mask_array)

    # Deep clean
    # Mask where clean components have been added
    mask_array = np.abs(model_fdf_spectrum) > 0
    resid_fdf_spectrum_mask = np.ma.array(resid_fdf_spectrum, mask=~mask_array)
    for iter_count in trange(
        iter_count, max_iter + 1, file=TQDM_OUT, desc="Deep clean"
    ):
        if resid_fdf_spectrum_mask.mask.all():
            logger.warning("All channels masked. Exiting loop...")
            break
        if iter_count == max_iter:
            logger.warning("Max iterations reached. Exiting loop...")
            break
        if np.ma.max(np.ma.abs(resid_fdf_spectrum_mask)) < threshold:
            logger.info("Thresold reached. Exiting loop....")
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
        # Don't remake the mask
        resid_fdf_spectrum_mask = np.ma.array(resid_fdf_spectrum, mask=~mask_array)

    clean_fdf_spectrum = np.squeeze(clean_fdf_spectrum)
    resid_fdf_spectrum = np.squeeze(resid_fdf_spectrum)
    model_fdf_spectrum = np.squeeze(model_fdf_spectrum)

    return CleanLoopResults(
        clean_fdf_spectrum=clean_fdf_spectrum,
        resid_fdf_spectrum=resid_fdf_spectrum,
        model_fdf_spectrum=model_fdf_spectrum,
        iter_count=iter_count,
    )
