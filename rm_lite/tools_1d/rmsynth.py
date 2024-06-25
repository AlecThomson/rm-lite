#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RM-synthesis on 1D data"""

import time
from typing import Literal, NamedTuple, Optional, Tuple

import numpy as np
from scipy import interpolate

from rm_lite.utils.logging import logger
from rm_lite.utils.synthesis import (
    FDFParameters,
    StokesIArray,
    StokesQArray,
    StokesUArray,
    compute_rmsynth_params,
    compute_theoretical_noise,
    get_fdf_parameters,
    get_rmsf_nufft,
    lambda2_to_freq,
    rmsynth_nufft,
    create_fractional_spectra,
)

logger.setLevel("WARNING")


class RMSynth1DArrays(NamedTuple):
    """Resulting arrays from RM-synthesis"""

    phi_arr_radm2: np.ndarray
    """ Array of Faraday depths """
    phi2_arr_radm2: np.ndarray
    """ Double length of Faraday depths """
    rmsf_array: np.ndarray
    """ Rotation Measure Spread Function """
    freq_array_hz: np.ndarray
    """ Frequency array """
    weight_array: np.ndarray
    """ Weight array """
    fdf_dirty_array: np.ndarray
    """ Dirty Faraday dispersion function """


def run_rmsynth(
    stokes_q_array: np.ndarray,
    stokes_u_array: np.ndarray,
    stokes_q_error_array: np.ndarray,
    stokes_u_error_array: np.ndarray,
    freq_array_hz: np.ndarray,
    stokes_i_array: Optional[np.ndarray] = None,
    stokes_i_error_array: Optional[np.ndarray] = None,
    stokes_i_model_array: Optional[np.ndarray] = None,
    fit_order: int = 2,
    phi_max_radm2: Optional[float] = None,
    d_phi_radm2: Optional[float] = None,
    n_samples: Optional[float] = 10.0,
    weight_type: Literal["variance", "uniform"] = "variance",
    do_fit_rmsf=False,
    fit_function: Literal["log", "linear"] = "log",
    super_resolution=False,
) -> Tuple[FDFParameters, RMSynth1DArrays]:
    stokes_q_array = StokesQArray(stokes_q_array)
    stokes_u_array = StokesUArray(stokes_u_array)
    stokes_q_error_array = StokesQArray(stokes_q_error_array)
    stokes_u_error_array = StokesUArray(stokes_u_error_array)

    lambda_sq_arr_m2, lam_sq_0_m2, phi_arr_radm2, weight_array = compute_rmsynth_params(
        freq_array_hz=freq_array_hz,
        pol_array=stokes_q_array + 1j * stokes_u_array,
        stokes_qu_error_array=np.abs(stokes_q_error_array + stokes_u_error_array) / 2.0,
        d_phi_radm2=d_phi_radm2,
        n_samples=n_samples,
        phi_max_radm2=phi_max_radm2,
        super_resolution=super_resolution,
        weight_type=weight_type,
    )

    if stokes_i_array is None or stokes_i_error_array is None:
        logger.warning(
            "Stokes I array/errors not provided. No fractional polarization will be calculated."
        )
        stokes_i_array = StokesIArray(np.ones_like(stokes_q_array))
        stokes_i_error_array = StokesIArray(np.zeros_like(stokes_q_error_array))

    else:
        stokes_i_array = StokesIArray(stokes_i_array)
        stokes_i_error_array = StokesIArray(stokes_i_error_array)

    (
        stokes_i_model_array,
        stokes_q_frac_array,
        stokes_u_frac_array,
        stokes_q_frac_error_array,
        stokes_u_frac_error_array,
        fit_result,
    ) = create_fractional_spectra(
        freq_array_hz=freq_array_hz,
        ref_freq_hz=lambda2_to_freq(lam_sq_0_m2),
        stokes_i_array=stokes_i_array,
        stokes_q_array=stokes_q_array,
        stokes_u_array=stokes_u_array,
        stokes_i_error_array=stokes_i_error_array,
        stokes_q_error_array=stokes_q_error_array,
        stokes_u_error_array=stokes_u_error_array,
        fit_order=fit_order,
        fit_function=fit_function,
        stokes_i_model_array=StokesIArray(stokes_i_model_array)
        if stokes_i_model_array
        else None,
    )

    # Compute after any fractional spectra have been created
    tick = time.time()

    # Perform RM-synthesis on the spectrum
    fdf_dirty_array = rmsynth_nufft(
        stokes_q_array=stokes_q_frac_array,
        stokes_u_array=stokes_u_frac_array,
        lambda_sq_arr_m2=lambda_sq_arr_m2,
        phi_arr_radm2=phi_arr_radm2,
        weight_array=weight_array,
        lam_sq_0_m2=0 if super_resolution else lam_sq_0_m2,
    )

    # Calculate the Rotation Measure Spread Function
    rmsf_array, phi_double_arr_radm2, fwhm_rmsf, fit_status_array = get_rmsf_nufft(
        lambda_sq_arr_m2=lambda_sq_arr_m2,
        phi_arr_radm2=phi_arr_radm2,
        weight_array=weight_array,
        lam_sq_0_m2=lam_sq_0_m2,
        super_resolution=super_resolution,
        mask_array=~np.isfinite(stokes_q_array) | ~np.isfinite(stokes_u_array),
        do_fit_rmsf=do_fit_rmsf or super_resolution,
        do_fit_rmsf_real=super_resolution,
    )

    tock = time.time()
    cpu_time = tock - tick
    logger.info(f"RM-synthesis completed in {cpu_time*1000:.2f}ms.")

    theoretical_noise = compute_theoretical_noise(
        stokes_q_error_array=stokes_q_frac_error_array,
        stokes_u_error_array=stokes_u_frac_error_array,
        weight_array=weight_array,
    )
    if stokes_i_model_array is not None:
        stokes_i_model = interpolate.interp1d(freq_array_hz, stokes_i_model_array)
        stokes_i_reference_flux = stokes_i_model(lambda2_to_freq(lam_sq_0_m2))
        fdf_dirty_array *= stokes_i_reference_flux
        theoretical_noise = theoretical_noise.with_options(
            fdf_error_noise=theoretical_noise.fdf_error_noise * stokes_i_reference_flux,
            fdf_q_noise=theoretical_noise.fdf_q_noise * stokes_i_reference_flux,
            fdf_u_noise=theoretical_noise.fdf_u_noise * stokes_i_reference_flux,
        )

    # Measure the parameters of the dirty FDF
    # Use the theoretical noise to calculate uncertainties
    fdf_parameters = get_fdf_parameters(
        fdf_array=fdf_dirty_array,
        phi_arr_radm2=phi_arr_radm2,
        fwhm_rmsf_radm2=fwhm_rmsf,
        freq_array_hz=freq_array_hz,
        stokes_q_array=stokes_q_array,
        stokes_u_array=stokes_u_array,
        stokes_q_error_array=stokes_q_error_array,
        stokes_u_error_array=stokes_u_error_array,
        lambda_sq_arr_m2=lambda_sq_arr_m2,
        lam_sq_0_m2=lam_sq_0_m2,
        stokes_i_reference_flux=stokes_i_reference_flux,
        theoretical_noise=theoretical_noise,
        fit_function=fit_function,
    )
    rmsyth_arrays = RMSynth1DArrays(
        phi_arr_radm2=phi_arr_radm2,
        phi2_arr_radm2=phi_double_arr_radm2,
        rmsf_array=rmsf_array,
        freq_array_hz=freq_array_hz,
        weight_array=weight_array,
        fdf_dirty_array=fdf_dirty_array,
    )

    return fdf_parameters, rmsyth_arrays
