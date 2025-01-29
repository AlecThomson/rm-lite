"""RM-synthesis on 1D data"""

from __future__ import annotations

import time
from typing import Literal, NamedTuple

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
    create_fractional_spectra,
    get_fdf_parameters,
    get_rmsf_nufft,
    lambda2_to_freq,
    rmsynth_nufft,
)

logger.setLevel("WARNING")


class RMSynth1DArrays(NamedTuple):
    """Resulting arrays from RM-synthesis"""

    phi_arr_radm2: np.ndarray
    """ Array of Faraday depths """
    phi2_arr_radm2: np.ndarray
    """ Double length of Faraday depths """
    rmsf_arr: np.ndarray
    """ Rotation Measure Spread Function """
    freq_arr_hz: np.ndarray
    """ Frequency array """
    weight_arr: np.ndarray
    """ Weight array """
    fdf_dirty_arr: np.ndarray
    """ Dirty Faraday dispersion function """


def run_rmsynth(
    stokes_q_arr: np.ndarray,
    stokes_u_arr: np.ndarray,
    stokes_q_error_arr: np.ndarray,
    stokes_u_error_arr: np.ndarray,
    freq_arr_hz: np.ndarray,
    stokes_i_arr: np.ndarray | None = None,
    stokes_i_error_arr: np.ndarray | None = None,
    stokes_i_model_arr: np.ndarray | None = None,
    fit_order: int = 2,
    phi_max_radm2: float | None = None,
    d_phi_radm2: float | None = None,
    n_samples: float | None = 10.0,
    weight_type: Literal["variance", "uniform"] = "variance",
    do_fit_rmsf=False,
    fit_function: Literal["log", "linear"] = "log",
    super_resolution=False,
) -> tuple[FDFParameters, RMSynth1DArrays]:
    stokes_q_arr = StokesQArray(stokes_q_arr)
    stokes_u_arr = StokesUArray(stokes_u_arr)
    stokes_q_error_arr = StokesQArray(stokes_q_error_arr)
    stokes_u_error_arr = StokesUArray(stokes_u_error_arr)

    lambda_sq_arr_m2, lam_sq_0_m2, phi_arr_radm2, weight_arr = compute_rmsynth_params(
        freq_arr_hz=freq_arr_hz,
        pol_arr=stokes_q_arr + 1j * stokes_u_arr,
        stokes_qu_error_arr=np.abs(stokes_q_error_arr + stokes_u_error_arr) / 2.0,
        d_phi_radm2=d_phi_radm2,
        n_samples=n_samples,
        phi_max_radm2=phi_max_radm2,
        super_resolution=super_resolution,
        weight_type=weight_type,
    )

    if stokes_i_arr is None or stokes_i_error_arr is None:
        logger.warning(
            "Stokes I array/errors not provided. No fractional polarization will be calculated."
        )
        stokes_i_arr = StokesIArray(np.ones_like(stokes_q_arr))
        stokes_i_error_arr = StokesIArray(np.zeros_like(stokes_q_error_arr))

    else:
        stokes_i_arr = StokesIArray(stokes_i_arr)
        stokes_i_error_arr = StokesIArray(stokes_i_error_arr)

    (
        stokes_i_model_arr,
        stokes_q_frac_arr,
        stokes_u_frac_arr,
        stokes_q_frac_error_arr,
        stokes_u_frac_error_arr,
        fit_result,
        no_nan_idx,
    ) = create_fractional_spectra(
        freq_arr_hz=freq_arr_hz,
        ref_freq_hz=lambda2_to_freq(lam_sq_0_m2),
        stokes_i_arr=stokes_i_arr,
        stokes_q_arr=stokes_q_arr,
        stokes_u_arr=stokes_u_arr,
        stokes_i_error_arr=stokes_i_error_arr,
        stokes_q_error_arr=stokes_q_error_arr,
        stokes_u_error_arr=stokes_u_error_arr,
        fit_order=fit_order,
        fit_function=fit_function,
        stokes_i_model_arr=StokesIArray(stokes_i_model_arr)
        if stokes_i_model_arr
        else None,
    )

    # Index down all arrays to remove NaNs
    freq_arr_hz = freq_arr_hz[no_nan_idx]
    lambda_sq_arr_m2 = lambda_sq_arr_m2[no_nan_idx]
    weight_arr = weight_arr[no_nan_idx]
    stokes_q_arr = stokes_q_arr[no_nan_idx]
    stokes_u_arr = stokes_u_arr[no_nan_idx]
    stokes_q_error_arr = stokes_q_error_arr[no_nan_idx]
    stokes_u_error_arr = stokes_u_error_arr[no_nan_idx]
    stokes_i_arr = stokes_i_arr[no_nan_idx]
    stokes_i_error_arr = stokes_i_error_arr[no_nan_idx]

    assert stokes_q_frac_arr.shape == stokes_u_frac_arr.shape
    assert stokes_q_frac_arr.shape == lambda_sq_arr_m2.shape

    # Compute after any fractional spectra have been created
    tick = time.time()

    # Perform RM-synthesis on the spectrum
    fdf_dirty_arr = rmsynth_nufft(
        stokes_q_arr=stokes_q_frac_arr,
        stokes_u_arr=stokes_u_frac_arr,
        lambda_sq_arr_m2=lambda_sq_arr_m2,
        phi_arr_radm2=phi_arr_radm2,
        weight_arr=weight_arr,
        lam_sq_0_m2=0 if super_resolution else lam_sq_0_m2,
    )

    # Calculate the Rotation Measure Spread Function
    rmsf_arr, phi_double_arr_radm2, fwhm_rmsf, fit_status_arr = get_rmsf_nufft(
        lambda_sq_arr_m2=lambda_sq_arr_m2,
        phi_arr_radm2=phi_arr_radm2,
        weight_arr=weight_arr,
        lam_sq_0_m2=lam_sq_0_m2,
        super_resolution=super_resolution,
        mask_arr=~np.isfinite(stokes_q_frac_arr) | ~np.isfinite(stokes_u_frac_arr),
        do_fit_rmsf=do_fit_rmsf or super_resolution,
        do_fit_rmsf_real=super_resolution,
    )

    tock = time.time()
    cpu_time = tock - tick
    logger.info(f"RM-synthesis completed in {cpu_time * 1000:.2f}ms.")

    theoretical_noise = compute_theoretical_noise(
        stokes_q_error_arr=stokes_q_frac_error_arr,
        stokes_u_error_arr=stokes_u_frac_error_arr,
        weight_arr=weight_arr,
    )
    if stokes_i_model_arr is not None:
        assert freq_arr_hz.shape == stokes_i_model_arr.shape
        stokes_i_model = interpolate.interp1d(freq_arr_hz, stokes_i_model_arr)
        stokes_i_reference_flux = stokes_i_model(lambda2_to_freq(lam_sq_0_m2))
        fdf_dirty_arr *= stokes_i_reference_flux
        theoretical_noise = theoretical_noise.with_options(
            fdf_error_noise=theoretical_noise.fdf_error_noise * stokes_i_reference_flux,
            fdf_q_noise=theoretical_noise.fdf_q_noise * stokes_i_reference_flux,
            fdf_u_noise=theoretical_noise.fdf_u_noise * stokes_i_reference_flux,
        )

    # Measure the parameters of the dirty FDF
    # Use the theoretical noise to calculate uncertainties
    fdf_parameters = get_fdf_parameters(
        fdf_arr=fdf_dirty_arr,
        phi_arr_radm2=phi_arr_radm2,
        fwhm_rmsf_radm2=fwhm_rmsf,
        freq_arr_hz=freq_arr_hz,
        stokes_q_arr=stokes_q_arr,
        stokes_u_arr=stokes_u_arr,
        stokes_q_error_arr=stokes_q_error_arr,
        stokes_u_error_arr=stokes_u_error_arr,
        lambda_sq_arr_m2=lambda_sq_arr_m2,
        lam_sq_0_m2=lam_sq_0_m2,
        stokes_i_reference_flux=stokes_i_reference_flux,
        theoretical_noise=theoretical_noise,
        fit_function=fit_function,
    )
    rmsyth_arrs = RMSynth1DArrays(
        phi_arr_radm2=phi_arr_radm2,
        phi2_arr_radm2=phi_double_arr_radm2,
        rmsf_arr=rmsf_arr,
        freq_arr_hz=freq_arr_hz,
        weight_arr=weight_arr,
        fdf_dirty_arr=fdf_dirty_arr,
    )

    return fdf_parameters, rmsyth_arrs
