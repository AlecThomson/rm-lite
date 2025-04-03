"""RM-synthesis on 1D data"""

from __future__ import annotations

import time
from typing import Literal, NamedTuple

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy import interpolate

from rm_lite.utils.logging import logger
from rm_lite.utils.synthesis import (
    FDFOptions,
    StokesData,
    compute_rmsynth_params,
    compute_theoretical_noise,
    create_fractional_spectra,
    get_fdf_parameters,
    get_rmsf_nufft,
    lambda2_to_freq,
    rmsynth_nufft,
)


class RMSynth1DResults(NamedTuple):
    """Resulting arrays from RM-synthesis"""

    fdf_parameters: pl.DataFrame
    """ FDF parameters """
    fdf_arrs: pl.DataFrame
    """ RMSynth arrays """
    rmsf_arrs: pl.DataFrame
    """ RMSF arrays """
    stokes_i_arrs: pl.DataFrame
    """ Stokes I arrays """


rmsyth_arrs_schema = pl.Schema(
    {
        "phi_arr_radm2": pl.Float64,
        "fdf_dirty_complex_arr": pl.Object,
    }
)
rmsyth_arrs_schema_df = rmsyth_arrs_schema.to_frame(eager=True)
rmsf_arrs_schema = pl.Schema(
    {
        "phi2_arr_radm2": pl.Float64,
        "rmsf_complex_arr": pl.Object,
    }
)
rmsf_arrs_schema_df = rmsf_arrs_schema.to_frame(eager=True)
stokes_i_arrs_schema = pl.Schema(
    {
        "freq_arr_hz": pl.Float64,
        "lambda_sq_arr_m2": pl.Float64,
        "stokes_i_model_arr": pl.Float64,
        "stokes_i_model_error": pl.Float64,
        "flag_arr": pl.Boolean,
        "complex_frac_pol_arr": pl.Object,
        "complex_frac_pol_error": pl.Object,
    }
)
stokes_i_arrs_schema_df = stokes_i_arrs_schema.to_frame(eager=True)


def run_rmsynth(
    freq_arr_hz: NDArray[np.float64],
    complex_pol_arr: NDArray[np.complex128],
    complex_pol_error: NDArray[np.complex128],
    stokes_i_arr: NDArray[np.float64] | None = None,
    stokes_i_error_arr: NDArray[np.float64] | None = None,
    stokes_i_model_arr: NDArray[np.float64] | None = None,
    stokes_i_model_error: NDArray[np.float64] | None = None,
    phi_max_radm2: float | None = None,
    d_phi_radm2: float | None = None,
    n_samples: float | None = 10.0,
    weight_type: Literal["variance", "uniform"] = "variance",
    do_fit_rmsf: bool = False,
    do_fit_rmsf_real: bool = False,
    fit_function: Literal["log", "linear"] = "log",
    fit_order: int = 2,
) -> RMSynth1DResults:
    """Run RM-synthesis on 1D data

    Args:
        freq_arr_hz (NDArray[np.float64]): Frequencies in Hz
        complex_pol_arr (NDArray[np.complex128]): Complex polarisation values (Q + iU)
        complex_pol_error (NDArray[np.float64]): Complex polarisation errors (dQ + idU)
        stokes_i_arr (NDArray[np.float64] | None, optional): Total itensity values. Defaults to None.
        stokes_i_error_arr (NDArray[np.float64] | None, optional): Total intensity errors. Defaults to None.
        stokes_i_model_arr (NDArray[np.float64] | None, optional): Total intensity model array. Defaults to None.
        stokes_i_model_error (NDArray[np.float64] | None, optional): Total intensity model error. Defaults to None.
        phi_max_radm2 (float | None, optional): Maximum Faraday depth. Defaults to None.
        d_phi_radm2 (float | None, optional): Spacing in Faraday depth. Defaults to None.
        n_samples (float | None, optional): Number of samples across the RMSF. Defaults to 10.0.
        weight_type ("variance", "uniform", optional): Type of weighting. Defaults to "variance".
        do_fit_rmsf (bool, optional): Fit the RMSF main lobe. Defaults to False.
        do_fit_rmsf_real (bool, optional): The the real part of the RMSF. Defaults to False.
        fit_function ("log" | "linear", optional): _description_. Defaults to "log".
        fit_order (int, optional): Polynomial fit order. Defaults to 2. Negative values will iterate until the fit is good.

    Returns:
        RMSynth1DResults:
            fdf_parameters (pl.DataFrame): FDF parameters
            fdf_arrs (pl.DataFrame): RMSynth arrays
            rmsf_arrs (pl.DataFrame): RMSF arrays
    """
    stokes_data = StokesData(
        freq_arr_hz=freq_arr_hz,
        complex_pol_arr=complex_pol_arr,
        complex_pol_error=complex_pol_error,
        stokes_i_arr=stokes_i_arr,
        stokes_i_error_arr=stokes_i_error_arr,
        stokes_i_model_arr=stokes_i_model_arr,
        stokes_i_model_error=stokes_i_model_error,
    )

    fdf_options = FDFOptions(
        phi_max_radm2=phi_max_radm2,
        d_phi_radm2=d_phi_radm2,
        n_samples=n_samples,
        weight_type=weight_type,
        do_fit_rmsf=do_fit_rmsf,
        do_fit_rmsf_real=do_fit_rmsf_real,
    )

    return _run_rmsynth(stokes_data, fdf_options, fit_function, fit_order)


def _run_rmsynth(
    stokes_data: StokesData,
    fdf_options: FDFOptions,
    fit_function: Literal["log", "linear"] = "log",
    fit_order: int = 2,
) -> RMSynth1DResults:
    """Run RM-synthesis on 1D data with packed data

    Args:
        stokes_data (StokesData): Frequency-dependent polarisation data
        fdf_options (FDFOptions): RM-synthesis options
        fit_function ("log", "linear", optional): Type of function to fit. Defaults to "log".
        fit_order (int, optional): Polynomial fit order. Defaults to 2. Negative values will iterate until the fit is good.

    Returns:
        RMSynth1DResults:
            fdf_parameters (pl.DataFrame): FDF parameters
            fdf_arrs (pl.DataFrame): RMSynth arrays
            rmsf_arrs (pl.DataFrame): RMSF arrays
    """

    rmsynth_params = compute_rmsynth_params(
        freq_arr_hz=stokes_data.freq_arr_hz,
        complex_pol_arr=stokes_data.complex_pol_arr,
        complex_pol_error=stokes_data.complex_pol_error,
        fdf_options=fdf_options,
    )

    fractional_spectra = create_fractional_spectra(
        stokes_data=stokes_data,
        ref_freq_hz=lambda2_to_freq(rmsynth_params.lam_sq_0_m2),
        fit_order=fit_order,
        fit_function=fit_function,
    )

    # Compute after any fractional spectra have been created
    tick = time.time()

    # Perform RM-synthesis on the spectrum
    no_nan_idx = fractional_spectra.no_nan_idx
    fdf_dirty_arr = rmsynth_nufft(
        complex_pol_arr=fractional_spectra.complex_pol_frac_arr[no_nan_idx],
        lambda_sq_arr_m2=rmsynth_params.lambda_sq_arr_m2[no_nan_idx],
        phi_arr_radm2=rmsynth_params.phi_arr_radm2,
        weight_arr=rmsynth_params.weight_arr[no_nan_idx],
        lam_sq_0_m2=rmsynth_params.lam_sq_0_m2,
    )

    # Calculate the Rotation Measure Spread Function
    rmsf_result = get_rmsf_nufft(
        lambda_sq_arr_m2=rmsynth_params.lambda_sq_arr_m2,
        phi_arr_radm2=rmsynth_params.phi_arr_radm2,
        weight_arr=rmsynth_params.weight_arr,
        lam_sq_0_m2=rmsynth_params.lam_sq_0_m2,
        mask_arr=~no_nan_idx,
        do_fit_rmsf=fdf_options.do_fit_rmsf,
        do_fit_rmsf_real=fdf_options.do_fit_rmsf_real,
    )

    tock = time.time()
    cpu_time = tock - tick
    logger.info(f"RM-synthesis completed in {cpu_time * 1000:.2f}ms.")

    theoretical_noise = compute_theoretical_noise(
        complex_pol_frac_error=fractional_spectra.complex_pol_frac_error[no_nan_idx],
        weight_arr=rmsynth_params.weight_arr[no_nan_idx],
    )

    assert stokes_data.freq_arr_hz.shape == fractional_spectra.stokes_i_model_arr.shape
    stokes_i_model = interpolate.interp1d(
        stokes_data.freq_arr_hz[no_nan_idx],
        fractional_spectra.stokes_i_model_arr[no_nan_idx],
    )

    stokes_i_reference_flux = stokes_i_model(
        lambda2_to_freq(rmsynth_params.lam_sq_0_m2)
    )
    fdf_dirty_arr *= stokes_i_reference_flux
    theoretical_noise = theoretical_noise.with_options(  # type: ignore[no-untyped-call]
        fdf_error_noise=theoretical_noise.fdf_error_noise * stokes_i_reference_flux,
        fdf_q_noise=theoretical_noise.fdf_q_noise * stokes_i_reference_flux,
        fdf_u_noise=theoretical_noise.fdf_u_noise * stokes_i_reference_flux,
    )

    # Measure the parameters of the dirty FDF
    # Use the theoretical noise to calculate uncertainties
    fdf_parameters = get_fdf_parameters(
        fdf_arr=fdf_dirty_arr,
        phi_arr_radm2=rmsynth_params.phi_arr_radm2,
        fwhm_rmsf_radm2=float(rmsf_result.fwhm_rmsf_arr),
        freq_arr_hz=stokes_data.freq_arr_hz,
        complex_pol_arr=fractional_spectra.complex_pol_frac_arr,
        complex_pol_error=fractional_spectra.complex_pol_frac_error,
        lambda_sq_arr_m2=rmsynth_params.lambda_sq_arr_m2,
        lam_sq_0_m2=rmsynth_params.lam_sq_0_m2,
        stokes_i_reference_flux=stokes_i_reference_flux,
        theoretical_noise=theoretical_noise,
        fit_function=fit_function,
    )
    rmsyth_arrs = rmsyth_arrs_schema_df.vstack(
        pl.DataFrame(
            {
                "phi_arr_radm2": rmsynth_params.phi_arr_radm2,
                "fdf_dirty_complex_arr": fdf_dirty_arr,
            }
        )
    )

    rmsf_arrs = rmsf_arrs_schema_df.vstack(
        pl.DataFrame(
            {
                "phi2_arr_radm2": rmsf_result.phi_double_arr_radm2,
                "rmsf_complex_arr": rmsf_result.rmsf_cube,
            }
        )
    )
    stokes_i_arrs = stokes_i_arrs_schema_df.vstack(
        pl.DataFrame(
            {
                "freq_arr_hz": stokes_data.freq_arr_hz,
                "lambda_sq_arr_m2": rmsynth_params.lambda_sq_arr_m2,
                "stokes_i_model_arr": fractional_spectra.stokes_i_model_arr,
                "stokes_i_model_error": fractional_spectra.stokes_i_model_error,
                "flag_arr": no_nan_idx,
                "complex_frac_pol_arr": fractional_spectra.complex_pol_frac_arr,
                "complex_frac_pol_error": fractional_spectra.complex_pol_frac_error,
            }
        )
    )

    return RMSynth1DResults(fdf_parameters, rmsyth_arrs, rmsf_arrs, stokes_i_arrs)
