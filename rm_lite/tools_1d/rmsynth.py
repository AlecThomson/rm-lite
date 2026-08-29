"""RM-synthesis on 1D data"""

from __future__ import annotations

import time
from typing import Literal, NamedTuple

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy import interpolate

from rm_lite.utils.fitting import (
    FitResult,
    StokesIFitOptions,
    coefficient_errors,
    coefficient_names,
)
from rm_lite.utils.logging import logger
from rm_lite.utils.synthesis import (
    FDFOptions,
    LamSq0Mode,
    StokesData,
    WeightType,
    compute_rmsynth_params,
    compute_theoretical_noise,
    create_fractional_spectra,
    get_fdf_parameters,
    get_mask_index,
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
    stokes_i_terms: pl.DataFrame
    """ Fitted Stokes I model terms, one row per term """


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
        "complex_pol_arr": pl.Object,
        "complex_pol_error": pl.Object,
    }
)
stokes_i_arrs_schema_df = stokes_i_arrs_schema.to_frame(eager=True)
stokes_i_terms_schema = pl.Schema(
    {
        "term_name": pl.String,
        "term_value": pl.Float64,
        "term_error": pl.Float64,
        "ref_freq_hz": pl.Float64,
        "fit_function": pl.String,
    }
)
stokes_i_terms_schema_df = stokes_i_terms_schema.to_frame(eager=True)


def _stokes_i_terms(
    fit_result: FitResult | None,
    ref_freq_hz: float,
    fit_function: Literal["log", "linear"],
) -> pl.DataFrame:
    """The fitted Stokes I model as one row per term.

    With `fit_function="log"` the terms are the usual radio ones and the model is
    `flux * 10**(alpha*log10(nu/nu_ref) + beta*log10(nu/nu_ref)**2 + ...)`; with
    "linear" they are `c0..cN` of `sum(c_i * (nu/nu_ref)**i)`. `ref_freq_hz` and
    `fit_function` ride along on every row so the frame describes the whole model
    on its own. Empty when nothing was fitted (a supplied model, or no Stokes I).
    """
    if fit_result is None:
        return stokes_i_terms_schema_df
    popt = np.asarray(fit_result.popt, dtype=np.float64)
    return stokes_i_terms_schema_df.vstack(
        pl.DataFrame(
            {
                "term_name": list(coefficient_names(popt.size, fit_function)),
                "term_value": popt,
                "term_error": coefficient_errors(fit_result.pcov, popt.size),
                "ref_freq_hz": np.full(popt.size, ref_freq_hz),
                "fit_function": [fit_function] * popt.size,
            }
        )
    )


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
    weight_type: WeightType = "variance",
    robust: float | None = None,
    lam_sq_0_m2: float | LamSq0Mode = "auto",
    do_fit_rmsf: bool = False,
    do_fit_rmsf_real: bool = False,
    fit_function: Literal["log", "linear"] = "log",
    fit_order: int = 2,
    ignore_stokes_i: bool = False,
    moment_threshold_snr: float = 5.0,
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
        weight_type (WeightType, optional): Weighting: 'variance' (1/sigma^2), 'uniform' (equal per channel), 'uniform_lsq' (equal per lambda^2 interval, narrows the RMSF), 'briggs' (robust). Defaults to "variance".
        lam_sq_0_m2 (float | LamSq0Mode, optional): Reference lambda^2 in m^2 the
            FDF is derotated to, or how to pick one: "auto" for the weighted mean
            of the observed lambda^2 (Brentjens & de Bruyn 2005, eq. 32), or a
            fixed value to share a reference with another spectrum or cube. There
            is one spectrum here, so "per_pixel" is the same as "auto". The Stokes
            I model's reference frequency is derived from it, so the flux and
            phase references match by construction. Defaults to "auto".
        robust (float | None, optional): Briggs robust parameter, required for weight_type='briggs'. Defaults to None.
        do_fit_rmsf (bool, optional): Fit the RMSF main lobe. Defaults to False.
        do_fit_rmsf_real (bool, optional): Fit only the real part of the RMSF. Defaults to False.
        fit_function ("log" | "linear", optional): RMSF fit function. Defaults to "log".
        fit_order (int, optional): Polynomial fit order. Defaults to 2. Negative values will iterate until the fit is good.
        moment_threshold_snr (float, optional): SNR cut (times the theoretical FDF noise) applied to FDF amplitudes before computing the Faraday moments. Defaults to 5.0.

    Returns:
        RMSynth1DResults:
            fdf_parameters (pl.DataFrame): FDF parameters
            fdf_arrs (pl.DataFrame): RMSynth arrays
            rmsf_arrs (pl.DataFrame): RMSF arrays
            stokes_i_arrs (pl.DataFrame): Stokes I arrays
            stokes_i_terms (pl.DataFrame): Fitted Stokes I model terms, empty when
                a model was supplied rather than fitted
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
        robust=robust,
        lam_sq_0_m2=lam_sq_0_m2,
        do_fit_rmsf=do_fit_rmsf,
        do_fit_rmsf_real=do_fit_rmsf_real,
    )
    # snr_cut=None: the 1D fractional fit has only one spectrum, so an SNR cut
    # would just silently drop fractional polarisation rather than fall back to
    # a flat per-pixel model as it does in 3D.
    fit_options = StokesIFitOptions(
        fit_order=fit_order,
        fit_function=fit_function,
        snr_cut=None,
    )

    if (
        stokes_i_arr is None or stokes_i_error_arr is None
    ) and stokes_data.stokes_i_model_arr is None:
        logger.warning(
            "Stokes I array/errors or model not provided. No fractional polarization will be calculated."
        )
        ignore_stokes_i = True

    return _run_rmsynth(
        stokes_data=stokes_data,
        fdf_options=fdf_options,
        fit_options=fit_options,
        ignore_stokes_i=ignore_stokes_i,
        moment_threshold_snr=moment_threshold_snr,
    )


def _run_rmsynth(
    stokes_data: StokesData,
    fdf_options: FDFOptions,
    fit_options: StokesIFitOptions,
    ignore_stokes_i: bool = False,
    moment_threshold_snr: float = 5.0,
) -> RMSynth1DResults:
    """Run RM-synthesis on 1D data with packed data

    Args:
        stokes_data (StokesData): Frequency-dependent polarisation data
        fdf_options (FDFOptions): RM-synthesis options
        fit_options (StokesIFitOptions): Stokes I model fitting options
        ignore_stokes_i (bool, optional): Skip the fractional-polarisation step. Defaults to False.
        moment_threshold_snr (float, optional): SNR cut for the Faraday moments. Defaults to 5.0.

    Returns:
        RMSynth1DResults:
            fdf_parameters (pl.DataFrame): FDF parameters
            fdf_arrs (pl.DataFrame): RMSynth arrays
            rmsf_arrs (pl.DataFrame): RMSF arrays
            stokes_i_arrs (pl.DataFrame): Stokes I arrays
            stokes_i_terms (pl.DataFrame): Fitted Stokes I model terms
    """

    rmsynth_params = compute_rmsynth_params(
        freq_arr_hz=stokes_data.freq_arr_hz,
        complex_pol_arr=stokes_data.complex_pol_arr,
        complex_pol_error=stokes_data.complex_pol_error,
        fdf_options=fdf_options,
    )

    no_nan_idx = get_mask_index(stokes_data=stokes_data)

    ref_freq_hz = float(lambda2_to_freq(rmsynth_params.lam_sq_0_m2))
    fit_result: FitResult | None = None
    if not ignore_stokes_i:
        fractional_stokes_data = create_fractional_spectra(
            stokes_data=stokes_data,
            ref_freq_hz=ref_freq_hz,
            fit_options=fit_options,
        )
        if fractional_stokes_data is not None:
            stokes_data = fractional_stokes_data.stokes_data
            no_nan_idx = fractional_stokes_data.no_nan_idx
            fit_result = fractional_stokes_data.fit_result

    # Compute after any fractional spectra have been created
    tick = time.time()

    # Perform RM-synthesis on the spectrum
    all_flagged = (~no_nan_idx).all()
    if all_flagged:
        msg = "All channels have been masked!"
        logger.warning(msg)

    fdf_dirty_arr = rmsynth_nufft(
        complex_pol_arr=stokes_data.complex_pol_arr[no_nan_idx],
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
        complex_pol_error=stokes_data.complex_pol_error[no_nan_idx],
        weight_arr=rmsynth_params.weight_arr[no_nan_idx],
    )

    if not ignore_stokes_i:
        assert stokes_data.stokes_i_model_arr is not None
        assert stokes_data.freq_arr_hz.shape == stokes_data.stokes_i_model_arr.shape
        if not all_flagged:
            stokes_i_model = interpolate.interp1d(
                stokes_data.freq_arr_hz[no_nan_idx],
                stokes_data.stokes_i_model_arr[no_nan_idx],
            )

            stokes_i_reference_flux = float(stokes_i_model(ref_freq_hz))
        else:
            logger.warning("Using mean as reference flux")
            stokes_i_reference_flux = float(np.nanmean(stokes_data.stokes_i_model_arr))

        fdf_dirty_arr *= stokes_i_reference_flux

        theoretical_noise = theoretical_noise._replace(
            fdf_error_noise=theoretical_noise.fdf_error_noise * stokes_i_reference_flux,
            fdf_q_noise=theoretical_noise.fdf_q_noise * stokes_i_reference_flux,
            fdf_u_noise=theoretical_noise.fdf_u_noise * stokes_i_reference_flux,
        )

    else:
        stokes_i_reference_flux = np.nan

    # Measure the parameters of the dirty FDF
    # Use the theoretical noise to calculate uncertainties
    fdf_parameters = get_fdf_parameters(
        fdf_arr=fdf_dirty_arr,
        phi_arr_radm2=rmsynth_params.phi_arr_radm2,
        fwhm_rmsf_radm2=float(rmsf_result.fwhm_rmsf_arr),
        freq_arr_hz=stokes_data.freq_arr_hz,
        complex_pol_arr=stokes_data.complex_pol_arr,
        complex_pol_error=stokes_data.complex_pol_error,
        lambda_sq_arr_m2=rmsynth_params.lambda_sq_arr_m2,
        lam_sq_0_m2=rmsynth_params.lam_sq_0_m2,
        stokes_i_reference_flux=stokes_i_reference_flux,
        theoretical_noise=theoretical_noise,
        fit_function=fit_options.fit_function,
        moment_threshold_snr=moment_threshold_snr,
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
                "stokes_i_model_arr": stokes_data.stokes_i_model_arr,
                "stokes_i_model_error": stokes_data.stokes_i_model_error,
                "flag_arr": no_nan_idx,
                "complex_pol_arr": stokes_data.complex_pol_arr,
                "complex_pol_error": stokes_data.complex_pol_error,
            }
        )
    )

    return RMSynth1DResults(
        fdf_parameters,
        rmsyth_arrs,
        rmsf_arrs,
        stokes_i_arrs,
        _stokes_i_terms(fit_result, ref_freq_hz, fit_options.fit_function),
    )
