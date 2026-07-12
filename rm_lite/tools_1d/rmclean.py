"""RM-CLEAN on 1D data"""

from __future__ import annotations

from typing import Literal, NamedTuple, cast

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy import interpolate

from rm_lite.tools_1d.rmsynth import RMSynth1DResults
from rm_lite.utils.clean import rmclean
from rm_lite.utils.logging import logger
from rm_lite.utils.synthesis import (
    TheoreticalNoise,
    get_fdf_parameters,
    lambda2_to_freq,
)

rmclean_arrs_schema = pl.Schema(
    {
        "phi_arr_radm2": pl.Float64,
        "fdf_dirty_complex_arr": pl.Object,
        "fdf_clean_complex_arr": pl.Object,
        "fdf_model_complex_arr": pl.Object,
        "fdf_residual_complex_arr": pl.Object,
    }
)
rmclean_arrs_schema_df = rmclean_arrs_schema.to_frame(eager=True)

rmclean_scalar_schema = pl.Schema(
    {
        "mask": pl.Float64,
        "threshold": pl.Float64,
        "n_iter": pl.Int64,
    }
)
rmclean_scalar_schema_df = rmclean_scalar_schema.to_frame(eager=True)


class RMClean1DResults(NamedTuple):
    """Resulting arrays from RM-synthesis"""

    fdf_parameters: pl.DataFrame
    """ FDF parameters """
    fdf_arrs: pl.DataFrame
    """ RMClean arrays """
    clean_parameters: pl.DataFrame
    """ RMClean parameters """


def run_rmclean_from_synth(
    rm_synth_1d_results: RMSynth1DResults,
    auto_mask: float = 7,
    auto_threshold: float = 1,
    max_iter: int = 10_000,
    gain: float = 0.1,
    mask_arr: NDArray[np.bool_] | None = None,
    moment_threshold_snr: float = 5.0,
    multiscale: bool = False,
    scale_bias: float | None = None,
    scales: NDArray[np.float64] | None = None,
    n_scales: int | None = None,
    kernel: Literal["tapered_quad", "gaussian"] | None = None,
    max_iter_sub_minor: int | None = None,
    sub_minor_fraction: float | None = None,
) -> RMClean1DResults:
    """Run RM-CLEAN on the results of RM-synth.

    Args:
        rm_synth_1d_results (RMSynth1DResults): Results from RM-synth.
        auto_mask (float, optional): Masking threshold in SNR. Defaults to 7.
        auto_threshold (float, optional): Cleaning threshold in SNR. Defaults to 1.
        max_iter (int, optional): Maximum CLEAN iterations. Defaults to 10_000.
        gain (float, optional): CLEAN gain. Defaults to 0.1.
        mask_arr (NDArray[np.bool_] | None, optional): Optional mask array. Defaults to None.
        moment_threshold_snr (float, optional): SNR cut (times the theoretical FDF noise) applied to the clean FDF amplitudes before computing the Faraday moments. Defaults to 5.0.
        multiscale (bool, optional): Use multiscale RM-CLEAN (recovers Faraday-thick structure). Defaults to False.
        scale_bias (float | None, optional): Multiscale scale-bias in (0, 1]; lower favours larger scales more. None uses the default.
        scales (NDArray[np.float64] | None, optional): Explicit multiscale scales (RMSF FWHM units); None auto-selects from the RMSF max scale.
        n_scales (int | None, optional): Cap on the auto scale count.
        kernel ("tapered_quad" | "gaussian" | None, optional): Multiscale scale kernel.
        max_iter_sub_minor (int | None, optional): Max sub-minor iterations per scale.
        sub_minor_fraction (float | None, optional): Sub-minor re-selection fraction.

    Returns:
        RMClean1DResults: RM-CLEAN results: `fdf_parameters`, `fdf_arrs`, `clean_parameters`.
    """
    rmsyth_arrs_df = rm_synth_1d_results.fdf_arrs
    rmsf_arrs_df = rm_synth_1d_results.rmsf_arrs
    fdf_parameters = rm_synth_1d_results.fdf_parameters
    stokes_i_arrs_df = rm_synth_1d_results.stokes_i_arrs

    theoretical_noise = TheoreticalNoise(
        fdf_error_noise=float(fdf_parameters["fdf_error_noise"].to_numpy().squeeze()),
        fdf_q_noise=float(fdf_parameters["fdf_q_noise"].to_numpy().squeeze()),
        fdf_u_noise=float(fdf_parameters["fdf_u_noise"].to_numpy().squeeze()),
    )

    logger.info(f"Theoretical noise: {theoretical_noise}")

    mask = auto_mask * theoretical_noise.fdf_error_noise
    threshold = auto_threshold * theoretical_noise.fdf_error_noise

    logger.info(
        f"Auto mask: {mask:0.2f}, Auto threshold: {threshold:0.2f}, Max iterations: {max_iter}, Gain: {gain}"
    )

    stokes_i_model = interpolate.interp1d(
        stokes_i_arrs_df["freq_arr_hz"],
        stokes_i_arrs_df["stokes_i_model_arr"],
    )

    stokes_i_reference_flux = float(
        stokes_i_model(
            lambda2_to_freq(float(fdf_parameters["lam_sq_0_m2"].to_numpy()[0]))
        )
    )

    fdf_dirty_arr = rmsyth_arrs_df["fdf_dirty_complex_arr"].to_numpy().astype(complex)

    rm_clean_results = rmclean(
        dirty_fdf_arr=fdf_dirty_arr,
        phi_arr_radm2=rmsyth_arrs_df["phi_arr_radm2"].to_numpy().astype(float),
        rmsf_arr=rmsf_arrs_df["rmsf_complex_arr"].to_numpy().astype(complex),
        phi_double_arr_radm2=rmsf_arrs_df["phi2_arr_radm2"].to_numpy().astype(float),
        fwhm_rmsf_arr=fdf_parameters["fwhm_rmsf_radm2"].to_numpy().astype(float),
        mask=mask,
        threshold=threshold,
        max_iter=max_iter,
        gain=gain,
        mask_arr=mask_arr,
        multiscale=multiscale,
        scale_bias=scale_bias,
        scales=scales,
        n_scales=n_scales,
        kernel=kernel,
        max_iter_sub_minor=max_iter_sub_minor,
        sub_minor_fraction=sub_minor_fraction,
        phi_max_scale_radm2=float(fdf_parameters["phi_max_scale_radm2"][0]),
    )
    clean_fdf_arr, model_fdf_arr, clean_iter_arr, resid_fdf_arr = rm_clean_results

    fdf_parameters = get_fdf_parameters(
        fdf_arr=rm_clean_results.clean_fdf_arr,
        phi_arr_radm2=rmsyth_arrs_df["phi_arr_radm2"].to_numpy().astype(float),
        fwhm_rmsf_radm2=float(
            fdf_parameters["fwhm_rmsf_radm2"].to_numpy().astype(float).squeeze()
        ),
        freq_arr_hz=stokes_i_arrs_df["freq_arr_hz"].to_numpy().astype(float),
        complex_pol_arr=stokes_i_arrs_df["complex_pol_arr"].to_numpy().astype(complex),
        complex_pol_error=stokes_i_arrs_df["complex_pol_error"]
        .to_numpy()
        .astype(complex),
        lambda_sq_arr_m2=stokes_i_arrs_df["lambda_sq_arr_m2"].to_numpy().astype(float),
        lam_sq_0_m2=float(fdf_parameters["lam_sq_0_m2"].to_numpy().squeeze()),
        stokes_i_reference_flux=stokes_i_reference_flux,
        theoretical_noise=theoretical_noise,
        fit_function=cast(
            "Literal['log', 'linear']",
            str(fdf_parameters["fit_function"].to_numpy().squeeze()),
        ),
        moment_threshold_snr=moment_threshold_snr,
    )

    rmclean_arrs = rmclean_arrs_schema_df.vstack(
        pl.DataFrame(
            {
                "phi_arr_radm2": rmsyth_arrs_df["phi_arr_radm2"]
                .to_numpy()
                .astype(float),
                "fdf_dirty_complex_arr": rmsyth_arrs_df["fdf_dirty_complex_arr"]
                .to_numpy()
                .astype(complex),
                "fdf_clean_complex_arr": clean_fdf_arr,
                "fdf_model_complex_arr": model_fdf_arr,
                "fdf_residual_complex_arr": resid_fdf_arr,
            }
        )
    )

    clean_parameters = rmclean_scalar_schema_df.vstack(
        pl.DataFrame(
            {
                "mask": mask,
                "threshold": threshold,
                "n_iter": clean_iter_arr,
            }
        )
    )

    return RMClean1DResults(
        fdf_parameters=fdf_parameters,
        fdf_arrs=rmclean_arrs,
        clean_parameters=clean_parameters,
    )
