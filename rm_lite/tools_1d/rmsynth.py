#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RM-synthesis on 1D data"""

import logging
import time
from typing import Literal, NamedTuple, Optional

from astropy.constants import c as speed_of_light
import numpy as np
from scipy.interpolate import interp1d

from rm_lite.utils.misc import (
    create_fractional_spectra,
    renormalize_StokesI_model,
)
from rm_lite.utils.rmsynth import (
    rmsynth_nufft,
    get_rmsf_nufft,
    measure_FDF_parms,
    measure_qu_complexity,
    make_phi_array,
    get_fwhm_rmsf,
)

from rm_lite.utils.logging import logger

logger.setLevel(logging.INFO)


class RMSynthParams(NamedTuple):
    lambda_sq_arr_m2: np.ndarray
    phi_arr_radm2: np.ndarray
    weight_array: np.ndarray


def compute_rmsynth_params(
    freq_array_hz: np.ndarray,
    stokes_qu_error_array: np.ndarray,
    d_phi_radm2: Optional[float] = None,
    n_samples: Optional[float] = 10.0,
    phi_max_radm2: Optional[float] = None,
    super_resolution: bool = False,
    weight_type: Literal["variance", "uniform"] = "variance",
) -> RMSynthParams:
    lambda_sq_arr_m2 = (speed_of_light.value / freq_array_hz) ** 2.0

    fwhm_rmsf_radm2, d_lambda_sq_max_m2, lambda_sq_range_m2 = get_fwhm_rmsf(
        lambda_sq_arr_m2, super_resolution
    )

    if d_phi_radm2 is None:
        if n_samples is None:
            raise ValueError("Either d_phi_radm2 or n_samples must be provided.")
        d_phi_radm2 = fwhm_rmsf_radm2 / n_samples
    if phi_max_radm2 is None:
        phi_max_radm2 = np.sqrt(3.0) / d_lambda_sq_max_m2
        phi_max_radm2 = max(
            phi_max_radm2, fwhm_rmsf_radm2 * 10.0
        )  # Force the minimum phiMax to 10 FWHM

    phi_arr_radm2 = make_phi_array(phi_max_radm2, d_phi_radm2)

    logger.info(
        f"phi = {phi_arr_radm2[0]:0.2f} to {phi_arr_radm2[-1]:0.2f} by {d_phi_radm2:0.2f} ({len(phi_arr_radm2)} chans)."
    )

    # Calculate the weighting as 1/sigma^2 or all 1s (uniform)
    if weight_type == "variance":
        weight_array = 1.0 / stokes_qu_error_array**2
    else:
        weight_array = np.ones_like(freq_array_hz)

    return RMSynthParams(
        lambda_sq_arr_m2=lambda_sq_arr_m2,
        phi_arr_radm2=phi_arr_radm2,
        weight_array=weight_array,
    )


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
    # phi_noise_radm2=1e6,
    units: str = "Jy/beam",
    fit_function: Literal["log", "linear"] = "log",
    super_resolution=False,
):
    if stokes_i_array is None:
        logger.warning(
            "Stokes I array not provided. No fractional polarization will be calculated."
        )
        stokes_i_array = np.ones_like(stokes_q_array)
        stokes_i_error_array = np.zeros_like(stokes_q_array)

    fractional_spectra = create_fractional_spectra(
        freq_array_hz=freq_array_hz,
        stokes_i_array=stokes_i_array,
        stokes_q_array=stokes_q_array,
        stokes_u_array=stokes_u_array,
        stokes_i_error_array=stokes_i_error_array,
        stokes_q_error_array=stokes_q_error_array,
        stokes_u_error_array=stokes_u_error_array,
        fit_order=fit_order,
        fit_function=fit_function,
        stokes_i_model_array=stokes_i_model_array,
    )
    stokes_q_array, stokes_u_array, stokes_q_error_array, stokes_u_error_array = (
        fractional_spectra.stokes_q_array,
        fractional_spectra.stokes_u_array,
        fractional_spectra.stokes_q_error_array,
        fractional_spectra.stokes_u_error_array,
    )

    lambda_sq_arr_m2, phi_arr_radm2, weight_array = compute_rmsynth_params(
        freq_array_hz=freq_array_hz,
        stokes_qu_error_array=np.sqrt(
            stokes_q_error_array**2 + stokes_u_error_array**2
        ),
        d_phi_radm2=d_phi_radm2,
        n_samples=n_samples,
        phi_max_radm2=phi_max_radm2,
        super_resolution=super_resolution,
        weight_type=weight_type,
    )

    tick = time.time()

    # Perform RM-synthesis on the spectrum
    fdf_dirty_cube, lam_sq_0_m2 = rmsynth_nufft(
        stokes_q_array=stokes_q_array,
        stokes_u_array=stokes_u_array,
        lambda_sq_arr_m2=lambda_sq_arr_m2,
        phi_arr_radm2=phi_arr_radm2,
        weight_array=weight_array,
        lam_sq_0_m2=0 if super_resolution else None,
    )

    # Calculate the Rotation Measure Spread Function
    rmsf_cube, phi_double_arr_radm2, fwhm_rmsf_arr, fit_status_array = get_rmsf_nufft(
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

    # Convert Stokes I model to polarization reference frequency. If lambda^2_0 is
    # non-zero, use that as polarization reference frequency and adapt Stokes I model.
    # If lambda^2_0 is zero, make polarization reference frequency equal to
    # Stokes I reference frequency.

    # if lam_sq_0_m2 == 0:  # Rudnick-Cotton adapatation
    #     freq0_Hz = fit_result.reference_frequency_Hz
    # else:  # standard RM-synthesis
    #     freq0_Hz = C / m.sqrt(lam_sq_0_m2)
    #     if stokes_i_model_array is None:
    #         fit_result = renormalize_StokesI_model(fit_result, freq0_Hz)
    #     else:
    #         fit_result = fit_result.with_options(reference_frequency_Hz=freq0_Hz)

    # # Set Ifreq0 (Stokes I at reference frequency) from either supplied model
    # # (interpolated as required) or fit model, as appropriate.
    # # Multiply the dirty FDF by Ifreq0 to recover the PI
    # if stokes_i_model_array is None:
    #     Ifreq0 = calculate_StokesI_model(fit_result, freq0_Hz)
    # elif stokes_i_model_array is not None:
    #     modStokesI_interp = interp1d(freqArr_Hz, stokes_i_model_array)
    #     Ifreq0 = modStokesI_interp(freq0_Hz)
    # dirtyFDF *= Ifreq0  # FDF is in fracpol units initially, convert back to flux

    # # Calculate the theoretical noise in the FDF !!Old formula only works for wariance weights!
    # weight_array = np.where(np.isnan(weight_array), 0.0, weight_array)
    # dFDFth = np.abs(Ifreq0) * np.sqrt(
    #     np.nansum(weight_array**2 * np.nan_to_num(stokes_qu_error_array) ** 2)
    #     / (np.sum(weight_array)) ** 2
    # )

    # # Measure the parameters of the dirty FDF
    # # Use the theoretical noise to calculate uncertainties
    # mDict = measure_FDF_parms(
    #     FDF=dirtyFDF,
    #     phiArr=phi_arr_radm2,
    #     fwhmRMSF=fwhmRMSF,
    #     dFDF=dFDFth,
    #     lamSqArr_m2=lambda_sq_arr_m2,
    #     lam0Sq=lam_sq_0_m2,
    # )
    # mDict["Ifreq0"] = toscalar(Ifreq0)
    # mDict["polyCoeffs"] = ",".join(
    #     [str(x.astype(np.float32)) for x in fit_result.params]
    # )
    # mDict["polyCoefferr"] = ",".join(
    #     [str(x.astype(np.float32)) for x in fit_result.perror]
    # )
    # mDict["fit_order"] = fit_result.fit_order
    # mDict["IfitStat"] = fit_result.fitStatus
    # mDict["IfitChiSqRed"] = fit_result.chiSqRed
    # mDict["fit_function"] = fit_function
    # mDict["lam_sq_0_m2"] = toscalar(lam_sq_0_m2)
    # mDict["freq0_Hz"] = toscalar(freq0_Hz)
    # mDict["fwhmRMSF"] = toscalar(fwhmRMSF)
    # mDict["dQU"] = toscalar(nanmedian(stokes_qu_error_array))
    # mDict["dFDFth"] = toscalar(dFDFth)
    # mDict["units"] = units

    # if (fit_result.fitStatus >= 128) and verbose:
    #     logger.warning("Stokes I model contains negative values!")
    # elif (fit_result.fitStatus >= 64) and verbose:
    #     logger.warning("Stokes I model has low signal-to-noise.")

    # # Add information on nature of channels:
    # good_channels = np.where(np.logical_and(weight_array != 0, np.isfinite(qArr)))[0]
    # mDict["min_freq"] = float(np.min(freqArr_Hz[good_channels]))
    # mDict["max_freq"] = float(np.max(freqArr_Hz[good_channels]))
    # mDict["N_channels"] = good_channels.size
    # mDict["median_channel_width"] = float(np.median(np.diff(freqArr_Hz)))

    # # Measure the complexity of the q and u spectra
    # # Use 'ampPeakPIfitEff' for bias correct PI
    # mDict["fracPol"] = toscalar(mDict["ampPeakPIfitEff"] / (Ifreq0))
    # mD, pD = measure_qu_complexity(
    #     freqArr_Hz=freqArr_Hz,
    #     qArr=qArr,
    #     uArr=uArr,
    #     dqArr=dqArr,
    #     duArr=duArr,
    #     fracPol=mDict["fracPol"],
    #     psi0_deg=mDict["polAngle0Fit_deg"],
    #     RM_radm2=mDict["phiPeakPIfit_rm2"],
    # )
    # mDict.update(mD)

    # # add array dictionary
    # aDict = dict()
    # aDict["phi_arr_radm2"] = phi_arr_radm2
    # aDict["phi2Arr_radm2"] = phi2Arr_radm2
    # aDict["RMSFArr"] = RMSFArr
    # aDict["freqArr_Hz"] = freqArr_Hz
    # aDict["weight_array"] = weight_array
    # aDict["dirtyFDF"] = dirtyFDF

    # if verbose:
    #     # Print the results to the screen
    #     log()
    #     log("-" * 80)
    #     log("RESULTS:\n")
    #     log("FWHM RMSF = %.4g rad/m^2" % (mDict["fwhmRMSF"]))

    #     log(
    #         "Pol Angle = %.4g (+/-%.4g) deg"
    #         % (mDict["polAngleFit_deg"], mDict["dPolAngleFit_deg"])
    #     )
    #     log(
    #         "Pol Angle 0 = %.4g (+/-%.4g) deg"
    #         % (mDict["polAngle0Fit_deg"], mDict["dPolAngle0Fit_deg"])
    #     )
    #     log(
    #         "Peak FD = %.4g (+/-%.4g) rad/m^2"
    #         % (mDict["phiPeakPIfit_rm2"], mDict["dPhiPeakPIfit_rm2"])
    #     )
    #     log("freq0_GHz = %.4g " % (mDict["freq0_Hz"] / 1e9))
    #     log("I freq0 = %.4g %s" % (mDict["Ifreq0"], units))
    #     log(
    #         "Peak PI = %.4g (+/-%.4g) %s"
    #         % (mDict["ampPeakPIfit"], mDict["dAmpPeakPIfit"], units)
    #     )
    #     log("QU Noise = %.4g %s" % (mDict["dQU"], units))
    #     log("FDF Noise (theory)   = %.4g %s" % (mDict["dFDFth"], units))
    #     log("FDF Noise (Corrected MAD) = %.4g %s" % (mDict["dFDFcorMAD"], units))
    #     log("FDF SNR = %.4g " % (mDict["snrPIfit"]))
    #     log(
    #         "sigma_add(q) = %.4g (+%.4g, -%.4g)"
    #         % (mDict["sigmaAddQ"], mDict["dSigmaAddPlusQ"], mDict["dSigmaAddMinusQ"])
    #     )
    #     log(
    #         "sigma_add(u) = %.4g (+%.4g, -%.4g)"
    #         % (mDict["sigmaAddU"], mDict["dSigmaAddPlusU"], mDict["dSigmaAddMinusU"])
    #     )
    #     log("Fitted polynomial order = {} ".format(mDict["fit_order"]))
    #     log()
    #     log("-" * 80)

    # return mDict, aDict
