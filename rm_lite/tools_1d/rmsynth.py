#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RM-synthesis on 1D data"""

import time
from typing import Literal, NamedTuple, Optional

import numpy as np
from scipy import interpolate

from rm_lite.utils.fitting import FitResult
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


class RMSynth1DResults(NamedTuple):
    """Results of RM-synthesis"""

    fdf_parameters: FDFParameters
    """ Parameters of the Faraday dispersion function """
    stokes_i_fit_result: FitResult
    """ Fit result of the Stokes I spectrum """
    arrays: RMSynth1DArrays
    """ Resulting arrays from RM-synthesis """


class RMSynthArrays(NamedTuple):
    """Result arrays from RM-synthesis"""


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
):
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
        lambda_sq_arr_m2=lambda_sq_arr_m2,
        lam_sq_0_m2=lam_sq_0_m2,
        fdf_error=theoretical_noise.fdf_error_noise,
    )
    fdf_arrays = RMSynth1DArrays(
        phi_arr_radm2=phi_arr_radm2,
        phi2_arr_radm2=phi_double_arr_radm2,
        rmsf_array=rmsf_array,
        freq_array_hz=freq_array_hz,
        weight_array=weight_array,
        fdf_dirty_array=fdf_dirty_array,
    )
    return RMSynth1DResults(
        fdf_parameters=fdf_parameters,
        stokes_i_fit_result=fit_result,
        arrays=fdf_arrays,
    )
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
