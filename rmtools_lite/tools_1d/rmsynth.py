#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RM-synthesis on 1D data"""

import math as m
import time
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from rmtools_lite.utils.misc import (
    calculate_StokesI_model,
    create_frac_spectra,
    nanmedian,
    renormalize_StokesI_model,
    toscalar,
)
from rmtools_lite.utils.rmsynth import (
    do_rmsynth_planes,
    get_rmsf_planes,
    measure_FDF_parms,
    measure_qu_complexity,
)


# -----------------------------------------------------------------------------#
def run_rmsynth(
    stokes_q_array: np.ndarray,
    stokes_u_array: np.ndarray,
    stokes_q_error_array: np.ndarray,
    stokes_u_error_array: np.ndarray,
    freq_array_hz: np.ndarray,
    stokes_i_array: Optional[np.ndarray] = None,
    stokes_i_error_array: Optional[np.ndarray] = None,
    stokes_i_model_array: Optional[np.ndarray] = None,
    poly_ord: int = 2,
    phi_max_radm2: Optional[float] = None,
    d_phi_radm2: Optional[float] = None,
    n_samples: Optional[float] = 10.0,
    weight_type: Literal["variance", "uniform"] = "variance",
    fit_rmsf=False,
    # phi_noise_radm2=1e6,
    units: str = "Jy/beam",
    fit_function: Literal["log", "linear"] = "log",
    super_resolution=False,
):
    stokes_qu_error_array = (stokes_q_error_array + stokes_u_error_array) / 2.0

    # Fit the Stokes I spectrum and create the fractional spectra
    fractional_spectra = create_frac_spectra(
        freq_array_hz=freq_array_hz,
        stokes_i_array=stokes_i_array,
        stokes_q_array=stokes_q_array,
        stokes_u_array=stokes_u_array,
        stokes_i_error_array=stokes_i_error_array,
        stokes_q_error_array=stokes_q_error_array,
        stokes_u_error_array=stokes_u_error_array,
        poly_ord=poly_ord,
        fit_function=fit_function,
        stokes_i_model_array=stokes_i_model_array,
    )

    stokes_qu_error_array = np.abs(dqArr + duArr) / 2.0
    stokes_qu_error_array = np.where(
        np.isfinite(stokes_qu_error_array), stokes_qu_error_array, np.nan
    )

    # Plot the data and the Stokes I model fit
    if verbose:
        log("Plotting the input data and spectral index fit.")
    freqHirArr_Hz = np.linspace(freqArr_Hz[0], freqArr_Hz[-1], 10000)
    if stokes_i_model_array is None:
        IModHirArr = calculate_StokesI_model(fit_result, freqHirArr_Hz)
    elif stokes_i_model_array is not None:
        modStokesI_interp = interp1d(freqArr_Hz, stokes_i_model_array)
        IModHirArr = modStokesI_interp(freqHirArr_Hz)
    if showPlots or saveFigures:
        specFig = plt.figure(facecolor="w", figsize=(12.0, 8))
        plot_Ipqu_spectra_fig(
            freqArr_Hz=freqArr_Hz,
            IArr=IArr,
            qArr=qArr,
            uArr=uArr,
            stokes_i_error_array=np.abs(stokes_i_error_array),
            dqArr=np.abs(dqArr),
            duArr=np.abs(duArr),
            freqHirArr_Hz=freqHirArr_Hz,
            IModArr=IModHirArr,
            fig=specFig,
            units=units,
        )

    # -------------------------------------------------------------------------#

    # Calculate some wavelength parameters
    lambdaSqArr_m2 = np.power(C / freqArr_Hz, 2.0)
    lambdaSqRange_m2 = np.nanmax(lambdaSqArr_m2) - np.nanmin(lambdaSqArr_m2)
    dLambdaSqMin_m2 = np.nanmin(np.abs(np.diff(lambdaSqArr_m2)))
    dLambdaSqMax_m2 = np.nanmax(np.abs(np.diff(lambdaSqArr_m2)))

    # Set the Faraday depth range
    if not super_resolution:
        fwhmRMSF_radm2 = 3.8 / lambdaSqRange_m2  # Dickey+2019 theoretical RMSF width
    else:  # If super resolution, use R&C23 theoretical width
        fwhmRMSF_radm2 = 2.0 / (np.nanmax(lambdaSqArr_m2) + np.nanmin(lambdaSqArr_m2))
    if dPhi_radm2 is None:
        dPhi_radm2 = fwhmRMSF_radm2 / nSamples
    if phiMax_radm2 is None:
        phiMax_radm2 = m.sqrt(3.0) / dLambdaSqMax_m2
        phiMax_radm2 = max(
            phiMax_radm2, fwhmRMSF_radm2 * 10.0
        )  # Force the minimum phiMax to 10 FWHM

    # Faraday depth sampling. Zero always centred on middle channel
    nChanRM = int(round(abs((phiMax_radm2 - 0.0) / dPhi_radm2)) * 2.0 + 1.0)
    startPhi_radm2 = -(nChanRM - 1.0) * dPhi_radm2 / 2.0
    stopPhi_radm2 = +(nChanRM - 1.0) * dPhi_radm2 / 2.0
    phiArr_radm2 = np.linspace(startPhi_radm2, stopPhi_radm2, nChanRM)
    phiArr_radm2 = phiArr_radm2.astype(dtFloat)
    if verbose:
        log(
            "PhiArr = %.2f to %.2f by %.2f (%d chans)."
            % (phiArr_radm2[0], phiArr_radm2[-1], float(dPhi_radm2), nChanRM)
        )

    # Calculate the weighting as 1/sigma^2 or all 1s (uniform)
    if weightType == "variance":
        weightArr = 1.0 / np.power(stokes_qu_error_array, 2.0)
    else:
        weightType = "uniform"
        weightArr = np.ones(freqArr_Hz.shape, dtype=dtFloat)
    if verbose:
        log("Weight type is '%s'." % weightType)

    startTime = time.time()

    # Perform RM-synthesis on the spectrum
    dirtyFDF, lam0Sq_m2 = do_rmsynth_planes(
        dataQ=qArr,
        dataU=uArr,
        lambdaSqArr_m2=lambdaSqArr_m2,
        phiArr_radm2=phiArr_radm2,
        weightArr=weightArr,
        nBits=nBits,
        log=log,
        lam0Sq_m2=0 if super_resolution else None,
    )

    # Calculate the Rotation Measure Spread Function
    RMSFArr, phi2Arr_radm2, fwhmRMSFArr, fitStatArr, _ = get_rmsf_planes(
        lambdaSqArr_m2=lambdaSqArr_m2,
        phiArr_radm2=phiArr_radm2,
        weightArr=weightArr,
        mskArr=~np.isfinite(qArr),
        lam0Sq_m2=lam0Sq_m2,
        double=True,
        fitRMSF=fitRMSF or super_resolution,
        fitRMSFreal=super_resolution,
        nBits=nBits,
        verbose=verbose,
        log=log,
    )
    fwhmRMSF = float(fwhmRMSFArr)

    # ALTERNATE RM-SYNTHESIS CODE --------------------------------------------#

    # dirtyFDF, [phi2Arr_radm2, RMSFArr], lam0Sq_m2, fwhmRMSF = \
    #          do_rmsynth(qArr, uArr, lambdaSqArr_m2, phiArr_radm2, weightArr)

    # -------------------------------------------------------------------------#

    endTime = time.time()
    cputime = endTime - startTime
    if verbose:
        log("> RM-synthesis completed in %.2f seconds." % cputime)

    # Convert Stokes I model to polarization reference frequency. If lambda^2_0 is
    # non-zero, use that as polarization reference frequency and adapt Stokes I model.
    # If lambda^2_0 is zero, make polarization reference frequency equal to
    # Stokes I reference frequency.

    if lam0Sq_m2 == 0:  # Rudnick-Cotton adapatation
        freq0_Hz = fit_result.reference_frequency_Hz
    else:  # standard RM-synthesis
        freq0_Hz = C / m.sqrt(lam0Sq_m2)
        if stokes_i_model_array is None:
            fit_result = renormalize_StokesI_model(fit_result, freq0_Hz)
        else:
            fit_result = fit_result.with_options(reference_frequency_Hz=freq0_Hz)

    # Set Ifreq0 (Stokes I at reference frequency) from either supplied model
    # (interpolated as required) or fit model, as appropriate.
    # Multiply the dirty FDF by Ifreq0 to recover the PI
    if stokes_i_model_array is None:
        Ifreq0 = calculate_StokesI_model(fit_result, freq0_Hz)
    elif stokes_i_model_array is not None:
        modStokesI_interp = interp1d(freqArr_Hz, stokes_i_model_array)
        Ifreq0 = modStokesI_interp(freq0_Hz)
    dirtyFDF *= Ifreq0  # FDF is in fracpol units initially, convert back to flux

    # Calculate the theoretical noise in the FDF !!Old formula only works for wariance weights!
    weightArr = np.where(np.isnan(weightArr), 0.0, weightArr)
    dFDFth = np.abs(Ifreq0) * np.sqrt(
        np.nansum(weightArr**2 * np.nan_to_num(stokes_qu_error_array) ** 2)
        / (np.sum(weightArr)) ** 2
    )

    # Measure the parameters of the dirty FDF
    # Use the theoretical noise to calculate uncertainties
    mDict = measure_FDF_parms(
        FDF=dirtyFDF,
        phiArr=phiArr_radm2,
        fwhmRMSF=fwhmRMSF,
        dFDF=dFDFth,
        lamSqArr_m2=lambdaSqArr_m2,
        lam0Sq=lam0Sq_m2,
    )
    mDict["Ifreq0"] = toscalar(Ifreq0)
    mDict["polyCoeffs"] = ",".join(
        [str(x.astype(np.float32)) for x in fit_result.params]
    )
    mDict["polyCoefferr"] = ",".join(
        [str(x.astype(np.float32)) for x in fit_result.perror]
    )
    mDict["poly_ord"] = fit_result.poly_ord
    mDict["IfitStat"] = fit_result.fitStatus
    mDict["IfitChiSqRed"] = fit_result.chiSqRed
    mDict["fit_function"] = fit_function
    mDict["lam0Sq_m2"] = toscalar(lam0Sq_m2)
    mDict["freq0_Hz"] = toscalar(freq0_Hz)
    mDict["fwhmRMSF"] = toscalar(fwhmRMSF)
    mDict["dQU"] = toscalar(nanmedian(stokes_qu_error_array))
    mDict["dFDFth"] = toscalar(dFDFth)
    mDict["units"] = units

    if (fit_result.fitStatus >= 128) and verbose:
        log("WARNING: Stokes I model contains negative values!")
    elif (fit_result.fitStatus >= 64) and verbose:
        log("Caution: Stokes I model has low signal-to-noise.")

    # Add information on nature of channels:
    good_channels = np.where(np.logical_and(weightArr != 0, np.isfinite(qArr)))[0]
    mDict["min_freq"] = float(np.min(freqArr_Hz[good_channels]))
    mDict["max_freq"] = float(np.max(freqArr_Hz[good_channels]))
    mDict["N_channels"] = good_channels.size
    mDict["median_channel_width"] = float(np.median(np.diff(freqArr_Hz)))

    # Measure the complexity of the q and u spectra
    # Use 'ampPeakPIfitEff' for bias correct PI
    mDict["fracPol"] = toscalar(mDict["ampPeakPIfitEff"] / (Ifreq0))
    mD, pD = measure_qu_complexity(
        freqArr_Hz=freqArr_Hz,
        qArr=qArr,
        uArr=uArr,
        dqArr=dqArr,
        duArr=duArr,
        fracPol=mDict["fracPol"],
        psi0_deg=mDict["polAngle0Fit_deg"],
        RM_radm2=mDict["phiPeakPIfit_rm2"],
    )
    mDict.update(mD)

    # Debugging plots for spectral complexity measure
    if debug:
        tmpFig = plot_complexity_fig(
            xArr=pD["xArrQ"],
            qArr=pD["yArrQ"],
            dqArr=pD["dyArrQ"],
            sigmaAddqArr=pD["sigmaAddArrQ"],
            chiSqRedqArr=pD["chiSqRedArrQ"],
            probqArr=pD["probArrQ"],
            uArr=pD["yArrU"],
            duArr=pD["dyArrU"],
            sigmaAdduArr=pD["sigmaAddArrU"],
            chiSqReduArr=pD["chiSqRedArrU"],
            probuArr=pD["probArrU"],
            mDict=mDict,
        )
        if saveFigures:
            if verbose:
                print("Saving debug plots:")
            outFilePlot = prefixOut + ".debug-plots.pdf"
            if verbose:
                print("> " + outFilePlot)
            tmpFig.savefig(outFilePlot, bbox_inches="tight")
        else:
            tmpFig.show()

    # add array dictionary
    aDict = dict()
    aDict["phiArr_radm2"] = phiArr_radm2
    aDict["phi2Arr_radm2"] = phi2Arr_radm2
    aDict["RMSFArr"] = RMSFArr
    aDict["freqArr_Hz"] = freqArr_Hz
    aDict["weightArr"] = weightArr
    aDict["dirtyFDF"] = dirtyFDF

    if verbose:
        # Print the results to the screen
        log()
        log("-" * 80)
        log("RESULTS:\n")
        log("FWHM RMSF = %.4g rad/m^2" % (mDict["fwhmRMSF"]))

        log(
            "Pol Angle = %.4g (+/-%.4g) deg"
            % (mDict["polAngleFit_deg"], mDict["dPolAngleFit_deg"])
        )
        log(
            "Pol Angle 0 = %.4g (+/-%.4g) deg"
            % (mDict["polAngle0Fit_deg"], mDict["dPolAngle0Fit_deg"])
        )
        log(
            "Peak FD = %.4g (+/-%.4g) rad/m^2"
            % (mDict["phiPeakPIfit_rm2"], mDict["dPhiPeakPIfit_rm2"])
        )
        log("freq0_GHz = %.4g " % (mDict["freq0_Hz"] / 1e9))
        log("I freq0 = %.4g %s" % (mDict["Ifreq0"], units))
        log(
            "Peak PI = %.4g (+/-%.4g) %s"
            % (mDict["ampPeakPIfit"], mDict["dAmpPeakPIfit"], units)
        )
        log("QU Noise = %.4g %s" % (mDict["dQU"], units))
        log("FDF Noise (theory)   = %.4g %s" % (mDict["dFDFth"], units))
        log("FDF Noise (Corrected MAD) = %.4g %s" % (mDict["dFDFcorMAD"], units))
        log("FDF SNR = %.4g " % (mDict["snrPIfit"]))
        log(
            "sigma_add(q) = %.4g (+%.4g, -%.4g)"
            % (mDict["sigmaAddQ"], mDict["dSigmaAddPlusQ"], mDict["dSigmaAddMinusQ"])
        )
        log(
            "sigma_add(u) = %.4g (+%.4g, -%.4g)"
            % (mDict["sigmaAddU"], mDict["dSigmaAddPlusU"], mDict["dSigmaAddMinusU"])
        )
        log("Fitted polynomial order = {} ".format(mDict["poly_ord"]))
        log()
        log("-" * 80)

    # Plot the RM Spread Function and dirty FDF
    if showPlots or saveFigures:
        fdfFig = plt.figure(facecolor="w", figsize=(12.0, 8))
        plot_rmsf_fdf_fig(
            phiArr=phiArr_radm2,
            FDF=dirtyFDF,
            phi2Arr=phi2Arr_radm2,
            RMSFArr=RMSFArr,
            fwhmRMSF=fwhmRMSF,
            vLine=mDict["phiPeakPIfit_rm2"],
            fig=fdfFig,
            units=units,
        )

    # Pause if plotting enabled
    if showPlots:
        plt.show()
    if saveFigures or debug:
        if verbose:
            print("Saving RMSF and dirty FDF plot:")
        outFilePlot = prefixOut + "_RMSF-dirtyFDF-plots.pdf"
        if verbose:
            print("> " + outFilePlot)
        fdfFig.savefig(outFilePlot, bbox_inches="tight")
        #        #if verbose: print "Press <RETURN> to exit ...",
    #        input()

    return mDict, aDict
