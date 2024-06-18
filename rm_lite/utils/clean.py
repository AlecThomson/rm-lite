#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RM-clean utils"""

from typing import NamedTuple

import numpy as np


class RMCleanResults(NamedTuple):
    """Results of the RM-CLEAN calculation"""

    cleanFDF: np.ndarray
    """The cleaned Faraday dispersion function cube"""
    ccArr: np.ndarray
    """The clean components cube"""
    iterCountArr: np.ndarray
    """The number of iterations for each pixel"""
    resifdf_error: np.ndarray
    """The residual Faraday dispersion function cube"""


def do_rmclean_hogbom(
    dirtyFDF,
    phi_arr_radm2,
    RMSFArr,
    phi_double_arr_radm2_radm2,
    fwhm_rmsf_arr,
    cutoff,
    maxIter=1000,
    gain=0.1,
    mask_array=None,
    nBits=32,
    verbose=False,
    doPlots=False,
    pool=None,
    chunksize=None,
    log=print,
    window=0,
) -> RMCleanResults:
    """Perform Hogbom CLEAN on a cube of complex Faraday dispersion functions
    given a cube of rotation measure spread functions.

    dirtyFDF       ... 1, 2 or 3D complex FDF array
    phi_arr_radm2   ... 1D Faraday depth array corresponding to the FDF
    RMSFArr        ... 1, 2 or 3D complex RMSF array
    phi_double_arr_radm2_radm2  ... double size 1D Faraday depth array of the RMSF
    fwhm_rmsf_arr    ... scalar, 1D or 2D array of RMSF main lobe widths
    cutoff         ... clean cutoff (+ve = absolute values, -ve = sigma) [-1]
    maxIter        ... maximun number of CLEAN loop interations [1000]
    gain           ... CLEAN loop gain [0.1]
    mask_array         ... scalar, 1D or 2D pixel mask array [None]
    nBits          ... precision of data arrays [32]
    verbose        ... print feedback during calculation [False]
    doPlots        ... plot the final CLEAN FDF [False]
    pool           ... thread pool for multithreading (from schwimmbad) [None]
    chunksize      ... number of pixels to be given per thread (for 3D) [None]
    log            ... function to be used to output messages [print]
    window         ... Only clean in ±RMSF_FWHM window around first peak [False]

    """

    # Default data types
    dtFloat = "float" + str(nBits)
    dtComplex = "complex" + str(2 * nBits)

    # Sanity checks on array sizes
    n_phi = phi_arr_radm2.shape[0]
    if n_phi != dirtyFDF.shape[0]:
        logger.error("'phi_arr_radm2' and 'dirtyFDF' are not the same length.")
        return None, None, None, None
    n_phi2 = phi_double_arr_radm2_radm2.shape[0]
    if not n_phi2 == RMSFArr.shape[0]:
        logger.error("missmatch in 'phi_double_arr_radm2_radm2' and 'RMSFArr' length.")
        return None, None, None, None
    if not (n_phi2 >= 2 * n_phi):
        logger.error("the Faraday depth of the RMSF must be twice the FDF.")
        return None, None, None, None
    n_dimension = len(dirtyFDF.shape)
    if not n_dimension <= 3:
        logger.error("FDF array dimensions must be <= 3.")
        return None, None, None, None
    if not n_dimension == len(RMSFArr.shape):
        logger.error("the input RMSF and FDF must have the same number of axes.")
        return None, None, None, None
    if not RMSFArr.shape[1:] == dirtyFDF.shape[1:]:
        logger.error("the xy dimesions of the RMSF and FDF must match.")
        return None, None, None, None
    if mask_array is not None:
        if not mask_array.shape == dirtyFDF.shape[1:]:
            logger.error("pixel mask must match xy dimesnisons of FDF cube.")
            log(
                "     FDF[z,y,z] = {:}, Mask[y,x] = {:}.".format(
                    dirtyFDF.shape, mask_array.shape
                ),
                end=" ",
            )

            return None, None, None, None
    else:
        mask_array = np.ones(dirtyFDF.shape[1:], dtype="bool")

    # Reshape the FDF & RMSF array to 3 dimensions and mask array to 2
    if n_dimension == 1:
        dirtyFDF = np.reshape(dirtyFDF, (dirtyFDF.shape[0], 1, 1))
        RMSFArr = np.reshape(RMSFArr, (RMSFArr.shape[0], 1, 1))
        mask_array = np.reshape(mask_array, (1, 1))
        fwhm_rmsf_arr = np.reshape(fwhm_rmsf_arr, (1, 1))
    elif n_dimension == 2:
        dirtyFDF = np.reshape(dirtyFDF, list(dirtyFDF.shape[:2]) + [1])
        RMSFArr = np.reshape(RMSFArr, list(RMSFArr.shape[:2]) + [1])
        mask_array = np.reshape(mask_array, (dirtyFDF.shape[1], 1))
        fwhm_rmsf_arr = np.reshape(fwhm_rmsf_arr, (dirtyFDF.shape[1], 1))
    iterCountArr = np.zeros_like(mask_array, dtype="int")

    # Determine which pixels have components above the cutoff
    abs_fdf_cube = np.abs(np.nan_to_num(dirtyFDF))
    mskCutoff = np.where(np.max(abs_fdf_cube, axis=0) >= cutoff, 1, 0)
    xyCoords = np.rot90(np.where(mskCutoff > 0))

    # Feeback to user
    if verbose:
        num_pixels = dirtyFDF.shape[-1] * dirtyFDF.shape[-2]
        nCleanum_pixels = len(xyCoords)
        log("Cleaning {:}/{:} spectra.".format(nCleanum_pixels, num_pixels))

    # Initialise arrays to hold the residual FDF, clean components, clean FDF
    # Residual is initially copies of dirty FDF, so that pixels that are not
    #  processed get correct values (but will be overridden when processed)
    resifdf_error = dirtyFDF.copy()
    ccArr = np.zeros(dirtyFDF.shape, dtype=dtComplex)
    cleanFDF = np.zeros_like(dirtyFDF)

    # Loop through the pixels containing a polarised signal
    inputs = [[yi, xi, dirtyFDF] for yi, xi in xyCoords]
    rmc = RMcleaner(
        RMSFArr,
        phi_double_arr_radm2_radm2,
        phi_arr_radm2,
        fwhm_rmsf_arr,
        iterCountArr,
        maxIter,
        gain,
        cutoff,
        nBits,
        verbose,
        window,
    )

    if pool is None:
        output = []
        for pix in inputs:
            output.append(rmc.cleanloop(pix))
    else:
        output = list(
            tqdm(
                pool.imap(
                    rmc.cleanloop,
                    inputs,
                    chunksize=chunksize if chunksize is not None else 1,
                ),
                desc="RM-CLEANing",
                disable=not verbose,
                total=len(inputs),
            )
        )
        pool.close()
    # Put data back in correct shape
    #    ccArr = np.reshape(np.rot90(np.stack([model for _, _, model in output]), k=-1),dirtyFDF.shape)
    #    cleanFDF = np.reshape(np.rot90(np.stack([clean for clean, _, _ in output]), k=-1),dirtyFDF.shape)
    #    resifdf_error = np.reshape(np.rot90(np.stack([resid for _, resid, _ in output]), k=-1),dirtyFDF.shape)
    for i in range(len(inputs)):
        yi = inputs[i][0]
        xi = inputs[i][1]
        ccArr[:, yi, xi] = output[i][2]
        cleanFDF[:, yi, xi] = output[i][0]
        resifdf_error[:, yi, xi] = output[i][1]

    # Restore the residual to the CLEANed FDF (moved outside of loop:
    # will now work for pixels/spectra without clean components)
    cleanFDF += resifdf_error

    # Remove redundant dimensions
    cleanFDF = np.squeeze(cleanFDF)
    ccArr = np.squeeze(ccArr)
    iterCountArr = np.squeeze(iterCountArr)
    resifdf_error = np.squeeze(resifdf_error)

    return RMCleanResults(cleanFDF, ccArr, iterCountArr, resifdf_error)


# -----------------------------------------------------------------------------#
class CleanLoopResults(NamedTuple):
    """Results of the RM-CLEAN loop"""

    cleanFDF: np.ndarray
    """The cleaned Faraday dispersion function cube"""
    resifdf_error: np.ndarray
    """The residual Faraday dispersion function cube"""
    ccArr: np.ndarray
    """The clean components cube"""


class RMcleaner:
    """Allows do_rmclean_hogbom to be run in parallel
    Designed around use of schwimmbad parallelization tools.
    """

    def __init__(
        self,
        RMSFArr,
        phi_double_arr_radm2_radm2,
        phi_arr_radm2,
        fwhm_rmsf_arr,
        iterCountArr,
        maxIter=1000,
        gain=0.1,
        cutoff=0,
        nbits=32,
        verbose=False,
        window=0,
    ):
        self.RMSFArr = RMSFArr
        self.phi_double_arr_radm2_radm2 = phi_double_arr_radm2_radm2
        self.phi_arr_radm2 = phi_arr_radm2
        self.fwhm_rmsf_arr = fwhm_rmsf_arr
        self.iterCountArr = iterCountArr
        self.maxIter = maxIter
        self.gain = gain
        self.cutoff = cutoff
        self.verbose = verbose
        self.nbits = nbits
        self.window = window

    def cleanloop(self, args) -> CleanLoopResults:
        return self._cleanloop(*args)

    def _cleanloop(self, yi, xi, dirtyFDF) -> CleanLoopResults:
        dirtyFDF = dirtyFDF[:, yi, xi]
        # Initialise arrays to hold the residual FDF, clean components, clean FDF
        resifdf_error = dirtyFDF.copy()
        ccArr = np.zeros_like(dirtyFDF)
        cleanFDF = np.zeros_like(dirtyFDF)
        RMSFArr = self.RMSFArr[:, yi, xi]
        fwhm_rmsf_arr = self.fwhm_rmsf_arr[yi, xi]

        # Find the index of the peak of the RMSF
        indxMaxRMSF = np.nanargmax(RMSFArr)

        # Calculate the padding in the sampled RMSF
        # Assumes only integer shifts and symmetric
        n_phiPad = int(
            (len(self.phi_double_arr_radm2_radm2) - len(self.phi_arr_radm2)) / 2
        )

        iterCount = 0
        while np.max(np.abs(resifdf_error)) >= self.cutoff and iterCount < self.maxIter:
            # Get the absolute peak channel, values and Faraday depth
            indxPeakFDF = np.argmax(np.abs(resifdf_error))
            peakFDFval = resifdf_error[indxPeakFDF]
            phiPeak = self.phi_arr_radm2[indxPeakFDF]

            # A clean component is "loop-gain * peakFDFval
            CC = self.gain * peakFDFval
            ccArr[indxPeakFDF] += CC

            # At which channel is the CC located at in the RMSF?
            indxPeakRMSF = indxPeakFDF + n_phiPad

            # Shift the RMSF & clip so that its peak is centred above this CC
            shiftedRMSFArr = np.roll(RMSFArr, indxPeakRMSF - indxMaxRMSF)[
                n_phiPad:-n_phiPad
            ]

            # Subtract the product of the CC shifted RMSF from the residual FDF
            resifdf_error -= CC * shiftedRMSFArr

            # Restore the CC * a Gaussian to the cleaned FDF
            # cleanFDF += gauss1D(CC, phiPeak, fwhm_rmsf_arr)(self.phi_arr_radm2)
            cleanFDF += Gaussian1D(
                amplitude=CC, mean=phiPeak, stddev=fwhm_rmsf_arr / 2.355
            )(self.phi_arr_radm2)
            iterCount += 1
            self.iterCountArr[yi, xi] = iterCount

        # Create a mask for the pixels that have been cleaned
        mask = np.abs(ccArr) > 0
        delta_phi = self.phi_arr_radm2[1] - self.phi_arr_radm2[0]
        fwhm_rmsf_arr_pix = fwhm_rmsf_arr / delta_phi
        for i in np.where(mask)[0]:
            start = int(i - fwhm_rmsf_arr_pix / 2)
            end = int(i + fwhm_rmsf_arr_pix / 2)
            mask[start:end] = True
        resifdf_error_mask = np.ma.array(resifdf_error, mask=~mask)
        # Clean again within mask
        while (
            np.ma.max(np.ma.abs(resifdf_error_mask)) >= self.window
            and iterCount < self.maxIter
        ):
            if resifdf_error_mask.mask.all():
                break
            # Get the absolute peak channel, values and Faraday depth
            indxPeakFDF = np.ma.argmax(np.abs(resifdf_error_mask))
            peakFDFval = resifdf_error_mask[indxPeakFDF]
            phiPeak = self.phi_arr_radm2[indxPeakFDF]

            # A clean component is "loop-gain * peakFDFval
            CC = self.gain * peakFDFval
            ccArr[indxPeakFDF] += CC

            # At which channel is the CC located at in the RMSF?
            indxPeakRMSF = indxPeakFDF + n_phiPad

            # Shift the RMSF & clip so that its peak is centred above this CC
            shiftedRMSFArr = np.roll(RMSFArr, indxPeakRMSF - indxMaxRMSF)[
                n_phiPad:-n_phiPad
            ]

            # Subtract the product of the CC shifted RMSF from the residual FDF
            resifdf_error -= CC * shiftedRMSFArr

            # Restore the CC * a Gaussian to the cleaned FDF
            cleanFDF += Gaussian1D(
                amplitude=CC, mean=phiPeak, stddev=fwhm_rmsf_arr / 2.355
            )(self.phi_arr_radm2)
            iterCount += 1
            self.iterCountArr[yi, xi] = iterCount

            # Remake masked residual FDF
            resifdf_error_mask = np.ma.array(resifdf_error, mask=~mask)

        cleanFDF = np.squeeze(cleanFDF)
        resifdf_error = np.squeeze(resifdf_error)
        ccArr = np.squeeze(ccArr)

        return CleanLoopResults(
            cleanFDF=cleanFDF, resifdf_error=resifdf_error, ccArr=ccArr
        )
