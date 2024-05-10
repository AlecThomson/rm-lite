#!/usr/bin/env python
# =============================================================================#
#                                                                             #
# NAME:     util_RM.py                                                        #
#                                                                             #
# PURPOSE:  Common procedures used with RM-synthesis scripts.                 #
#                                                                             #
# REQUIRED: Requires the numpy and scipy modules.                             #
#                                                                             #
# MODIFIED: 16-Nov-2018 by J. West                                            #
#                                                                             #
# CONTENTS:                                                                   #
#                                                                             #
#  do_rmsynth_planes   ... perform RM-synthesis on Q & U data cubes           #
#  get_rmsf_planes     ... calculate the RMSF for a cube of data              #
#  do_rmclean_hogbom   ... perform Hogbom RM-clean on a dirty FDF             #
#  fits_make_lin_axis  ... create an array of absica values for a lin axis    #
#  extrap              ... interpolate and extrapolate an array               #
#  fit_rmsf            ... fit a Gaussian to the main lobe of the RMSF        #
#  gauss1D             ... return a function to evaluate a 1D Gaussian        #
#  detect_peak         ... detect the extent of a peak in a 1D array          #
#  measure_FDF_parms   ... measure parameters of a Faraday dispersion func    #
#  norm_cdf            ... calculate the CDF of a Normal distribution         #
#  cdf_percentile      ... return the value at the given percentile of a CDF  #
#  calc_sigma_add      ... calculate most likely additional scatter           #
#  calc_normal_tests   ... calculate metrics measuring deviation from normal  #
#  measure_qu_complexity  ... measure the complexity of a q & u spectrum      #
#  measure_fdf_complexity  ... measure the complexity of a clean FDF spectrum #
#                                                                             #
# DEPRECATED CODE ------------------------------------------------------------#
#                                                                             #
#  do_rmsynth          ... perform RM-synthesis on Q & U data by spectrum     #
#  get_RMSF            ... calculate the RMSF for a 1D wavelength^2 array     #
#  do_rmclean          ... perform Hogbom RM-clean on a dirty FDF             #
#  plot_complexity     ... plot the residual, PDF and CDF (deprecated)        #
#                                                                             #
# =============================================================================#
#                                                                             #
# The MIT License (MIT)                                                       #
#                                                                             #
# Copyright (c) 2015 - 2018 Cormac R. Purcell                                 #
#                                                                             #
# Permission is hereby granted, free of charge, to any person obtaining a     #
# copy of this software and associated documentation files (the "Software"),  #
# to deal in the Software without restriction, including without limitation   #
# the rights to use, copy, modify, merge, publish, distribute, sublicense,    #
# and/or sell copies of the Software, and to permit persons to whom the       #
# Software is furnished to do so, subject to the following conditions:        #
#                                                                             #
# The above copyright notice and this permission notice shall be included in  #
# all copies or substantial portions of the Software.                         #
#                                                                             #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,    #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER      #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING     #
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER         #
# DEALINGS IN THE SOFTWARE.                                                   #
#                                                                             #
# =============================================================================#

import gc
import math as m
from typing import NamedTuple

import finufft
import numpy as np
from scipy.stats import anderson, kstest, kurtosis, kurtosistest, norm, skew, skewtest
from tqdm.auto import tqdm, trange
from utils.mpfit import mpfit
from utils.util_misc import (
    MAD,
    calc_mom2_FDF,
    calc_parabola_vertex,
    create_pqu_spectra_burn,
    toscalar,
)

# Constants
C = 2.99792458e8


# -----------------------------------------------------------------------------#
class RMsynthResults(NamedTuple):
    """Results of the RM-synthesis calculation"""

    FDFcube: np.ndarray
    """The Faraday dispersion function cube"""
    lam0Sq_m2: float
    """The reference lambda^2 value"""


def do_rmsynth_planes(
    dataQ,
    dataU,
    lambdaSqArr_m2,
    phiArr_radm2,
    weightArr=None,
    lam0Sq_m2=None,
    nBits=32,
    eps=1e-6,
    log=print,
) -> RMsynthResults:
    """Perform RM-synthesis on Stokes Q and U cubes (1,2 or 3D). This version
    of the routine loops through spectral planes and is faster than the pixel-
    by-pixel code. This version also correctly deals with isolated clumps of
    NaN-flagged voxels within the data-cube (unlikely in interferometric cubes,
    but possible in single-dish cubes). Input data must be in standard python
    [z,y,x] order, where z is the frequency axis in ascending order.

    dataQ           ... 1, 2 or 3D Stokes Q data array
    dataU           ... 1, 2 or 3D Stokes U data array
    lambdaSqArr_m2  ... vector of wavelength^2 values (assending freq order)
    phiArr_radm2    ... vector of trial Faraday depth values
    weightArr       ... vector of weights, default [None] is Uniform (all 1s)
    lam0Sq_m2       ... force a reference lambda^2 value [None, calculated]
    nBits           ... precision of data arrays [32]
    eps             ... NUFFT tolerance [1e-6] (see https://finufft.readthedocs.io/en/latest/python.html#module-finufft)
    log             ... function to be used to output messages [print]

    """

    # Default data types
    dtFloat = "float" + str(nBits)

    # Set the weight array
    if weightArr is None:
        weightArr = np.ones(lambdaSqArr_m2.shape, dtype=dtFloat)
    weightArr = np.where(np.isnan(weightArr), 0.0, weightArr)

    # Sanity check on array sizes
    if not weightArr.shape == lambdaSqArr_m2.shape:
        log("Err: Lambda^2 and weight arrays must be the same shape.")
        return None, None
    if not dataQ.shape == dataU.shape:
        log("Err: Stokes Q and U data arrays must be the same shape.")
        return None, None
    nDims = len(dataQ.shape)
    if not nDims <= 3:
        log("Err: data dimensions must be <= 3.")
        return None, None
    if not dataQ.shape[0] == lambdaSqArr_m2.shape[0]:
        log(
            "Err: Data depth does not match lambda^2 vector ({} vs {}).".format(
                dataQ.shape[0], lambdaSqArr_m2.shape[0]
            ),
            end=" ",
        )
        log("     Check that data is in [z, y, x] order.")
        return None, None

    # Reshape the data arrays to 2 dimensions
    if nDims == 1:
        dataQ = np.reshape(dataQ, (dataQ.shape[0], 1))
        dataU = np.reshape(dataU, (dataU.shape[0], 1))
    elif nDims == 3:
        old_data_shape = dataQ.shape
        dataQ = np.reshape(dataQ, (dataQ.shape[0], dataQ.shape[1] * dataQ.shape[2]))
        dataU = np.reshape(dataU, (dataU.shape[0], dataU.shape[1] * dataU.shape[2]))

    # Create a complex polarised cube, B&dB Eqns. (8) and (14)
    # Array has dimensions [nFreq, nY * nX]
    pCube = (dataQ + 1j * dataU) * weightArr[:, np.newaxis]

    # Check for NaNs (flagged data) in the cube & set to zero
    mskCube = np.isnan(pCube)
    pCube = np.nan_to_num(pCube)

    # If full planes are flagged then set corresponding weights to zero
    mskPlanes = np.sum(~mskCube, axis=1)
    mskPlanes = np.where(mskPlanes == 0, 0, 1)
    weightArr *= mskPlanes

    # lam0Sq_m2 is the weighted mean of lambda^2 distribution (B&dB Eqn. 32)
    # Calculate a global lam0Sq_m2 value, ignoring isolated flagged voxels
    K = 1.0 / np.sum(weightArr)
    if lam0Sq_m2 is None:
        lam0Sq_m2 = K * np.sum(weightArr * lambdaSqArr_m2)
    if not np.isfinite(lam0Sq_m2):  # Can happen if all channels are NaNs/zeros
        lam0Sq_m2 = 0.0

    # The K value used to scale each FDF spectrum must take into account
    # flagged voxels data in the datacube and can be position dependent
    weightCube = np.invert(mskCube) * weightArr[:, np.newaxis]
    with np.errstate(divide="ignore", invalid="ignore"):
        KArr = np.true_divide(1.0, np.sum(weightCube, axis=0))
        KArr[KArr == np.inf] = 0
        KArr = np.nan_to_num(KArr)

    # Clean up one cube worth of memory
    del weightCube
    gc.collect()

    # Do the RM-synthesis on each plane
    # finufft must have matching dtypes, so complex64 matches float32
    a = (lambdaSqArr_m2 - lam0Sq_m2).astype(f"float{pCube.itemsize*8/2:.0f}")
    FDFcube = (
        finufft.nufft1d3(
            x=a,
            c=np.ascontiguousarray(pCube.T),
            s=(phiArr_radm2[::-1] * 2).astype(a.dtype),
            eps=eps,
        )
        * KArr[..., None]
    ).T

    # Check for pixels that have Re(FDF)=Im(FDF)=0. across ALL Faraday depths
    # These pixels will be changed to NaN in the output
    zeromap = np.all(FDFcube == 0.0, axis=0)
    FDFcube[..., zeromap] = np.nan + 1.0j * np.nan

    # Restore if 3D shape
    if nDims == 3:
        FDFcube = np.reshape(
            FDFcube, (FDFcube.shape[0], old_data_shape[1], old_data_shape[2])
        )

    # Remove redundant dimensions in the FDF array
    FDFcube = np.squeeze(FDFcube)

    return RMsynthResults(FDFcube=FDFcube, lam0Sq_m2=lam0Sq_m2)


# -----------------------------------------------------------------------------#
class RMSFResults(NamedTuple):
    """Results of the RMSF calculation"""

    RMSFcube: np.ndarray
    """The RMSF cube"""
    phi2Arr: np.ndarray
    """The (double length) Faraday depth array"""
    fwhmRMSFArr: np.ndarray
    """The FWHM of the RMSF main lobe"""
    statArr: np.ndarray
    """The status of the RMSF fit"""
    lam0Sq_m2: float
    """The reference lambda^2 value"""


def get_rmsf_planes(
    lambdaSqArr_m2,
    phiArr_radm2,
    weightArr=None,
    mskArr=None,
    lam0Sq_m2=None,
    double=True,
    fitRMSF=False,
    fitRMSFreal=False,
    nBits=32,
    eps=1e-6,
    verbose=False,
    log=print,
) -> RMSFResults:
    """Calculate the Rotation Measure Spread Function from inputs. This version
    returns a cube (1, 2 or 3D) of RMSF spectra based on the shape of a
    boolean mask array, where flagged data are True and unflagged data False.
    If only whole planes (wavelength channels) are flagged then the RMSF is the
    same for all pixels and the calculation is done once and replicated to the
    dimensions of the mask. If some isolated voxels are flagged then the RMSF
    is calculated by looping through each wavelength plane, which can take some
    time. By default the routine returns the analytical width of the RMSF main
    lobe but can also use MPFIT to fit a Gaussian.

    lambdaSqArr_m2  ... vector of wavelength^2 values (assending freq order)
    phiArr_radm2    ... vector of trial Faraday depth values
    weightArr       ... vector of weights, default [None] is no weighting
    maskArr         ... cube of mask values used to shape return cube [None]
    lam0Sq_m2       ... force a reference lambda^2 value (def=calculate) [None]
    double          ... pad the Faraday depth to double-size [True]
    fitRMSF         ... fit the main lobe of the RMSF with a Gaussian [False]
    fitRMSFreal     ... fit RMSF.real, rather than abs(RMSF) [False]
    nBits           ... precision of data arrays [32]
    eps             ... NUFFT tolerance [1e-6] (see https://finufft.readthedocs.io/en/latest/python.html#module-finufft)
    verbose         ... print feedback during calculation [False]
    log             ... function to be used to output messages [print]

    """

    # Default data types
    dtFloat = "float" + str(nBits)
    dtComplex = "complex" + str(2 * nBits)

    # For cleaning the RMSF should extend by 1/2 on each side in phi-space
    if double:
        nPhi = phiArr_radm2.shape[0]
        nExt = np.ceil(nPhi / 2.0)
        resampIndxArr = np.arange(2.0 * nExt + nPhi) - nExt
        phi2Arr = extrap(resampIndxArr, np.arange(nPhi, dtype="int"), phiArr_radm2)
    else:
        phi2Arr = phiArr_radm2

    # Set the weight array
    if weightArr is None:
        weightArr = np.ones(lambdaSqArr_m2.shape, dtype=dtFloat)
    weightArr = np.where(np.isnan(weightArr), 0.0, weightArr)

    # Set the mask array (default to 1D, no masked channels)
    if mskArr is None:
        mskArr = np.zeros_like(lambdaSqArr_m2, dtype="bool")
        nDims = 1
    else:
        mskArr = mskArr.astype("bool")
        nDims = len(mskArr.shape)

    # Sanity checks on array sizes
    if not weightArr.shape == lambdaSqArr_m2.shape:
        log("Err: wavelength^2 and weight arrays must be the same shape.")
        return None, None, None, None
    if not nDims <= 3:
        log("Err: mask dimensions must be <= 3.")
        return None, None, None, None
    if not mskArr.shape[0] == lambdaSqArr_m2.shape[0]:
        log(
            f"Err: mask depth does not match lambda^2 vector ({mskArr.shape[0]} vs {lambdaSqArr_m2.shape[-1]}).",
            end=" ",
        )
        log("     Check that the mask is in [z, y, x] order.")
        return None, None, None, None

    # Reshape the mask array to 2 dimensions
    if nDims == 1:
        mskArr = np.reshape(mskArr, (mskArr.shape[0], 1))
    elif nDims == 3:
        old_data_shape = mskArr.shape
        mskArr = np.reshape(
            mskArr, (mskArr.shape[0], mskArr.shape[1] * mskArr.shape[2])
        )
    nPix = mskArr.shape[-1]

    # If full planes are flagged then set corresponding weights to zero
    xySum = np.sum(mskArr, axis=1)
    mskPlanes = np.where(xySum == nPix, 0, 1)
    weightArr *= mskPlanes

    # Check for isolated clumps of flags (# flags in a plane not 0 or nPix)
    flagTotals = np.unique(xySum).tolist()
    try:
        flagTotals.remove(0)
    except Exception:
        pass
    try:
        flagTotals.remove(nPix)
    except Exception:
        pass
    do1Dcalc = True
    if len(flagTotals) > 0:
        do1Dcalc = False

    # lam0Sq is the weighted mean of LambdaSq distribution (B&dB Eqn. 32)
    # Calculate a single lam0Sq_m2 value, ignoring isolated flagged voxels
    K = 1.0 / np.nansum(weightArr)
    if lam0Sq_m2 is None:
        lam0Sq_m2 = K * np.nansum(weightArr * lambdaSqArr_m2)

    # Calculate the analytical FWHM width of the main lobe
    fwhmRMSF = 3.8 / (np.nanmax(lambdaSqArr_m2) - np.nanmin(lambdaSqArr_m2))

    # Do a simple 1D calculation and replicate along X & Y axes
    if do1Dcalc:
        if verbose:
            log("Calculating 1D RMSF and replicating along X & Y axes.")

        # Calculate the RMSF
        a = (-2.0 * 1j * phi2Arr).astype(dtComplex)
        b = lambdaSqArr_m2 - lam0Sq_m2
        RMSFArr = K * np.sum(weightArr * np.exp(np.outer(a, b)), 1)

        # Fit the RMSF main lobe
        fitStatus = -1
        if fitRMSF:
            if verbose:
                log("Fitting Gaussian to the main lobe.")
            mp = fit_rmsf(phi2Arr, RMSFArr.real if fitRMSFreal else np.abs(RMSFArr))
            if mp is None or mp.status < 1:
                log("Err: failed to fit the RMSF.")
                log("     Defaulting to analytical value.")
            else:
                fwhmRMSF = mp.params[2]
                fitStatus = mp.status

        # Replicate along X and Y axes
        RMSFcube = np.tile(RMSFArr[:, np.newaxis], (1, nPix))
        fwhmRMSFArr = np.ones((nPix), dtype=dtFloat) * fwhmRMSF
        statArr = np.ones((nPix), dtype="int") * fitStatus

    # Calculate the RMSF at each pixel
    else:
        # The K value used to scale each RMSF must take into account
        # isolated flagged voxels data in the datacube
        weightCube = (np.invert(mskArr) * weightArr[:, np.newaxis]).astype(dtComplex)
        with np.errstate(divide="ignore", invalid="ignore"):
            KArr = np.true_divide(1.0, np.sum(weightCube, axis=0))
            KArr[KArr == np.inf] = 0
            KArr = np.nan_to_num(KArr)

        # Calculate the RMSF for each plane
        a = (lambdaSqArr_m2 - lam0Sq_m2).astype(dtFloat)
        RMSFcube = (
            finufft.nufft1d3(
                x=a,
                c=np.ascontiguousarray(weightCube.T),
                s=(phiArr_radm2[::-1] * 2).astype(a.dtype),
                eps=eps,
            )
            * KArr[..., None]
        ).T

        # Clean up one cube worth of memory
        del weightCube
        gc.collect()

        # Default to the analytical RMSF
        fwhmRMSFArr = np.ones((nPix), dtype=dtFloat) * fwhmRMSF
        statArr = np.ones((nPix), dtype="int") * (-1)

        # Fit the RMSF main lobe
        if fitRMSF:
            if verbose:
                log("Fitting main lobe in each RMSF spectrum.")
                log("> This may take some time!")
            k = 0
            for i in trange(nPix, desc="Fitting RMSF by pixel", disable=not verbose):
                k += 1
                if fitRMSFreal:
                    mp = fit_rmsf(phi2Arr, RMSFcube[:, i].real)
                else:
                    mp = fit_rmsf(phi2Arr, np.abs(RMSFcube[:, i]))
                if not (mp is None or mp.status < 1):
                    fwhmRMSFArr[i] = mp.params[2]
                    statArr[i] = mp.status

    # Remove redundant dimensions
    RMSFcube = np.squeeze(RMSFcube)
    fwhmRMSFArr = np.squeeze(fwhmRMSFArr)
    statArr = np.squeeze(statArr)

    # Restore if 3D shape
    if nDims == 3:
        RMSFcube = np.reshape(
            RMSFcube, (RMSFcube.shape[0], old_data_shape[1], old_data_shape[2])
        )
        fwhmRMSFArr = np.reshape(fwhmRMSFArr, (old_data_shape[1], old_data_shape[2]))
        statArr = np.reshape(statArr, (old_data_shape[1], old_data_shape[2]))

    return RMSFResults(
        RMSFcube=RMSFcube,
        phi2Arr=phi2Arr,
        fwhmRMSFArr=fwhmRMSFArr,
        statArr=statArr,
        lam0Sq_m2=lam0Sq_m2,
    )


# -----------------------------------------------------------------------------#
class RMCleanResults(NamedTuple):
    """Results of the RM-CLEAN calculation"""

    cleanFDF: np.ndarray
    """The cleaned Faraday dispersion function cube"""
    ccArr: np.ndarray
    """The clean components cube"""
    iterCountArr: np.ndarray
    """The number of iterations for each pixel"""
    residFDF: np.ndarray
    """The residual Faraday dispersion function cube"""


def do_rmclean_hogbom(
    dirtyFDF,
    phiArr_radm2,
    RMSFArr,
    phi2Arr_radm2,
    fwhmRMSFArr,
    cutoff,
    maxIter=1000,
    gain=0.1,
    mskArr=None,
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
    phiArr_radm2   ... 1D Faraday depth array corresponding to the FDF
    RMSFArr        ... 1, 2 or 3D complex RMSF array
    phi2Arr_radm2  ... double size 1D Faraday depth array of the RMSF
    fwhmRMSFArr    ... scalar, 1D or 2D array of RMSF main lobe widths
    cutoff         ... clean cutoff (+ve = absolute values, -ve = sigma) [-1]
    maxIter        ... maximun number of CLEAN loop interations [1000]
    gain           ... CLEAN loop gain [0.1]
    mskArr         ... scalar, 1D or 2D pixel mask array [None]
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
    nPhi = phiArr_radm2.shape[0]
    if nPhi != dirtyFDF.shape[0]:
        log("Err: 'phiArr_radm2' and 'dirtyFDF' are not the same length.")
        return None, None, None, None
    nPhi2 = phi2Arr_radm2.shape[0]
    if not nPhi2 == RMSFArr.shape[0]:
        log("Err: missmatch in 'phi2Arr_radm2' and 'RMSFArr' length.")
        return None, None, None, None
    if not (nPhi2 >= 2 * nPhi):
        log("Err: the Faraday depth of the RMSF must be twice the FDF.")
        return None, None, None, None
    nDims = len(dirtyFDF.shape)
    if not nDims <= 3:
        log("Err: FDF array dimensions must be <= 3.")
        return None, None, None, None
    if not nDims == len(RMSFArr.shape):
        log("Err: the input RMSF and FDF must have the same number of axes.")
        return None, None, None, None
    if not RMSFArr.shape[1:] == dirtyFDF.shape[1:]:
        log("Err: the xy dimesions of the RMSF and FDF must match.")
        return None, None, None, None
    if mskArr is not None:
        if not mskArr.shape == dirtyFDF.shape[1:]:
            log("Err: pixel mask must match xy dimesnisons of FDF cube.")
            log(
                "     FDF[z,y,z] = {:}, Mask[y,x] = {:}.".format(
                    dirtyFDF.shape, mskArr.shape
                ),
                end=" ",
            )

            return None, None, None, None
    else:
        mskArr = np.ones(dirtyFDF.shape[1:], dtype="bool")

    # Reshape the FDF & RMSF array to 3 dimensions and mask array to 2
    if nDims == 1:
        dirtyFDF = np.reshape(dirtyFDF, (dirtyFDF.shape[0], 1, 1))
        RMSFArr = np.reshape(RMSFArr, (RMSFArr.shape[0], 1, 1))
        mskArr = np.reshape(mskArr, (1, 1))
        fwhmRMSFArr = np.reshape(fwhmRMSFArr, (1, 1))
    elif nDims == 2:
        dirtyFDF = np.reshape(dirtyFDF, list(dirtyFDF.shape[:2]) + [1])
        RMSFArr = np.reshape(RMSFArr, list(RMSFArr.shape[:2]) + [1])
        mskArr = np.reshape(mskArr, (dirtyFDF.shape[1], 1))
        fwhmRMSFArr = np.reshape(fwhmRMSFArr, (dirtyFDF.shape[1], 1))
    iterCountArr = np.zeros_like(mskArr, dtype="int")

    # Determine which pixels have components above the cutoff
    absFDF = np.abs(np.nan_to_num(dirtyFDF))
    mskCutoff = np.where(np.max(absFDF, axis=0) >= cutoff, 1, 0)
    xyCoords = np.rot90(np.where(mskCutoff > 0))

    # Feeback to user
    if verbose:
        nPix = dirtyFDF.shape[-1] * dirtyFDF.shape[-2]
        nCleanPix = len(xyCoords)
        log("Cleaning {:}/{:} spectra.".format(nCleanPix, nPix))

    # Initialise arrays to hold the residual FDF, clean components, clean FDF
    # Residual is initially copies of dirty FDF, so that pixels that are not
    #  processed get correct values (but will be overridden when processed)
    residFDF = dirtyFDF.copy()
    ccArr = np.zeros(dirtyFDF.shape, dtype=dtComplex)
    cleanFDF = np.zeros_like(dirtyFDF)

    # Loop through the pixels containing a polarised signal
    inputs = [[yi, xi, dirtyFDF] for yi, xi in xyCoords]
    rmc = RMcleaner(
        RMSFArr,
        phi2Arr_radm2,
        phiArr_radm2,
        fwhmRMSFArr,
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
    #    residFDF = np.reshape(np.rot90(np.stack([resid for _, resid, _ in output]), k=-1),dirtyFDF.shape)
    for i in range(len(inputs)):
        yi = inputs[i][0]
        xi = inputs[i][1]
        ccArr[:, yi, xi] = output[i][2]
        cleanFDF[:, yi, xi] = output[i][0]
        residFDF[:, yi, xi] = output[i][1]

    # Restore the residual to the CLEANed FDF (moved outside of loop:
    # will now work for pixels/spectra without clean components)
    cleanFDF += residFDF

    # Remove redundant dimensions
    cleanFDF = np.squeeze(cleanFDF)
    ccArr = np.squeeze(ccArr)
    iterCountArr = np.squeeze(iterCountArr)
    residFDF = np.squeeze(residFDF)

    return RMCleanResults(cleanFDF, ccArr, iterCountArr, residFDF)


# -----------------------------------------------------------------------------#
class CleanLoopResults(NamedTuple):
    """Results of the RM-CLEAN loop"""

    cleanFDF: np.ndarray
    """The cleaned Faraday dispersion function cube"""
    residFDF: np.ndarray
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
        phi2Arr_radm2,
        phiArr_radm2,
        fwhmRMSFArr,
        iterCountArr,
        maxIter=1000,
        gain=0.1,
        cutoff=0,
        nbits=32,
        verbose=False,
        window=0,
    ):
        self.RMSFArr = RMSFArr
        self.phi2Arr_radm2 = phi2Arr_radm2
        self.phiArr_radm2 = phiArr_radm2
        self.fwhmRMSFArr = fwhmRMSFArr
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
        residFDF = dirtyFDF.copy()
        ccArr = np.zeros_like(dirtyFDF)
        cleanFDF = np.zeros_like(dirtyFDF)
        RMSFArr = self.RMSFArr[:, yi, xi]
        fwhmRMSFArr = self.fwhmRMSFArr[yi, xi]

        # Find the index of the peak of the RMSF
        indxMaxRMSF = np.nanargmax(RMSFArr)

        # Calculate the padding in the sampled RMSF
        # Assumes only integer shifts and symmetric
        nPhiPad = int((len(self.phi2Arr_radm2) - len(self.phiArr_radm2)) / 2)

        iterCount = 0
        while np.max(np.abs(residFDF)) >= self.cutoff and iterCount < self.maxIter:
            # Get the absolute peak channel, values and Faraday depth
            indxPeakFDF = np.argmax(np.abs(residFDF))
            peakFDFval = residFDF[indxPeakFDF]
            phiPeak = self.phiArr_radm2[indxPeakFDF]

            # A clean component is "loop-gain * peakFDFval
            CC = self.gain * peakFDFval
            ccArr[indxPeakFDF] += CC

            # At which channel is the CC located at in the RMSF?
            indxPeakRMSF = indxPeakFDF + nPhiPad

            # Shift the RMSF & clip so that its peak is centred above this CC
            shiftedRMSFArr = np.roll(RMSFArr, indxPeakRMSF - indxMaxRMSF)[
                nPhiPad:-nPhiPad
            ]

            # Subtract the product of the CC shifted RMSF from the residual FDF
            residFDF -= CC * shiftedRMSFArr

            # Restore the CC * a Gaussian to the cleaned FDF
            cleanFDF += gauss1D(CC, phiPeak, fwhmRMSFArr)(self.phiArr_radm2)
            iterCount += 1
            self.iterCountArr[yi, xi] = iterCount

        # Create a mask for the pixels that have been cleaned
        mask = np.abs(ccArr) > 0
        dPhi = self.phiArr_radm2[1] - self.phiArr_radm2[0]
        fwhmRMSFArr_pix = fwhmRMSFArr / dPhi
        for i in np.where(mask)[0]:
            start = int(i - fwhmRMSFArr_pix / 2)
            end = int(i + fwhmRMSFArr_pix / 2)
            mask[start:end] = True
        residFDF_mask = np.ma.array(residFDF, mask=~mask)
        # Clean again within mask
        while (
            np.ma.max(np.ma.abs(residFDF_mask)) >= self.window
            and iterCount < self.maxIter
        ):
            if residFDF_mask.mask.all():
                break
            # Get the absolute peak channel, values and Faraday depth
            indxPeakFDF = np.ma.argmax(np.abs(residFDF_mask))
            peakFDFval = residFDF_mask[indxPeakFDF]
            phiPeak = self.phiArr_radm2[indxPeakFDF]

            # A clean component is "loop-gain * peakFDFval
            CC = self.gain * peakFDFval
            ccArr[indxPeakFDF] += CC

            # At which channel is the CC located at in the RMSF?
            indxPeakRMSF = indxPeakFDF + nPhiPad

            # Shift the RMSF & clip so that its peak is centred above this CC
            shiftedRMSFArr = np.roll(RMSFArr, indxPeakRMSF - indxMaxRMSF)[
                nPhiPad:-nPhiPad
            ]

            # Subtract the product of the CC shifted RMSF from the residual FDF
            residFDF -= CC * shiftedRMSFArr

            # Restore the CC * a Gaussian to the cleaned FDF
            cleanFDF += gauss1D(CC, phiPeak, fwhmRMSFArr)(self.phiArr_radm2)
            iterCount += 1
            self.iterCountArr[yi, xi] = iterCount

            # Remake masked residual FDF
            residFDF_mask = np.ma.array(residFDF, mask=~mask)

        cleanFDF = np.squeeze(cleanFDF)
        residFDF = np.squeeze(residFDF)
        ccArr = np.squeeze(ccArr)

        return CleanLoopResults(cleanFDF=cleanFDF, residFDF=residFDF, ccArr=ccArr)


# -----------------------------------------------------------------------------#
def fits_make_lin_axis(head, axis=0, dtype="f4"):
    """Create an array containing the axis values, assuming a simple linear
    projection scheme. Axis selection is zero-indexed."""

    axis = int(axis)
    if head["NAXIS"] < axis + 1:
        return []

    i = str(int(axis) + 1)
    start = head["CRVAL" + i] + (1 - head["CRPIX" + i]) * head["CDELT" + i]
    stop = (
        head["CRVAL" + i] + (head["NAXIS" + i] - head["CRPIX" + i]) * head["CDELT" + i]
    )
    nChan = int(abs(start - stop) / head["CDELT" + i] + 1)

    return np.linspace(start, stop, nChan).astype(dtype)


# -----------------------------------------------------------------------------#
def extrap(x, xp, yp):
    """
    Wrapper to allow np.interp to linearly extrapolate at function ends.

    np.interp function with linear extrapolation
    http://stackoverflow.com/questions/2745329/how-to-make-scipy-interpolate
    -give-a-an-extrapolated-result-beyond-the-input-ran

    """
    y = np.interp(x, xp, yp)
    y = np.where(x < xp[0], yp[0] + (x - xp[0]) * (yp[0] - yp[1]) / (xp[0] - xp[1]), y)
    y = np.where(
        x > xp[-1], yp[-1] + (x - xp[-1]) * (yp[-1] - yp[-2]) / (xp[-1] - xp[-2]), y
    )
    return y


# -----------------------------------------------------------------------------#
def fit_rmsf(xData, yData, thresh=0.4, ampThresh=0.4):
    """
    Fit the main lobe of the RMSF with a Gaussian function.
    """

    try:
        # Detect the peak and mask off the sidelobes
        msk1 = detect_peak(yData, thresh)
        msk2 = np.where(yData < ampThresh, 0.0, msk1)
        if sum(msk2) < 4:
            msk2 = msk1
        validIndx = np.where(msk2 == 1.0)
        xData = xData[validIndx]
        yData = yData[validIndx]

        # Estimate starting parameters
        a = 1.0
        b = xData[np.argmax(yData)]
        w = np.nanmax(xData) - np.nanmin(xData)
        inParms = [
            {"value": a, "fixed": False, "parname": "amp"},
            {"value": b, "fixed": False, "parname": "offset"},
            {"value": w, "fixed": False, "parname": "width"},
        ]

        # Function which returns another function to evaluate a Gaussian
        def gauss(p):
            a, b, w = p
            gfactor = 2.0 * m.sqrt(2.0 * m.log(2.0))
            s = w / gfactor

            def rfunc(x):
                y = a * np.exp(-((x - b) ** 2.0) / (2.0 * s**2.0))
                return y

            return rfunc

        # Function to evaluate the difference between the model and data.
        # This is minimised in the least-squared sense by the fitter
        def errFn(p, fjac=None):
            status = 0
            return status, gauss(p)(xData) - yData

        # Use mpfit to perform the fitting
        mp = mpfit(errFn, parinfo=inParms, quiet=True)

        return mp

    except Exception:
        return None


# -----------------------------------------------------------------------------#
def gauss1D(amp=1.0, mean=0.0, fwhm=1.0):
    """Function which returns another function to evaluate a Gaussian"""

    gfactor = 2.0 * m.sqrt(2.0 * m.log(2.0))
    sigma = fwhm / gfactor

    def rfunc(x):
        return amp * np.exp(-((x - mean) ** 2.0) / (2.0 * sigma**2.0))

    return rfunc


# -----------------------------------------------------------------------------#


def detect_peak(a, thresh=0.4):
    """Detect the extent of the peak in the array by moving away, in both
    directions, from the peak channel amd looking for where the value drops
    below a threshold.
    Returns a mask array like the input array with 1s over the extent of the
    peak and 0s elsewhere."""

    # Find the peak
    iPk = np.argmax(a)  # If the peak is flat, this is the left index

    # find first point below threshold right of peak
    ishift = np.where(a[iPk:] < thresh)[0][0]
    iR = iPk + ishift
    iL = iPk - ishift + 1

    msk = np.zeros_like(a)
    msk[iL:iR] = 1

    # DEBUG PLOTTING
    if False:
        from matplotlib import pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.step(np.arange(len(a)), a, where="mid", label="arr")
        ax.step(np.arange(len(msk)), msk * 0.5, where="mid", label="msk")
        ax.axhline(0, color="grey")
        ax.axvline(iPk, color="k", linewidth=3.0)
        ax.axhline(thresh, color="magenta", ls="--")
        ax.set_xlim([iL - 20, iR + 20])
        leg = ax.legend(
            numpoints=1,
            loc="upper right",
            shadow=False,
            borderaxespad=0.3,
            ncol=1,
            bbox_to_anchor=(1.00, 1.00),
        )
        fig.show()
    return msk


# -----------------------------------------------------------------------------#
# def detect_peak(a, thresh=0.3):
#     """Detect the extent of the peak in the array by moving away, in both
#     directions, from the peak channel amd looking for where the slope changes
#     to some shallow value. The triggering slope is 'thresh*max(slope)'.
#     Returns a mask array like the input array with 1s over the extent of the
#     peak and 0s elsewhere."""

#     # Find the peak and take the 1st derivative
#     iPkL= np.argmax(a)  # If the peak is flat, this is the left index
#     g1 = np.abs(np.gradient(a))

#     # Specify a threshold for the 1st derivative. Channels between the peak
#     # and the first crossing point will be included in the mask.
#     threshPos = np.nanmax(g1) * thresh

#     # Determine the right-most index of flat peaks
#     iPkR = iPkL
#     d = np.diff(a)
#     flatIndxLst = np.argwhere(d[iPkL:]==0).flatten()
#     if len(flatIndxLst)>0:
#         iPkR += (np.max(flatIndxLst)+1)

#     # Search for the left & right crossing point
#     iL = np.max(np.argwhere(g1[:iPkL]<=threshPos).flatten())
#     iR = iPkR + np.min(np.argwhere(g1[iPkR+1:]<=threshPos).flatten()) + 2
#     msk = np.zeros_like(a)
#     msk[iL:iR] = 1

#     # DEBUG PLOTTING
#     if False:
#         from matplotlib import pyplot as plt
#         fig = plt.figure()
#         ax = fig.add_subplot(111)
#         ax.step(np.arange(len(a)),a, where="mid", label="arr")
#         ax.step(np.arange(len(g1)), np.abs(g1), where="mid", label="g1")
#         ax.step(np.arange(len(msk)), msk*0.5, where="mid", label="msk")
#         ax.axhline(0, color='grey')
#         ax.axvline(iPkL, color='k', linewidth=3.0)
#         ax.axhline(threshPos, color='magenta', ls="--")
#         ax.set_xlim([iPkL-20, iPkL+20])
#         leg = ax.legend(numpoints=1, loc='upper right', shadow=False,
#                         borderaxespad=0.3, ncol=1,
#                         bbox_to_anchor=(1.00, 1.00))
#         fig.show()
#         input()

#     return msk


# -----------------------------------------------------------------------------#
def measure_FDF_parms(
    FDF,
    phiArr,
    fwhmRMSF,
    dFDF=None,
    lamSqArr_m2=None,
    lam0Sq=None,
    snrDoBiasCorrect=5.0,
):
    """
    Measure standard parameters from a complex Faraday Dispersion Function.
    Currently this function assumes that the noise levels in the Stokes Q
    and U spectra are the same.
    Returns a dictionary containing measured parameters.
    """

    # Determine the peak channel in the FDF, its amplitude and index
    absFDF = np.abs(FDF)
    indxPeakPIchan = (
        np.nanargmax(absFDF[1:-1]) + 1
    )  # Masks out the edge channels, since they can't be fit to.

    # Measure the RMS noise in the spectrum after masking the peak
    dPhi = np.nanmin(np.diff(phiArr))
    fwhmRMSF_chan = np.ceil(fwhmRMSF / dPhi)
    iL = int(max(0, indxPeakPIchan - fwhmRMSF_chan * 2))
    iR = int(min(len(absFDF), indxPeakPIchan + fwhmRMSF_chan * 2))
    FDFmsked = FDF.copy()
    FDFmsked[iL:iR] = np.nan
    FDFmsked = FDFmsked[np.where(FDFmsked == FDFmsked)]
    if float(len(FDFmsked)) / len(FDF) < 0.3:
        dFDFcorMAD = MAD(np.concatenate((np.real(FDF), np.imag(FDF))))
    else:
        dFDFcorMAD = MAD(np.concatenate((np.real(FDFmsked), np.imag(FDFmsked))))

    # Default to using the measured FDF if a noise value has not been provided
    if dFDF is None:
        dFDF = dFDFcorMAD

    nChansGood = np.sum(np.where(lamSqArr_m2 == lamSqArr_m2, 1.0, 0.0))
    varLamSqArr_m2 = (
        np.sum(lamSqArr_m2**2.0) - np.sum(lamSqArr_m2) ** 2.0 / nChansGood
    ) / (nChansGood - 1)

    # Determine the peak in the FDF, its amplitude and Phi using a
    # 3-point parabolic interpolation
    phiPeakPIfit = None
    dPhiPeakPIfit = None
    ampPeakPIfit = None
    snrPIfit = None
    ampPeakPIfitEff = None
    indxPeakPIfit = None
    peakFDFimagFit = None
    peakFDFrealFit = None
    polAngleFit_deg = None
    dPolAngleFit_deg = None
    polAngle0Fit_deg = None
    dPolAngle0Fit_deg = None

    # Only do the 3-point fit if peak is 1-channel from either edge
    if indxPeakPIchan > 0 and indxPeakPIchan < len(FDF) - 1:
        phiPeakPIfit, ampPeakPIfit = calc_parabola_vertex(
            phiArr[indxPeakPIchan - 1],
            absFDF[indxPeakPIchan - 1],
            phiArr[indxPeakPIchan],
            absFDF[indxPeakPIchan],
            phiArr[indxPeakPIchan + 1],
            absFDF[indxPeakPIchan + 1],
        )

        snrPIfit = ampPeakPIfit / dFDF

        # In rare cases, a parabola can be fitted to the edge of the spectrum,
        # producing a unreasonably large RM and polarized intensity.
        # In these cases, everything should get NaN'd out.
        if np.abs(phiPeakPIfit) > np.max(np.abs(phiArr)):
            phiPeakPIfit = np.nan
            ampPeakPIfit = np.nan

        # Error on fitted Faraday depth (RM) is same as channel, but using fitted PI
        dPhiPeakPIfit = fwhmRMSF * dFDF / (2.0 * ampPeakPIfit)

        # Correct the peak for polarisation bias (POSSUM report 11)
        ampPeakPIfitEff = ampPeakPIfit
        if snrPIfit >= snrDoBiasCorrect:
            ampPeakPIfitEff = np.sqrt(ampPeakPIfit**2.0 - 2.3 * dFDF**2.0)

        # Calculate the polarisation angle from the fitted peak
        # Uncertainty from Eqn A.12 in Brentjens & De Bruyn 2005
        indxPeakPIfit = np.interp(
            phiPeakPIfit, phiArr, np.arange(phiArr.shape[-1], dtype="f4")
        )
        peakFDFimagFit = np.interp(phiPeakPIfit, phiArr, FDF.imag)
        peakFDFrealFit = np.interp(phiPeakPIfit, phiArr, FDF.real)
        polAngleFit_deg = (
            0.5 * np.degrees(np.arctan2(peakFDFimagFit, peakFDFrealFit)) % 180
        )
        dPolAngleFit_deg = np.degrees(dFDF / (2.0 * ampPeakPIfit))

        # Calculate the derotated polarisation angle and uncertainty
        # Uncertainty from Eqn A.20 in Brentjens & De Bruyn 2005
        polAngle0Fit_deg = (
            np.degrees(np.radians(polAngleFit_deg) - phiPeakPIfit * lam0Sq)
        ) % 180
        dPolAngle0Fit_rad = np.sqrt(
            dFDF**2.0
            * nChansGood
            / (4.0 * (nChansGood - 2.0) * ampPeakPIfit**2.0)
            * ((nChansGood - 1) / nChansGood + lam0Sq**2.0 / varLamSqArr_m2)
        )
        dPolAngle0Fit_deg = np.degrees(dPolAngle0Fit_rad)

    # Store the measurements in a dictionary and return
    mDict = {
        "dFDFcorMAD": toscalar(dFDFcorMAD),
        "phiPeakPIfit_rm2": toscalar(phiPeakPIfit),
        "dPhiPeakPIfit_rm2": toscalar(dPhiPeakPIfit),
        "ampPeakPIfit": toscalar(ampPeakPIfit),
        "ampPeakPIfitEff": toscalar(ampPeakPIfitEff),
        "dAmpPeakPIfit": toscalar(dFDF),
        "snrPIfit": toscalar(snrPIfit),
        "indxPeakPIfit": toscalar(indxPeakPIfit),
        "peakFDFimagFit": toscalar(peakFDFimagFit),
        "peakFDFrealFit": toscalar(peakFDFrealFit),
        "polAngleFit_deg": toscalar(polAngleFit_deg),
        "dPolAngleFit_deg": toscalar(dPolAngleFit_deg),
        "polAngle0Fit_deg": toscalar(polAngle0Fit_deg),
        "dPolAngle0Fit_deg": toscalar(dPolAngle0Fit_deg),
    }

    return mDict


# -----------------------------------------------------------------------------#
def norm_cdf(mean=0.0, std=1.0, N=50, xArr=None):
    """Return the CDF of a normal distribution between -6 and 6 sigma, or at
    the values of an input array."""

    if xArr is None:
        x = np.linspace(-6.0 * std, 6.0 * std, N)
    else:
        x = xArr
    y = norm.cdf(x, loc=mean, scale=std)

    return x, y


# -----------------------------------------------------------------------------#
def cdf_percentile(x, p, q=50.0):
    """Return the value at a given percentile of a cumulative distribution
    function."""

    # Determine index where cumulative percentage is achieved
    try:  # Can fail if NaNs present, so return NaN in this case.
        i = np.where(p > q / 100.0)[0][0]
    except:
        return np.nan

    # If at extremes of the distribution, return the limiting value
    if i == 0 or i == len(x):
        return x[i]

    # or interpolate between the two bracketing values in the CDF
    else:
        m = (p[i] - p[i - 1]) / (x[i] - x[i - 1])
        c = p[i] - m * x[i]
        return (q / 100.0 - c) / m


# -----------------------------------------------------------------------------#
def calc_sigma_add(xArr, yArr, dyArr, yMed=None, noise=None, nSamp=1000, suffix=""):
    """Calculate the most likely value of additional scatter, assuming the
    input data is drawn from a normal distribution. The total uncertainty on
    each data point Y_i is modelled as dYtot_i**2 = dY_i**2 + dYadd**2."""

    # Measure the median and MADFM of the input data if not provided.
    # Used to overplot a normal distribution when debugging.
    if yMed is None:
        yMed = np.median(yArr)
    if noise is None:
        noise = MAD(yArr)

    # Sample the PDF of the additional noise term from a limit near zero to
    # a limit of the range of the data, including error bars
    yRng = np.nanmax(yArr + dyArr) - np.nanmin(yArr - dyArr)
    sigmaAddArr = np.linspace(yRng / nSamp, yRng, nSamp)

    # Model deviation from Gaussian as an additional noise term.
    # Loop through the range of i additional noise samples and calculate
    # chi-squared and sum(ln(sigma_total)), used later to calculate likelihood.
    nData = len(xArr)
    chiSqArr = np.zeros_like(sigmaAddArr)
    lnSigmaSumArr = np.zeros_like(sigmaAddArr)
    for i, sigmaAdd in enumerate(sigmaAddArr):
        sigmaSqTot = dyArr**2.0 + sigmaAdd**2.0
        lnSigmaSumArr[i] = np.nansum(np.log(np.sqrt(sigmaSqTot)))
        chiSqArr[i] = np.nansum((yArr - yMed) ** 2.0 / sigmaSqTot)
    dof = nData - 1
    chiSqRedArr = chiSqArr / dof

    # Calculate the PDF in log space and normalise the peak to 1
    lnProbArr = (
        -np.log(sigmaAddArr)
        - nData * np.log(2.0 * np.pi) / 2.0
        - lnSigmaSumArr
        - chiSqArr / 2.0
    )
    lnProbArr -= np.nanmax(lnProbArr)
    probArr = np.exp(lnProbArr)

    # Normalise the area under the PDF to be 1
    A = np.nansum(probArr * np.diff(sigmaAddArr)[0])
    probArr /= A

    # Calculate the cumulative PDF
    CPDF = np.cumsum(probArr) / np.nansum(probArr)

    # Calculate the mean of the distribution and the +/- 1-sigma limits
    sigmaAdd = cdf_percentile(x=sigmaAddArr, p=CPDF, q=50.0)
    sigmaAddMinus = cdf_percentile(x=sigmaAddArr, p=CPDF, q=15.72)
    sigmaAddPlus = cdf_percentile(x=sigmaAddArr, p=CPDF, q=84.27)
    mDict = {
        "sigmaAdd" + suffix: toscalar(sigmaAdd),
        "dSigmaAddMinus" + suffix: toscalar(sigmaAdd - sigmaAddMinus),
        "dSigmaAddPlus" + suffix: toscalar(sigmaAddPlus - sigmaAdd),
    }

    # Return the curves to be plotted in a separate dictionary
    pltDict = {
        "sigmaAddArr" + suffix: sigmaAddArr,
        "chiSqRedArr" + suffix: chiSqRedArr,
        "probArr" + suffix: probArr,
        "xArr" + suffix: xArr,
        "yArr" + suffix: yArr,
        "dyArr" + suffix: dyArr,
    }

    return mDict, pltDict


# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#
def measure_qu_complexity(
    freqArr_Hz, qArr, uArr, dqArr, duArr, fracPol, psi0_deg, RM_radm2, specF=1
):
    # Create a RM-thin model to subtract
    pModArr, qModArr, uModArr = create_pqu_spectra_burn(
        freqArr_Hz=freqArr_Hz,
        fracPolArr=[fracPol],
        psi0Arr_deg=[psi0_deg],
        RMArr_radm2=[RM_radm2],
    )
    lamSqArr_m2 = np.power(C / freqArr_Hz, 2.0)
    ndata = len(lamSqArr_m2)

    # Subtract the RM-thin model to create a residual q & u
    qResidArr = qArr - qModArr
    uResidArr = uArr - uModArr

    # Calculate value of additional scatter term for q & u (max likelihood)
    mDict = {}
    pDict = {}
    m1D, p1D = calc_sigma_add(
        xArr=lamSqArr_m2[: int(ndata / specF)],
        yArr=(qResidArr / dqArr)[: int(ndata / specF)],
        dyArr=(dqArr / dqArr)[: int(ndata / specF)],
        yMed=0.0,
        noise=1.0,
        suffix="Q",
    )
    mDict.update(m1D)
    pDict.update(p1D)
    m2D, p2D = calc_sigma_add(
        xArr=lamSqArr_m2[: int(ndata / specF)],
        yArr=(uResidArr / duArr)[: int(ndata / specF)],
        dyArr=(duArr / duArr)[: int(ndata / specF)],
        yMed=0.0,
        noise=1.0,
        suffix="U",
    )
    mDict.update(m2D)
    pDict.update(p2D)

    # Calculate the deviations statistics
    # Done as a test for the paper, not usually offered to user.
    # mDict.update( calc_normal_tests(qResidArr/dqArr, suffix="Q") )
    # mDict.update( calc_normal_tests(uResidArr/duArr, suffix="U") )

    return mDict, pDict


# -----------------------------------------------------------------------------#
def measure_fdf_complexity(phiArr, FDF):
    # Second moment of clean component spectrum
    mom2FDF = calc_mom2_FDF(FDF, phiArr)

    return toscalar(mom2FDF)


# -----------------------------------------------------------------------------#
# 3D noise functions are still in prototype stage! Proper function is not guaranteed!
