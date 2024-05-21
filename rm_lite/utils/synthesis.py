#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gc
from typing import Literal, NamedTuple, Optional

import finufft
import numpy as np
from astropy.constants import c as speed_of_light
from astropy.modeling.models import Gaussian1D
from scipy import optimize
from tqdm.auto import tqdm, trange

from rm_lite.utils.fitting import (
    calc_mom2_FDF,
    calc_parabola_vertex,
    create_pqu_spectra_burn,
)
from rm_lite.utils.logging import logger


def freq_to_lambda2(freq_hz: float) -> float:
    """Convert frequency to lambda^2."""
    return (speed_of_light.value / freq_hz) ** 2.0


def lambda2_to_freq(lambda_sq_m2: float) -> float:
    """Convert lambda^2 to frequency."""
    return speed_of_light.value / np.sqrt(lambda_sq_m2)


def compute_theoretical_noise(
    stokes_qu_error_array: np.ndarray,
    weight_array: np.ndarray,
) -> float:
    weight_array = np.nan_to_num(weight_array, nan=0.0, posinf=0.0, neginf=0.0)
    stokes_qu_error_array = np.nan_to_num(
        stokes_qu_error_array, nan=0.0, posinf=0.0, neginf=0.0
    )
    fdf_error_noise = np.sqrt(
        np.nansum(weight_array**2 * stokes_qu_error_array**2)
        / (np.sum(weight_array)) ** 2
    )
    return fdf_error_noise


class RMSynthParams(NamedTuple):
    lambda_sq_arr_m2: np.ndarray
    lam_sq_0_m2: float
    phi_arr_radm2: np.ndarray
    weight_array: np.ndarray


def compute_rmsynth_params(
    freq_array_hz: np.ndarray,
    pol_array: np.ndarray,
    stokes_qu_error_array: np.ndarray,
    d_phi_radm2: Optional[float] = None,
    n_samples: Optional[float] = 10.0,
    phi_max_radm2: Optional[float] = None,
    super_resolution: bool = False,
    weight_type: Literal["variance", "uniform"] = "variance",
) -> RMSynthParams:
    lambda_sq_arr_m2 = freq_to_lambda2(freq_array_hz)

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

    logger.debug(
        f"phi = {phi_arr_radm2[0]:0.2f} to {phi_arr_radm2[-1]:0.2f} by {d_phi_radm2:0.2f} ({len(phi_arr_radm2)} chans)."
    )

    # Calculate the weighting as 1/sigma^2 or all 1s (uniform)
    if weight_type == "variance":
        weight_array = 1.0 / stokes_qu_error_array**2
    else:
        weight_array = np.ones_like(freq_array_hz)

    mask = ~np.isfinite(pol_array)
    weight_array[mask] = 0.0

    # lam_sq_0_m2 is the weighted mean of lambda^2 distribution (B&dB Eqn. 32)
    # Calculate a global lam_sq_0_m2 value, ignoring isolated flagged voxels
    scale_factor = 1.0 / np.nansum(weight_array)
    lam_sq_0_m2 = scale_factor * np.nansum(weight_array * lambda_sq_arr_m2)
    if not np.isfinite(lam_sq_0_m2):  # Can happen if all channels are NaNs/zeros
        lam_sq_0_m2 = 0.0

    return RMSynthParams(
        lambda_sq_arr_m2=lambda_sq_arr_m2,
        lam_sq_0_m2=lam_sq_0_m2,
        phi_arr_radm2=phi_arr_radm2,
        weight_array=weight_array,
    )


def make_phi_array(
    phi_max_radm2: float,
    d_phi_radm2: float,
) -> np.ndarray:
    # Faraday depth sampling. Zero always centred on middle channel
    n_chan_rm = int(np.round(abs((phi_max_radm2 - 0.0) / d_phi_radm2)) * 2.0 + 1.0)
    max_phi_radm2 = (n_chan_rm - 1.0) * d_phi_radm2 / 2.0
    phi_arr_radm2 = np.linspace(-max_phi_radm2, max_phi_radm2, n_chan_rm)
    return phi_arr_radm2


class FWHMRMSF(NamedTuple):
    fwhm_rmsf_radm2: float
    """The FWHM of the RMSF main lobe"""
    d_lambda_sq_max_m2: float
    """The maximum difference in lambda^2 values"""
    lambda_sq_range_m2: float
    """The range of lambda^2 values"""


def get_fwhm_rmsf(
    lambda_sq_arr_m2: np.ndarray,
    super_resolution: bool = False,
) -> FWHMRMSF:
    lambda_sq_range_m2 = np.nanmax(lambda_sq_arr_m2) - np.nanmin(lambda_sq_arr_m2)
    d_lambda_sq_max_m2 = np.nanmax(np.abs(np.diff(lambda_sq_arr_m2)))

    # Set the Faraday depth range
    if not super_resolution:
        fwhm_rmsf_radm2 = 3.8 / lambda_sq_range_m2  # Dickey+2019 theoretical RMSF width
    else:  # If super resolution, use R&C23 theoretical width
        fwhm_rmsf_radm2 = 2.0 / (
            np.nanmax(lambda_sq_arr_m2) + np.nanmin(lambda_sq_arr_m2)
        )
    return FWHMRMSF(
        fwhm_rmsf_radm2=fwhm_rmsf_radm2,
        d_lambda_sq_max_m2=d_lambda_sq_max_m2,
        lambda_sq_range_m2=lambda_sq_range_m2,
    )


class RMsynthResults(NamedTuple):
    """Results of the RM-synthesis calculation"""

    fdf_dirty_cube: np.ndarray
    """The Faraday dispersion function cube"""
    lam_sq_0_m2: float
    """The reference lambda^2 value"""


def rmsynth_nufft(
    stokes_q_array: np.ndarray,
    stokes_u_array: np.ndarray,
    lambda_sq_arr_m2: np.ndarray,
    phi_arr_radm2: np.ndarray,
    weight_array: np.ndarray,
    lam_sq_0_m2: Optional[float] = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """Run RM-synthesis on a cube of Stokes Q and U data using the NUFFT method.

    Args:
        stokes_q_array (np.ndarray): Stokes Q data array
        stokes_u_array (np.ndarray): Stokes U data array
        lambda_sq_arr_m2 (np.ndarray): Wavelength^2 values in m^2
        phi_arr_radm2 (np.ndarray): Faraday depth values in rad/m^2
        weight_array (np.ndarray): Weight array
        lam_sq_0_m2 (Optional[float], optional): Reference wavelength^2 in m^2. Defaults to None.
        eps (float, optional): NUFFT tolerance. Defaults to 1e-6.

    Raises:
        ValueError: If the weight and lambda^2 arrays are not the same shape.
        ValueError: If the Stokes Q and U data arrays are not the same shape.
        ValueError: If the data dimensions are > 3.
        ValueError: If the data depth does not match the lambda^2 vector.

    Returns:
        np.ndarray: Dirty Faraday dispersion function cube
    """
    weight_array = np.nan_to_num(weight_array, nan=0.0, posinf=0.0, neginf=0.0)

    # Sanity check on array sizes
    if not weight_array.shape == lambda_sq_arr_m2.shape:
        raise ValueError(
            f"Weight and lambda^2 arrays must be the same shape. Got {weight_array.shape} and {lambda_sq_arr_m2.shape}"
        )

    if not stokes_q_array.shape == stokes_u_array.shape:
        raise ValueError("Stokes Q and U data arrays must be the same shape.")

    n_dims = len(stokes_q_array.shape)
    if not n_dims <= 3:
        raise ValueError(f"Data dimensions must be <= 3. Got {n_dims}")

    if not stokes_q_array.shape[0] == lambda_sq_arr_m2.shape[0]:
        raise ValueError(
            f"Data depth does not match lambda^2 vector ({stokes_q_array.shape[0]} vs {lambda_sq_arr_m2.shape[0]})."
        )

    # Reshape the data arrays to 2 dimensions
    if n_dims == 1:
        stokes_q_array = np.reshape(stokes_q_array, (stokes_q_array.shape[0], 1))
        stokes_u_array = np.reshape(stokes_u_array, (stokes_u_array.shape[0], 1))
    elif n_dims == 3:
        old_data_shape = stokes_q_array.shape
        stokes_q_array = np.reshape(
            stokes_q_array,
            (
                stokes_q_array.shape[0],
                stokes_q_array.shape[1] * stokes_q_array.shape[2],
            ),
        )
        stokes_u_array = np.reshape(
            stokes_u_array,
            (
                stokes_u_array.shape[0],
                stokes_u_array.shape[1] * stokes_u_array.shape[2],
            ),
        )

    # Create a complex polarised cube, B&dB Eqns. (8) and (14)
    # Array has dimensions [nFreq, nY * nX]
    pol_cube = (stokes_q_array + 1j * stokes_u_array) * weight_array[:, np.newaxis]

    # Check for NaNs (flagged data) in the cube & set to zero
    mask_cube = ~np.isfinite(pol_cube)
    pol_cube = np.nan_to_num(pol_cube, nan=0.0, posinf=0.0, neginf=0.0)

    # If full planes are flagged then set corresponding weights to zero
    mask_planes = np.sum(~mask_cube, axis=1)
    mask_planes = np.where(mask_planes == 0, 0, 1)
    weight_array *= mask_planes

    # The K value used to scale each FDF spectrum must take into account
    # flagged voxels data in the datacube and can be position dependent
    weight_cube = np.invert(mask_cube) * weight_array[:, np.newaxis]
    with np.errstate(divide="ignore", invalid="ignore"):
        scale_array = np.true_divide(1.0, np.sum(weight_cube, axis=0))
        scale_array[scale_array == np.inf] = 0
        scale_array = np.nan_to_num(scale_array)

    # Clean up one cube worth of memory
    del weight_cube
    gc.collect()

    # Do the RM-synthesis on each plane
    # finufft must have matching dtypes, so complex64 matches float32
    exponent = (lambda_sq_arr_m2 - lam_sq_0_m2).astype(
        f"float{pol_cube.itemsize*8/2:.0f}"
    )
    fdf_dirty_cube = (
        finufft.nufft1d3(
            x=exponent,
            c=np.ascontiguousarray(pol_cube.T),
            s=(phi_arr_radm2[::-1] * 2).astype(exponent.dtype),
            eps=eps,
        )
        * scale_array[..., None]
    ).T

    # Check for pixels that have Re(FDF)=Im(FDF)=0. across ALL Faraday depths
    # These pixels will be changed to NaN in the output
    zeromap = np.all(fdf_dirty_cube == 0.0, axis=0)
    fdf_dirty_cube[..., zeromap] = np.nan + 1.0j * np.nan

    # Restore if 3D shape
    if n_dims == 3:
        fdf_dirty_cube = np.reshape(
            fdf_dirty_cube,
            (fdf_dirty_cube.shape[0], old_data_shape[1], old_data_shape[2]),
        )

    # Remove redundant dimensions in the FDF array
    fdf_dirty_cube = np.squeeze(fdf_dirty_cube)

    return fdf_dirty_cube


# -----------------------------------------------------------------------------#
class RMSFResults(NamedTuple):
    """Results of the RMSF calculation"""

    rmsf_cube: np.ndarray
    """The RMSF cube"""
    phi_double_arr_radm2: np.ndarray
    """The (double length) Faraday depth array"""
    fwhm_rmsf_arr: np.ndarray
    """The FWHM of the RMSF main lobe"""
    fit_status_array: np.ndarray
    """The status of the RMSF fit"""


def get_rmsf_nufft(
    lambda_sq_arr_m2: np.ndarray,
    phi_arr_radm2: np.ndarray,
    weight_array: np.ndarray,
    lam_sq_0_m2: float,
    super_resolution: bool = False,
    mask_array: Optional[np.ndarray] = None,
    do_fit_rmsf: bool = False,
    do_fit_rmsf_real=False,
    eps: float = 1e-6,
) -> RMSFResults:
    """Compute the RMSF for a given set of lambda^2 values.

    Args:
        lambda_sq_arr_m2 (np.ndarray): Wavelength^2 values in m^2
        phi_arr_radm2 (np.ndarray): Faraday depth values in rad/m^2
        weight_array (np.ndarray): Weight array
        lam_sq_0_m2 (float): Reference wavelength^2 value
        super_resolution (bool, optional): Use superresolution. Defaults to False.
        mask_array (Optional[np.ndarray], optional): Mask array. Defaults to None.
        do_fit_rmsf (bool, optional): Fit the RMSF with a Gaussian. Defaults to False.
        do_fit_rmsf_real (bool, optional): Fit the *real* part of the. Defaults to False.
        eps (float, optional): NUFFT tolerance. Defaults to 1e-6.

    Raises:
        ValueError: If the wavelength^2 and weight arrays are not the same shape.
        ValueError: If the mask dimensions are > 3.
        ValueError: If the mask depth does not match the lambda^2 vector.

    Returns:
        RMSFResults: rmsf_cube, phi_double_arr_radm2, fwhm_rmsf_arr, fit_status_array
    """
    phi_double_arr_radm2 = make_phi_array(
        phi_max_radm2=np.max(phi_arr_radm2) * 2,
        d_phi_radm2=phi_arr_radm2[1] - phi_arr_radm2[0],
    )

    weight_array = np.nan_to_num(weight_array, nan=0.0, posinf=0.0, neginf=0.0)

    # Set the mask array (default to 1D, no masked channels)
    if mask_array is None:
        mask_array = np.zeros_like(lambda_sq_arr_m2, dtype=bool)
        n_dimension = 1
    else:
        mask_array = mask_array.astype(bool)
        n_dimension = len(mask_array.shape)

    # Sanity checks on array sizes
    if not weight_array.shape == lambda_sq_arr_m2.shape:
        raise ValueError("wavelength^2 and weight arrays must be the same shape.")

    if not n_dimension <= 3:
        raise ValueError("mask dimensions must be <= 3.")

    if not mask_array.shape[0] == lambda_sq_arr_m2.shape[0]:
        raise ValueError(
            f"Mask depth does not match lambda^2 vector ({mask_array.shape[0]} vs {lambda_sq_arr_m2.shape[-1]})."
        )

    # Reshape the mask array to 2 dimensions
    if n_dimension == 1:
        mask_array = np.reshape(mask_array, (mask_array.shape[0], 1))
    elif n_dimension == 3:
        old_data_shape = mask_array.shape
        mask_array = np.reshape(
            mask_array, (mask_array.shape[0], mask_array.shape[1] * mask_array.shape[2])
        )
    num_pixels = mask_array.shape[-1]

    # If full planes are flagged then set corresponding weights to zero
    flag_xy_sum = np.sum(mask_array, axis=1)
    mskPlanes = np.where(flag_xy_sum == num_pixels, 0, 1)
    weight_array *= mskPlanes

    # Check for isolated clumps of flags (# flags in a plane not 0 or num_pixels)
    flag_totals_list = np.unique(flag_xy_sum).tolist()
    try:
        flag_totals_list.remove(0)
    except Exception as e:
        logger.warning(e)
    try:
        flag_totals_list.remove(num_pixels)
    except Exception as e:
        logger.warning(e)

    fwhm_rmsf_radm2, _, _ = get_fwhm_rmsf(lambda_sq_arr_m2, super_resolution)
    # Calculate the RMSF at each pixel
    # The K value used to scale each RMSF must take into account
    # isolated flagged voxels data in the datacube
    weight_cube = np.invert(mask_array) * weight_array[:, np.newaxis]
    with np.errstate(divide="ignore", invalid="ignore"):
        scale_factor_array = 1.0 / np.sum(weight_cube, axis=0)
        scale_factor_array = np.nan_to_num(
            scale_factor_array, nan=0.0, posinf=0.0, neginf=0.0
        )

    # Calculate the RMSF for each plane
    exponent = lambda_sq_arr_m2 - lam_sq_0_m2
    rmsf_cube = (
        finufft.nufft1d3(
            x=exponent,
            c=np.ascontiguousarray(weight_cube.T).astype(complex),
            s=(phi_double_arr_radm2[::-1] * 2).astype(exponent.dtype),
            eps=eps,
        )
        * scale_factor_array[..., None]
    ).T

    # Clean up one cube worth of memory
    del weight_cube
    gc.collect()

    # Default to the analytical RMSF
    fwhm_rmsf_arr = np.ones(num_pixels) * fwhm_rmsf_radm2
    fit_status_array = np.zeros(num_pixels, dtype=bool)

    # Fit the RMSF main lobe
    if do_fit_rmsf:
        logger.info("Fitting main lobe in each RMSF spectrum.")
        logger.info("> This may take some time!")
        for i in trange(num_pixels, desc="Fitting RMSF by pixel"):
            try:
                fitted_rmsf = fit_rmsf(
                    rmsf_to_fit_array=(
                        rmsf_cube[:, i].real
                        if do_fit_rmsf_real
                        else np.abs(rmsf_cube[:, i])
                    ),
                    phi_double_arr_radm2=phi_double_arr_radm2,
                    fwhm_rmsf_radm2=fwhm_rmsf_radm2,
                )
                fit_status = True
            except Exception as e:
                logger.error(f"Failed to fit RMSF at pixel {i}.")
                logger.error(e)
                logger.warning("Setting RMSF FWHM to default value.")
                fitted_rmsf = fwhm_rmsf_radm2
                fit_status = False

            fwhm_rmsf_arr[i] = fitted_rmsf
            fit_status_array[i] = fit_status

    # Remove redundant dimensions
    rmsf_cube = np.squeeze(rmsf_cube)
    fwhm_rmsf_arr = np.squeeze(fwhm_rmsf_arr)
    fit_status_array = np.squeeze(fit_status_array)

    # Restore if 3D shape
    if n_dimension == 3:
        rmsf_cube = np.reshape(
            rmsf_cube, (rmsf_cube.shape[0], old_data_shape[1], old_data_shape[2])
        )
        fwhm_rmsf_arr = np.reshape(
            fwhm_rmsf_arr, (old_data_shape[1], old_data_shape[2])
        )
        fit_status_array = np.reshape(
            fit_status_array, (old_data_shape[1], old_data_shape[2])
        )

    return RMSFResults(
        rmsf_cube=rmsf_cube,
        phi_double_arr_radm2=phi_double_arr_radm2,
        fwhm_rmsf_arr=fwhm_rmsf_arr,
        fit_status_array=fit_status_array,
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
    absFDF = np.abs(np.nan_to_num(dirtyFDF))
    mskCutoff = np.where(np.max(absFDF, axis=0) >= cutoff, 1, 0)
    xyCoords = np.rot90(np.where(mskCutoff > 0))

    # Feeback to user
    if verbose:
        num_pixels = dirtyFDF.shape[-1] * dirtyFDF.shape[-2]
        nCleanum_pixels = len(xyCoords)
        log("Cleaning {:}/{:} spectra.".format(nCleanum_pixels, num_pixels))

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
        residFDF = dirtyFDF.copy()
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
        while np.max(np.abs(residFDF)) >= self.cutoff and iterCount < self.maxIter:
            # Get the absolute peak channel, values and Faraday depth
            indxPeakFDF = np.argmax(np.abs(residFDF))
            peakFDFval = residFDF[indxPeakFDF]
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
            residFDF -= CC * shiftedRMSFArr

            # Restore the CC * a Gaussian to the cleaned FDF
            cleanFDF += gauss1D(CC, phiPeak, fwhm_rmsf_arr)(self.phi_arr_radm2)
            iterCount += 1
            self.iterCountArr[yi, xi] = iterCount

        # Create a mask for the pixels that have been cleaned
        mask = np.abs(ccArr) > 0
        dPhi = self.phi_arr_radm2[1] - self.phi_arr_radm2[0]
        fwhm_rmsf_arr_pix = fwhm_rmsf_arr / dPhi
        for i in np.where(mask)[0]:
            start = int(i - fwhm_rmsf_arr_pix / 2)
            end = int(i + fwhm_rmsf_arr_pix / 2)
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
            residFDF -= CC * shiftedRMSFArr

            # Restore the CC * a Gaussian to the cleaned FDF
            cleanFDF += gauss1D(CC, phiPeak, fwhm_rmsf_arr)(self.phi_arr_radm2)
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


def gaussian(x, amplitude, mean, stddev):
    return Gaussian1D(amplitude=amplitude, mean=mean, stddev=stddev)(x)


def unit_gaussian(x, mean, stddev):
    return Gaussian1D(amplitude=1, mean=mean, stddev=stddev)(x)


def unit_centred_gaussian(x, stddev):
    return Gaussian1D(amplitude=1, mean=0, stddev=stddev)(x)


def fit_rmsf(
    rmsf_to_fit_array: np.ndarray,
    phi_double_arr_radm2: np.ndarray,
    fwhm_rmsf_radm2: float,
) -> float:
    d_phi = phi_double_arr_radm2[1] - phi_double_arr_radm2[0]
    mask = np.zeros_like(phi_double_arr_radm2, dtype=bool)
    mask[np.argmax(rmsf_to_fit_array)] = 1
    fwhm_rmsf_arr_pix = fwhm_rmsf_radm2 / d_phi
    for i in np.where(mask)[0]:
        start = int(i - fwhm_rmsf_arr_pix / 2)
        end = int(i + fwhm_rmsf_arr_pix / 2)
        mask[start : end + 2] = True
    popt, pcov = optimize.curve_fit(
        unit_centred_gaussian,
        phi_double_arr_radm2[mask],
        rmsf_to_fit_array[mask],
        p0=[fwhm_rmsf_radm2],
    )
    return popt[0]


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
