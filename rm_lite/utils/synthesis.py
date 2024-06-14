#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gc
from typing import Literal, NamedTuple, Optional

from astropy.stats import mad_std
import finufft
import numpy as np
from astropy.constants import c as speed_of_light
from scipy.stats import multivariate_normal, scoreatpercentile
from tqdm.auto import tqdm, trange
from uncertainties import unumpy

from rm_lite.utils.fitting import FitResult, fit_fdf, fit_stokes_i_model
from rm_lite.utils.logging import logger


class StokesIArray(np.ndarray):
    pass


class StokesQArray(np.ndarray):
    pass


class StokesUArray(np.ndarray):
    pass


class FWHM(NamedTuple):
    fwhm_rmsf_radm2: float
    """The FWHM of the RMSF main lobe"""
    d_lambda_sq_max_m2: float
    """The maximum difference in lambda^2 values"""
    lambda_sq_range_m2: float
    """The range of lambda^2 values"""


class RMsynthResults(NamedTuple):
    """Results of the RM-synthesis calculation"""

    fdf_dirty_cube: np.ndarray
    """The Faraday dispersion function cube"""
    lam_sq_0_m2: float
    """The reference lambda^2 value"""


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


class FractionalSpectra(NamedTuple):
    stokes_i_model_array: Optional[StokesIArray]
    stokes_q_frac_array: StokesQArray
    stokes_u_frac_array: StokesUArray
    stokes_q_frac_error_array: StokesQArray
    stokes_u_frac_error_array: StokesUArray
    fit_result: FitResult


class FDFParameters(NamedTuple):
    """Parameters of the Faraday dispersion function"""

    fdf_error_mad: float
    """Median absolute deviation error of the FDF"""
    peak_pi_fit: float
    """Peak polarised intensity of the FDF"""
    peak_pi_error: float
    """Error on the peak polarised intensity"""
    peak_pi_fit_debias: float
    """Debiased peak polarised intensity of the FDF"""
    peak_pi_fit_snr: float
    """Signal-to-noise ratio of the peak polarised intensity"""
    peak_pi_fit_index: float
    """Index of the fitted peak polarised intensity"""
    peak_rm_fit: float
    """Peak Faraday depth of the FDF"""
    peak_rm_fit_error: float
    """Error on the peak Faraday depth"""
    peak_q_fit: float
    """Peak Stokes Q of the FDF"""
    peak_u_fit: float
    """Peak Stokes U of the FDF"""
    peak_pa_fit_deg: float
    """Peak position angle of the FDF in degrees"""
    peak_pa_fit_deg_error: float
    """Error on the peak position angle of the FDF in degrees"""
    peak_pa0_fit_deg: float
    """Peak deroated position angle of the FDF in degrees"""
    peak_pa0_fit_deg_error: float
    """Error on the peak deroated position angle of the FDF in degrees"""
    fit_function: Literal["log", "linear"]
    """The function used to fit the FDF"""
    lam_sq_0_m2: float
    """Reference wavelength^2 value"""
    ref_freq_hz: float
    """Reference frequency in Hz"""
    fwhm_rmsf_radm2: float
    """The FWHM of the RMSF main lobe"""
    fdf_error_noise: float
    """Theoretical noise of the FDF"""
    fdf_q_noise: float
    """Theoretical noise of the real FDF"""
    fdf_u_noise: float
    """Theoretical noise of the imaginary FDF"""
    min_freq_hz: float
    """Minimum frequency in Hz"""
    max_freq_hz: float
    """Maximum frequency in Hz"""
    n_channels: int
    """Number of channels"""
    median_d_freq_hz: float
    """Channel width in Hz"""
    frac_pol: float
    """Fractional linear polarisation"""


class TheoreticalNoise(NamedTuple):
    """Theoretical noise of the FDF"""

    fdf_error_noise: float
    """Theoretical noise of the FDF"""
    fdf_q_noise: float
    """Theoretical noise of the real FDF"""
    fdf_u_noise: float
    """Theoretical noise of the imaginary FDF"""

    def with_options(self, **kwargs):
        """Create a new TheoreticalNoise instance with keywords updated

        Returns:
            TheoreticalNoise: New TheoreticalNoise instance with updated attributes
        """
        # TODO: Update the signature to have the actual attributes to
        # help keep mypy and other linters happy
        as_dict = self._asdict()
        as_dict.update(kwargs)

        return TheoreticalNoise(**as_dict)


def calc_mom2_FDF(FDF, phi_arr_radm2):
    """
    Calculate the 2nd moment of the polarised intensity FDF. Can be applied to
    a clean component spectrum or a standard FDF
    """

    K = np.sum(np.abs(FDF))
    phiMean = np.sum(phi_arr_radm2 * np.abs(FDF)) / K
    phiMom2 = np.sqrt(
        np.sum(np.power((phi_arr_radm2 - phiMean), 2.0) * np.abs(FDF)) / K
    )

    return phiMom2


def create_fractional_spectra(
    freq_array_hz: np.ndarray,
    ref_freq_hz: float,
    stokes_i_array: StokesIArray,
    stokes_q_array: StokesQArray,
    stokes_u_array: StokesUArray,
    stokes_i_error_array: StokesIArray,
    stokes_q_error_array: StokesQArray,
    stokes_u_error_array: StokesUArray,
    fit_order: int = 2,
    fit_function: Literal["log", "linear"] = "log",
    stokes_i_model_array: Optional[StokesIArray] = None,
    stokes_i_model_error: Optional[StokesIArray] = None,
    n_error_samples: int = 10_000,
) -> FractionalSpectra:
    no_nan_idx = (
        np.isfinite(stokes_i_array)
        & np.isfinite(stokes_i_error_array)
        & np.isfinite(stokes_q_array)
        & np.isfinite(stokes_q_error_array)
        & np.isfinite(stokes_u_array)
        & np.isfinite(stokes_u_error_array)
        & np.isfinite(freq_array_hz)
    )
    logger.debug(f"{ref_freq_hz=}")
    freq_array_hz = freq_array_hz[no_nan_idx]
    stokes_i_array = stokes_i_array[no_nan_idx]
    stokes_q_array = stokes_q_array[no_nan_idx]
    stokes_u_array = stokes_u_array[no_nan_idx]
    stokes_i_error_array = stokes_i_error_array[no_nan_idx]
    stokes_q_error_array = stokes_q_error_array[no_nan_idx]
    stokes_u_error_array = stokes_u_error_array[no_nan_idx]

    # stokes_i_stokes_u_arrayay = unumpy.stokes_u_arrayay(stokes_i_array, stokes_i_error_array)
    stokes_q_stokes_u_arrayay = unumpy.stokes_u_arrayay(
        stokes_q_array, stokes_q_error_array
    )
    stokes_u_stokes_u_arrayay = unumpy.stokes_u_arrayay(
        stokes_u_array, stokes_u_error_array
    )
    if stokes_i_model_array is not None:
        if stokes_i_model_error is None:
            raise ValueError(
                "If `stokes_i_model_array` is provided, `stokes_i_model_error` must also be provided."
            )
        stokes_i_model_stokes_u_arrayay = unumpy.stokes_u_arrayay(
            stokes_i_model_array, stokes_i_model_error
        )
    else:
        fit_result = fit_stokes_i_model(
            freq_array_hz,
            ref_freq_hz,
            stokes_i_array,
            stokes_i_error_array,
            fit_order,
            fit_function,
        )
        popt, pcov, stokes_i_model_func, aic = fit_result
        error_distribution = multivariate_normal(
            mean=popt, cov=pcov, allow_singular=True
        )
        error_samples = error_distribution.rvs(n_error_samples)

        model_samples = np.array(
            [
                stokes_i_model_func(freq_array_hz / ref_freq_hz, *sample)
                for sample in error_samples
            ]
        )
        stokes_i_model_low, stokes_i_model_array, stokes_i_model_high = np.percentile(
            model_samples, [16, 50, 84], axis=0
        )
        stokes_i_model_stokes_u_arrayay = unumpy.stokes_u_arrayay(
            stokes_i_model_array,
            np.abs((stokes_i_model_high - stokes_i_model_low)),
        )

    stokes_q_frac_stokes_u_arrayay = (
        stokes_q_stokes_u_arrayay / stokes_i_model_stokes_u_arrayay
    )
    stokes_u_frac_stokes_u_arrayay = (
        stokes_u_stokes_u_arrayay / stokes_i_model_stokes_u_arrayay
    )

    stokes_q_frac_array = StokesQArray(
        unumpy.nominal_values(stokes_q_frac_stokes_u_arrayay)
    )
    stokes_u_frac_array = StokesUArray(
        unumpy.nominal_values(stokes_u_frac_stokes_u_arrayay)
    )
    stokes_q_frac_error_array = StokesQArray(
        unumpy.std_devs(stokes_q_frac_stokes_u_arrayay)
    )
    stokes_u_frac_error_array = StokesUArray(
        unumpy.std_devs(stokes_u_frac_stokes_u_arrayay)
    )

    return FractionalSpectra(
        stokes_i_model_array=stokes_i_model_array,
        stokes_q_frac_array=stokes_q_frac_array,
        stokes_u_frac_array=stokes_u_frac_array,
        stokes_q_frac_error_array=stokes_q_frac_error_array,
        stokes_u_frac_error_array=stokes_u_frac_error_array,
        fit_result=fit_result,
    )


def freq_to_lambda2(freq_hz: float) -> float:
    """Convert frequency to lambda^2.

    Args:
        freq_hz (float): Frequency in Hz

    Returns:
        float: Wavelength^2 in m^2
    """
    return (speed_of_light.value / freq_hz) ** 2.0


def lambda2_to_freq(lambda_sq_m2: float) -> float:
    """Convert lambda^2 to frequency.

    Args:
        lambda_sq_m2 (float): Wavelength^2 in m^2

    Returns:
        float: Frequency in Hz
    """
    return speed_of_light.value / np.sqrt(lambda_sq_m2)


def compute_theoretical_noise(
    stokes_q_error_array: StokesQArray,
    stokes_u_error_array: StokesUArray,
    weight_array: np.ndarray,
) -> TheoreticalNoise:
    weight_array = np.nan_to_num(weight_array, nan=0.0, posinf=0.0, neginf=0.0)
    stokes_qu_error_array = np.abs(stokes_q_error_array + stokes_u_error_array) / 2.0
    stokes_qu_error_array = np.nan_to_num(
        stokes_qu_error_array, nan=0.0, posinf=0.0, neginf=0.0
    )
    fdf_error_noise = np.sqrt(
        np.nansum(weight_array**2 * stokes_qu_error_array**2)
        / (np.sum(weight_array)) ** 2
    )
    fdf_q_noise = np.average(stokes_q_error_array, weights=weight_array)
    fdf_u_noise = np.average(stokes_u_error_array, weights=weight_array)
    return TheoreticalNoise(
        fdf_error_noise=fdf_error_noise,
        fdf_q_noise=fdf_q_noise,
        fdf_u_noise=fdf_u_noise,
    )


class RMSynthParams(NamedTuple):
    """Parameters for RM-synthesis calculation"""

    lambda_sq_arr_m2: np.ndarray
    """ Wavelength^2 values in m^2 """
    lam_sq_0_m2: float
    """ Reference wavelength^2 value """
    phi_arr_radm2: np.ndarray
    """ Faraday depth values in rad/m^2 """
    weight_array: np.ndarray
    """ Weight array """


class SigmaAdd(NamedTuple):
    """Sigma_add complexity metrics"""

    sigma_add: float
    """Sigma_add median value"""
    sigma_add_plus: float
    """Sigma_add upper quartile"""
    sigma_add_minus: float
    """Sigma_add lower quartile"""
    sigma_add_cdf: np.ndarray
    """Sigma_add CDF"""
    sigma_add_pdf: np.ndarray
    """Sigma_add PDF"""
    sigma_add_array: np.ndarray
    """Sigma_add array"""


class StokesSigmaAdd(NamedTuple):
    """Stokes Sigma_add complexity metrics"""

    sigma_add_q: SigmaAdd
    """Sigma_add for Stokes Q"""
    sigma_add_u: SigmaAdd
    """Sigma_add for Stokes U"""
    sigma_add_p: SigmaAdd
    """Sigma_add for polarised intensity"""


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
    """Calculate the parameters for RM-synthesis.

    Args:
        freq_array_hz (np.ndarray): Frequency array in Hz
        pol_array (np.ndarray): Complex polarisation array
        stokes_qu_error_array (np.ndarray): Error in Stokes Q and U
        d_phi_radm2 (Optional[float], optional): Pixel spacing in Faraday depth in rad/m^2. Defaults to None.
        n_samples (Optional[float], optional): Number of samples across the RMSF main lobe. Defaults to 10.0.
        phi_max_radm2 (Optional[float], optional): Maximum Faraday depth in rad/m^2. Defaults to None.
        super_resolution (bool, optional): Use Cotton+Rudnick superresolution. Defaults to False.
        weight_type (Literal["variance", "uniform"], optional): Type of weighting to use. Defaults to "variance".

    Raises:
        ValueError: If d_phi_radm2 is not provided and n_samples is None.

    Returns:
        RMSynthParams: Wavelength^2 values, reference wavelength^2, Faraday depth values, weight array
    """
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
    """Construct a Faraday depth array.

    Args:
        phi_max_radm2 (float): Maximum Faraday depth in rad/m^2
        d_phi_radm2 (float): Spacing in Faraday depth in rad/m^2

    Returns:
        np.ndarray: Faraday depth array in rad/m^2
    """
    # Faraday depth sampling. Zero always centred on middle channel
    n_chan_rm = int(np.round(abs((phi_max_radm2 - 0.0) / d_phi_radm2)) * 2.0 + 1.0)
    max_phi_radm2 = (n_chan_rm - 1.0) * d_phi_radm2 / 2.0
    phi_arr_radm2 = np.linspace(-max_phi_radm2, max_phi_radm2, n_chan_rm)
    return phi_arr_radm2


def get_fwhm_rmsf(
    lambda_sq_arr_m2: np.ndarray,
    super_resolution: bool = False,
) -> FWHM:
    """Calculate the FWHM of the RMSF.

    Args:
        lambda_sq_arr_m2 (np.ndarray): Wavelength^2 values in m^2
        super_resolution (bool, optional): Use Cotton+Rudnick superresolution. Defaults to False.

    Returns:
        fwhm_rmsf_arr: FWHM of the RMSF main lobe, maximum difference in lambda^2 values, range of lambda^2 values
    """
    lambda_sq_range_m2 = np.nanmax(lambda_sq_arr_m2) - np.nanmin(lambda_sq_arr_m2)
    d_lambda_sq_max_m2 = np.nanmax(np.abs(np.diff(lambda_sq_arr_m2)))

    # Set the Faraday depth range
    if not super_resolution:
        fwhm_rmsf_radm2 = 3.8 / lambda_sq_range_m2  # Dickey+2019 theoretical RMSF width
    else:  # If super resolution, use R&C23 theoretical width
        fwhm_rmsf_radm2 = 2.0 / (
            np.nanmax(lambda_sq_arr_m2) + np.nanmin(lambda_sq_arr_m2)
        )
    return FWHM(
        fwhm_rmsf_radm2=fwhm_rmsf_radm2,
        d_lambda_sq_max_m2=d_lambda_sq_max_m2,
        lambda_sq_range_m2=lambda_sq_range_m2,
    )


def rmsynth_nufft(
    stokes_q_array: StokesQArray,
    stokes_u_array: StokesUArray,
    lambda_sq_arr_m2: np.ndarray,
    phi_arr_radm2: np.ndarray,
    weight_array: np.ndarray,
    lam_sq_0_m2: float,
    eps: float = 1e-6,
) -> np.ndarray:
    """Run RM-synthesis on a cube of Stokes Q and U data using the NUFFT method.

    Args:
        stokes_q_array (StokesQArray): Stokes Q data array
        stokes_u_array (StokesUArray): Stokes U data array
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
            cleanFDF += gauss1D(CC, phiPeak, fwhm_rmsf_arr)(self.phi_arr_radm2)
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
            cleanFDF += gauss1D(CC, phiPeak, fwhm_rmsf_arr)(self.phi_arr_radm2)
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


def get_fdf_parameters(
    fdf_array: np.ndarray,
    phi_arr_radm2: np.ndarray,
    fwhm_rmsf_radm2: float,
    freq_array_hz: np.ndarray,
    stokes_q_array: StokesQArray,
    stokes_u_array: StokesUArray,
    stokes_q_error_array: StokesQArray,
    stokes_u_error_array: StokesUArray,
    lambda_sq_arr_m2: np.ndarray,
    lam_sq_0_m2: float,
    stokes_i_reference_flux: float,
    theoretical_noise: TheoreticalNoise,
    fit_function: Literal["log", "linear"],
    bias_correction_snr: float = 5.0,
) -> FDFParameters:
    """
    Measure standard parameters from a complex Faraday Dispersion Function.
    Currently this function assumes that the noise levels in the Stokes Q
    and U spectra are the same.
    Returns a dictionary containing measured parameters.
    """

    abs_fdf_array = np.abs(fdf_array)
    peak_pi_index = np.nanargmax(abs_fdf_array)

    # Measure the RMS noise in the spectrum after masking the peak
    d_phi = phi_arr_radm2[1] - phi_arr_radm2[0]
    mask = np.ones_like(phi_arr_radm2, dtype=bool)
    mask[peak_pi_index] = False
    fwhm_rmsf_arr_pix = fwhm_rmsf_radm2 / d_phi
    for i in np.where(mask)[0]:
        start = int(i - fwhm_rmsf_arr_pix / 2)
        end = int(i + fwhm_rmsf_arr_pix / 2)
        mask[start : end + 2] = False

    fdf_error_mad: float = mad_std(
        np.concatenate([fdf_array[mask].real, fdf_array[mask].imag])
    )

    n_good_phi = np.isfinite(fdf_array).sum()
    lambda_sq_arr_m2_variance = (
        np.sum(lambda_sq_arr_m2**2.0) - np.sum(lambda_sq_arr_m2) ** 2.0 / n_good_phi
    ) / (n_good_phi - 1)

    good_chan_idx = np.isfinite(freq_array_hz)
    n_good_chan = good_chan_idx.sum()

    if not (peak_pi_index > 0 and peak_pi_index < len(abs_fdf_array) - 1):
        return FDFParameters(
            fdf_error_mad=fdf_error_mad,
            peak_pi_fit=np.nan,
            peak_pi_error=theoretical_noise.fdf_error_noise,
            peak_pi_fit_debias=np.nan,
            peak_pi_fit_snr=np.nan,
            peak_pi_fit_index=np.nan,
            peak_rm_fit=np.nan,
            peak_rm_fit_error=np.nan,
            peak_q_fit=np.nan,
            peak_u_fit=np.nan,
            peak_pa_fit_deg=np.nan,
            peak_pa_fit_deg_error=np.nan,
            peak_pa0_fit_deg=np.nan,
            peak_pa0_fit_deg_error=np.nan,
            fit_function=fit_function,
            lam_sq_0_m2=lam_sq_0_m2,
            ref_freq_hz=lambda2_to_freq(lam_sq_0_m2),
            fwhm_rmsf_radm2=fwhm_rmsf_radm2,
            fdf_error_noise=theoretical_noise.fdf_error_noise,
            fdf_q_noise=theoretical_noise.fdf_q_noise,
            fdf_u_noise=theoretical_noise.fdf_u_noise,
            min_freq_hz=freq_array_hz[good_chan_idx].min(),
            max_freq_hz=freq_array_hz[good_chan_idx].max(),
            n_channels=n_good_chan,
            median_d_freq_hz=np.nanmedian(np.diff(freq_array_hz[good_chan_idx])),
            frac_pol=np.nan,
        )
    peak_pi_fit, peak_rm_fit, _ = fit_fdf(
        fdf_to_fit_array=abs_fdf_array,
        phi_arr_radm2=phi_arr_radm2,
        fwhm_fdf_radm2=fwhm_rmsf_radm2,
    )
    peak_pi_fit_snr = peak_pi_fit / theoretical_noise.fdf_error_noise
    # In rare cases, a parabola can be fitted to the edge of the spectrum,
    # producing a unreasonably large RM and polarized intensity.
    # In these cases, everything should get NaN'd out.
    if np.abs(peak_rm_fit) > np.max(np.abs(phi_arr_radm2)):
        peak_rm_fit = np.nan
        peak_pi_fit = np.nan

    # Error on fitted Faraday depth (RM) is same as channel, but using fitted PI
    peak_rm_fit_err = (
        fwhm_rmsf_radm2 * theoretical_noise.fdf_error_noise / (2.0 * peak_pi_fit)
    )

    # Correct the peak for polarisation bias (POSSUM report 11)
    peak_pi_fit_debias = peak_pi_fit
    if peak_pi_fit_snr >= bias_correction_snr:
        peak_pi_fit_debias = np.sqrt(
            peak_pi_fit**2.0 - 2.3 * theoretical_noise.fdf_error_noise**2.0
        )

    # Calculate the polarisation angle from the fitted peak
    # Uncertainty from Eqn A.12 in Brentjens & De Bruyn 2005
    peak_pi_fit_index = np.interp(
        peak_rm_fit, phi_arr_radm2, np.arange(phi_arr_radm2.shape[-1], dtype="f4")
    )
    peak_u_fit = np.interp(peak_rm_fit, phi_arr_radm2, fdf_array.imag)
    peak_q_fit = np.interp(peak_rm_fit, phi_arr_radm2, fdf_array.real)
    peak_pa_fit_deg = 0.5 * np.degrees(np.arctan2(peak_u_fit, peak_q_fit)) % 180
    peak_pa_fit_deg_err = np.degrees(
        theoretical_noise.fdf_error_noise / (2.0 * peak_pi_fit)
    )

    # Calculate the derotated polarisation angle and uncertainty
    # Uncertainty from Eqn A.20 in Brentjens & De Bruyn 2005
    peak_pa0_fit_deg = (
        float(np.degrees(np.radians(peak_pa_fit_deg) - peak_rm_fit * lam_sq_0_m2))
        % 180.0
    )
    peak_pa0_fit_rad_err = np.sqrt(
        theoretical_noise.fdf_error_noise**2.0
        * n_good_phi
        / (4.0 * (n_good_phi - 2.0) * peak_pi_fit**2.0)
        * ((n_good_phi - 1) / n_good_phi + lam_sq_0_m2**2.0 / lambda_sq_arr_m2_variance)
    )
    peak_pa0_fit_deg_err = float(np.degrees(peak_pa0_fit_rad_err))

    stokes_sigma_add = measure_qu_complexity(
        freq_arr_hz=freq_array_hz,
        stokes_q_array=stokes_q_array,
        stokes_u_array=stokes_u_array,
        stokes_q_err_array=stokes_q_error_array,
        stokes_u_err_array=stokes_u_error_array,
        frac_pol=peak_pi_fit_debias / stokes_i_reference_flux,
        psi0_deg=peak_pa0_fit_deg,
        rm_radm2=peak_rm_fit,
    )

    return FDFParameters(
        fdf_error_mad=fdf_error_mad,
        peak_pi_fit=peak_pi_fit,
        peak_pi_error=fdf_error,
        peak_pi_fit_debias=peak_pi_fit_debias,
        peak_pi_fit_snr=peak_pi_fit_snr,
        peak_pi_fit_index=peak_pi_fit_index,
        peak_rm_fit=peak_rm_fit,
        peak_rm_fit_error=peak_rm_fit_err,
        peak_q_fit=peak_q_fit,
        peak_u_fit=peak_u_fit,
        peak_pa_fit_deg=peak_pa_fit_deg,
        peak_pa_fit_deg_error=peak_pa_fit_deg_err,
        peak_pa0_fit_deg=peak_pa0_fit_deg,
        peak_pa0_fit_deg_error=peak_pa0_fit_deg_err,
        fit_function=fit_function,
        lam_sq_0_m2=lam_sq_0_m2,
        ref_freq_hz=lambda2_to_freq(lam_sq_0_m2),
        fwhm_rmsf_radm2=fwhm_rmsf_radm2,
        fdf_error_noise=theoretical_noise.fdf_error_noise,
        fdf_q_noise=theoretical_noise.fdf_q_noise,
        fdf_u_noise=theoretical_noise.fdf_u_noise,
        min_freq_hz=freq_array_hz[good_chan_idx].min(),
        max_freq_hz=freq_array_hz[good_chan_idx].max(),
        n_channels=n_good_chan,
        median_d_freq_hz=np.nanmedian(np.diff(freq_array_hz[good_chan_idx])),
        frac_pol=peak_pi_fit_debias / stokes_i_reference_flux,
    )


# # -----------------------------------------------------------------------------#
# def norm_cdf(mean=0.0, std=1.0, N=50, x_array=None):
#     """Return the CDF of a normal distribution between -6 and 6 sigma, or at
#     the values of an input array."""

#     if x_array is None:
#         x = np.linspace(-6.0 * std, 6.0 * std, N)
#     else:
#         x = x_array
#     y = norm.cdf(x, loc=mean, scale=std)

#     return x, y


def cdf_percentile(values: np.ndarray, cdf: np.ndarray, q=50.0) -> float:
    """Return the value at a given percentile of a cumulative distribution function

    Args:
        values (np.ndarray): Array of values
        cdf (np.ndarray): Cumulative distribution function
        q (float, optional): Percentile. Defaults to 50.0.

    Returns:
        float: Interpolated value at the given percentile
    """
    return np.interp(q / 100.0, cdf, values)


def calculate_sigma_add(
    y_array: np.ndarray,
    dy_array: np.ndarray,
    median: Optional[float] = None,
    noise: Optional[float] = None,
    n_samples: int = 1000,
) -> SigmaAdd:
    """Calculate the most likely value of additional scatter, assuming the
    input data is drawn from a normal distribution. The total uncertainty on
    each data point Y_i is modelled as dYtot_i**2 = dY_i**2 + dYadd**2."""

    # Measure the median and MADFM of the input data if not provided.
    # Used to overplot a normal distribution when debugging.
    if median is None:
        median = np.nanmedian(y_array)
    if noise is None:
        noise = mad_std(y_array)

    # Sample the PDF of the additional noise term from a limit near zero to
    # a limit of the range of the data, including error bars
    y_range = np.nanmax(y_array + dy_array) - np.nanmin(y_array - dy_array)
    sigma_add_arr = np.linspace(y_range / n_samples, y_range, n_samples)

    # Model deviation from Gaussian as an additional noise term.
    # Loop through the range of i additional noise samples and calculate
    # chi-squared and sum(ln(sigma_total)), used later to calculate likelihood.
    n_data = len(y_array)

    # Calculate sigma_sq_tot for all sigma_add values
    sigma_sq_tot = dy_array**2.0 + sigma_add_arr[:, None] ** 2.0

    # Calculate ln_sigma_sum_arr for all sigma_add values
    ln_sigma_sum_arr = np.nansum(np.log(np.sqrt(sigma_sq_tot)), axis=1)

    # Calculate chi_sq_arr for all sigma_add values
    chi_sq_arr = np.nansum((y_array - median) ** 2.0 / sigma_sq_tot, axis=1)
    ln_prob_arr = (
        -np.log(sigma_add_arr)
        - n_data * np.log(2.0 * np.pi) / 2.0
        - ln_sigma_sum_arr
        - chi_sq_arr / 2.0
    )
    ln_prob_arr -= np.nanmax(ln_prob_arr)
    prob_arr = np.exp(ln_prob_arr)
    # Normalize the area under the PDF to be 1
    prob_arr /= np.nansum(prob_arr * np.diff(sigma_add_arr)[0])
    # Calculate the CDF
    cdf = np.cumsum(prob_arr) / np.nansum(prob_arr)

    # Calculate the mean of the distribution and the +/- 1-sigma limits
    sigma_add = cdf_percentile(values=sigma_add_arr, cdf=cdf, q=50.0)
    sigma_add_minus = cdf_percentile(values=sigma_add_arr, cdf=cdf, q=15.72)
    sigma_add_plus = cdf_percentile(values=sigma_add_arr, cdf=cdf, q=84.27)

    return SigmaAdd(
        sigma_add=sigma_add,
        sigma_add_minus=sigma_add_minus,
        sigma_add_plus=sigma_add_plus,
        sigma_add_cdf=cdf,
        sigma_add_pdf=prob_arr,
        sigma_add_array=sigma_add_arr,
    )


def faraday_simple_spectrum(
    freq_arr_hz: np.ndarray,
    frac_pol: float,
    psi0_deg: float,
    rm_radm2: float,
) -> np.ndarray:
    """Create a simple Faraday spectrum with a single component.

    Args:
        freq_arr_hz (np.ndarray): Frequency array in Hz
        frac_pol (float): Fractional polarization
        psi0_deg (float): Initial polarization angle in degrees
        rm_radm2 (float): RM in rad/m^2

    Returns:
        np.ndarray: Complex polarization spectrum
    """
    lambda_sq_arr_m2 = freq_to_lambda2(freq_arr_hz)

    complex_polarization = frac_pol * np.exp(
        2j * np.radians(psi0_deg + rm_radm2 * lambda_sq_arr_m2)
    )

    return complex_polarization


def measure_qu_complexity(
    freq_arr_hz: np.ndarray,
    stokes_q_array: StokesQArray,
    stokes_u_array: StokesUArray,
    stokes_q_err_array: StokesQArray,
    stokes_u_err_array: StokesUArray,
    frac_pol: float,
    psi0_deg: float,
    rm_radm2: float,
) -> StokesSigmaAdd:
    # Create a RM-thin model to subtract
    complex_polarisation = faraday_simple_spectrum(
        freq_arr_hz=freq_arr_hz,
        frac_pol=frac_pol,
        psi0_deg=psi0_deg,
        rm_radm2=rm_radm2,
    )

    # Subtract the RM-thin model to create a residual q & u
    stokes_q_residual = stokes_q_array - complex_polarisation.real
    stokes_u_residual = stokes_u_array - complex_polarisation.imag

    sigma_add_q = calculate_sigma_add(
        y_array=stokes_q_residual / stokes_q_err_array,
        dy_array=np.ones_like(stokes_q_residual),
        median=0.0,
        noise=1.0,
    )
    sigma_add_u = calculate_sigma_add(
        y_array=stokes_u_residual / stokes_u_err_array,
        dy_array=np.ones_like(stokes_u_residual),
        median=0.0,
        noise=1.0,
    )

    sigma_add_p_array = np.hypot(sigma_add_q.sigma_add, sigma_add_u.sigma_add)
    sigma_add_p_pdf = np.hypot(sigma_add_q.sigma_add_pdf, sigma_add_u.sigma_add_pdf)
    sigma_add_p_cdf = np.cumsum(sigma_add_p_pdf) / np.nansum(sigma_add_p_pdf)
    sigma_add_p_val = cdf_percentile(
        values=sigma_add_p_array, cdf=sigma_add_p_cdf, q=50.0
    )
    sigma_add_p_minus = cdf_percentile(
        values=sigma_add_p_array, cdf=sigma_add_p_cdf, q=15.72
    )
    sigma_add_p_plus = cdf_percentile(
        values=sigma_add_p_array, cdf=sigma_add_p_cdf, q=84.27
    )
    sigma_add_p = SigmaAdd(
        sigma_add=sigma_add_p_val,
        sigma_add_minus=sigma_add_p_minus,
        sigma_add_plus=sigma_add_p_plus,
        sigma_add_cdf=sigma_add_p_cdf,
        sigma_add_pdf=sigma_add_p_pdf,
        sigma_add_array=sigma_add_p_array,
    )

    return StokesSigmaAdd(
        sigma_add_q=sigma_add_q,
        sigma_add_u=sigma_add_u,
        sigma_add_p=sigma_add_p,
    )


def measure_fdf_complexity(phi_arr_radm2, FDF):
    # Second moment of clean component spectrum
    return calc_mom2_FDF(FDF, phi_arr_radm2)
