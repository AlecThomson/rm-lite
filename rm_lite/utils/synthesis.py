#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RM-synthesis utils"""

from typing import Literal, NamedTuple, Optional

from astropy.stats import mad_std
import finufft
import numpy as np
from astropy.constants import c as speed_of_light
from scipy.stats import multivariate_normal
from scipy.interpolate import interp1d
from tqdm.auto import trange
from uncertainties import unumpy

from rm_lite.utils.fitting import FitResult, fit_fdf, fit_stokes_i_model, fit_rmsf
from rm_lite.utils.logging import logger


class StokesIArray(np.ndarray):
    def __new__(cls, input_arr):
        return np.asarray(input_arr).view(cls)

    def __arr_finalize__(self, obj) -> None:
        if obj is None:
            return
        # This attribute should be maintained!
        default_attributes = {"attr": 1}
        self.__dict__.update(default_attributes)


class StokesQArray(np.ndarray):
    def __new__(cls, input_arr):
        return np.asarray(input_arr).view(cls)


class StokesUArray(np.ndarray):
    def __new__(cls, input_arr):
        return np.asarray(input_arr).view(cls)


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
    fit_status_arr: np.ndarray
    """The status of the RMSF fit"""


class FractionalSpectra(NamedTuple):
    stokes_i_model_arr: Optional[StokesIArray]
    stokes_q_frac_arr: StokesQArray
    stokes_u_frac_arr: StokesUArray
    stokes_q_frac_error_arr: StokesQArray
    stokes_u_frac_error_arr: StokesUArray
    fit_result: FitResult
    no_nan_idx: np.ndarray


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

    def __str__(self):
        return (
            f"RMSF FWHM: {self.fwhm_rmsf_radm2:.2f}\n"
            f"Peak pI: {self.peak_pi_fit:.2f} ± {self.peak_pi_error:.2f}\n"
            f"Peak RM: {self.peak_rm_fit:.2f} ± {self.peak_rm_fit_error:.2f}\n"
            f"Peak PA: {self.peak_pa_fit_deg:.2f} ± {self.peak_pa_fit_deg_error:.2f}\n"
            f"Peak PA0: {self.peak_pa0_fit_deg:.2f} ± {self.peak_pa0_fit_deg_error:.2f}\n"
            f"Peak Q: {self.peak_q_fit:.2f} ± {self.fdf_q_noise:.2f}, Peak U: {self.peak_u_fit:.2f} ± {self.fdf_u_noise:.2f}\n"
            f"freq0: {self.ref_freq_hz:.2f}, lam0: {self.lam_sq_0_m2:.2f}\n"
            f"Fit function: {self.fit_function}\n"
            f"Frac pol: {self.frac_pol:.2f}\n"
            f"pI error (MAD): {self.fdf_error_mad:.2f}\n"
            f"pI error (noise): {self.fdf_error_noise:.2f}\n"
            f"pI SNR: {self.peak_pi_fit_snr:.2f}\n"
            f"Min freq: {self.min_freq_hz:.2f}, Max freq: {self.max_freq_hz:.2f}\n"
            f"n_channels: {self.n_channels}, Median d_freq: {self.median_d_freq_hz:.2f}\n"
        )

    def __repr__(self):
        return self.__str__()


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
    freq_arr_hz: np.ndarray,
    ref_freq_hz: float,
    stokes_i_arr: StokesIArray,
    stokes_q_arr: StokesQArray,
    stokes_u_arr: StokesUArray,
    stokes_i_error_arr: StokesIArray,
    stokes_q_error_arr: StokesQArray,
    stokes_u_error_arr: StokesUArray,
    fit_order: int = 2,
    fit_function: Literal["log", "linear"] = "log",
    stokes_i_model_arr: Optional[StokesIArray] = None,
    stokes_i_model_error: Optional[StokesIArray] = None,
    n_error_samples: int = 10_000,
) -> FractionalSpectra:
    no_nan_idx = (
        np.isfinite(stokes_i_arr)
        & np.isfinite(stokes_i_error_arr)
        & np.isfinite(stokes_q_arr)
        & np.isfinite(stokes_q_error_arr)
        & np.isfinite(stokes_u_arr)
        & np.isfinite(stokes_u_error_arr)
        & np.isfinite(freq_arr_hz)
    )
    logger.debug(f"{ref_freq_hz=}")
    freq_arr_hz = freq_arr_hz[no_nan_idx]
    stokes_i_arr = stokes_i_arr[no_nan_idx]
    stokes_q_arr = stokes_q_arr[no_nan_idx]
    stokes_u_arr = stokes_u_arr[no_nan_idx]
    stokes_i_error_arr = stokes_i_error_arr[no_nan_idx]
    stokes_q_error_arr = stokes_q_error_arr[no_nan_idx]
    stokes_u_error_arr = stokes_u_error_arr[no_nan_idx]

    # stokes_i_stokes_u_array = unumpy.stokes_u_array(stokes_i_arr, stokes_i_error_arr)
    stokes_q_uarray = unumpy.uarray(stokes_q_arr, stokes_q_error_arr)
    stokes_u_uarray = unumpy.uarray(stokes_u_arr, stokes_u_error_arr)
    if stokes_i_model_arr is not None:
        stokes_i_model_arr = stokes_i_model_arr[no_nan_idx]
        if stokes_i_model_error is None:
            raise ValueError(
                "If `stokes_i_model_arr` is provided, `stokes_i_model_error` must also be provided."
            )
        stokes_i_model_error = stokes_i_model_error[no_nan_idx]
        stokes_i_model_stokes_u_array = unumpy.stokes_u_array(
            stokes_i_model_arr, stokes_i_model_error
        )
    else:
        fit_result = fit_stokes_i_model(
            freq_arr_hz,
            ref_freq_hz,
            stokes_i_arr,
            stokes_i_error_arr,
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
                stokes_i_model_func(freq_arr_hz / ref_freq_hz, *sample)
                for sample in error_samples
            ]
        )
        stokes_i_model_low, stokes_i_model_arr, stokes_i_model_high = np.percentile(
            model_samples, [16, 50, 84], axis=0
        )
        stokes_i_model_stokes_u_array = unumpy.uarray(
            stokes_i_model_arr,
            np.abs((stokes_i_model_high - stokes_i_model_low)),
        )

    stokes_q_frac_uarray = stokes_q_uarray / stokes_i_model_stokes_u_array
    stokes_u_frac_uarray = stokes_u_uarray / stokes_i_model_stokes_u_array

    stokes_q_frac_arr = StokesQArray(unumpy.nominal_values(stokes_q_frac_uarray))
    stokes_u_frac_arr = StokesUArray(unumpy.nominal_values(stokes_u_frac_uarray))
    stokes_q_frac_error_arr = StokesQArray(unumpy.std_devs(stokes_q_frac_uarray))
    stokes_u_frac_error_arr = StokesUArray(unumpy.std_devs(stokes_u_frac_uarray))

    assert len(stokes_i_arr) == len(stokes_q_frac_arr)
    assert len(stokes_i_arr) == len(stokes_u_frac_arr)
    assert len(stokes_i_arr) == len(stokes_q_frac_error_arr)
    assert len(stokes_i_arr) == len(stokes_u_frac_error_arr)

    return FractionalSpectra(
        stokes_i_model_arr=stokes_i_model_arr,
        stokes_q_frac_arr=stokes_q_frac_arr,
        stokes_u_frac_arr=stokes_u_frac_arr,
        stokes_q_frac_error_arr=stokes_q_frac_error_arr,
        stokes_u_frac_error_arr=stokes_u_frac_error_arr,
        fit_result=fit_result,
        no_nan_idx=no_nan_idx,
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
    stokes_q_error_arr: StokesQArray,
    stokes_u_error_arr: StokesUArray,
    weight_arr: np.ndarray,
) -> TheoreticalNoise:
    weight_arr = np.nan_to_num(weight_arr, nan=0.0, posinf=0.0, neginf=0.0)
    stokes_qu_error_arr = np.abs(stokes_q_error_arr + stokes_u_error_arr) / 2.0
    stokes_qu_error_arr = np.nan_to_num(
        stokes_qu_error_arr, nan=0.0, posinf=0.0, neginf=0.0
    )
    fdf_error_noise = np.sqrt(
        np.nansum(weight_arr**2 * stokes_qu_error_arr**2) / (np.sum(weight_arr)) ** 2
    )
    fdf_q_noise = np.average(stokes_q_error_arr, weights=weight_arr)
    fdf_u_noise = np.average(stokes_u_error_arr, weights=weight_arr)
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
    weight_arr: np.ndarray
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
    sigma_add_arr: np.ndarray
    """Sigma_add array"""


class StokesSigmaAdd(NamedTuple):
    """Stokes Sigma_add complexity metrics"""

    sigma_add_q: SigmaAdd
    """Sigma_add for Stokes Q"""
    sigma_add_u: SigmaAdd
    """Sigma_add for Stokes U"""
    sigma_add_p: SigmaAdd
    """Sigma_add for polarised intensity"""


class RMSFParams(NamedTuple):
    """RM spread function parameters"""

    rmsf_fwhm_theory: float
    """ Theoretical FWHM of the RMSF """
    rmsf_fwhm_meas: float
    """ Measured FWHM of the RMSF """
    phi_max: float
    """ Maximum Faraday depth """
    phi_max_scale: float
    """ Maximum Faraday depth scale """


def compute_rmsf_params(
    freq_arr_hz: np.ndarray,
    weight_arr: np.ndarray,
    super_resolution: bool = False,
) -> RMSFParams:
    lambda_sq_arr_m2 = freq_to_lambda2(freq_arr_hz)
    # lam_sq_0_m2 is the weighted mean of lambda^2 distribution (B&dB Eqn. 32)
    # Calculate a global lam_sq_0_m2 value, ignoring isolated flagged voxels
    scale_factor = 1.0 / np.nansum(weight_arr)
    lam_sq_0_m2 = scale_factor * np.nansum(weight_arr * lambda_sq_arr_m2)
    if not np.isfinite(lam_sq_0_m2):
        lam_sq_0_m2 = np.nanmean(lambda_sq_arr_m2)
    if super_resolution:
        lam_sq_0_m2 = 0.0

    lambda_sq_m2_max = np.nanmax(lambda_sq_arr_m2)
    lambda_sq_m2_min = np.nanmin(lambda_sq_arr_m2)
    delta_lambda_sq_m2 = np.median(np.abs(np.diff(lambda_sq_arr_m2)))

    rmsf_fwhm_theory = 3.8 / (lambda_sq_m2_max - lambda_sq_m2_min)
    phi_max = np.sqrt(3.0) / delta_lambda_sq_m2
    phi_max_scale = np.pi / lambda_sq_m2_min
    dphi = 0.1 * rmsf_fwhm_theory

    phi_arr_radm2 = make_phi_arr(phi_max * 10 * 2, dphi)

    rmsf_results = get_rmsf_nufft(
        lambda_sq_arr_m2=lambda_sq_arr_m2,
        phi_arr_radm2=phi_arr_radm2,
        weight_arr=weight_arr,
        lam_sq_0_m2=lam_sq_0_m2,
        super_resolution=super_resolution,
    )

    rmsf_fwhm_meas = float(rmsf_results.fwhm_rmsf_arr)

    return RMSFParams(
        rmsf_fwhm_theory=rmsf_fwhm_theory,
        rmsf_fwhm_meas=rmsf_fwhm_meas,
        phi_max=phi_max,
        phi_max_scale=phi_max_scale,
    )


def compute_rmsynth_params(
    freq_arr_hz: np.ndarray,
    pol_arr: np.ndarray,
    stokes_qu_error_arr: np.ndarray,
    d_phi_radm2: Optional[float] = None,
    n_samples: Optional[float] = 10.0,
    phi_max_radm2: Optional[float] = None,
    super_resolution: bool = False,
    weight_type: Literal["variance", "uniform"] = "variance",
) -> RMSynthParams:
    """Calculate the parameters for RM-synthesis.

    Args:
        freq_arr_hz (np.ndarray): Frequency array in Hz
        pol_arr (np.ndarray): Complex polarisation array
        stokes_qu_error_arr (np.ndarray): Error in Stokes Q and U
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
    lambda_sq_arr_m2 = freq_to_lambda2(freq_arr_hz)

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

    phi_arr_radm2 = make_phi_arr(phi_max_radm2, d_phi_radm2)

    logger.debug(
        f"phi = {phi_arr_radm2[0]:0.2f} to {phi_arr_radm2[-1]:0.2f} by {d_phi_radm2:0.2f} ({len(phi_arr_radm2)} chans)."
    )

    # Calculate the weighting as 1/sigma^2 or all 1s (uniform)
    if weight_type == "variance":
        if (stokes_qu_error_arr == 0).all():
            stokes_qu_error_arr = np.ones(len(stokes_qu_error_arr))
        weight_arr = 1.0 / stokes_qu_error_arr**2
    else:
        weight_arr = np.ones_like(freq_arr_hz)

    logger.debug(f"Weighting type: {weight_type}")

    mask = ~np.isfinite(pol_arr)
    weight_arr[mask] = 0.0

    # lam_sq_0_m2 is the weighted mean of lambda^2 distribution (B&dB Eqn. 32)
    # Calculate a global lam_sq_0_m2 value, ignoring isolated flagged voxels
    scale_factor = 1.0 / np.nansum(weight_arr)
    lam_sq_0_m2 = scale_factor * np.nansum(weight_arr * lambda_sq_arr_m2)
    if not np.isfinite(lam_sq_0_m2):
        lam_sq_0_m2 = np.nanmean(lambda_sq_arr_m2)

    logger.debug(f"lam_sq_0_m2 = {lam_sq_0_m2:0.2f} m^2")

    return RMSynthParams(
        lambda_sq_arr_m2=lambda_sq_arr_m2,
        lam_sq_0_m2=lam_sq_0_m2,
        phi_arr_radm2=phi_arr_radm2,
        weight_arr=weight_arr,
    )


def make_phi_arr(
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


def make_double_phi_arr(
    phi_arr_radm2: np.ndarray,
) -> np.ndarray:
    n_phi = len(phi_arr_radm2)
    n_ext = np.ceil(n_phi / 2.0)
    resamp_index = np.arange(2.0 * n_ext + n_phi) - n_ext
    phi_double_arr_radm2 = interp1d(
        np.arange(n_phi), phi_arr_radm2, fill_value="extrapolate"
    )(resamp_index)
    return phi_double_arr_radm2


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
    stokes_q_arr: StokesQArray,
    stokes_u_arr: StokesUArray,
    lambda_sq_arr_m2: np.ndarray,
    phi_arr_radm2: np.ndarray,
    weight_arr: np.ndarray,
    lam_sq_0_m2: float,
    eps: float = 1e-6,
) -> np.ndarray:
    """Run RM-synthesis on a cube of Stokes Q and U data using the NUFFT method.

    Args:
        stokes_q_arr (StokesQArray): Stokes Q data array
        stokes_u_arr (StokesUArray): Stokes U data array
        lambda_sq_arr_m2 (np.ndarray): Wavelength^2 values in m^2
        phi_arr_radm2 (np.ndarray): Faraday depth values in rad/m^2
        weight_arr (np.ndarray): Weight array
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
    weight_arr = np.nan_to_num(weight_arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Sanity check on array sizes
    if not weight_arr.shape == lambda_sq_arr_m2.shape:
        raise ValueError(
            f"Weight and lambda^2 arrays must be the same shape. Got {weight_arr.shape} and {lambda_sq_arr_m2.shape}"
        )

    if not stokes_q_arr.shape == stokes_u_arr.shape:
        raise ValueError("Stokes Q and U data arrays must be the same shape.")

    n_dims = len(stokes_q_arr.shape)
    if not n_dims <= 3:
        raise ValueError(f"Data dimensions must be <= 3. Got {n_dims}")

    if not stokes_q_arr.shape[0] == lambda_sq_arr_m2.shape[0]:
        raise ValueError(
            f"Data depth does not match lambda^2 vector ({stokes_q_arr.shape[0]} vs {lambda_sq_arr_m2.shape[0]})."
        )

    # Reshape the data arrays to 2 dimensions
    if n_dims == 1:
        stokes_q_arr = np.reshape(stokes_q_arr, (stokes_q_arr.shape[0], 1))
        stokes_u_arr = np.reshape(stokes_u_arr, (stokes_u_arr.shape[0], 1))
    elif n_dims == 3:
        old_data_shape = stokes_q_arr.shape
        stokes_q_arr = np.reshape(
            stokes_q_arr,
            (
                stokes_q_arr.shape[0],
                stokes_q_arr.shape[1] * stokes_q_arr.shape[2],
            ),
        )
        stokes_u_arr = np.reshape(
            stokes_u_arr,
            (
                stokes_u_arr.shape[0],
                stokes_u_arr.shape[1] * stokes_u_arr.shape[2],
            ),
        )

    # Create a complex polarised cube, B&dB Eqns. (8) and (14)
    # Array has dimensions [nFreq, nY * nX]
    pol_cube = (stokes_q_arr + 1j * stokes_u_arr) * weight_arr[:, np.newaxis]

    # Check for NaNs (flagged data) in the cube & set to zero
    mask_cube = ~np.isfinite(pol_cube)
    pol_cube = np.nan_to_num(pol_cube, nan=0.0, posinf=0.0, neginf=0.0)

    # If full planes are flagged then set corresponding weights to zero
    mask_planes = np.sum(~mask_cube, axis=1)
    mask_planes = np.where(mask_planes == 0, 0, 1)
    weight_arr *= mask_planes

    # The K value used to scale each FDF spectrum must take into account
    # flagged voxels data in the datacube and can be position dependent
    weight_cube = np.invert(mask_cube) * weight_arr[:, np.newaxis]
    with np.errstate(divide="ignore", invalid="ignore"):
        scale_arr = np.true_divide(1.0, np.sum(weight_cube, axis=0))
        scale_arr[scale_arr == np.inf] = 0
        scale_arr = np.nan_to_num(scale_arr)

    # Clean up one cube worth of memory
    del weight_cube

    # Do the RM-synthesis on each plane
    # finufft must have matching dtypes, so complex64 matches float32
    exponent = (lambda_sq_arr_m2 - lam_sq_0_m2).astype(
        f"float{pol_cube.itemsize*8/2:.0f}"
    )
    fdf_dirty_cube = (
        finufft.nufft1d3(
            x=exponent,
            c=np.ascontiguousarray(pol_cube.T),
            s=(phi_arr_radm2 * 2).astype(exponent.dtype),
            eps=eps,
            isign=-1,
        )
        * scale_arr[..., None]
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


def inverse_rmsynth_nufft(
    fdf_q_arr: StokesQArray,
    fdf_u_arr: StokesUArray,
    lambda_sq_arr_m2: np.ndarray,
    phi_arr_radm2: np.ndarray,
    lam_sq_0_m2: float,
    eps: float = 1e-6,
) -> np.ndarray:
    """Inverse RM-synthesis - FDF to Stokes Q and U in wavelength^2 space.

    Args:
        fdf_q_arr (StokesQArray): FDF Stokes Q data array
        fdf_u_arr (StokesUArray): FDF Stokes U data array
        lambda_sq_arr_m2 (np.ndarray): Wavelength^2 values in m^2
        phi_arr_radm2 (np.ndarray): Faraday depth values in rad/m^2
        lam_sq_0_m2 (float): Reference wavelength^2 value
        eps (float, optional): NUFFT tolerance. Defaults to 1e-6.

    Raises:
        ValueError: If the Stokes Q and U data arrays are not the same shape.
        ValueError: If the data dimensions are > 3.
        ValueError: If the data depth does not match the lambda^2 vector.

    Returns:
        np.ndarray: Complex polarisation array in wavelength^2 space
    """
    if not fdf_q_arr.shape == fdf_u_arr.shape:
        raise ValueError("Stokes Q and U data arrays must be the same shape.")

    n_dims = len(fdf_q_arr.shape)
    if not n_dims <= 3:
        raise ValueError(f"Data dimensions must be <= 3. Got {n_dims}")

    if not fdf_q_arr.shape[0] == phi_arr_radm2.shape[0]:
        raise ValueError(
            f"Data depth does not match Faraday depth vector ({fdf_q_arr.shape[0]} vs {phi_arr_radm2.shape[0]})."
        )

    # Reshape the data arrays to 2 dimensions
    if n_dims == 1:
        fdf_q_arr = np.reshape(fdf_q_arr, (fdf_q_arr.shape[0], 1))
        fdf_u_arr = np.reshape(fdf_u_arr, (fdf_u_arr.shape[0], 1))
    elif n_dims == 3:
        old_data_shape = fdf_q_arr.shape
        fdf_q_arr = np.reshape(
            fdf_q_arr,
            (
                fdf_q_arr.shape[0],
                fdf_q_arr.shape[1] * fdf_q_arr.shape[2],
            ),
        )
        fdf_u_arr = np.reshape(
            fdf_u_arr,
            (
                fdf_u_arr.shape[0],
                fdf_u_arr.shape[1] * fdf_u_arr.shape[2],
            ),
        )

    fdf_pol_cube = fdf_q_arr + 1j * fdf_u_arr
    exponent = (lambda_sq_arr_m2 - lam_sq_0_m2).astype(
        f"float{fdf_pol_cube.itemsize*8/2:.0f}"
    )
    pol_cube_inv = (
        finufft.nufft1d3(
            x=(phi_arr_radm2 * 2).astype(exponent.dtype),
            c=fdf_pol_cube.T.astype(complex),
            s=exponent,
            eps=eps,
            isign=1,
        )
    ).T

    # Restore if 3D shape
    if n_dims == 3:
        pol_cube_inv = np.reshape(
            pol_cube_inv,
            (pol_cube_inv.shape[0], old_data_shape[1], old_data_shape[2]),
        )

    # Remove redundant dimensions in the FDF array
    pol_cube_inv = np.squeeze(pol_cube_inv)

    return pol_cube_inv


def get_rmsf_nufft(
    lambda_sq_arr_m2: np.ndarray,
    phi_arr_radm2: np.ndarray,
    weight_arr: np.ndarray,
    lam_sq_0_m2: float,
    super_resolution: bool = False,
    mask_arr: Optional[np.ndarray] = None,
    do_fit_rmsf: bool = False,
    do_fit_rmsf_real=False,
    eps: float = 1e-6,
) -> RMSFResults:
    """Compute the RMSF for a given set of lambda^2 values.

    Args:
        lambda_sq_arr_m2 (np.ndarray): Wavelength^2 values in m^2
        phi_arr_radm2 (np.ndarray): Faraday depth values in rad/m^2
        weight_arr (np.ndarray): Weight array
        lam_sq_0_m2 (float): Reference wavelength^2 value
        super_resolution (bool, optional): Use superresolution. Defaults to False.
        mask_arr (Optional[np.ndarray], optional): Mask array. Defaults to None.
        do_fit_rmsf (bool, optional): Fit the RMSF with a Gaussian. Defaults to False.
        do_fit_rmsf_real (bool, optional): Fit the *real* part of the. Defaults to False.
        eps (float, optional): NUFFT tolerance. Defaults to 1e-6.

    Raises:
        ValueError: If the wavelength^2 and weight arrays are not the same shape.
        ValueError: If the mask dimensions are > 3.
        ValueError: If the mask depth does not match the lambda^2 vector.

    Returns:
        RMSFResults: rmsf_cube, phi_double_arr_radm2, fwhm_rmsf_arr, fit_status_arr
    """
    phi_double_arr_radm2 = make_double_phi_arr(phi_arr_radm2)

    weight_arr = np.nan_to_num(weight_arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Set the mask array (default to 1D, no masked channels)
    if mask_arr is None:
        mask_arr = np.zeros_like(lambda_sq_arr_m2, dtype=bool)
        n_dimension = 1
    else:
        mask_arr = mask_arr.astype(bool)
        n_dimension = len(mask_arr.shape)

    # Sanity checks on array sizes
    if not weight_arr.shape == lambda_sq_arr_m2.shape:
        raise ValueError("wavelength^2 and weight arrays must be the same shape.")

    if not n_dimension <= 3:
        raise ValueError("mask dimensions must be <= 3.")

    if not mask_arr.shape[0] == lambda_sq_arr_m2.shape[0]:
        raise ValueError(
            f"Mask depth does not match lambda^2 vector ({mask_arr.shape[0]} vs {lambda_sq_arr_m2.shape[-1]})."
        )

    # Reshape the mask array to 2 dimensions
    if n_dimension == 1:
        mask_arr = np.reshape(mask_arr, (mask_arr.shape[0], 1))
    elif n_dimension == 3:
        old_data_shape = mask_arr.shape
        mask_arr = np.reshape(
            mask_arr, (mask_arr.shape[0], mask_arr.shape[1] * mask_arr.shape[2])
        )
    num_pixels = mask_arr.shape[-1]

    # If full planes are flagged then set corresponding weights to zero
    flag_xy_sum = np.sum(mask_arr, axis=1)
    mskPlanes = np.where(flag_xy_sum == num_pixels, 0, 1)
    weight_arr *= mskPlanes

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
    weight_cube = np.invert(mask_arr) * weight_arr[:, np.newaxis]
    with np.errstate(divide="ignore", invalid="ignore"):
        scale_factor_arr = 1.0 / np.sum(weight_cube, axis=0)
        scale_factor_arr = np.nan_to_num(
            scale_factor_arr, nan=0.0, posinf=0.0, neginf=0.0
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
        * scale_factor_arr[..., None]
    ).T

    # Clean up one cube worth of memory
    del weight_cube

    # Default to the analytical RMSF
    fwhm_rmsf_arr = np.ones(num_pixels) * fwhm_rmsf_radm2
    fit_status_arr = np.zeros(num_pixels, dtype=bool)

    # Fit the RMSF main lobe
    if do_fit_rmsf:
        logger.info("Fitting main lobe in each RMSF spectrum.")
        logger.info("> This may take some time!")
        for i in trange(num_pixels, desc="Fitting RMSF by pixel"):
            try:
                fitted_rmsf = fit_rmsf(
                    rmsf_to_fit_arr=(
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
            fit_status_arr[i] = fit_status

    # Remove redundant dimensions
    rmsf_cube = np.squeeze(rmsf_cube)
    fwhm_rmsf_arr = np.squeeze(fwhm_rmsf_arr)
    fit_status_arr = np.squeeze(fit_status_arr)

    # Restore if 3D shape
    if n_dimension == 3:
        rmsf_cube = np.reshape(
            rmsf_cube, (rmsf_cube.shape[0], old_data_shape[1], old_data_shape[2])
        )
        fwhm_rmsf_arr = np.reshape(
            fwhm_rmsf_arr, (old_data_shape[1], old_data_shape[2])
        )
        fit_status_arr = np.reshape(
            fit_status_arr, (old_data_shape[1], old_data_shape[2])
        )

    return RMSFResults(
        rmsf_cube=rmsf_cube,
        phi_double_arr_radm2=phi_double_arr_radm2,
        fwhm_rmsf_arr=fwhm_rmsf_arr,
        fit_status_arr=fit_status_arr,
    )


def get_fdf_parameters(
    fdf_arr: np.ndarray,
    phi_arr_radm2: np.ndarray,
    fwhm_rmsf_radm2: float,
    freq_arr_hz: np.ndarray,
    stokes_q_arr: StokesQArray,
    stokes_u_arr: StokesUArray,
    stokes_q_error_arr: StokesQArray,
    stokes_u_error_arr: StokesUArray,
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

    abs_fdf_arr = np.abs(fdf_arr)
    peak_pi_index = np.nanargmax(abs_fdf_arr)

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
        np.concatenate([fdf_arr[mask].real, fdf_arr[mask].imag])
    )

    n_good_phi = np.isfinite(fdf_arr).sum()
    lambda_sq_arr_m2_variance = (
        np.sum(lambda_sq_arr_m2**2.0) - np.sum(lambda_sq_arr_m2) ** 2.0 / n_good_phi
    ) / (n_good_phi - 1)

    good_chan_idx = np.isfinite(freq_arr_hz)
    n_good_chan = good_chan_idx.sum()

    if not (peak_pi_index > 0 and peak_pi_index < len(abs_fdf_arr) - 1):
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
            min_freq_hz=freq_arr_hz[good_chan_idx].min(),
            max_freq_hz=freq_arr_hz[good_chan_idx].max(),
            n_channels=n_good_chan,
            median_d_freq_hz=np.nanmedian(np.diff(freq_arr_hz[good_chan_idx])),
            frac_pol=np.nan,
        )
    peak_pi_fit, peak_rm_fit, _ = fit_fdf(
        fdf_to_fit_arr=abs_fdf_arr,
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
    peak_u_fit = np.interp(peak_rm_fit, phi_arr_radm2, fdf_arr.imag)
    peak_q_fit = np.interp(peak_rm_fit, phi_arr_radm2, fdf_arr.real)
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
        freq_arr_hz=freq_arr_hz,
        stokes_q_arr=stokes_q_arr,
        stokes_u_arr=stokes_u_arr,
        stokes_q_err_arr=stokes_q_error_arr,
        stokes_u_err_arr=stokes_u_error_arr,
        frac_pol=peak_pi_fit_debias / stokes_i_reference_flux,
        psi0_deg=peak_pa0_fit_deg,
        rm_radm2=peak_rm_fit,
    )

    return FDFParameters(
        fdf_error_mad=fdf_error_mad,
        peak_pi_fit=peak_pi_fit,
        peak_pi_error=theoretical_noise.fdf_error_noise,
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
        min_freq_hz=freq_arr_hz[good_chan_idx].min(),
        max_freq_hz=freq_arr_hz[good_chan_idx].max(),
        n_channels=n_good_chan,
        median_d_freq_hz=np.nanmedian(np.diff(freq_arr_hz[good_chan_idx])),
        frac_pol=peak_pi_fit_debias / stokes_i_reference_flux,
    )


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
    y_arr: np.ndarray,
    dy_arr: np.ndarray,
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
        median = np.nanmedian(y_arr)
    if noise is None:
        noise = mad_std(y_arr)

    # Sample the PDF of the additional noise term from a limit near zero to
    # a limit of the range of the data, including error bars
    y_range = np.nanmax(y_arr + dy_arr) - np.nanmin(y_arr - dy_arr)
    sigma_add_arr = np.linspace(y_range / n_samples, y_range, n_samples)

    # Model deviation from Gaussian as an additional noise term.
    # Loop through the range of i additional noise samples and calculate
    # chi-squared and sum(ln(sigma_total)), used later to calculate likelihood.
    n_data = len(y_arr)

    # Calculate sigma_sq_tot for all sigma_add values
    sigma_sq_tot = dy_arr**2.0 + sigma_add_arr[:, None] ** 2.0

    # Calculate ln_sigma_sum_arr for all sigma_add values
    ln_sigma_sum_arr = np.nansum(np.log(np.sqrt(sigma_sq_tot)), axis=1)

    # Calculate chi_sq_arr for all sigma_add values
    chi_sq_arr = np.nansum((y_arr - median) ** 2.0 / sigma_sq_tot, axis=1)
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
        sigma_add_arr=sigma_add_arr,
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
        2j * (np.deg2rad(psi0_deg) + rm_radm2 * lambda_sq_arr_m2)
    )

    return complex_polarization


def measure_qu_complexity(
    freq_arr_hz: np.ndarray,
    stokes_q_arr: StokesQArray,
    stokes_u_arr: StokesUArray,
    stokes_q_err_arr: StokesQArray,
    stokes_u_err_arr: StokesUArray,
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
    stokes_q_residual = stokes_q_arr - complex_polarisation.real
    stokes_u_residual = stokes_u_arr - complex_polarisation.imag

    sigma_add_q = calculate_sigma_add(
        y_arr=stokes_q_residual / stokes_q_err_arr,
        dy_arr=np.ones_like(stokes_q_residual),
        median=0.0,
        noise=1.0,
    )
    sigma_add_u = calculate_sigma_add(
        y_arr=stokes_u_residual / stokes_u_err_arr,
        dy_arr=np.ones_like(stokes_u_residual),
        median=0.0,
        noise=1.0,
    )

    sigma_add_p_arr = np.hypot(sigma_add_q.sigma_add_arr, sigma_add_u.sigma_add_arr)
    sigma_add_p_pdf = np.hypot(sigma_add_q.sigma_add_pdf, sigma_add_u.sigma_add_pdf)
    sigma_add_p_cdf = np.cumsum(sigma_add_p_pdf) / np.nansum(sigma_add_p_pdf)
    sigma_add_p_val = cdf_percentile(
        values=sigma_add_p_arr, cdf=sigma_add_p_cdf, q=50.0
    )
    sigma_add_p_minus = cdf_percentile(
        values=sigma_add_p_arr, cdf=sigma_add_p_cdf, q=15.72
    )
    sigma_add_p_plus = cdf_percentile(
        values=sigma_add_p_arr, cdf=sigma_add_p_cdf, q=84.27
    )
    sigma_add_p = SigmaAdd(
        sigma_add=sigma_add_p_val,
        sigma_add_minus=sigma_add_p_minus,
        sigma_add_plus=sigma_add_p_plus,
        sigma_add_cdf=sigma_add_p_cdf,
        sigma_add_pdf=sigma_add_p_pdf,
        sigma_add_arr=sigma_add_p_arr,
    )

    return StokesSigmaAdd(
        sigma_add_q=sigma_add_q,
        sigma_add_u=sigma_add_u,
        sigma_add_p=sigma_add_p,
    )


def measure_fdf_complexity(phi_arr_radm2, FDF):
    # Second moment of clean component spectrum
    return calc_mom2_FDF(FDF, phi_arr_radm2)
