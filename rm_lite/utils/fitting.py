#!/usr/bin/env python3
from typing import Callable, Literal, NamedTuple, Tuple

import numpy as np
from astropy.modeling.models import Gaussian1D
from scipy.optimize import curve_fit

from rm_lite.utils.logging import logger

logger.setLevel("INFO")


class FitResult(NamedTuple):
    """Results of a Stokes I fit"""

    popt: np.ndarray
    """Best fit parameters"""
    pcov: np.ndarray
    """Covariance matrix of the fit"""
    stokes_i_model_func: Callable
    """Function of the best fit model"""
    aic: float
    """Akaike Information Criterion of the fit"""


class FDFFitResult(NamedTuple):
    """Results of a Gaussian FDF fit"""

    amplitude_fit: float
    """Amplitude of the best fit model"""
    mean_fit: float
    """Mean (Faraday depth) of the best fit model"""
    stddev_fit: float
    """Standard deviation (Faraday depth) of the best fit model"""


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
    popt, pcov = curve_fit(
        unit_centred_gaussian,
        phi_double_arr_radm2[mask],
        rmsf_to_fit_array[mask],
        p0=[fwhm_rmsf_radm2 / (2 * np.sqrt(2 * np.log(2)))],
    )
    return popt[0]


def fit_fdf(
    fdf_to_fit_array: np.ndarray,
    phi_arr_radm2: np.ndarray,
    fwhm_fdf_radm2: float,
) -> FDFFitResult:
    d_phi = phi_arr_radm2[1] - phi_arr_radm2[0]
    mask = np.zeros_like(phi_arr_radm2, dtype=bool)
    mask[np.argmax(fdf_to_fit_array)] = 1
    fwhm_fdf_arr_pix = fwhm_fdf_radm2 / d_phi
    for i in np.where(mask)[0]:
        start = int(i - fwhm_fdf_arr_pix / 2)
        end = int(i + fwhm_fdf_arr_pix / 2)
        mask[start : end + 2] = True

    amplitude_guess = np.nanmax(fdf_to_fit_array[mask])
    mean_guess = phi_arr_radm2[np.argmax(fdf_to_fit_array[mask])]
    stddev_guess = fwhm_fdf_radm2 / (2 * np.sqrt(2 * np.log(2)))
    popt, pcov = curve_fit(
        gaussian,
        phi_arr_radm2[mask],
        fdf_to_fit_array[mask],
        p0=[amplitude_guess, mean_guess, stddev_guess],
    )
    amplitude_fit, mean_fit, stddev_fit = popt
    return FDFFitResult(
        amplitude_fit=amplitude_fit,
        mean_fit=mean_fit,
        stddev_fit=stddev_fit,
    )


def polynomial(order: int) -> Callable:
    def poly_func(x: np.ndarray, *params) -> np.ndarray:
        if len(params) != order + 1:
            raise ValueError(
                f"Polynomial function of order {order} requires {order + 1} parameters, {len(params)} given."
            )
        result = 0
        for i in range(order + 1):
            result += params[i] * x**i
        return result

    return poly_func


def power_law(order: int) -> Callable:
    def power_func(x: np.ndarray, *params) -> np.ndarray:
        if len(params) != order + 1:
            raise ValueError(
                f"Power law function of order {order} requires {order + 1} parameters, {len(params)} given."
            )
        power = 0
        for i in range(1, order + 1):
            power += params[i] * np.log10(x) ** i
        return params[0] * 10**power

    return power_func


def best_aic_func(aics: np.ndarray, n_param: np.ndarray) -> Tuple[float, int, int]:
    """Find the best AIC for a set of AICs using Occam's razor."""
    # Find the best AIC
    best_aic_idx = int(np.nanargmin(aics))
    best_aic = float(aics[best_aic_idx])
    best_n = int(n_param[best_aic_idx])
    logger.debug(f"Lowest AIC is {best_aic}, with {best_n} params.")
    # Check if lower have diff < 2 in AIC
    aic_abs_diff = np.abs(aics - best_aic)
    bool_min_idx = np.zeros_like(aics).astype(bool)
    bool_min_idx[best_aic_idx] = True
    potential_idx = (aic_abs_diff[~bool_min_idx] < 2) & (
        n_param[~bool_min_idx] < best_n
    )
    if not any(potential_idx):
        return best_aic, best_n, best_aic_idx

    bestest_n = int(np.min(n_param[~bool_min_idx][potential_idx]))
    bestest_aic_idx = int(np.where(n_param == bestest_n)[0][0])
    bestest_aic = float(aics[bestest_aic_idx])
    logger.debug(
        f"Model within 2 of lowest AIC found. Occam says to take AIC of {bestest_aic}, with {bestest_n} params."
    )
    return bestest_aic, bestest_n, bestest_aic_idx


def static_fit(
    freq_array_hz: np.ndarray,
    ref_freq_hz: float,
    stokes_i_array: np.ndarray,
    stokes_i_error_array: np.ndarray,
    fit_order: int = 2,
    fit_type: Literal["log", "linear"] = "log",
) -> FitResult:
    if fit_type == "linear":
        fit_func = polynomial(fit_order)
    elif fit_type == "log":
        fit_func = power_law(fit_order)
    else:
        raise ValueError(
            f"Unknown fit type {fit_type} provided. Must be 'log' or 'linear'."
        )

    logger.debug(f"Fitting Stokes I model with {fit_type} model of order {fit_order}.")
    initital_guess = np.zeros(fit_order + 1)
    initital_guess[0] = np.nanmean(stokes_i_array)
    bounds = (
        [-np.inf] * (fit_order + 1),
        [np.inf] * (fit_order + 1),
    )
    bounds[0][0] = 0.0
    popt, pcov = curve_fit(
        fit_func,
        freq_array_hz / ref_freq_hz,
        stokes_i_array,
        sigma=stokes_i_error_array,
        absolute_sigma=True,
        p0=initital_guess,
        bounds=bounds,
    )
    stokes_i_model_array = fit_func(freq_array_hz / ref_freq_hz, *popt)
    ssr = np.sum((stokes_i_array - stokes_i_model_array) ** 2)
    aic = akaike_info_criterion_lsq(
        ssr=ssr, n_params=fit_order + 1, n_samples=len(freq_array_hz)
    )

    return FitResult(
        popt=popt,
        pcov=pcov,
        stokes_i_model_func=fit_func,
        aic=aic,
    )


def dynamic_fit(
    freq_array_hz: np.ndarray,
    ref_freq_hz: float,
    stokes_i_array: np.ndarray,
    stokes_i_error_array: np.ndarray,
    fit_order: int = 2,
    fit_type: Literal["log", "linear"] = "log",
) -> FitResult:
    orders = np.arange(fit_order + 1)
    n_parameters = orders + 1
    fit_results = []

    for i, order in enumerate(orders):
        fit_result = static_fit(
            freq_array_hz,
            ref_freq_hz,
            stokes_i_array,
            stokes_i_error_array,
            order,
            fit_type,
        )
        fit_results.append(fit_result)

    logger.debug(f"Fit results for orders {orders}:")
    aics = np.array([fit_result.aic for fit_result in fit_results])
    bestest_aic, bestest_n, bestest_aic_idx = best_aic_func(aics, n_parameters)
    logger.debug(f"Best fit found with {bestest_n} parameters.")
    logger.debug(f"Best fit found with AIC {bestest_aic}.")
    logger.debug(f"Best fit found at index {bestest_aic_idx}.")
    logger.debug(f"Best fit found with order {orders[bestest_aic_idx]}.")

    return fit_results[bestest_aic_idx]


def fit_stokes_i_model(
    freq_array_hz: np.ndarray,
    ref_freq_hz: float,
    stokes_i_array: np.ndarray,
    stokes_i_error_array: np.ndarray,
    fit_order: int = 2,
    fit_type: Literal["log", "linear"] = "log",
) -> FitResult:
    if fit_order < 0:
        return dynamic_fit(
            freq_array_hz,
            ref_freq_hz,
            stokes_i_array,
            stokes_i_error_array,
            abs(fit_order),
            fit_type,
        )

    return static_fit(
        freq_array_hz,
        ref_freq_hz,
        stokes_i_array,
        stokes_i_error_array,
        fit_order,
        fit_type,
    )
