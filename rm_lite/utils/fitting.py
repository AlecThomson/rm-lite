from __future__ import annotations

from typing import Literal, NamedTuple, Protocol

import numpy as np
from astropy.modeling.models import Gaussian1D
from astropy.stats import akaike_info_criterion_lsq
from numpy.typing import ArrayLike, NDArray
from scipy import optimize

from rm_lite.utils.logging import logger

logger.setLevel("INFO")

GAUSSIAN_SIGMA_TO_FWHM = float(2.0 * np.sqrt(2.0 * np.log(2.0)))


class StokesIModel(Protocol):
    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...


class FitResult(NamedTuple):
    """Results of a Stokes I fit"""

    popt: ArrayLike
    """Best fit parameters"""
    pcov: ArrayLike
    """Covariance matrix of the fit"""
    stokes_i_model_func: StokesIModel
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


def fwhm_to_sigma(fwhm: float) -> float:
    return float(fwhm / GAUSSIAN_SIGMA_TO_FWHM)


def sigma_to_fwhm(sigma: float) -> float:
    return float(sigma * GAUSSIAN_SIGMA_TO_FWHM)


def gaussian_integrand(
    amplitude: float,
    stddev: float | None = None,
    fwhm: float | None = None,
) -> float:
    if stddev is None and fwhm is None:
        msg = "Must provide either stddev or fwhm."
        raise ValueError(msg)
    if stddev is None and fwhm is not None:
        stddev = fwhm_to_sigma(fwhm)
    if stddev is None:
        msg = "stddev cannot be None"
        raise ValueError(msg)
    return float(amplitude * stddev * np.sqrt(2 * np.pi))


def gaussian(
    x: NDArray[np.float64],
    amplitude: float | complex,
    mean: float,
    stddev: float | None = None,
    fwhm: float | None = None,
) -> NDArray[np.float64]:
    if stddev is None and fwhm is None:
        msg = "Must provide either stddev or fwhm."
        raise ValueError(msg)
    if stddev is None and fwhm is not None:
        stddev = fwhm_to_sigma(fwhm)
    if isinstance(amplitude, complex):
        return Gaussian1D(amplitude=amplitude.real, mean=mean, stddev=stddev)(
            x
        ) + 1j * Gaussian1D(amplitude=amplitude.imag, mean=mean, stddev=stddev)(x)
    return Gaussian1D(amplitude=amplitude, mean=mean, stddev=stddev)(x)


def unit_gaussian(
    x: NDArray[np.float64],
    mean: float,
    stddev: float | None = None,
    fwhm: float | None = None,
) -> NDArray[np.float64]:
    if stddev is None and fwhm is None:
        msg = "Must provide either stddev or fwhm."
        raise ValueError(msg)
    if stddev is None and fwhm is not None:
        stddev = fwhm_to_sigma(fwhm)
    return Gaussian1D(amplitude=1, mean=mean, stddev=stddev)(x)


def unit_centred_gaussian(
    x: NDArray[np.float64], stddev: float | None = None, fwhm: float | None = None
) -> NDArray[np.float64]:
    if stddev is None and fwhm is None:
        msg = "Must provide either stddev or fwhm."
        raise ValueError(msg)
    if stddev is None and fwhm is not None:
        stddev = fwhm_to_sigma(fwhm)
    return Gaussian1D(amplitude=1, mean=0, stddev=stddev)(x)


def fit_rmsf(
    rmsf_to_fit_arr: NDArray[np.float64],
    phi_double_arr_radm2: NDArray[np.float64],
    fwhm_rmsf_radm2: float,
) -> float:
    rmsf_to_fit_arr = rmsf_to_fit_arr.copy()
    rmsf_to_fit_arr /= np.nanmax(rmsf_to_fit_arr)
    d_phi = phi_double_arr_radm2[1] - phi_double_arr_radm2[0]
    mask = np.zeros_like(phi_double_arr_radm2, dtype=bool)
    mask[np.argmax(rmsf_to_fit_arr)] = True
    sigma_rmsf_radm2 = fwhm_to_sigma(fwhm_rmsf_radm2)
    sigma_rmsf_arr_pix = sigma_rmsf_radm2 / d_phi
    for i in np.where(mask)[0]:
        start = int(i - sigma_rmsf_arr_pix / 2)
        end = int(i + sigma_rmsf_arr_pix / 2)
        mask[start : end + 2] = True
    popt, pcov = optimize.curve_fit(
        unit_centred_gaussian,
        phi_double_arr_radm2[mask],
        rmsf_to_fit_arr[mask],
        p0=[sigma_rmsf_radm2],
        bounds=([0], [np.inf]),
    )
    return sigma_to_fwhm(popt[0])


def fit_fdf(
    fdf_to_fit_arr: NDArray[np.float64],
    phi_arr_radm2: NDArray[np.float64],
    fwhm_fdf_radm2: float,
) -> FDFFitResult:
    d_phi = phi_arr_radm2[1] - phi_arr_radm2[0]
    mask = np.zeros_like(phi_arr_radm2, dtype=bool)
    mask[np.argmax(fdf_to_fit_arr)] = 1
    fwhm_fdf_arr_pix = fwhm_fdf_radm2 / d_phi
    for i in np.where(mask)[0]:
        start = int(i - fwhm_fdf_arr_pix / 2)
        end = int(i + fwhm_fdf_arr_pix / 2)
        mask[start : end + 2] = True

    amplitude_guess = np.nanmax(fdf_to_fit_arr[mask])
    mean_guess = phi_arr_radm2[mask][np.argmax(fdf_to_fit_arr[mask])]
    stddev_guess = fwhm_fdf_radm2 / (2 * np.sqrt(2 * np.log(2)))
    popt, pcov = optimize.curve_fit(
        gaussian,
        phi_arr_radm2[mask],
        fdf_to_fit_arr[mask],
        p0=[amplitude_guess, mean_guess, stddev_guess],
    )
    logger.debug(f"Fit results: {popt}")
    amplitude_fit, mean_fit, stddev_fit = popt
    return FDFFitResult(
        amplitude_fit=amplitude_fit,
        mean_fit=mean_fit,
        stddev_fit=stddev_fit,
    )


def polynomial(order: int) -> StokesIModel:
    def poly_func(x: NDArray[np.float64], *params) -> NDArray[np.float64]:
        if len(params) != order + 1:
            msg = f"Polynomial function of order {order} requires {order + 1} parameters, {len(params)} given."
            raise ValueError(msg)
        result = 0
        for i in range(order + 1):
            result += params[i] * x**i
        return result

    return poly_func


def power_law(order: int) -> StokesIModel:
    def power_func(x: NDArray[np.float64], *params) -> NDArray[np.float64]:
        if len(params) != order + 1:
            msg = f"Power law function of order {order} requires {order + 1} parameters, {len(params)} given."
            raise ValueError(msg)
        power = 0
        for i in range(1, order + 1):
            power += params[i] * np.log10(x) ** i
        return params[0] * 10**power

    return power_func


def best_aic_func(
    aics: NDArray[np.float64], n_param: NDArray[np.int32]
) -> tuple[float, int, int]:
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
    freq_arr_hz: NDArray[np.float64],
    ref_freq_hz: float,
    stokes_i_arr: NDArray[np.float64],
    stokes_i_error_arr: NDArray[np.float64],
    fit_order: int = 2,
    fit_type: Literal["log", "linear"] = "log",
) -> FitResult:
    if fit_type == "linear":
        fit_func = polynomial(fit_order)
    elif fit_type == "log":
        fit_func = power_law(fit_order)
    else:
        msg = f"Unknown fit type {fit_type} provided. Must be 'log' or 'linear'."  # type: ignore[unreachable]
        raise ValueError(msg)

    logger.debug(f"Fitting Stokes I model with {fit_type} model of order {fit_order}.")
    initital_guess = np.zeros(fit_order + 1)
    initital_guess[0] = np.nanmean(stokes_i_arr)
    bounds = (
        [-np.inf] * (fit_order + 1),
        [np.inf] * (fit_order + 1),
    )
    bounds[0][0] = 0.0
    if (stokes_i_error_arr == 0).all():
        stokes_i_error_arr = None
    popt, pcov = optimize.curve_fit(
        fit_func,
        freq_arr_hz / ref_freq_hz,
        stokes_i_arr,
        sigma=stokes_i_error_arr,
        absolute_sigma=True,
        p0=initital_guess,
        bounds=bounds,
    )
    stokes_i_model_arr = fit_func(freq_arr_hz / ref_freq_hz, *popt)
    ssr = np.sum((stokes_i_arr - stokes_i_model_arr) ** 2)
    with np.errstate(divide="ignore"):
        aic = akaike_info_criterion_lsq(
            ssr=ssr, n_params=fit_order + 1, n_samples=len(freq_arr_hz)
        )

    return FitResult(
        popt=popt,
        pcov=pcov,
        stokes_i_model_func=fit_func,
        aic=aic,
    )


def dynamic_fit(
    freq_arr_hz: NDArray[np.float64],
    ref_freq_hz: float,
    stokes_i_arr: NDArray[np.float64],
    stokes_i_error_arr: NDArray[np.float64],
    fit_order: int = 2,
    fit_type: Literal["log", "linear"] = "log",
) -> FitResult:
    orders = np.arange(fit_order + 1)
    n_parameters = orders + 1
    fit_results: list[FitResult] = []

    for _i, order in enumerate(orders):
        fit_result = static_fit(
            freq_arr_hz,
            ref_freq_hz,
            stokes_i_arr,
            stokes_i_error_arr,
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
    freq_arr_hz: NDArray[np.float64],
    ref_freq_hz: float,
    stokes_i_arr: NDArray[np.float64],
    stokes_i_error_arr: NDArray[np.float64],
    fit_order: int = 2,
    fit_type: Literal["log", "linear"] = "log",
) -> FitResult:
    if fit_order < 0:
        return dynamic_fit(
            freq_arr_hz,
            ref_freq_hz,
            stokes_i_arr,
            stokes_i_error_arr,
            abs(fit_order),
            fit_type,
        )

    return static_fit(
        freq_arr_hz,
        ref_freq_hz,
        stokes_i_arr,
        stokes_i_error_arr,
        fit_order,
        fit_type,
    )
