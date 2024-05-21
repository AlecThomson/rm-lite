#!/usr/bin/env python3
import warnings
from typing import Callable, Literal, NamedTuple, Optional, Tuple

import numpy as np
from astropy.stats import akaike_info_criterion_lsq
from scipy.optimize import curve_fit
from scipy.stats import norm, multivariate_normal
from uncertainties import unumpy

from rm_lite.utils.logging import logger

logger.setLevel("INFO")


class FitResult(NamedTuple):
    popt: np.ndarray
    pcov: np.ndarray
    stokes_i_model_func: Callable
    aic: float


def calc_mom2_FDF(FDF, phiArr):
    """
    Calculate the 2nd moment of the polarised intensity FDF. Can be applied to
    a clean component spectrum or a standard FDF
    """

    K = np.sum(np.abs(FDF))
    phiMean = np.sum(phiArr * np.abs(FDF)) / K
    phiMom2 = np.sqrt(np.sum(np.power((phiArr - phiMean), 2.0) * np.abs(FDF)) / K)

    return phiMom2


def calc_parabola_vertex(x1, y1, x2, y2, x3, y3):
    """
    Calculate the vertex of a parabola given three adjacent points.
    Normalization of coordinates must be performed first to reduce risk of
    floating point errors.
    """
    midpoint = x2
    deltax = x2 - x3
    yscale = y2
    (x1, x2, x3) = [(x - x2) / deltax for x in (x1, x2, x3)]  # slide spectrum to zero
    (y1, y2, y3) = [y / yscale for y in (y1, y2, y3)]

    D = (x1 - x2) * (x1 - x3) * (x2 - x3)
    A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / D
    B = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / D
    C = (
        x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3
    ) / D

    xv = -B / (2.0 * A)
    yv = C - B * B / (4.0 * A)

    return xv * deltax + midpoint, yv * yscale


def renormalize_StokesI_model(
    fit_result: FitResult, new_reference_frequency: float
) -> FitResult:
    """Adjust the reference frequency for the Stokes I model and fix the fit
    parameters such that the the model is the same.

    This is important because the initial Stokes I fitted model uses an arbitrary
    reference frequency, and it may be desirable for users to know the exact
    reference frequency of the model.

    This function now includes the ability to transform the model parameter
    errors to the new reference frequency. This feature uses a first order
    approximation, that scales with the ratio of new to old reference frequencies.
    Large changes in reference frequency may be outside the linear valid regime
    of the first order approximation, and thus should be avoided.

    Args:
        fit_result (FitResult): the result of a Stokes I model fit.
        new_reference_frequency (float): the new reference frequency for the model.

    Returns:
        FitResult: the fit results with the reference frequency adjusted to the new value.


    """
    # Renormalization ratio:
    x = new_reference_frequency / fit_result.reference_frequency_Hz

    # Check if ratio is within zone of probable accuracy (approx. 10%, from empirical tests)
    if (x < 0.9) or (x > 1.1):
        warnings.warn(
            "New Stokes I reference frequency more than 10% different than original, uncertainties may be unreliable",
            UserWarning,
        )

    (a, b, c, d, f, g) = fit_result.params

    # Modify fit parameters to new reference frequency.
    # I have derived all these conversion equations analytically for the
    # linear- and log-polynomial models.
    if fit_result.fit_function == "linear":
        new_parms = [a * x**5, b * x**4, c * x**3, d * x**2, f * x, g]
    elif fit_result.fit_function == "log":
        lnx = np.log10(x)
        new_parms = [
            a,
            5 * a * lnx + b,
            10 * a * lnx**2 + 4 * b * lnx + c,
            10 * a * lnx**3 + 6 * b * lnx**2 + 3 * c * lnx + d,
            5 * a * lnx**4 + 4 * b * lnx**3 + 3 * c * lnx**2 + 2 * d * lnx + f,
            g
            * np.power(10, a * lnx**5 + b * lnx**4 + c * lnx**3 + d * lnx**2 + f * lnx),
        ]

    # Modify fit parameter errors to new reference frequency.
    # Note this implicitly makes a first-order approximation in the correletion
    # structure between uncertainties
    # The general equation for the transformation of uncertainties is:
    #   var(p) = sum_i,j((\partial p / \partial a_i) * (\partial p / \partial a_j) * cov(a_i,a_j))
    # where a_i are the initial parameters, p is a final parameter,
    # and the partial derivatives are evaluated at the fit parameter values (and frequency ratio).
    # The partial derivatives all come from the parameter conversion equations above.

    cov = fit_result.pcov
    if fit_result.fit_function == "linear":
        new_errors = [
            np.sqrt(x**10 * cov[0, 0]),
            np.sqrt(x**8 * cov[1, 1]),
            np.sqrt(x**6 * cov[2, 2]),
            np.sqrt(x**4 * cov[3, 3]),
            np.sqrt(x**2 * cov[4, 4]),
            np.sqrt(cov[5, 5]),
        ]
    elif fit_result.fit_function == "log":
        g2 = new_parms[5]  # Convenient shorthand for new value of g variable.
        new_errors = [
            np.sqrt(cov[0, 0]),
            np.sqrt(25 * lnx**2 * cov[0, 0] + 10 * lnx * cov[0, 1] + cov[1, 1]),
            np.sqrt(
                100 * lnx**4 * cov[0, 0]
                + 80 * lnx**3 * cov[0, 1]
                + 20 * lnx**2 * cov[0, 2]
                + 16 * lnx**2 * cov[1, 1]
                + 8 * lnx * cov[1, 2]
                + cov[2, 2]
            ),
            np.sqrt(
                100 * lnx**6 * cov[0, 0]
                + 120 * lnx**5 * cov[0, 1]
                + 60 * lnx**4 * cov[0, 2]
                + 20 * lnx**3 * cov[0, 3]
                + 36 * lnx**4 * cov[1, 1]
                + 36 * lnx**3 * cov[1, 2]
                + 12 * lnx**2 * cov[1, 3]
                + 9 * lnx**2 * cov[2, 2]
                + 6 * lnx * cov[2, 3]
                + cov[3, 3]
            ),
            np.sqrt(
                25 * lnx**8 * cov[0, 0]
                + 40 * lnx**7 * cov[0, 1]
                + 30 * lnx**6 * cov[0, 2]
                + 20 * lnx**5 * cov[0, 3]
                + 10 * lnx**4 * cov[0, 4]
                + 16 * lnx**6 * cov[0, 5]
                + 24 * lnx**5 * cov[1, 2]
                + 16 * lnx**4 * cov[1, 3]
                + 8 * lnx**3 * cov[1, 4]
                + 9 * lnx**4 * cov[2, 2]
                + 12 * lnx**3 * cov[2, 3]
                + 6 * lnx**2 * cov[2, 4]
                + 4 * lnx**2 * cov[3, 3]
                + 4 * lnx * cov[3, 4]
                + cov[4, 4]
            ),
            np.abs(g2)
            * np.sqrt(
                lnx**10 * cov[0, 0]
                + 2 * lnx**9 * cov[0, 1]
                + 2 * lnx**8 * cov[0, 2]
                + 2 * lnx**7 * cov[0, 3]
                + 2 * lnx**6 * cov[0, 4]
                + 2 * lnx**5 / g * np.log(10) * cov[0, 5]
                + lnx**8 * cov[1, 1]
                + 2 * lnx**7 * cov[1, 2]
                + 2 * lnx**6 * cov[1, 3]
                + 2 * lnx**5 * cov[1, 4]
                + 2 * lnx**4 / g * np.log(10) * cov[1, 5]
                + lnx**6 * cov[2, 2]
                + 2 * lnx**5 * cov[2, 3]
                + 2 * lnx**4 * cov[2, 4]
                + 2 * lnx**3 / g * np.log(10) * cov[2, 5]
                + lnx**4 * cov[3, 3]
                + 2 * lnx**3 * cov[3, 4]
                + 2 * lnx**2 / g * np.log(10) * cov[3, 5]
                + lnx**2 * cov[4, 4]
                + 2 * lnx / g * np.log(10) * cov[4, 5]
                + 1 / g**2 * cov[5, 5]
            ),
        ]

    return fit_result.with_options(
        params=new_parms,
        reference_frequency_Hz=new_reference_frequency,
        perror=new_errors,
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


class FractionalSpectra(NamedTuple):
    stokes_i_model_array: np.ndarray
    stokes_q_frac_array: np.ndarray
    stokes_u_frac_array: np.ndarray
    stokes_q_frac_error_array: np.ndarray
    stokes_u_frac_error_array: np.ndarray


def create_fractional_spectra(
    freq_array_hz: np.ndarray,
    ref_freq_hz: float,
    stokes_i_array: np.ndarray,
    stokes_q_array: np.ndarray,
    stokes_u_array: np.ndarray,
    stokes_i_error_array: np.ndarray,
    stokes_q_error_array: np.ndarray,
    stokes_u_error_array: np.ndarray,
    fit_order: int = 2,
    fit_function: Literal["log", "linear"] = "log",
    stokes_i_model_array: Optional[np.ndarray] = None,
    stokes_i_model_error: Optional[np.ndarray] = None,
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

    # stokes_i_uarray = unumpy.uarray(stokes_i_array, stokes_i_error_array)
    stokes_q_uarray = unumpy.uarray(stokes_q_array, stokes_q_error_array)
    stokes_u_uarray = unumpy.uarray(stokes_u_array, stokes_u_error_array)
    if stokes_i_model_array is not None:
        if stokes_i_model_error is None:
            raise ValueError(
                "If `stokes_i_model_array` is provided, `stokes_i_model_error` must also be provided."
            )
        stokes_i_model_uarray = unumpy.uarray(
            stokes_i_model_array, stokes_i_model_error
        )
    else:
        popt, pcov, stokes_i_model_func, aic = fit_stokes_i_model(
            freq_array_hz,
            ref_freq_hz,
            stokes_i_array,
            stokes_i_error_array,
            fit_order,
            fit_function,
        )
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
        stokes_i_model_uarray = unumpy.uarray(
            stokes_i_model_array,
            np.abs((stokes_i_model_high - stokes_i_model_low)),
        )

    stokes_q_frac_uarray = stokes_q_uarray / stokes_i_model_uarray
    stokes_u_frac_uarray = stokes_u_uarray / stokes_i_model_uarray

    stokes_q_frac_array = unumpy.nominal_values(stokes_q_frac_uarray)
    stokes_u_frac_array = unumpy.nominal_values(stokes_u_frac_uarray)
    stokes_q_frac_error_array = unumpy.std_devs(stokes_q_frac_uarray)
    stokes_u_frac_error_array = unumpy.std_devs(stokes_u_frac_uarray)

    return FractionalSpectra(
        stokes_i_model_array=stokes_i_model_array,
        stokes_q_frac_array=stokes_q_frac_array,
        stokes_u_frac_array=stokes_u_frac_array,
        stokes_q_frac_error_array=stokes_q_frac_error_array,
        stokes_u_frac_error_array=stokes_u_frac_error_array,
    )


def create_pqu_spectra_burn(
    freqArr_Hz, fracPolArr, psi0Arr_deg, RMArr_radm2, sigmaRMArr_radm2=None
):
    """Return fractional P/I, Q/I & U/I spectra for a sum of Faraday thin
    components (multiple values may be given as a list for each argument).
    Burn-law external depolarisation may be applied to each
    component via the optional 'sigmaRMArr_radm2' argument. If
    sigmaRMArr_radm2=None, all values are set to zero, i.e., no
    depolarisation."""

    # Convert lists to arrays
    freqArr_Hz = np.array(freqArr_Hz, dtype="f8")
    fracPolArr = np.array(fracPolArr, dtype="f8")
    psi0Arr_deg = np.array(psi0Arr_deg, dtype="f8")
    RMArr_radm2 = np.array(RMArr_radm2, dtype="f8")
    if sigmaRMArr_radm2 is None:
        sigmaRMArr_radm2 = np.zeros_like(fracPolArr)
    else:
        sigmaRMArr_radm2 = np.array(sigmaRMArr_radm2, dtype="f8")

    # Calculate some prerequsites
    nChans = len(freqArr_Hz)
    nComps = len(fracPolArr)
    lamArr_m = C / freqArr_Hz
    lamSqArr_m2 = np.power(lamArr_m, 2.0)

    # Convert the inputs to column vectors
    fracPolArr = fracPolArr.reshape((nComps, 1))
    psi0Arr_deg = psi0Arr_deg.reshape((nComps, 1))
    RMArr_radm2 = RMArr_radm2.reshape((nComps, 1))
    sigmaRMArr_radm2 = sigmaRMArr_radm2.reshape((nComps, 1))

    # Calculate the p, q and u Spectra for all components
    pArr = fracPolArr * np.ones((nComps, nChans), dtype="f8")
    quArr = pArr * (
        np.exp(2j * (np.radians(psi0Arr_deg) + RMArr_radm2 * lamSqArr_m2))
        * np.exp(-2.0 * sigmaRMArr_radm2 * np.power(lamArr_m, 4.0))
    )

    # Sum along the component axis to create the final spectra
    quArr = quArr.sum(0)
    qArr = quArr.real
    uArr = quArr.imag
    pArr = np.abs(quArr)

    return pArr, qArr, uArr


def norm_cdf(mean=0.0, std=1.0, N=50, xArr=None):
    """Return the CDF of a normal distribution between -6 and 6 sigma, or at
    the values of an input array."""

    if xArr is None:
        x = np.linspace(-6.0 * std, 6.0 * std, N)
    else:
        x = xArr
    y = norm.cdf(x, loc=mean, scale=std)

    return x, y
