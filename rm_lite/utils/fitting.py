from __future__ import annotations

from typing import Any, Literal, NamedTuple, Protocol

import numpy as np
import sigfig as sf
from astropy.modeling.models import Gaussian1D
from astropy.stats import akaike_info_criterion_lsq
from numpy.typing import ArrayLike, NDArray
from scipy import optimize, stats

from rm_lite.utils.logging import logger

GAUSSIAN_SIGMA_TO_FWHM = float(2.0 * np.sqrt(2.0 * np.log(2.0)))


class StokesIModel(Protocol):
    def __call__(
        self, x: NDArray[np.float64], *params: float
    ) -> NDArray[np.float64]: ...


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
    mean: float | NDArray[np.float64],
    stddev: float | None = None,
    fwhm: float | None = None,
) -> NDArray[np.float64]:
    if stddev is None and fwhm is None:
        msg = "Must provide either stddev or fwhm."
        raise ValueError(msg)
    if stddev is None and fwhm is not None:
        stddev = fwhm_to_sigma(fwhm)
    if isinstance(amplitude, complex):
        return np.array(
            Gaussian1D(amplitude=amplitude.real, mean=mean, stddev=stddev)(x)
            + 1j * Gaussian1D(amplitude=amplitude.imag, mean=mean, stddev=stddev)(x)
        )
    return np.array(Gaussian1D(amplitude=amplitude, mean=mean, stddev=stddev)(x))


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
    return np.array(Gaussian1D(amplitude=1, mean=mean, stddev=stddev)(x))


def unit_centred_gaussian(
    x: NDArray[np.float64], stddev: float | None = None, fwhm: float | None = None
) -> NDArray[np.float64]:
    if stddev is None and fwhm is None:
        msg = "Must provide either stddev or fwhm."
        raise ValueError(msg)
    if stddev is None and fwhm is not None:
        stddev = fwhm_to_sigma(fwhm)
    return np.array(Gaussian1D(amplitude=1, mean=0, stddev=stddev)(x))


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
    popt, _ = optimize.curve_fit(
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
    fwhm_fdf_arr_pix /= 2  # fit within half the FWHM
    for i in np.where(mask)[0]:
        start = int(i - fwhm_fdf_arr_pix / 2)
        end = int(i + fwhm_fdf_arr_pix / 2)
        mask[start : end + 2] = True

    amplitude_guess = float(np.nanmax(fdf_to_fit_arr[mask]))
    mean_guess = float(phi_arr_radm2[mask][np.argmax(fdf_to_fit_arr[mask])])
    stddev_guess = float(fwhm_fdf_radm2 / (2 * np.sqrt(2 * np.log(2))))
    if mask.sum() > 1:
        popt, _ = optimize.curve_fit(
            gaussian,
            phi_arr_radm2[mask],
            fdf_to_fit_arr[mask],
            p0=[amplitude_guess, mean_guess, stddev_guess],
        )
        logger.debug(f"Fit results: {popt}")
        amplitude_fit, mean_fit, stddev_fit = popt
    else:
        msg = "Can't fit single data point - just returning peak"
        logger.warning(msg)
        amplitude_fit, mean_fit, stddev_fit = amplitude_guess, mean_guess, stddev_guess
    return FDFFitResult(
        amplitude_fit=amplitude_fit,
        mean_fit=mean_fit,
        stddev_fit=stddev_fit,
    )


def polynomial(order: int) -> StokesIModel:
    def poly_func(x: NDArray[np.float64], *params: float) -> NDArray[np.float64]:
        if len(params) != order + 1:
            msg = f"Polynomial function of order {order} requires {order + 1} parameters, {len(params)} given."
            raise ValueError(msg)
        result = np.zeros_like(x)
        for i in range(order + 1):
            result = result + params[i] * x**i
        return result

    return poly_func


def power_law(order: int) -> StokesIModel:
    def power_func(x: NDArray[np.float64], *params: float) -> NDArray[np.float64]:
        if len(params) != order + 1:
            msg = f"Power law function of order {order} requires {order + 1} parameters, {len(params)} given."
            raise ValueError(msg)
        power = np.zeros_like(x)
        for i in range(1, order + 1):
            power = power + params[i] * np.log10(x) ** i
        return np.asarray(params[0] * 10**power)

    return power_func


def best_aic_func(
    aics: NDArray[np.float64], n_param: NDArray[np.integer[Any]]
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
    msg = f"Fitting Stokes I model of type {fit_type} with order {fit_order}."
    logger.info(msg)
    if fit_type == "linear":
        fit_func = polynomial(fit_order)
    elif fit_type == "log":
        fit_func = power_law(fit_order)
    else:
        msg = f"Unknown fit type {fit_type} provided. Must be 'log' or 'linear'."  # type: ignore[unreachable]
        raise ValueError(msg)

    logger.debug(f"Fitting Stokes I model with {fit_type} model of order {fit_order}.")
    initial_guess = np.zeros(fit_order + 1)
    mean_spectrum = float(np.nanmean(stokes_i_arr))
    # Use 0 if errors are large and spectrum ends up negative
    mean_spectrum = max(mean_spectrum, 0.0)
    initial_guess[0] = mean_spectrum
    bounds = (
        [-np.inf] * (fit_order + 1),
        [np.inf] * (fit_order + 1),
    )
    bounds[0][0] = 0.0
    sigma_arr: NDArray[np.float64] | None = stokes_i_error_arr
    if (stokes_i_error_arr == 0).all():
        sigma_arr = None

    try:
        popt, pcov = optimize.curve_fit(
            fit_func,
            freq_arr_hz / ref_freq_hz,
            stokes_i_arr,
            sigma=sigma_arr,
            absolute_sigma=True,
            p0=initial_guess,
            bounds=bounds,
        )
    except (ValueError, RuntimeError) as e:
        logger.warning(f"Stokes I fit with errors failed ({e}); retrying unweighted.")
        try:
            popt, pcov = optimize.curve_fit(
                fit_func,
                freq_arr_hz / ref_freq_hz,
                stokes_i_arr,
                p0=initial_guess,
            )
        except (ValueError, RuntimeError) as e2:
            # Fall back to a flat model at the mean. With only params[0] set,
            # both power_law and polynomial give that constant, so the caller
            # gets a usable model rather than an exception. pcov is zero.
            logger.warning(
                f"Stokes I fit failed ({e2}); falling back to a flat (mean) model."
            )
            popt = np.zeros(fit_order + 1)
            popt[0] = mean_spectrum
            pcov = np.zeros((fit_order + 1, fit_order + 1))
    stokes_i_model_arr = fit_func(freq_arr_hz / ref_freq_hz, *popt)
    ssr = float(np.sum((stokes_i_arr - stokes_i_model_arr) ** 2))
    with np.errstate(divide="ignore"):
        aic = akaike_info_criterion_lsq(
            ssr=ssr, n_params=fit_order + 1, n_samples=len(freq_arr_hz)
        )

    errors = np.sqrt(np.diag(pcov))
    fit_vals = [sf.round(p, e) for p, e in zip(popt, errors, strict=False)]
    logger.info(f"Fit results: {fit_vals}")

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
    msg = f"Iteratively fitting Stokes I model of type {fit_type} with max order {fit_order}."
    logger.info(msg)
    orders = np.arange(fit_order + 1)
    n_parameters = orders + 1
    fit_results: list[FitResult] = []

    for order in orders:
        fit_result = static_fit(
            freq_arr_hz,
            ref_freq_hz,
            stokes_i_arr,
            stokes_i_error_arr,
            int(order),
            fit_type,
        )
        fit_results.append(fit_result)

    logger.info(f"Fit results for orders {orders}:")
    aics = np.array([fit_result.aic for fit_result in fit_results])
    bestest_aic, bestest_n, bestest_aic_idx = best_aic_func(aics, n_parameters)
    logger.info(f"Best fit found with {bestest_n} parameters.")
    logger.debug(f"Best fit found with AIC {bestest_aic}.")
    logger.debug(f"Best fit found at index {bestest_aic_idx}.")
    logger.debug(f"Best fit found with order {orders[bestest_aic_idx]}.")

    return fit_results[bestest_aic_idx]


def stokes_i_snr(i_spec: NDArray[np.float64], e_spec: NDArray[np.float64]) -> float:
    """Frequency-averaged Stokes I SNR: `mean(I) * sqrt(n) / rms(error)`.

    Averaging `n` channels beats the noise down by `sqrt(n)`, hence the factor.
    Returns inf when there is no usable noise (all-zero or non-finite error), so
    an SNR cut becomes a no-op instead of rejecting everything.
    """
    n = e_spec.size
    rms_err = float(np.sqrt(np.mean(e_spec**2))) if n else 0.0
    if not np.isfinite(rms_err) or rms_err <= 0:
        return np.inf
    return float(np.mean(i_spec) * np.sqrt(n) / rms_err)


def sample_model_error(
    fit: FitResult,
    x_arr: NDArray[np.float64],
    n_error_samples: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Median model and 1-sigma (16th/84th-percentile) error via Monte-Carlo
    over the fit covariance, evaluated at `x_arr = freq / ref_freq`."""
    dist = stats.multivariate_normal(
        mean=np.asarray(fit.popt),
        cov=np.asarray(fit.pcov),
        allow_singular=True,
    )
    # reshape (not atleast_2d): a length-1 popt gives rvs shape (n,), which must
    # become (n, 1) rather than (1, n).
    samples = np.asarray(dist.rvs(n_error_samples)).reshape(n_error_samples, -1)
    model_samples = np.array([fit.stokes_i_model_func(x_arr, *s) for s in samples])
    low, median, high = np.nanpercentile(model_samples, [16, 50, 84], axis=0)
    error = np.abs(high - low)
    error[error > 1e99] = np.nan  # guard numerical overflow
    return median, error


def fit_stokes_i_model(
    freq_arr_hz: NDArray[np.float64],
    ref_freq_hz: float,
    stokes_i_arr: NDArray[np.float64],
    stokes_i_error_arr: NDArray[np.float64],
    fit_order: int = 2,
    fit_type: Literal["log", "linear"] = "log",
    snr_cut: float | None = None,
) -> FitResult | None:
    """Fit a Stokes I spectrum, or return None when it should not be fitted.

    Masks non-finite channels first, then returns None if too few finite
    channels remain (`< abs(fit_order) + 2`) or, when `snr_cut` is given, the
    frequency-averaged SNR is below it -- letting the caller impose a flat model
    for that spectrum. A fit that cannot converge does not raise: `static_fit`
    falls back to a flat (mean) model.
    """
    good = np.isfinite(stokes_i_arr) & np.isfinite(stokes_i_error_arr)
    if int(good.sum()) < abs(fit_order) + 2:
        return None
    if (
        snr_cut is not None
        and stokes_i_snr(stokes_i_arr[good], stokes_i_error_arr[good]) < snr_cut
    ):
        return None

    freq_g = freq_arr_hz[good]
    i_g = stokes_i_arr[good]
    e_g = stokes_i_error_arr[good]
    if fit_order < 0:
        return dynamic_fit(freq_g, ref_freq_hz, i_g, e_g, abs(fit_order), fit_type)
    return static_fit(freq_g, ref_freq_hz, i_g, e_g, fit_order, fit_type)
