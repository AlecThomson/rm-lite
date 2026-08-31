from __future__ import annotations

import logging
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, Protocol, TypeAlias, cast

import dask.array as da
import numpy as np
from astropy.stats import akaike_info_criterion_lsq
from numpy.typing import ArrayLike, NDArray
from scipy import optimize, stats

from rm_lite.utils.logging import logger, quiet_logs

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


@dataclass(frozen=True, kw_only=True, slots=True)
class StokesIFitOptions:
    """Options for Stokes I model fitting, shared by the 1D and 3D tools"""

    fit_order: int = 2
    """Fit order; negative iterates orders and picks the best by AIC"""
    fit_function: Literal["log", "linear"] = "log"
    """"log" fits a power law, "linear" a polynomial"""
    snr_cut: float | None = 5.0
    """Skip the fit below this frequency-averaged SNR; None fits everything"""
    compute_model_error: bool = False
    """Also compute the per-pixel model error (3D fit path only)"""
    n_error_samples: int = 1000
    """Monte-Carlo samples for the model error"""

    def __post_init__(self) -> None:
        if self.fit_function not in ("log", "linear"):
            msg = f"fit_function must be 'log' or 'linear', got {self.fit_function!r}."
            raise ValueError(msg)
        if self.snr_cut is not None and self.snr_cut < 0:
            msg = f"snr_cut must be non-negative, got {self.snr_cut}."
            raise ValueError(msg)
        if self.n_error_samples < 1:
            msg = f"n_error_samples must be >= 1, got {self.n_error_samples}."
            raise ValueError(msg)


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
    if stddev is None:
        msg = "stddev cannot be None"
        raise ValueError(msg)
    return np.asarray(amplitude * np.exp(-0.5 * ((x - mean) / stddev) ** 2))


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
    if stddev is None:
        msg = "stddev cannot be None"
        raise ValueError(msg)
    return np.asarray(np.exp(-0.5 * ((x - mean) / stddev) ** 2))


def unit_centred_gaussian(
    x: NDArray[np.float64], stddev: float | None = None, fwhm: float | None = None
) -> NDArray[np.float64]:
    if stddev is None and fwhm is None:
        msg = "Must provide either stddev or fwhm."
        raise ValueError(msg)
    if stddev is None and fwhm is not None:
        stddev = fwhm_to_sigma(fwhm)
    if stddev is None:
        msg = "stddev cannot be None"
        raise ValueError(msg)
    return np.asarray(np.exp(-0.5 * (x / stddev) ** 2))


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
    stddev_guess = fwhm_to_sigma(fwhm_fdf_radm2)
    if mask.sum() > 1:
        # pcov is discarded, so a "covariance could not be estimated" warning on a
        # near-flat (depolarised) peak is irrelevant here; the params still fit.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", optimize.OptimizeWarning)
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


class SampledPeakFit(NamedTuple):
    """Sub-sample peak from the three samples straddling it."""

    amplitude: NDArray[np.float64]
    """Interpolated peak amplitude"""
    offset: NDArray[np.float64]
    """Peak position in samples from the middle one, within [-0.5, 0.5]"""
    value: NDArray[np.complex128]
    """The sampled curve interpolated to `offset`, phase kept"""


def fit_sampled_peak(
    below: NDArray[np.complex128],
    at: NDArray[np.complex128],
    above: NDArray[np.complex128],
) -> SampledPeakFit:
    """Fit a peak sub-sample from the three samples straddling it.

    A parabola through the three amplitudes gives the peak amplitude and its
    offset from the middle sample; the samples themselves are interpolated
    linearly to that offset, so the phase comes along. Elementwise, so one peak
    or a whole cube of them (numpy or dask) costs the same code, which is what
    makes this the cheap alternative to `fit_fdf`'s per-spectrum Gaussian fit.

    Everything is NaN where the three samples do not describe a maximum: a curve
    that does not turn over, or a non-finite sample.
    """
    amp_below, amp_at, amp_above = np.abs(below), np.abs(at), np.abs(above)
    # A real maximum curves down, so a non-negative curvature is not a peak.
    curvature = amp_below - 2.0 * amp_at + amp_above
    offset = 0.5 * (amp_below - amp_above) / np.where(curvature < 0, curvature, np.nan)
    amplitude = amp_at - 0.25 * (amp_below - amp_above) * offset
    # Linear interpolation towards whichever neighbour the fit moved.
    neighbour_weight = np.abs(offset)
    value = (1.0 - neighbour_weight) * at + neighbour_weight * np.where(
        offset >= 0, above, below
    )
    return SampledPeakFit(amplitude=amplitude, offset=offset, value=value)


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
    fit_function: Literal["log", "linear"] = "log",
) -> FitResult:
    msg = f"Fitting Stokes I model of type {fit_function} with order {fit_order}."
    logger.info(msg)
    if fit_function == "linear":
        fit_func = polynomial(fit_order)
    elif fit_function == "log":
        fit_func = power_law(fit_order)
    else:
        msg = f"Unknown fit type {fit_function} provided. Must be 'log' or 'linear'."  # type: ignore[unreachable]
        raise ValueError(msg)

    logger.debug(
        f"Fitting Stokes I model with {fit_function} model of order {fit_order}."
    )
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
    fit_vals = [f"{p:.3g} +/- {e:.3g}" for p, e in zip(popt, errors, strict=False)]
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
    fit_function: Literal["log", "linear"] = "log",
) -> FitResult:
    msg = f"Iteratively fitting Stokes I model of type {fit_function} with max order {fit_order}."
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
            fit_function,
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


def check_snr_cut_has_error(
    options: StokesIFitOptions,
    stokes_i_error_arr: NDArray[np.float64] | None,
) -> None:
    """Raise if `snr_cut` is set but there is no error to measure SNR against.

    `stokes_i_snr` returns inf without one, so the cut silently passes every
    spectrum and noise gets fitted as if it were signal. This is a property of
    the call rather than of any one spectrum, so it is checked once up front:
    a cube fails in seconds instead of an hour in.
    """
    if options.snr_cut is None:
        return
    msg = (
        f"snr_cut={options.snr_cut} needs a Stokes I error to measure SNR "
        "against, and none was given (or it is all zero). Pass an error, or "
        "set snr_cut=None to fit every spectrum."
    )
    if stokes_i_error_arr is None:
        raise ValueError(msg)
    err = np.asarray(stokes_i_error_arr, dtype=np.float64)
    if not bool(np.any(np.isfinite(err) & (err > 0))):
        raise ValueError(msg)


def model_is_usable(model: NDArray[np.float64]) -> bool:
    """Whether a fitted Stokes I model can safely divide Q/U.

    Q/U are divided by the model, so one that reaches zero or goes negative
    anywhere in the band flips or blows up the fractional polarisation instead
    of correcting it.
    """
    return bool(np.all(np.isfinite(model)) and np.min(model) > 0.0)


def coefficient_names(
    n_coeff: int, fit_function: Literal["log", "linear"]
) -> tuple[str, ...]:
    """Names for `n_coeff` Stokes I model terms, in `popt` order.

    The power law is `I = flux * 10**(alpha*log10(x) + beta*log10(x)**2 + ...)`
    at `x = nu/nu_ref`, so its first three terms get the usual radio names and the
    rest are `p3` up. A polynomial's are `c0..cN`, coefficients of `x**i` rather
    than spectral indices.
    """
    if fit_function == "linear":
        return tuple(f"c{i}" for i in range(n_coeff))
    named = ("flux", "alpha", "beta")
    return tuple(named[i] if i < len(named) else f"p{i}" for i in range(n_coeff))


def pad_coefficients(coeffs: ArrayLike, n_coeff: int) -> NDArray[np.float64]:
    """`coeffs` widened to `n_coeff` terms, zero-filling the ones the fit dropped.

    A negative `fit_order` picks each pixel's order by AIC, so pixels come back
    with different term counts. A term the fit did not use contributes nothing to
    either model, so zero is its value rather than a gap.
    """
    arr = np.asarray(coeffs, dtype=np.float64)
    if arr.size > n_coeff:
        msg = f"Got {arr.size} coefficients, more than the {n_coeff} expected."
        raise ValueError(msg)
    out = np.zeros(n_coeff, dtype=np.float64)
    out[: arr.size] = arr
    return out


def coefficient_errors(pcov: ArrayLike, n_coeff: int) -> NDArray[np.float64]:
    """1-sigma marginal error per coefficient, `sqrt(diag(pcov))`, padded to `n_coeff`.

    Marginal, so it ignores the off-diagonal correlations, which are strong
    between power-law terms. To propagate the model itself use the full
    covariance via `draw_model_samples`, not these.
    """
    diag = np.diag(np.asarray(pcov, dtype=np.float64))
    with np.errstate(invalid="ignore"):
        return pad_coefficients(np.sqrt(diag), n_coeff)


def flat_fit_result(
    mean_flux: float,
    fit_order: int,
    fit_function: Literal["log", "linear"],
) -> FitResult:
    """A `FitResult` holding a flat model at `mean_flux`, with zero covariance.

    With only params[0] set both `power_law` and `polynomial` give that
    constant, so a caller that cannot use a fitted model still gets a valid one.
    """
    fit_func = power_law(fit_order) if fit_function == "log" else polynomial(fit_order)
    popt = np.zeros(fit_order + 1)
    popt[0] = mean_flux
    return FitResult(
        popt=popt,
        pcov=np.zeros((fit_order + 1, fit_order + 1)),
        stokes_i_model_func=fit_func,
        aic=np.inf,
    )


def draw_model_samples(
    fit: FitResult,
    x_arr: NDArray[np.float64],
    n_error_samples: int,
) -> NDArray[np.float64]:
    """Monte-Carlo model realisations over the fit covariance, shape
    (n_error_samples, len(x_arr)), evaluated at `x_arr = freq / ref_freq`."""
    dist = stats.multivariate_normal(
        mean=np.asarray(fit.popt),
        cov=np.asarray(fit.pcov),
        allow_singular=True,
    )
    # reshape (not atleast_2d): a length-1 popt gives rvs shape (n,), which must
    # become (n, 1) rather than (1, n).
    samples = np.asarray(dist.rvs(n_error_samples)).reshape(n_error_samples, -1)
    return np.array([fit.stokes_i_model_func(x_arr, *s) for s in samples])


def sample_model_error(
    fit: FitResult,
    x_arr: NDArray[np.float64],
    n_error_samples: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Median model and 1-sigma (16th/84th-percentile) error via Monte-Carlo
    over the fit covariance, evaluated at `x_arr = freq / ref_freq`."""
    model_samples = draw_model_samples(fit, x_arr, n_error_samples)
    low, median, high = np.nanpercentile(model_samples, [16, 50, 84], axis=0)
    error = np.abs(high - low)
    error[error > 1e99] = np.nan  # guard numerical overflow
    return median, error


def fit_stokes_i_model(
    freq_arr_hz: NDArray[np.float64],
    ref_freq_hz: float,
    stokes_i_arr: NDArray[np.float64],
    stokes_i_error_arr: NDArray[np.float64],
    options: StokesIFitOptions,
) -> FitResult | None:
    """Fit a Stokes I spectrum, or return None when it should not be fitted.

    Masks non-finite channels first, then returns None if too few finite
    channels remain (`< abs(options.fit_order) + 2`) or, when `options.snr_cut`
    is given, the frequency-averaged SNR is below it -- letting the caller
    impose a flat model for that spectrum. A fit that cannot converge does not
    raise: `static_fit` falls back to a flat (mean) model.
    """
    fit_order = options.fit_order
    good = np.isfinite(stokes_i_arr) & np.isfinite(stokes_i_error_arr)
    if int(good.sum()) < abs(fit_order) + 2:
        return None
    if (
        options.snr_cut is not None
        and stokes_i_snr(stokes_i_arr[good], stokes_i_error_arr[good]) < options.snr_cut
    ):
        return None

    freq_g = freq_arr_hz[good]
    i_g = stokes_i_arr[good]
    e_g = stokes_i_error_arr[good]
    if fit_order < 0:
        return dynamic_fit(
            freq_g, ref_freq_hz, i_g, e_g, abs(fit_order), options.fit_function
        )
    return static_fit(freq_g, ref_freq_hz, i_g, e_g, fit_order, options.fit_function)


class StokesIFitCubes(NamedTuple):
    """Lazy outputs of the per-pixel Stokes I fit pass, all from one `map_blocks`."""

    model_cube: da.Array
    """Model cube (n_freq, ny, nx), chunked like the Stokes I cube."""
    alpha_map: da.Array
    """Spectral index at the reference frequency, (ny, nx)."""
    order_map: da.Array
    """Fitted polynomial order, (ny, nx)."""
    coeff_cube: da.Array
    """Fitted model terms, (n_coeff, ny, nx)."""
    coeff_error_cube: da.Array
    """Marginal 1-sigma error per term, (n_coeff, ny, nx)."""
    model_error_cube: da.Array | None
    """Monte-Carlo model error (n_freq, ny, nx); None without `compute_model_error`."""
    alpha_error_map: da.Array | None
    """Monte-Carlo alpha error (ny, nx); None without `compute_model_error`."""


def fit_stokes_cube(
    stokes_i: da.Array,
    stokes_i_error: NDArray[np.float64] | da.Array | None,
    freq_arr_hz: NDArray[np.float64],
    ref_freq_hz: RefFreqHz | da.Array,
    fit_options: StokesIFitOptions,
    log_level: int,
) -> StokesIFitCubes:
    """Lazy per-pixel Stokes I model cube, maps and fitted terms.

    All come from one per-pixel fit pass (`_fit_stokes_i_block`), so the terms and
    their marginal errors cost nothing beyond the fit already being run, and the
    optional Monte-Carlo error cube and alpha error reuse the same fit rather than
    refitting. Alpha, order and the terms are NaN for pixels that were not fitted.
    `stokes_i_error` is a per-channel 1D array (n_freq,), a per-pixel error cube
    (n_freq, ny, nx), or None.
    """
    check_snr_cut_has_error(fit_options, _error_to_check(stokes_i_error))
    err_1d, err_cube = _split_stokes_i_error(stokes_i_error, stokes_i.chunks)

    n_freq = int(stokes_i.shape[0])
    planes = _block_planes(
        n_freq, abs(fit_options.fit_order) + 1, fit_options.compute_model_error
    )
    # A per-pixel reference must be sliced per block, so it travels as a
    # positional array argument, last in `arrays`.
    ref_blocks: tuple[da.Array, ...] = ()
    if np.ndim(ref_freq_hz) != 0:
        ref_blocks = (cast("da.Array", ref_freq_hz).rechunk(stokes_i.chunks[1:]),)

    stacked = da.map_blocks(
        _fit_stokes_i_block,
        *((stokes_i,) if err_cube is None else (stokes_i, err_cube)),
        *ref_blocks,
        dtype=np.float64,
        chunks=((planes.n_out,), stokes_i.chunks[1], stokes_i.chunks[2]),
        freq_arr_hz=freq_arr_hz,
        ref_freq_hz=ref_freq_hz,
        err_1d=err_1d,
        fit_options=fit_options,
        log_level=log_level,
        has_ref_block=bool(ref_blocks),
    )
    return StokesIFitCubes(
        model_cube=stacked[planes.model],
        alpha_map=stacked[planes.alpha],
        order_map=stacked[planes.order],
        coeff_cube=stacked[planes.coeff],
        coeff_error_cube=stacked[planes.coeff_error],
        model_error_cube=(
            None if planes.model_error is None else stacked[planes.model_error]
        ),
        alpha_error_map=(
            None if planes.alpha_error is None else stacked[planes.alpha_error]
        ),
    )


def _pixel_stokes_i_error(
    err_block: NDArray[np.float64] | None,
    err_1d: NDArray[np.float64] | None,
    n_freq: int,
    y: int,
    x: int,
) -> NDArray[np.float64]:
    """Per-pixel Stokes I error spectrum from a 3D error cube, 1D per-channel
    array, or neither (zeros -> `fit_stokes_i_model` fits unweighted)."""
    if err_block is not None:
        return err_block[:, y, x]
    if err_1d is not None:
        return err_1d
    return np.zeros(n_freq, dtype=np.float64)


def _alpha_at_ref(
    spec: NDArray[np.float64],
    freq_arr_hz: NDArray[np.float64],
    ref_freq_hz: float,
) -> float:
    """Spectral index (d ln I / d ln nu) of `spec` at the reference frequency.

    Sorts by frequency first: `np.interp`/`np.gradient` need an ascending axis,
    and a descending-frequency cube (negative CDELT3) otherwise gives garbage.
    """
    order = np.argsort(freq_arr_hz)
    ln_freq = np.log(freq_arr_hz[order])
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.gradient(np.log(np.abs(spec[order])), ln_freq)
    return float(np.interp(np.log(ref_freq_hz), ln_freq, slope))


class PixelFit(NamedTuple):
    """One pixel's Stokes I fit within a chunk (see `_iter_pixel_fits`)."""

    y: int
    """y pixel"""
    x: int
    """x pixel"""
    i_spec: NDArray[np.float64]
    """The pixel's Stokes I spectrum (unmasked), for the flat-model fallback."""
    good: NDArray[np.bool_]
    """Finite-channel mask, for the flat-model fallback."""
    fit: FitResult | None
    """The fit, or None if the pixel was skipped (too few channels / low SNR)."""


def _iter_pixel_fits(
    i_block: NDArray[np.float64],
    err_block: NDArray[np.float64] | None,
    err_1d: NDArray[np.float64] | None,
    freq_arr_hz: NDArray[np.float64],
    ref_freq_hz: RefFreqHz,
    fit_options: StokesIFitOptions,
) -> Iterator[PixelFit]:
    """Yield a `PixelFit` for every pixel in a chunk.

    `fit` is None for a skipped pixel (`fit_stokes_i_model` returns None when
    there are too few finite channels or the SNR is below `fit_options.snr_cut`).
    Shared by the model/alpha pass and the opt-in error pass so the setup lives
    once.
    """
    n_freq, cy, cx = i_block.shape
    for y in range(cy):
        for x in range(cx):
            i_spec = i_block[:, y, x]
            e_spec = _pixel_stokes_i_error(err_block, err_1d, n_freq, y, x)
            good = np.isfinite(i_spec) & np.isfinite(e_spec)
            fit = fit_stokes_i_model(
                freq_arr_hz=freq_arr_hz,
                ref_freq_hz=ref_freq_for_pixel(ref_freq_hz, y, x),
                stokes_i_arr=i_spec,
                stokes_i_error_arr=e_spec,
                options=fit_options,
            )
            yield PixelFit(y, x, i_spec, good, fit)


class BlockPlanes(NamedTuple):
    """Where each output of `_fit_stokes_i_block` sits in its stacked block."""

    model: slice
    alpha: int
    order: int
    coeff: slice
    coeff_error: slice
    model_error: slice | None
    """None unless the Monte-Carlo error pass ran."""
    alpha_error: int | None
    """None unless the Monte-Carlo error pass ran."""
    n_out: int
    """Total planes in the block."""


def _block_planes(n_freq: int, n_coeff: int, with_error: bool) -> BlockPlanes:
    """Lay out the block `_fit_stokes_i_block` stacks its outputs into.

    Everything the fit hands over for free comes first, so those planes sit in the
    same place whether or not the Monte-Carlo error pass was asked for.
    """
    coeff = slice(n_freq + 2, n_freq + 2 + n_coeff)
    coeff_error = slice(coeff.stop, coeff.stop + n_coeff)
    model_error = (
        slice(coeff_error.stop, coeff_error.stop + n_freq) if with_error else None
    )
    return BlockPlanes(
        model=slice(0, n_freq),
        alpha=n_freq,
        order=n_freq + 1,
        coeff=coeff,
        coeff_error=coeff_error,
        model_error=model_error,
        alpha_error=None if model_error is None else model_error.stop,
        n_out=coeff_error.stop if model_error is None else model_error.stop + 1,
    )


def _write_model_planes(
    out: NDArray[np.float64],
    y: int,
    x: int,
    planes: BlockPlanes,
    fit: FitResult,
    model: NDArray[np.float64],
    freq_arr_hz: NDArray[np.float64],
    ref_freq_hz: float,
) -> None:
    """Model cube, alpha at the reference frequency, fitted order, and the fit's
    own terms with their marginal errors."""
    n_coeff = planes.coeff.stop - planes.coeff.start
    popt = np.asarray(fit.popt)
    out[planes.model, y, x] = model
    out[planes.alpha, y, x] = _alpha_at_ref(model, freq_arr_hz, ref_freq_hz)
    out[planes.order, y, x] = popt.size - 1
    out[planes.coeff, y, x] = pad_coefficients(popt, n_coeff)
    out[planes.coeff_error, y, x] = coefficient_errors(fit.pcov, n_coeff)


def _write_error_planes(
    out: NDArray[np.float64],
    y: int,
    x: int,
    planes: BlockPlanes,
    fit: FitResult,
    freq_arr_hz: NDArray[np.float64],
    ref_freq_hz: float,
    n_error_samples: int,
) -> None:
    """Per-channel model error and alpha error, from one Monte-Carlo draw."""
    assert planes.model_error is not None
    assert planes.alpha_error is not None
    samples = draw_model_samples(fit, freq_arr_hz / ref_freq_hz, n_error_samples)
    low, high = np.nanpercentile(samples, [16, 84], axis=0)
    model_error = np.abs(high - low)
    model_error[model_error > 1e99] = np.nan
    out[planes.model_error, y, x] = model_error
    alpha_samples = np.array(
        [_alpha_at_ref(m, freq_arr_hz, ref_freq_hz) for m in samples]
    )
    a_low, a_high = np.nanpercentile(alpha_samples, [16, 84])
    out[planes.alpha_error, y, x] = abs(a_high - a_low)


def _write_flat_model(
    out: NDArray[np.float64],
    y: int,
    x: int,
    planes: BlockPlanes,
    mean_flux: float,
) -> None:
    """Flat model at the pixel's mean Stokes I: no correction, everything the fit
    would have said (alpha, order, terms, errors) stays NaN."""
    out[planes.model, y, x] = mean_flux


RefFreqHz: TypeAlias = float | NDArray[np.float64]
"""A reference frequency in Hz: one for the whole image, or one per pixel."""


def ref_freq_for_pixel(ref_freq_hz: RefFreqHz, y: int, x: int) -> float:
    """This pixel's reference frequency, shared or its own.

    Resolved once per pixel so everything below stays scalar.
    """
    if np.ndim(ref_freq_hz) == 0:
        return float(cast("float", ref_freq_hz))
    return float(cast("NDArray[np.float64]", ref_freq_hz)[y, x])


def _fit_stokes_i_block(
    *arrays: NDArray[np.float64],
    freq_arr_hz: NDArray[np.float64],
    ref_freq_hz: RefFreqHz,
    err_1d: NDArray[np.float64] | None,
    fit_options: StokesIFitOptions,
    log_level: int,
    has_ref_block: bool = False,
) -> NDArray[np.float64]:
    """Fit a Stokes I model per pixel over one spatial chunk, in a single pass.

    Returns a stacked block of shape (n_out, cy, cx), laid out by `_block_planes`:
    the model cube, the fitted spectral index alpha at the reference frequency,
    the fitted polynomial order (`len(popt) - 1`), and the fit's own terms and
    their marginal errors. With `fit_options.compute_model_error` a per-channel
    model error and an alpha error follow, both from one Monte-Carlo over the same
    per-pixel fit covariance, so they cost no extra fit.

    `arrays` is `(i_block,)` or `(i_block, err_block)`, optionally followed by a
    per-pixel `ref_freq_hz` block when `has_ref_block`; the error cube is
    optional (see `_pixel_stokes_i_error`). A pixel that was not fitted (too few
    finite channels or SNR below `fit_options.snr_cut`) or whose model is
    unusable (see `rm_lite.utils.fitting.model_is_usable`) falls back to a flat
    model at its mean Stokes I, so it gets no spectral correction and its alpha,
    order, terms and errors stay NaN. A pixel with no finite channels stays NaN.
    """
    i_block = arrays[0]
    # A per-pixel reference arrives as a (cy, cx) block after the data;
    # `has_ref_block` disambiguates (data, error) from (data, reference).
    if has_ref_block:
        ref_freq_hz = arrays[-1]
    n_data = len(arrays) - (1 if has_ref_block else 0)
    err_block = arrays[1] if n_data > 1 else None
    n_freq, cy, cx = i_block.shape
    planes = _block_planes(
        n_freq, abs(fit_options.fit_order) + 1, fit_options.compute_model_error
    )
    out = np.full((planes.n_out, cy, cx), np.nan, dtype=np.float64)
    n_rejected = 0
    # The 1D fitter logs per fit and per failure. At cube scale that floods, so
    # quiet it to at least ERROR whatever the caller's log_level.
    with quiet_logs(max(log_level, logging.ERROR)):
        for y, x, i_spec, good, fit in _iter_pixel_fits(
            i_block, err_block, err_1d, freq_arr_hz, ref_freq_hz, fit_options
        ):
            if not good.any():
                continue
            mean_flux = float(np.mean(i_spec[good]))
            if fit is None:
                _write_flat_model(out, y, x, planes, mean_flux)
                continue
            pixel_ref_hz = ref_freq_for_pixel(ref_freq_hz, y, x)
            model = fit.stokes_i_model_func(
                freq_arr_hz / pixel_ref_hz, *np.asarray(fit.popt)
            )
            if not model_is_usable(model[good]):
                n_rejected += 1
                _write_flat_model(out, y, x, planes, mean_flux)
                continue
            _write_model_planes(
                out, y, x, planes, fit, model, freq_arr_hz, pixel_ref_hz
            )
            if fit_options.compute_model_error:
                _write_error_planes(
                    out,
                    y,
                    x,
                    planes,
                    fit,
                    freq_arr_hz,
                    pixel_ref_hz,
                    fit_options.n_error_samples,
                )
    if n_rejected:
        logger.warning(
            f"{n_rejected} of {cy * cx} pixels in this chunk fitted an unusable "
            "Stokes I model and fell back to a flat one (see "
            "`rm_lite.utils.fitting.model_is_usable`). Expect this on pixels with "
            "no real Stokes I signal, i.e. when `stokes_i_snr_cut` is None."
        )
    return out


def ref_flux_from_block(
    model_block: NDArray[np.float64],
    freq_arr_hz: NDArray[np.float64],
    ref_freq_hz: RefFreqHz,
) -> NDArray[np.float64]:
    """Interpolate a Stokes I model cube block at the reference frequency ->
    (cy, cx) reference-flux map block."""
    _, cy, cx = model_block.shape
    # np.interp needs an ascending axis; sort so a descending-frequency cube
    # (negative CDELT3) doesn't silently interpolate to garbage.
    order = np.argsort(freq_arr_hz)
    freq_sorted = freq_arr_hz[order]
    ref_flux = np.full((cy, cx), np.nan, dtype=np.float64)
    for y in range(cy):
        for x in range(cx):
            spec = model_block[:, y, x]
            if np.isfinite(spec).all():
                ref_flux[y, x] = np.interp(
                    ref_freq_for_pixel(ref_freq_hz, y, x), freq_sorted, spec[order]
                )
    return ref_flux


def alpha_from_model_block(
    model_block: NDArray[np.float64],
    freq_arr_hz: NDArray[np.float64],
    ref_freq_hz: RefFreqHz,
) -> NDArray[np.float64]:
    """Spectral index alpha at the reference frequency from a supplied model cube.

    For a model given directly (not fitted), every pixel is modelled, so alpha is
    finite wherever the model is finite (0 for a flat model). NaN only where the
    model is not finite.
    """
    _, cy, cx = model_block.shape
    alpha = np.full((cy, cx), np.nan, dtype=np.float64)
    for y in range(cy):
        for x in range(cx):
            spec = model_block[:, y, x]
            if not np.isfinite(spec).all():
                continue
            value = _alpha_at_ref(
                spec, freq_arr_hz, ref_freq_for_pixel(ref_freq_hz, y, x)
            )
            alpha[y, x] = value if np.isfinite(value) else 0.0
    return alpha


def _error_to_check(
    stokes_i_error: NDArray[np.float64] | da.Array | None,
) -> NDArray[np.float64] | None:
    """The part of a Stokes I error that `check_snr_cut_has_error` can inspect.

    A dask cube would have to be computed to look at its values, so only its
    presence is checked; an all-zero one still falls to the per-pixel path.
    """
    if stokes_i_error is None:
        return None
    if isinstance(stokes_i_error, da.Array):
        return np.ones(1, dtype=np.float64)
    return np.asarray(stokes_i_error, dtype=np.float64)


def _split_stokes_i_error(
    stokes_i_error: NDArray[np.float64] | da.Array | None,
    chunks: tuple[tuple[int, ...], ...],
) -> tuple[NDArray[np.float64] | None, da.Array | None]:
    """Split a Stokes I error into (per-channel 1D array, per-pixel dask cube).

    A 3D error (numpy or dask) becomes a dask cube rechunked to `chunks`; a 1D
    per-channel error becomes a numpy array; None stays None. `da.asarray` wraps
    a numpy cube so `.rechunk` works either way.
    """
    if stokes_i_error is None:
        return None, None
    if getattr(stokes_i_error, "ndim", 1) == 3:
        return None, da.asarray(stokes_i_error).rechunk(chunks)
    return np.asarray(stokes_i_error, dtype=np.float64), None
