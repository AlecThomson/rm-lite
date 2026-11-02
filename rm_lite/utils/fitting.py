from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Literal, NamedTuple, Protocol, TypeAlias, cast

import dask.array as da
import numba
import numpy as np
from dask.base import compute
from numpy.typing import ArrayLike, NDArray
from scipy import optimize, stats

from rm_lite.utils.arrays import nb_median
from rm_lite.utils.lmfit_numba import ModelSettings, lm_fit
from rm_lite.utils.logging import logger, quiet_logs

GAUSSIAN_SIGMA_TO_FWHM = float(2.0 * np.sqrt(2.0 * np.log(2.0)))

RobustLoss: TypeAlias = Literal["cauchy", "linear"]
"""Stokes I fit loss, passed to `scipy.optimize.least_squares`"""


class StokesIModel(Protocol):
    def __call__(
        self, x: NDArray[np.float64], *params: float | NDArray[np.float64]
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
    robust_loss: RobustLoss = "cauchy"
    """Downweight channels far from the model; "linear" is plain least squares"""
    f_scale: float = 3.0
    """How far, in sigma, before a channel is downweighted. Flat from 1 to 10"""
    model_floor_sigma: float = 0.01
    """Reject a model dipping this many sigma below the band-averaged Stokes I
    noise (see `model_noise_floor`): 3D blanks the pixel, 1D leaves Q/U
    uncorrected. 0 disables"""

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
        if self.robust_loss not in ("cauchy", "linear"):
            msg = f"robust_loss must be 'cauchy' or 'linear', got {self.robust_loss!r}."
            raise ValueError(msg)
        if self.f_scale <= 0:
            msg = f"f_scale must be positive, got {self.f_scale}."
            raise ValueError(msg)
        if self.model_floor_sigma < 0:
            msg = (
                f"model_floor_sigma must be non-negative, got {self.model_floor_sigma}."
            )
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
    fitting_size: float = 1.25,
) -> float:
    """FWHM of a Gaussian fitted to the RMSF main lobe.

    The fit spans `fitting_size` times `fwhm_rmsf_radm2`, cut short at the
    lobe's first minimum on each side so no sidelobe gets in (cf. WSClean's
    `-beam-fitting-size`).
    """
    rmsf_to_fit_arr = rmsf_to_fit_arr / np.nanmax(rmsf_to_fit_arr)
    d_phi = phi_double_arr_radm2[1] - phi_double_arr_radm2[0]
    peak = int(np.argmax(rmsf_to_fit_arr))
    half_width = max(1, round(fitting_size * fwhm_rmsf_radm2 / 2 / d_phi))
    # Samples from the peak outwards until the profile stops falling.
    right = rmsf_to_fit_arr[peak:]
    left = rmsf_to_fit_arr[peak::-1]
    to_right_min = int(np.argmax(np.append(np.diff(right) >= 0, True)))
    to_left_min = int(np.argmax(np.append(np.diff(left) >= 0, True)))
    # Stop one short of each minimum, but keep a point either side of the peak.
    start = peak - max(1, min(half_width, to_left_min - 1))
    end = peak + max(1, min(half_width, to_right_min - 1))
    start, end = max(0, start), min(len(rmsf_to_fit_arr) - 1, end)
    popt, _ = optimize.curve_fit(
        unit_centred_gaussian,
        phi_double_arr_radm2[start : end + 1],
        rmsf_to_fit_arr[start : end + 1],
        p0=[fwhm_to_sigma(fwhm_rmsf_radm2)],
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
    value: NDArray[np.complexfloating]
    """The sampled curve interpolated to `offset`, phase kept"""


def fit_sampled_peak(
    below: NDArray[np.complexfloating],
    at: NDArray[np.complexfloating],
    above: NDArray[np.complexfloating],
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
    def poly_func(
        x: NDArray[np.float64], *params: float | NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if len(params) != order + 1:
            msg = f"Polynomial function of order {order} requires {order + 1} parameters, {len(params)} given."
            raise ValueError(msg)
        return _eval_model_at(params, x, log_model=False)

    return poly_func


def power_law(order: int) -> StokesIModel:
    def power_func(
        x: NDArray[np.float64], *params: float | NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if len(params) != order + 1:
            msg = f"Power law function of order {order} requires {order + 1} parameters, {len(params)} given."
            raise ValueError(msg)
        return _eval_model_at(params, x, log_model=True)

    return power_func


def _eval_model_at(
    params: tuple[float | NDArray[np.float64], ...],
    x: NDArray[np.float64],
    log_model: bool,
) -> NDArray[np.float64]:
    """`_eval_model` in NumPy for the model functions, so it broadcasts"""
    # Uncompiled (NUMBA_DISABLE_JIT), _eval_model is the function itself
    py_eval_model = getattr(_eval_model, "py_func", _eval_model)
    return np.asarray(
        py_eval_model(params, np.asarray(x), ModelSettings(log_model, 0.0))
    )


def best_aic_func(
    aics: NDArray[np.float64], n_param: NDArray[np.integer[Any]]
) -> tuple[float, int, int]:
    """Find the best AIC for a set of AICs using Occam's razor."""
    best_aic_idx = _best_aic_index(
        np.asarray(aics, dtype=np.float64), np.asarray(n_param, dtype=np.int64)
    )
    best_aic, best_n = float(aics[best_aic_idx]), int(n_param[best_aic_idx])
    logger.debug(f"Best AIC is {best_aic}, with {best_n} params.")
    return best_aic, best_n, best_aic_idx


class FitStatus(IntEnum):
    """What `fit_stokes_i_spectra` did with a spectrum"""

    FITTED = 0
    """Fitted, with a usable model"""
    SKIPPED = 1
    """Not fitted: too few channels, or below the SNR cut"""
    UNUSABLE = 2
    """Fitted, but the model cannot divide Q/U (see `model_is_usable`)"""
    NEEDS_SCIPY = 3
    """`lm_fit` did not converge, so the Python fit takes it"""


class StokesISpectra(NamedTuple):
    """Stokes I spectra for `fit_stokes_i_spectra`, one per row"""

    i_rows: NDArray[np.float64]
    """Stokes I"""
    err_rows: NDArray[np.float64]
    """Errors; zeros for none"""
    freq_arr_hz: NDArray[np.float64]
    """Channel frequencies"""
    ref_freq_hz: NDArray[np.float64]
    """Reference frequency per spectrum"""


class SpectraFitOptions(NamedTuple):
    """Options for `fit_stokes_i_spectra`"""

    fit_order: int
    """Model order; negative for every order up to it, chosen by AIC"""
    model: ModelSettings
    """Model and loss"""
    snr_cut: float
    """`stokes_i_snr` below which a spectrum is skipped; -inf for no cut"""
    model_floor_sigma: float
    """`model_noise_floor` sigma"""


class SpectraFit(NamedTuple):
    """Results of `fit_stokes_i_spectra`, one row per spectrum"""

    model: NDArray[np.float64]
    """Model spectra"""
    alpha: NDArray[np.float64]
    """Spectral index at the reference frequency"""
    order: NDArray[np.float64]
    """Order fitted"""
    coeff: NDArray[np.float64]
    """Coefficients, zero-padded to `abs(fit_order) + 1`"""
    coeff_error: NDArray[np.float64]
    """Their errors"""
    status: NDArray[np.int64]
    """`FitStatus`"""


@numba.njit(cache=True, nogil=True, error_model="numpy")  # type: ignore[misc]
def _eval_model(
    params: NDArray[np.float64], x_arr: NDArray[np.float64], settings: ModelSettings
) -> NDArray[np.float64]:
    """A `power_law` or `polynomial` model, compiled. Uncompiled (`py_func`) it
    broadcasts `x_arr` against parameters that are arrays, as the model
    functions need.

    Args:
        params (NDArray[np.float64]): Model parameters.
        x_arr (NDArray[np.float64]): freq / ref_freq.
        settings (ModelSettings): Model (the loss is not used).

    Returns:
        NDArray[np.float64]: The model at `x_arr`.
    """
    model = np.zeros_like(x_arr)
    if settings.log_model:
        log_x = np.log10(x_arr)
        for i in range(1, len(params)):
            model = model + params[i] * log_x**i
        power_law_model: NDArray[np.float64] = params[0] * 10.0**model
        return power_law_model
    for i in range(len(params)):
        model = model + params[i] * x_arr**i
    return model


@numba.njit(cache=True, nogil=True, error_model="numpy")  # type: ignore[misc]
def _robust_aic(
    sigma_resid: NDArray[np.float64], n_params: int, f_scale: float
) -> float:
    """AIC of a least-squares fit, as astropy's `akaike_info_criterion_lsq`, with
    each residual capped at `f_scale` so one bad channel cannot pick the order.

    Uncapped, a bad channel inflates every order's residual by about the same
    amount, the scores land within 2 of each other, and the fewest parameters
    (a flat model) win. Not capped for a linear loss (`f_scale` 0), where a
    missing error would leave `f_scale` in flux units rather than sigma.

    Args:
        sigma_resid (NDArray[np.float64]): Residuals in sigma (or flux, unweighted).
        n_params (int): Parameters fitted.
        f_scale (float): Cauchy loss scale in sigma; 0 for a linear loss.

    Returns:
        float: AIC, corrected below 40 samples per parameter; inf where the
            correction diverges.
    """
    if f_scale > 0:
        sigma_resid = np.minimum(np.abs(sigma_resid), f_scale)
    return float(aic_lsq(float(np.sum(sigma_resid**2)), n_params, sigma_resid.size))


@numba.njit(cache=True, nogil=True, error_model="numpy")  # type: ignore[misc]
def aic_lsq(ssr: float, n_params: int, n_samples: int) -> float:
    """AIC of a least-squares fit, as astropy's `akaike_info_criterion_lsq`.

    Args:
        ssr (float): Sum of squared residuals.
        n_params (int): Parameters fitted.
        n_samples (int): Samples fitted.

    Returns:
        float: AIC, corrected below 40 samples per parameter; inf where the
            correction diverges.
    """
    if n_samples - n_params - 1 <= 0:
        return np.inf
    log_likelihood = -0.5 * n_samples * np.log(ssr / n_samples)
    aic = 2.0 * (n_params - log_likelihood)
    if n_samples / n_params < 40.0:
        aic += 2.0 * n_params * (n_params + 1.0) / (n_samples - n_params - 1.0)
    return float(aic)


@numba.njit(cache=True, nogil=True, error_model="numpy")  # type: ignore[misc]
def _best_aic_index(aics: NDArray[np.float64], n_param: NDArray[np.int64]) -> int:
    """Occam's razor on AIC: the lowest, then the fewest parameters within 2 of
    it; the fewest parameters if none can be scored.

    Args:
        aics (NDArray[np.float64]): AIC of each model.
        n_param (NDArray[np.int64]): Parameters of each model.

    Returns:
        int: Index of the model to keep.
    """
    best = int(np.nanargmin(aics))
    if not np.isfinite(aics[best]):
        return int(np.argmin(n_param))
    fewer = (np.abs(aics - aics[best]) < 2) & (n_param < n_param[best])
    if not fewer.any():
        return best
    candidates = np.flatnonzero(fewer)
    return int(candidates[np.argmin(n_param[candidates])])


@numba.njit(cache=True, nogil=True, error_model="numpy")  # type: ignore[misc]
def _spectral_index(
    models: NDArray[np.float64], freq_arr_hz: NDArray[np.float64], ref_freq_hz: float
) -> NDArray[np.float64]:
    """Spectral index (d ln I / d ln nu) of each model (row) at the reference
    frequency, as `np.gradient` and `np.interp` on ascending frequency.

    Args:
        models (NDArray[np.float64]): Model spectra, one per row.
        freq_arr_hz (NDArray[np.float64]): Channel frequencies, in any order.
        ref_freq_hz (float): Reference frequency.

    Raises:
        ValueError: With fewer than two channels.

    Returns:
        NDArray[np.float64]: Spectral index of each model.
    """
    if freq_arr_hz.size < 2:
        msg = "A spectral index needs at least two channels."
        raise ValueError(msg)
    sort = np.argsort(freq_arr_hz)
    ln_freq = np.log(freq_arr_hz[sort])
    ln_ref = np.log(ref_freq_hz)
    # np.gradient: second order inside, one-sided at the ends
    dx1, dx2 = np.diff(ln_freq)[:-1], np.diff(ln_freq)[1:]
    alpha = np.empty(models.shape[0])
    slope = np.empty(ln_freq.size)
    for row in range(models.shape[0]):
        ln_model = np.log(np.abs(models[row][sort]))
        slope[0] = (ln_model[1] - ln_model[0]) / (ln_freq[1] - ln_freq[0])
        slope[-1] = (ln_model[-1] - ln_model[-2]) / (ln_freq[-1] - ln_freq[-2])
        slope[1:-1] = (
            -dx2 / (dx1 * (dx1 + dx2)) * ln_model[:-2]
            + (dx2 - dx1) / (dx1 * dx2) * ln_model[1:-1]
            + dx1 / (dx2 * (dx1 + dx2)) * ln_model[2:]
        )
        alpha[row] = np.interp(ln_ref, ln_freq, slope)
    return alpha


@numba.njit(cache=True, nogil=True, error_model="numpy")  # type: ignore[misc]
def _fittable_channels(
    stokes_i_arr: NDArray[np.float64], stokes_i_error_arr: NDArray[np.float64]
) -> NDArray[np.bool_]:
    """Channels a Stokes I fit can use.

    An error of zero or less cannot weight a fit, so those channels go. All of
    them at once is how callers say "no error given", and then every finite
    channel is kept, to fit unweighted.

    Args:
        stokes_i_arr (NDArray[np.float64]): Stokes I.
        stokes_i_error_arr (NDArray[np.float64]): Errors; zeros for none.

    Returns:
        NDArray[np.bool_]: Channels to fit.
    """
    weightable = np.isfinite(stokes_i_error_arr) & (stokes_i_error_arr > 0)
    if not weightable.any():
        weightable = np.isfinite(stokes_i_error_arr)
    return np.isfinite(stokes_i_arr) & weightable


@numba.njit(cache=True, nogil=True, error_model="numpy")  # type: ignore[misc]
def _initial_guess(
    stokes_i_arr: NDArray[np.float64], n_param: int
) -> NDArray[np.float64]:
    """Starting parameters: the mean flux (0 if negative, as `p0 >= 0`), then
    zeros.

    Args:
        stokes_i_arr (NDArray[np.float64]): Stokes I.
        n_param (int): Parameters to fit.

    Returns:
        NDArray[np.float64]: Starting parameters.
    """
    params = np.zeros(n_param)
    params[0] = max(float(np.nanmean(stokes_i_arr)), 0.0)
    return params


class _FitWeights(NamedTuple):
    """How a Stokes I fit weights its channels (see `_fit_weights`)"""

    sigma: NDArray[np.float64]
    """1-sigma errors to divide residuals by; ones for an unweighted fit"""
    weighted: bool
    """Whether `sigma` are the errors"""
    settings: ModelSettings
    """Model and loss, linear when unweighted"""


@numba.njit(cache=True, nogil=True, error_model="numpy")  # type: ignore[misc]
def _fit_weights(
    stokes_i_error_arr: NDArray[np.float64], settings: ModelSettings
) -> _FitWeights:
    """Weight by the errors if every one is finite and positive. Otherwise fit
    unweighted, with a linear loss, as residuals in flux have no sigma for
    `f_scale` to count.

    Args:
        stokes_i_error_arr (NDArray[np.float64]): Errors.
        settings (ModelSettings): Model and loss asked for.

    Returns:
        _FitWeights: Errors to divide by, whether they are real, and the loss.
    """
    if np.all(np.isfinite(stokes_i_error_arr) & (stokes_i_error_arr > 0)):
        return _FitWeights(stokes_i_error_arr, True, settings)
    return _FitWeights(
        np.ones(stokes_i_error_arr.size), False, ModelSettings(settings.log_model, 0.0)
    )


class _SpectrumFit(NamedTuple):
    """`_fit_spectrum` of one spectrum"""

    status: int
    """`FitStatus`"""
    popt: NDArray[np.float64]
    """Parameters, if fitted"""
    pcov: NDArray[np.float64]
    """Their covariance, if fitted"""


@numba.njit(cache=True, nogil=True, error_model="numpy")  # type: ignore[misc]
def _fit_spectrum(
    stokes_i_arr: NDArray[np.float64],
    stokes_i_error_arr: NDArray[np.float64],
    x_arr: NDArray[np.float64],
    options: SpectraFitOptions,
) -> _SpectrumFit:
    """`fit_stokes_i_model` on one spectrum, compiled: the channel and SNR
    gates, then `lm_fit` at the order, or each order up to it with Occam's rule
    on `_robust_aic`.

    A dynamic fit scores an order `lm_fit` left to scipy where it stopped, near
    scipy's fit. That can pick another order than the Python fit, which scores
    scipy's: for polynomials about 1 pixel in 150 at `fit_order=-4`, whose
    models differ by about 5e-4. Any order stopped on the `p0 >= 0` bound, or a
    chosen order `lm_fit` left, needs scipy.

    Args:
        stokes_i_arr (NDArray[np.float64]): Stokes I.
        stokes_i_error_arr (NDArray[np.float64]): Errors; zeros for none.
        x_arr (NDArray[np.float64]): freq / ref_freq.
        options (SpectraFitOptions): Fit options.

    Returns:
        _SpectrumFit: `FitStatus`, and popt and pcov if fitted.
    """
    n_coeff = abs(options.fit_order) + 1
    good = _fittable_channels(stokes_i_arr, stokes_i_error_arr)
    if np.count_nonzero(good) < n_coeff + 1:
        return _SpectrumFit(FitStatus.SKIPPED, np.empty(0), np.empty((0, 0)))
    i_good, e_good, x_good = stokes_i_arr[good], stokes_i_error_arr[good], x_arr[good]
    if _stokes_i_snr(i_good, e_good) < options.snr_cut:
        return _SpectrumFit(FitStatus.SKIPPED, np.empty(0), np.empty((0, 0)))
    weights = _fit_weights(e_good, options.model)
    # Every order up to |fit_order| when it is negative
    first = 0 if options.fit_order < 0 else options.fit_order
    aics = np.full(n_coeff, np.inf)
    fits = []
    for order in range(first, n_coeff):
        fit = lm_fit(
            x_good,
            i_good,
            weights.sigma,
            _initial_guess(i_good, order + 1),
            weights.settings,
        )
        # On the p0 >= 0 bound, scipy's bounded fit lands elsewhere, and its AIC
        # could pick another order
        if not fit.converged and fit.popt[0] <= 0:
            return _SpectrumFit(FitStatus.NEEDS_SCIPY, np.empty(0), np.empty((0, 0)))
        fits.append(fit)
        model = _eval_model(fit.popt, x_good, weights.settings)
        aics[order] = _robust_aic(
            (model - i_good) / weights.sigma, order + 1, weights.settings.f_scale
        )
    best = (
        _best_aic_index(aics, np.arange(1, n_coeff + 1))
        if options.fit_order < 0
        else first
    )
    fit = fits[best - first]
    if not fit.converged:
        return _SpectrumFit(FitStatus.NEEDS_SCIPY, np.empty(0), np.empty((0, 0)))
    return _SpectrumFit(FitStatus.FITTED, fit.popt, fit.pcov)


@numba.njit(cache=True, nogil=True, error_model="numpy")  # type: ignore[misc]
def _empty_spectra_fit(n_spec: int, n_freq: int, n_coeff: int) -> SpectraFit:
    """A `SpectraFit` with every spectrum skipped.

    Args:
        n_spec (int): Spectra.
        n_freq (int): Channels.
        n_coeff (int): Coefficients.

    Returns:
        SpectraFit: NaN everywhere, and every status `FitStatus.SKIPPED`.
    """
    return SpectraFit(
        model=np.full((n_spec, n_freq), np.nan),
        alpha=np.full(n_spec, np.nan),
        order=np.full(n_spec, np.nan),
        coeff=np.full((n_spec, n_coeff), np.nan),
        coeff_error=np.full((n_spec, n_coeff), np.nan),
        status=np.full(n_spec, FitStatus.SKIPPED, dtype=np.int64),
    )


@numba.njit(cache=True, nogil=True, error_model="numpy")  # type: ignore[misc]
def _record_fit(
    fit: SpectraFit,
    spec: int,
    spectra: StokesISpectra,
    options: SpectraFitOptions,
    popt: NDArray[np.float64],
    pcov: NDArray[np.float64],
) -> None:
    """Fill one spectrum's row of `fit` from its fitted parameters: the model
    and alpha, order, padded coefficients and their errors, or
    `FitStatus.UNUSABLE` if the model cannot divide Q/U.

    Args:
        fit (SpectraFit): Filled in place.
        spec (int): The spectrum (row).
        spectra (StokesISpectra): The spectra.
        options (SpectraFitOptions): Fit options.
        popt (NDArray[np.float64]): Fitted parameters.
        pcov (NDArray[np.float64]): Their covariance.
    """
    x_arr = spectra.freq_arr_hz / spectra.ref_freq_hz[spec]
    model = _eval_model(popt, x_arr, options.model)
    floor = _noise_floor(spectra.err_rows[spec], options.model_floor_sigma)
    # The whole model divides Q/U, including channels Stokes I had flagged and
    # Q/U did not, so all of it has to be usable
    if not _model_is_usable(model, floor):
        fit.status[spec] = FitStatus.UNUSABLE
        return
    fit.status[spec] = FitStatus.FITTED
    fit.model[spec] = model
    fit.alpha[spec] = _spectral_index(
        model.reshape(1, -1), spectra.freq_arr_hz, spectra.ref_freq_hz[spec]
    )[0]
    fit.order[spec] = popt.size - 1
    fit.coeff[spec] = 0.0
    fit.coeff[spec, : popt.size] = popt
    fit.coeff_error[spec] = 0.0
    fit.coeff_error[spec, : popt.size] = np.sqrt(np.diag(pcov))


@numba.njit(cache=True, nogil=True, error_model="numpy")  # type: ignore[misc]
def fit_stokes_i_spectra(
    spectra: StokesISpectra, options: SpectraFitOptions
) -> SpectraFit:
    """`_fit_spectrum` and `_record_fit` on every spectrum, without the GIL.

    Args:
        spectra (StokesISpectra): Stokes I, errors and frequencies.
        options (SpectraFitOptions): Fit options.

    Returns:
        SpectraFit: Models, alpha, order, coefficients and errors, NaN where
            not fitted, and each spectrum's `FitStatus`; a spectrum `lm_fit`
            left to scipy is `FitStatus.NEEDS_SCIPY`, for the Python fit.
    """
    n_spec, n_freq = spectra.i_rows.shape
    fit: SpectraFit = _empty_spectra_fit(n_spec, n_freq, abs(options.fit_order) + 1)
    for spec in range(n_spec):
        result = _fit_spectrum(
            spectra.i_rows[spec],
            spectra.err_rows[spec],
            spectra.freq_arr_hz / spectra.ref_freq_hz[spec],
            options,
        )
        fit.status[spec] = result.status
        if result.status == FitStatus.FITTED:
            _record_fit(fit, spec, spectra, options, result.popt, result.pcov)
    return fit


def _model_settings(
    fit_function: Literal["log", "linear"], robust_loss: RobustLoss, f_scale: float
) -> ModelSettings:
    """`ModelSettings` for a fit function and loss.

    Args:
        fit_function (Literal["log", "linear"]): Power law or polynomial.
        robust_loss (RobustLoss): Loss.
        f_scale (float): Cauchy loss scale, in sigma.

    Raises:
        ValueError: For a loss other than "cauchy" or "linear".

    Returns:
        ModelSettings: The model, and `f_scale`, or 0 for a linear loss.
    """
    if robust_loss not in ("cauchy", "linear"):
        msg = f"robust_loss must be 'cauchy' or 'linear', got {robust_loss!r}."
        raise ValueError(msg)
    return ModelSettings(
        fit_function == "log", f_scale if robust_loss == "cauchy" else 0.0
    )


def _spectra_fit_options(fit_options: StokesIFitOptions) -> SpectraFitOptions:
    """`SpectraFitOptions` from `StokesIFitOptions`.

    Args:
        fit_options (StokesIFitOptions): Fit options.

    Returns:
        SpectraFitOptions: The same, for `fit_stokes_i_spectra`.
    """
    return SpectraFitOptions(
        fit_order=fit_options.fit_order,
        model=_model_settings(
            fit_options.fit_function, fit_options.robust_loss, fit_options.f_scale
        ),
        snr_cut=-np.inf if fit_options.snr_cut is None else fit_options.snr_cut,
        model_floor_sigma=fit_options.model_floor_sigma,
    )


def static_fit(
    freq_arr_hz: NDArray[np.float64],
    ref_freq_hz: float,
    stokes_i_arr: NDArray[np.float64],
    stokes_i_error_arr: NDArray[np.float64],
    fit_order: int = 2,
    fit_function: Literal["log", "linear"] = "log",
    robust_loss: RobustLoss = "cauchy",
    f_scale: float = 3.0,
) -> FitResult:
    """Fit one Stokes I spectrum at a fixed order.

    `lm_fit` first, then `curve_fit` (scipy's `least_squares`) for any fit it
    leaves. With `robust_loss` "cauchy" a channel more than `f_scale` sigma out
    is downweighted instead of dragging the model; the bounds pick
    `method="trf"`, the one method that takes a loss.

    Raises:
        ValueError: For an unknown `fit_function` or `robust_loss`.
    """
    msg = f"Fitting Stokes I model of type {fit_function} with order {fit_order}."
    logger.info(msg)
    if fit_function == "linear":
        fit_func = polynomial(fit_order)
    elif fit_function == "log":
        fit_func = power_law(fit_order)
    else:
        msg = f"Unknown fit type {fit_function} provided. Must be 'log' or 'linear'."  # type: ignore[unreachable]
        raise ValueError(msg)
    settings = _model_settings(fit_function, robust_loss, f_scale)
    x_arr = np.ascontiguousarray(freq_arr_hz / ref_freq_hz, dtype=np.float64)
    stokes_i_arr = np.ascontiguousarray(stokes_i_arr, dtype=np.float64)
    weights = _fit_weights(
        np.ascontiguousarray(stokes_i_error_arr, dtype=np.float64), settings
    )
    p0 = _initial_guess(stokes_i_arr, fit_order + 1)
    bounds = ([0.0] + [-np.inf] * fit_order, [np.inf] * (fit_order + 1))
    loss_kwargs: dict[str, Any] = (
        {"loss": "cauchy", "f_scale": f_scale} if weights.settings.f_scale > 0 else {}
    )
    fitted = True
    try:
        # Compiled first, so threads can run it; scipy for any fit it leaves
        popt, pcov, converged = lm_fit(
            x_arr, stokes_i_arr, weights.sigma, p0, weights.settings
        )
        if not converged:
            popt, pcov = optimize.curve_fit(
                fit_func,
                x_arr,
                stokes_i_arr,
                sigma=weights.sigma if weights.weighted else None,
                absolute_sigma=True,
                p0=p0,
                bounds=bounds,
                **loss_kwargs,
            )
    except (ValueError, RuntimeError) as e:
        logger.warning(f"Stokes I fit with errors failed ({e}); retrying unweighted.")
        try:
            popt, pcov = optimize.curve_fit(fit_func, x_arr, stokes_i_arr, p0=p0)
        except (ValueError, RuntimeError) as e2:
            # Fall back to a flat model at the mean. With only params[0] set,
            # both power_law and polynomial give that constant, so the caller
            # gets a usable model rather than an exception. pcov is zero.
            logger.warning(
                f"Stokes I fit failed ({e2}); falling back to a flat (mean) model."
            )
            popt = p0
            pcov = np.zeros((fit_order + 1, fit_order + 1))
            fitted = False
    stokes_i_model_arr = fit_func(x_arr, *popt)
    # Nothing converged, so there is no fit for `dynamic_fit` to rank.
    aic = (
        _robust_aic(
            (stokes_i_arr - stokes_i_model_arr) / weights.sigma,
            fit_order + 1,
            weights.settings.f_scale,
        )
        if fitted
        else float(np.inf)
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
    robust_loss: RobustLoss = "cauchy",
    f_scale: float = 3.0,
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
            robust_loss,
            f_scale,
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


@numba.njit(cache=True, nogil=True, error_model="numpy")  # type: ignore[misc]
def _stokes_i_snr(i_spec: NDArray[np.float64], e_spec: NDArray[np.float64]) -> float:
    """Frequency-averaged Stokes I SNR: `median(I) * sqrt(n) / median(error)`.

    Averaging `n` channels beats the noise down by `sqrt(n)`, hence the factor.
    Returns inf when there is no usable noise (all-zero or non-finite error), so
    an SNR cut becomes a no-op instead of rejecting everything.

    Medians, not the mean and rms: one oversized error channel took an rms SNR
    from 1200 to 1.4 and skipped a good pixel, and one negative flux spike made
    it negative. Only usable channels count, as in the fit.
    """
    usable = np.isfinite(i_spec) & np.isfinite(e_spec) & (e_spec > 0)
    n = np.count_nonzero(usable)
    if not n:
        return np.inf
    median_err = nb_median(e_spec[usable])
    if not np.isfinite(median_err) or median_err <= 0:
        return np.inf
    return float(nb_median(i_spec[usable]) * np.sqrt(n) / median_err)


def stokes_i_snr(i_spec: ArrayLike, e_spec: ArrayLike) -> float:
    """`_stokes_i_snr` of any arrays"""
    return float(
        _stokes_i_snr(
            np.ascontiguousarray(i_spec, dtype=np.float64),
            np.ascontiguousarray(e_spec, dtype=np.float64),
        )
    )


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


def model_noise_floor(stokes_i_error_arr: ArrayLike | None, sigma: float) -> float:
    """`_noise_floor`, or 0.0 without an error spectrum"""
    if stokes_i_error_arr is None:
        return 0.0
    return float(
        _noise_floor(np.ascontiguousarray(stokes_i_error_arr, dtype=np.float64), sigma)
    )


@numba.njit(cache=True, nogil=True, error_model="numpy")  # type: ignore[misc]
def _noise_floor(stokes_i_error_arr: NDArray[np.float64], sigma: float) -> float:
    """Flux `sigma` times the band-averaged noise, `median(error)/sqrt(n)`.

    Same noise as `stokes_i_snr`, so this and `snr_cut` share units. 0.0 without
    a usable error.
    """
    usable = np.isfinite(stokes_i_error_arr) & (stokes_i_error_arr > 0)
    n = np.count_nonzero(usable)
    if sigma <= 0 or not n:
        return 0.0
    return float(sigma * nb_median(stokes_i_error_arr[usable]) / np.sqrt(n))


@numba.njit(cache=True, nogil=True, error_model="numpy")  # type: ignore[misc]
def _model_is_usable(model: NDArray[np.float64], noise_floor: float = 0.0) -> bool:
    """Whether a Stokes I model can safely divide Q/U: finite, and never below
    `noise_floor` or outside what float32 can hold.

    Dividing by it amplifies Q/U by `model(ref_freq) / min(model)`, unbounded
    once a fit runs to zero in a channel. Both bare guards are float32's, the
    dtype it divides in: the fit works in float64, but the model is cast to the
    cube's precision before it divides anything, and a fit that runs away past
    float32's range is inf by the time the reference flux is read off it, which
    blanks that pixel's whole FDF spectrum.
    """
    floor = max(noise_floor, float(np.finfo(np.float32).tiny))
    return bool(
        np.all(np.isfinite(model))
        and np.min(model) > floor
        and np.max(model) <= np.finfo(np.float32).max
    )


def model_is_usable(model: ArrayLike, noise_floor: float = 0.0) -> bool:
    """`_model_is_usable` of any array"""
    return bool(
        _model_is_usable(np.ascontiguousarray(model, dtype=np.float64), noise_floor)
    )


def flat_model_value(mean_flux: float) -> float:
    """Divisor to use where no usable model was fitted. The mean cancels out of
    the FDF, leaving Q/U uncorrected; 1.0 where the mean cannot divide."""
    return mean_flux if model_is_usable(np.array([mean_flux])) else 1.0


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
    # Each parameter a column, so every sample evaluates in one call
    return fit.stokes_i_model_func(x_arr, *samples.T[:, :, np.newaxis])


def sample_model_error(
    fit: FitResult,
    x_arr: NDArray[np.float64],
    n_error_samples: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Median model and 1-sigma (16th/84th-percentile) error via Monte-Carlo
    over the fit covariance, evaluated at `x_arr = freq / ref_freq`."""
    return _model_percentiles(draw_model_samples(fit, x_arr, n_error_samples))


def _model_percentiles(
    samples: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Median and 16th to 84th percentile spread of model samples, per channel"""
    # nanpercentile goes column by column; percentile is the same without NaN
    percentile = np.nanpercentile if np.isnan(samples).any() else np.percentile
    low, median, high = percentile(samples, [16, 50, 84], axis=0)
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

    Masks channels that cannot be fitted, then returns None if too few remain
    (`< abs(options.fit_order) + 2`) or, when `options.snr_cut` is given, the
    frequency-averaged SNR is below it, so the caller leaves the pixel out. A
    fit that cannot converge does not raise: `static_fit` falls back to a flat
    (mean) model. Right at the minimum channel count the fit is real but its AIC
    is inf (see `_robust_aic`), so a negative `fit_order` settles on a lower order
    there. A bad flux is left to `options.robust_loss` in `static_fit`.
    """
    fit_order = options.fit_order
    stokes_i_arr = np.ascontiguousarray(stokes_i_arr, dtype=np.float64)
    stokes_i_error_arr = np.ascontiguousarray(stokes_i_error_arr, dtype=np.float64)
    good = _fittable_channels(stokes_i_arr, stokes_i_error_arr)
    if int(good.sum()) < abs(fit_order) + 2:
        return None
    if (
        options.snr_cut is not None
        and _stokes_i_snr(stokes_i_arr[good], stokes_i_error_arr[good])
        < options.snr_cut
    ):
        return None

    freq_g = freq_arr_hz[good]
    i_g = stokes_i_arr[good]
    e_g = stokes_i_error_arr[good]
    if fit_order < 0:
        return dynamic_fit(
            freq_g,
            ref_freq_hz,
            i_g,
            e_g,
            abs(fit_order),
            options.fit_function,
            options.robust_loss,
            options.f_scale,
        )
    return static_fit(
        freq_g,
        ref_freq_hz,
        i_g,
        e_g,
        fit_order,
        options.fit_function,
        options.robust_loss,
        options.f_scale,
    )


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


def field_spectral_index(
    stokes_i: da.Array,
    stokes_i_error: NDArray[np.float64] | da.Array | None,
    freq_arr_hz: NDArray[np.float64],
    fit_options: StokesIFitOptions,
) -> float:
    """Power-law index of the field's mean Stokes I spectrum.

    One pass over the cube: each channel is averaged over the pixels that have
    it, and the errors are carried in quadrature. 0 if the fit gives nothing.
    """
    finite = da.isfinite(stokes_i)
    total = da.sum(da.where(finite, stokes_i, 0.0), axis=(1, 2))
    count = da.sum(finite, axis=(1, 2))
    variance: da.Array | NDArray[np.float64] | None = None
    if stokes_i_error is not None and np.ndim(stokes_i_error) == 3:
        variance = da.sum(
            da.where(finite, da.asarray(stokes_i_error) ** 2, 0.0), axis=(1, 2)
        )
    total, count, variance = compute(total, count, variance)
    count = np.asarray(count, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.asarray(total, dtype=np.float64) / count
        if variance is not None:
            error = np.sqrt(np.asarray(variance, dtype=np.float64)) / count
        elif stokes_i_error is not None:
            error = np.asarray(stokes_i_error, dtype=np.float64) / np.sqrt(count)
        else:
            error = np.zeros_like(mean)
    good = np.isfinite(mean) & np.isfinite(error)
    if int(good.sum()) < 3:
        logger.warning("Too few channels to fit the field's Stokes I; using alpha=0.")
        return 0.0
    fit = static_fit(
        freq_arr_hz[good],
        float(np.exp(np.mean(np.log(freq_arr_hz[good])))),
        mean[good],
        error[good],
        fit_order=1,
        fit_function="log",
        robust_loss=fit_options.robust_loss,
        f_scale=fit_options.f_scale,
    )
    alpha = float(np.asarray(fit.popt)[1])
    if not np.isfinite(alpha):
        logger.warning("The field's Stokes I gave no spectral index; using alpha=0.")
        return 0.0
    logger.info(f"Field-wide Stokes I spectral index for the weights: {alpha:.3f}")
    return alpha


def _block_stokes_i_error(
    err_block: NDArray[np.float64] | None,
    err_1d: NDArray[np.float64] | None,
    shape: tuple[int, ...],
) -> NDArray[np.float64]:
    """A block's Stokes I errors, shaped as the block, from a 3D error cube, a
    1D per-channel array (broadcast, not copied), or neither (zeros, so the fit
    is unweighted)."""
    if err_block is not None:
        return err_block
    if err_1d is not None:
        return np.broadcast_to(err_1d[:, np.newaxis, np.newaxis], shape)
    return np.zeros(shape, dtype=np.float64)


def _alpha_at_ref(
    spec: NDArray[np.float64],
    freq_arr_hz: NDArray[np.float64],
    ref_freq_hz: float,
) -> NDArray[np.float64]:
    """`_spectral_index` of `spec`, or of each of its rows"""
    rows = np.ascontiguousarray(spec, dtype=np.float64).reshape(-1, spec.shape[-1])
    alpha = _spectral_index(
        rows, np.ascontiguousarray(freq_arr_hz, dtype=np.float64), ref_freq_hz
    )
    return np.asarray(alpha.reshape(spec.shape[:-1]))


class PixelFit(NamedTuple):
    """One pixel's Stokes I fit within a chunk (see `_iter_pixel_fits`)."""

    y: int
    """y pixel"""
    x: int
    """x pixel"""
    fit: FitResult | None
    """The fit, or None if the pixel was skipped (too few channels / low SNR)."""


def _iter_pixel_fits(
    i_block: NDArray[np.float64],
    err_block: NDArray[np.float64],
    freq_arr_hz: NDArray[np.float64],
    ref_freq_hz: RefFreqHz,
    fit_options: StokesIFitOptions,
    pixels: Iterable[tuple[int, int]],
) -> Iterator[PixelFit]:
    """Yield a `PixelFit` for each (y, x) in `pixels` of a chunk.

    `fit` is None for a skipped pixel (`fit_stokes_i_model` returns None when
    there are too few finite channels or the SNR is below `fit_options.snr_cut`).
    Shared by the model/alpha pass and the opt-in error pass so the setup lives
    once.
    """
    for y, x in pixels:
        fit = fit_stokes_i_model(
            freq_arr_hz=freq_arr_hz,
            ref_freq_hz=ref_freq_for_pixel(ref_freq_hz, y, x),
            stokes_i_arr=i_block[:, y, x],
            stokes_i_error_arr=err_block[:, y, x],
            options=fit_options,
        )
        yield PixelFit(y, x, fit)


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
    out[planes.model_error, y, x] = _model_percentiles(samples)[1]
    alpha_samples = _alpha_at_ref(samples, freq_arr_hz, ref_freq_hz)
    a_low, a_high = np.nanpercentile(alpha_samples, [16, 84])
    out[planes.alpha_error, y, x] = abs(a_high - a_low)


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
    optional (see `_block_stokes_i_error`). A pixel that was not fitted (too few
    finite channels or SNR below `fit_options.snr_cut`) or whose model is
    unusable (non-finite anywhere, or below `fit_options.model_floor_sigma`
    times the pixel's band-averaged noise, see `model_is_usable`) keeps a NaN
    model, so its Q/U are blanked rather than left uncorrected beside corrected
    neighbours.
    """
    i_block = arrays[0]
    # A per-pixel reference arrives as a (cy, cx) block after the data;
    # `has_ref_block` disambiguates (data, error) from (data, reference).
    if has_ref_block:
        ref_freq_hz = arrays[-1]
    n_data = len(arrays) - (1 if has_ref_block else 0)
    n_freq, cy, cx = i_block.shape
    err_block = _block_stokes_i_error(
        arrays[1] if n_data > 1 else None, err_1d, i_block.shape
    )
    # One layout for every block, so numba compiles one specialisation
    spectra = StokesISpectra(
        i_rows=np.ascontiguousarray(i_block.reshape(n_freq, -1).T, dtype=np.float64),
        err_rows=np.ascontiguousarray(
            err_block.reshape(n_freq, -1).T, dtype=np.float64
        ),
        freq_arr_hz=np.ascontiguousarray(freq_arr_hz, dtype=np.float64),
        ref_freq_hz=np.ascontiguousarray(
            np.broadcast_to(np.asarray(ref_freq_hz, dtype=np.float64), (cy, cx)).ravel()
        ),
    )
    options = _spectra_fit_options(fit_options)
    planes = _block_planes(
        n_freq, abs(fit_options.fit_order) + 1, fit_options.compute_model_error
    )
    out = np.full((planes.n_out, cy, cx), np.nan, dtype=np.float64)
    # The model errors need each fit's covariance, so every pixel takes the
    # per-pixel path. Otherwise only those the compiled fit leaves to scipy do.
    if fit_options.compute_model_error:
        fit = _empty_spectra_fit(cy * cx, n_freq, abs(fit_options.fit_order) + 1)
        pixels = np.arange(cy * cx)
    else:
        fit = fit_stokes_i_spectra(spectra, options)
        pixels = np.flatnonzero(fit.status == FitStatus.NEEDS_SCIPY)
    # The 1D fitter logs per fit and per failure. At cube scale that floods, so
    # quiet it to at least ERROR whatever the caller's log_level.
    with quiet_logs(max(log_level, logging.ERROR)):
        for y, x, pixel_fit in _iter_pixel_fits(
            i_block,
            err_block,
            freq_arr_hz,
            ref_freq_hz,
            fit_options,
            (divmod(int(pixel), cx) for pixel in pixels),
        ):
            spec = y * cx + x
            fit.status[spec] = FitStatus.SKIPPED
            if pixel_fit is None:
                continue
            _record_fit(
                fit,
                spec,
                spectra,
                options,
                np.ascontiguousarray(pixel_fit.popt, dtype=np.float64),
                np.ascontiguousarray(pixel_fit.pcov, dtype=np.float64),
            )
            if fit_options.compute_model_error and fit.status[spec] == FitStatus.FITTED:
                _write_error_planes(
                    out,
                    y,
                    x,
                    planes,
                    pixel_fit,
                    freq_arr_hz,
                    ref_freq_for_pixel(ref_freq_hz, y, x),
                    fit_options.n_error_samples,
                )
    out[planes.model] = fit.model.T.reshape(n_freq, cy, cx)
    out[planes.alpha] = fit.alpha.reshape(cy, cx)
    out[planes.order] = fit.order.reshape(cy, cx)
    out[planes.coeff] = fit.coeff.T.reshape(-1, cy, cx)
    out[planes.coeff_error] = fit.coeff_error.T.reshape(-1, cy, cx)
    n_rejected = int(np.count_nonzero(fit.status == FitStatus.UNUSABLE))
    if n_rejected:
        logger.warning(
            f"{n_rejected} of {cy * cx} pixels in this chunk fitted an unusable "
            "Stokes I model and were blanked (see "
            "`rm_lite.utils.fitting.model_is_usable`). Expect this on pixels with "
            "no real Stokes I signal, i.e. when `stokes_i_snr_cut` is None, and "
            "on artefacts, whose Stokes I passes through zero."
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
            value = float(
                _alpha_at_ref(spec, freq_arr_hz, ref_freq_for_pixel(ref_freq_hz, y, x))
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
