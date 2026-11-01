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
    """`eval_model` for the model functions, broadcasting `x` and the params"""
    x_arr, *param_arrs = np.broadcast_arrays(
        np.asarray(x, dtype=np.float64),
        *(np.asarray(p, dtype=np.float64) for p in params),
    )
    params_2d = np.stack([p.ravel() for p in param_arrs])
    model = eval_model(params_2d, x_arr.ravel(), log_model)
    return np.asarray(model.reshape(x_arr.shape))


def best_aic_func(
    aics: NDArray[np.float64], n_param: NDArray[np.integer[Any]]
) -> tuple[float, int, int]:
    """Find the best AIC for a set of AICs using Occam's razor."""
    best_aic_idx = best_aic_index(
        np.asarray(aics, dtype=np.float64), np.asarray(n_param, dtype=np.int64)
    )
    best_aic, best_n = float(aics[best_aic_idx]), int(n_param[best_aic_idx])
    logger.debug(f"Best AIC is {best_aic}, with {best_n} params.")
    return best_aic, best_n, best_aic_idx


class ModelSettings(NamedTuple):
    """Stokes I model and loss, as the compiled fit takes them"""

    log_model: bool
    """Fit `power_law` rather than `polynomial`"""
    f_scale: float
    """Cauchy loss scale in sigma; 0 for a linear loss"""


class RobustCost(NamedTuple):
    """A Stokes I fit's cost at a set of parameters (see `_robust_cost`)"""

    cost: float
    """Cost under the loss"""
    sigma_resid: NDArray[np.float64]
    """Residuals in sigma"""
    scaled_resid: NDArray[np.float64]
    """Residuals scaled for the loss"""
    jac: NDArray[np.float64]
    """Jacobian scaled for the loss"""


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


@numba.njit(cache=True, nogil=True)  # type: ignore[misc]
def eval_model(
    params: NDArray[np.float64], x_arr: NDArray[np.float64], log_model: bool
) -> NDArray[np.float64]:
    """A `power_law` or `polynomial` model, compiled.

    Args:
        params (NDArray[np.float64]): Model parameters, or a row of them for
            each point of `x_arr`.
        x_arr (NDArray[np.float64]): freq / ref_freq.
        log_model (bool): Power law rather than polynomial.

    Returns:
        NDArray[np.float64]: The model at `x_arr`.
    """
    model: NDArray[np.float64] = np.zeros(x_arr.size)
    if log_model:
        log_x = np.log10(x_arr)
        for i in range(1, len(params)):
            model += params[i] * log_x**i
        model = params[0] * 10.0**model
    else:
        for i in range(len(params)):
            model += params[i] * x_arr**i
    return model


@numba.njit(cache=True, nogil=True)  # type: ignore[misc]
def _robust_cost(
    params: NDArray[np.float64],
    basis: NDArray[np.float64],
    stokes_i_arr: NDArray[np.float64],
    sigma_arr: NDArray[np.float64],
    settings: ModelSettings,
) -> RobustCost:
    """Cost of a Stokes I fit at `params`, with the residuals and Jacobian
    scaled for the loss as scipy's `scale_for_robust_loss_function`, so a
    Gauss-Newton step and the covariance match `least_squares`.

    Args:
        params (NDArray[np.float64]): Model parameters.
        basis (NDArray[np.float64]): log10(freq / ref_freq) for a power law,
            freq / ref_freq for a polynomial.
        stokes_i_arr (NDArray[np.float64]): Stokes I.
        sigma_arr (NDArray[np.float64]): 1-sigma errors.
        settings (ModelSettings): Model and loss.

    Returns:
        RobustCost: Cost, residuals, and the scaled residuals and Jacobian.
    """
    # One pass per channel, as lm_fit calls this on every trial step: the model
    # and its Jacobian row, then the loss
    n_param = params.size
    f_scale = settings.f_scale
    eps = np.finfo(np.float64).eps
    sigma_resid = np.empty(basis.size)
    scaled_resid = np.empty(basis.size)
    jac = np.empty((basis.size, n_param))
    cost = 0.0
    for chan in range(basis.size):
        if settings.log_model:
            # p0 * 10**(sum p_i L**i), L = log10(x): d/dp0 = 10**(...), and
            # d/dp_i = model * ln(10) * L**i
            power, log_x_i = 0.0, 1.0
            for i in range(1, n_param):
                log_x_i *= basis[chan]
                power += params[i] * log_x_i
            jac[chan, 0] = np.exp(np.log(10.0) * power)
            model = params[0] * jac[chan, 0]
            log_x_i = 1.0
            for i in range(1, n_param):
                log_x_i *= basis[chan]
                jac[chan, i] = model * np.log(10.0) * log_x_i
        else:
            model, x_i = 0.0, 1.0
            for i in range(n_param):
                jac[chan, i] = x_i
                model += params[i] * x_i
                x_i *= basis[chan]
        sigma_resid[chan] = (model - stokes_i_arr[chan]) / sigma_arr[chan]
        if f_scale > 0:
            # Cauchy: rho(z) = f**2 log(1 + z/f**2), on z = resid**2
            z = (sigma_resid[chan] / f_scale) ** 2
            rho1 = 1 / (1 + z)
            rho2 = -(rho1**2) / f_scale**2
            scale = np.sqrt(max(rho1 + 2 * rho2 * sigma_resid[chan] ** 2, eps))
            cost += 0.5 * f_scale**2 * np.log1p(z)
            scaled_resid[chan] = sigma_resid[chan] * rho1 / scale
        else:
            scale = 1.0
            cost += 0.5 * sigma_resid[chan] ** 2
            scaled_resid[chan] = sigma_resid[chan]
        for i in range(n_param):
            jac[chan, i] *= scale / sigma_arr[chan]
    return RobustCost(cost, sigma_resid, scaled_resid, jac)


@numba.njit(cache=True, nogil=True)  # type: ignore[misc]
def lm_fit(
    x_arr: NDArray[np.float64],
    stokes_i_arr: NDArray[np.float64],
    sigma_arr: NDArray[np.float64],
    initial_guess: NDArray[np.float64],
    settings: ModelSettings,
) -> tuple[NDArray[np.float64], NDArray[np.float64], bool]:
    """Levenberg-Marquardt fit of a Stokes I model, compiled so threads can run it.

    Same optimum as `static_fit`'s `curve_fit` call: residuals divided by
    `sigma_arr`, the Cauchy loss with scipy's robust scaling, and the covariance
    as curve_fit's (pseudo-inverse of the loss-scaled Jacobian, absolute sigma).
    Anything else is left to scipy by returning not converged: no convergence,
    a non-finite value, or `p0` starting or ending on its >= 0 bound.

    Args:
        x_arr (NDArray[np.float64]): freq / ref_freq.
        stokes_i_arr (NDArray[np.float64]): Stokes I.
        sigma_arr (NDArray[np.float64]): 1-sigma errors; ones for an unweighted fit.
        initial_guess (NDArray[np.float64]): Starting parameters.
        settings (ModelSettings): Model and loss.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64], bool]: popt, pcov, and
            whether it converged.
    """
    n_param = initial_guess.size
    params = initial_guess.copy()
    pcov = np.full((n_param, n_param), np.nan)
    if x_arr.size < n_param or params[0] <= 0:
        return params, pcov, False
    basis = np.log10(x_arr) if settings.log_model else x_arr
    current = _robust_cost(params, basis, stokes_i_arr, sigma_arr, settings)
    # Steps are measured against the errors, or against the data where that is
    # smaller (an unweighted fit has sigma = 1, in flux units): the mean squared
    # SNR, capped at 1
    snr_sq = min(1.0, float(np.mean((stokes_i_arr / sigma_arr) ** 2)))
    damping = 1e-3
    converged = False
    for _ in range(100):
        # Columns scaled to unit norm, as scipy's x_scale="jac": the powers of
        # log10(x) are nearly parallel over a band, and J^T J of the raw columns
        # too ill-conditioned for pinv to keep every direction.
        norm = np.sqrt(np.sum(current.jac**2, axis=0))
        norm[norm == 0] = 1.0
        jac = current.jac / norm
        hess = jac.T @ jac
        grad = jac.T @ current.scaled_resid
        if not np.isfinite(current.cost) or not np.all(np.isfinite(hess)):
            return params, pcov, False
        # Converged once a full Gauss-Newton step is under 1e-6 of the errors:
        # it lowers the cost by 0.5 dp.H.dp, and H is the inverse covariance.
        # Tested before stepping, as at the optimum no step lowers it, in
        # floats. pinv (n_param square) rather than solve, which can raise.
        predicted = 0.5 * grad @ np.linalg.pinv(hess) @ grad
        if predicted <= max(0.5 * 1e-6**2 * snr_sq, 1e-15 * current.cost):
            converged = True
            break
        # Marquardt: damp along the curvature, more after a step that failed
        while True:
            damped = hess + damping * np.diag(np.diag(hess))
            if not np.all(np.isfinite(damped)):
                return params, pcov, False
            trial = params - (np.linalg.pinv(damped) @ grad) / norm
            trial[0] = max(trial[0], 0.0)
            trial_cost = _robust_cost(trial, basis, stokes_i_arr, sigma_arr, settings)
            if np.isfinite(trial_cost.cost) and trial_cost.cost <= current.cost:
                break
            damping *= 10
            if damping > 1e16:
                return params, pcov, False
        params, current = trial, trial_cost
        damping = max(damping / 10, 1e-15)
    if not converged or params[0] <= 0:
        return params, pcov, False
    # Many channels past f_scale: the loss is far from quadratic and has other
    # minima, where solvers part ways. Leave those to scipy.
    f_scale = settings.f_scale
    if f_scale > 0 and np.mean(np.abs(current.sigma_resid) > f_scale) > 0.1:
        return params, pcov, False
    # As curve_fit: Moore-Penrose inverse of J^T J, dropping tiny singular values
    _, sing, vt = np.linalg.svd(current.jac, full_matrices=False)
    eps = np.finfo(np.float64).eps
    keep = sing > eps * max(current.jac.shape) * sing[0]
    vt = vt[keep]
    pcov = (vt.T / sing[keep] ** 2) @ vt
    if not np.all(np.isfinite(pcov)):
        return params, pcov, False
    return params, pcov, True


@numba.njit(cache=True, nogil=True)  # type: ignore[misc]
def robust_aic(
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
    n_samples = sigma_resid.size
    if n_samples - n_params - 1 <= 0:
        return np.inf
    if f_scale > 0:
        sigma_resid = np.minimum(np.abs(sigma_resid), f_scale)
    log_likelihood = -0.5 * n_samples * np.log(np.sum(sigma_resid**2) / n_samples)
    aic = 2.0 * (n_params - log_likelihood)
    if n_samples / n_params < 40.0:
        aic += 2.0 * n_params * (n_params + 1.0) / (n_samples - n_params - 1.0)
    return float(aic)


@numba.njit(cache=True, nogil=True)  # type: ignore[misc]
def best_aic_index(aics: NDArray[np.float64], n_param: NDArray[np.int64]) -> int:
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


@numba.njit(cache=True, nogil=True)  # type: ignore[misc]
def spectral_index(
    models: NDArray[np.float64], freq_arr_hz: NDArray[np.float64], ref_freq_hz: float
) -> NDArray[np.float64]:
    """Spectral index (d ln I / d ln nu) of each model (row) at the reference
    frequency, as `np.gradient` and `np.interp` on ascending frequency.

    Args:
        models (NDArray[np.float64]): Model spectra, one per row.
        freq_arr_hz (NDArray[np.float64]): Channel frequencies, in any order.
        ref_freq_hz (float): Reference frequency.

    Returns:
        NDArray[np.float64]: Spectral index of each model.
    """
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


@numba.njit(cache=True, nogil=True)  # type: ignore[misc]
def _compiled_fit_spectrum(
    i_spec: NDArray[np.float64],
    e_spec: NDArray[np.float64],
    x_arr: NDArray[np.float64],
    options: SpectraFitOptions,
) -> tuple[int, NDArray[np.float64], NDArray[np.float64]]:
    """`fit_stokes_i_model` on one spectrum: gates, then `static_fit` or
    `dynamic_fit` with `lm_fit`.

    Args:
        i_spec (NDArray[np.float64]): Stokes I.
        e_spec (NDArray[np.float64]): Errors; zeros for none.
        x_arr (NDArray[np.float64]): freq / ref_freq.
        options (SpectraFitOptions): Fit options.

    Returns:
        tuple[int, NDArray[np.float64], NDArray[np.float64]]: `FitStatus`, and
            popt and pcov if fitted.
    """
    no_popt, no_pcov = np.empty(0), np.empty((0, 0))
    n_coeff = abs(options.fit_order) + 1
    # No positive error anywhere means none given
    weightable = np.isfinite(e_spec) & (e_spec > 0)
    if not weightable.any():
        weightable = np.isfinite(e_spec)
    good = np.isfinite(i_spec) & weightable
    if np.count_nonzero(good) < n_coeff + 1:
        return FitStatus.SKIPPED, no_popt, no_pcov
    i_good, e_good, x_good = i_spec[good], e_spec[good], x_arr[good]
    usable = e_good > 0
    if stokes_i_snr(i_good, e_good) < options.snr_cut:
        return FitStatus.SKIPPED, no_popt, no_pcov
    # Without errors, an unweighted fit with a linear loss
    weighted = usable.all()
    sigma = e_good if weighted else np.ones(i_good.size)
    settings = ModelSettings(
        options.model.log_model, options.model.f_scale if weighted else 0.0
    )
    # Every order up to |fit_order| when it is negative
    first = 0 if options.fit_order < 0 else options.fit_order
    aics = np.full(n_coeff, np.inf)
    converged = np.zeros(n_coeff, dtype=np.bool_)
    fits = []
    for order in range(n_coeff):
        initial_guess = np.zeros(order + 1)
        initial_guess[0] = max(float(np.mean(i_good)), 0.0)
        if order < first:
            fits.append((initial_guess, np.empty((0, 0))))
            continue
        popt, pcov, converged[order] = lm_fit(
            x_good, i_good, sigma, initial_guess, settings
        )
        fits.append((popt, pcov))
        # On the p0 >= 0 bound, scipy's bounded fit lands elsewhere, and its AIC
        # could pick another order. Other orders lm_fit leaves to scipy are
        # scored where it stopped, which is near scipy's fit, and only go to
        # scipy below if they are the one chosen.
        if not converged[order] and popt[0] <= 0:
            return FitStatus.NEEDS_SCIPY, no_popt, no_pcov
        sigma_resid = (eval_model(popt, x_good, settings.log_model) - i_good) / sigma
        aics[order] = robust_aic(sigma_resid, order + 1, settings.f_scale)
    n_param = np.arange(1, n_coeff + 1)
    best = best_aic_index(aics, n_param) if options.fit_order < 0 else first
    if not converged[best]:
        return FitStatus.NEEDS_SCIPY, no_popt, no_pcov
    popt, pcov = fits[best]
    return FitStatus.FITTED, popt, pcov


@numba.njit(cache=True, nogil=True)  # type: ignore[misc]
def fit_stokes_i_spectra(
    spectra: StokesISpectra, options: SpectraFitOptions
) -> SpectraFit:
    """`_fit_stokes_i_block`'s per-pixel fit, compiled, for every spectrum.

    A spectrum `lm_fit` leaves to scipy gets `FitStatus.NEEDS_SCIPY` and
    nothing else, for the caller to fit the Python way.

    Args:
        spectra (StokesISpectra): Stokes I, errors and frequencies.
        options (SpectraFitOptions): Fit options.

    Returns:
        SpectraFit: Models, alpha, order, coefficients and errors, NaN where
            not fitted, and each spectrum's `FitStatus`.
    """
    n_spec, n_freq = spectra.i_rows.shape
    n_coeff = abs(options.fit_order) + 1
    fit = SpectraFit(
        model=np.full((n_spec, n_freq), np.nan),
        alpha=np.full(n_spec, np.nan),
        order=np.full(n_spec, np.nan),
        coeff=np.full((n_spec, n_coeff), np.nan),
        coeff_error=np.full((n_spec, n_coeff), np.nan),
        status=np.zeros(n_spec, dtype=np.int64),
    )
    for spec in range(n_spec):
        e_spec = spectra.err_rows[spec]
        x_arr = spectra.freq_arr_hz / spectra.ref_freq_hz[spec]
        status, popt, pcov = _compiled_fit_spectrum(
            spectra.i_rows[spec], e_spec, x_arr, options
        )
        fit.status[spec] = status
        if status != FitStatus.FITTED:
            continue
        model = eval_model(popt, x_arr, options.model.log_model)
        floor = noise_floor(e_spec, options.model_floor_sigma)
        if not model_is_usable(model, floor):
            fit.status[spec] = FitStatus.UNUSABLE
            continue
        fit.model[spec] = model
        fit.alpha[spec] = spectral_index(
            model.reshape(1, -1), spectra.freq_arr_hz, spectra.ref_freq_hz[spec]
        )[0]
        fit.order[spec] = popt.size - 1
        fit.coeff[spec] = 0.0
        fit.coeff[spec, : popt.size] = popt
        fit.coeff_error[spec] = 0.0
        fit.coeff_error[spec, : popt.size] = np.sqrt(np.diag(pcov))
    return fit


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

    `robust_loss` is passed to `scipy.optimize.least_squares` through
    `curve_fit`, so a channel more than `f_scale` sigma out is downweighted
    instead of dragging the model. The bounds already pick `method="trf"`, the
    one method that takes a loss.
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
    # Callers pass all-zero errors to mean "no error given". Without an error
    # there is no sigma for `f_scale` to count, so the loss goes back to plain.
    x_arr = freq_arr_hz / ref_freq_hz
    sigma_arr: NDArray[np.float64] | None = stokes_i_error_arr
    if not bool(np.all(np.isfinite(stokes_i_error_arr) & (stokes_i_error_arr > 0))):
        sigma_arr = None
        robust_loss = "linear"
    loss_kwargs: dict[str, Any] = (
        {} if robust_loss == "linear" else {"loss": robust_loss, "f_scale": f_scale}
    )
    fitted = True
    try:
        # Compiled first, so threads can run it; scipy for any fit it leaves
        popt, pcov, converged = lm_fit(
            np.asarray(x_arr, dtype=np.float64),
            np.asarray(stokes_i_arr, dtype=np.float64),
            np.ones(x_arr.size)
            if sigma_arr is None
            else np.asarray(sigma_arr, dtype=np.float64),
            initial_guess,
            ModelSettings(
                fit_function == "log", f_scale if robust_loss == "cauchy" else 0.0
            ),
        )
        if not converged:
            popt, pcov = optimize.curve_fit(
                fit_func,
                x_arr,
                stokes_i_arr,
                sigma=sigma_arr,
                absolute_sigma=True,
                p0=initial_guess,
                bounds=bounds,
                **loss_kwargs,
            )
    except (ValueError, RuntimeError) as e:
        logger.warning(f"Stokes I fit with errors failed ({e}); retrying unweighted.")
        try:
            popt, pcov = optimize.curve_fit(
                fit_func,
                x_arr,
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
            fitted = False
    stokes_i_model_arr = fit_func(x_arr, *popt)
    # Nothing converged, so there is no fit for `dynamic_fit` to rank.
    resid = stokes_i_arr - stokes_i_model_arr
    aic = (
        robust_aic(
            np.asarray(resid if sigma_arr is None else resid / sigma_arr, dtype=float),
            fit_order + 1,
            f_scale if robust_loss != "linear" else 0.0,
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


@numba.njit(cache=True, nogil=True)  # type: ignore[misc]
def stokes_i_snr(i_spec: NDArray[np.float64], e_spec: NDArray[np.float64]) -> float:
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


def model_noise_floor(
    stokes_i_error_arr: NDArray[np.float64] | None, sigma: float
) -> float:
    """`noise_floor`, or 0.0 without an error spectrum"""
    if stokes_i_error_arr is None:
        return 0.0
    return float(noise_floor(np.asarray(stokes_i_error_arr, dtype=np.float64), sigma))


@numba.njit(cache=True, nogil=True)  # type: ignore[misc]
def noise_floor(stokes_i_error_arr: NDArray[np.float64], sigma: float) -> float:
    """Flux `sigma` times the band-averaged noise, `median(error)/sqrt(n)`.

    Same noise as `stokes_i_snr`, so this and `snr_cut` share units. 0.0 without
    a usable error.
    """
    usable = np.isfinite(stokes_i_error_arr) & (stokes_i_error_arr > 0)
    n = np.count_nonzero(usable)
    if sigma <= 0 or not n:
        return 0.0
    return float(sigma * nb_median(stokes_i_error_arr[usable]) / np.sqrt(n))


@numba.njit(cache=True, nogil=True)  # type: ignore[misc]
def model_is_usable(model: NDArray[np.float64], noise_floor: float = 0.0) -> bool:
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
    return np.stack([fit.stokes_i_model_func(x_arr, *sample) for sample in samples])


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
    is inf (see `robust_aic`), so a negative `fit_order` settles on a lower order
    there. A bad flux is left to `options.robust_loss` in `static_fit`.
    """
    fit_order = options.fit_order
    # An error of zero or less cannot weight a fit, so those channels go. All of
    # them at once is how the callers say "no error given", and then the whole
    # spectrum is kept and fitted unweighted.
    weightable = np.isfinite(stokes_i_error_arr) & (stokes_i_error_arr > 0)
    if not weightable.any():
        weightable = np.isfinite(stokes_i_error_arr)
    good = np.isfinite(stokes_i_arr) & weightable
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
) -> NDArray[np.float64]:
    """`spectral_index` of `spec`, or of each of its rows"""
    rows = np.asarray(spec, dtype=np.float64).reshape(-1, spec.shape[-1])
    alpha = spectral_index(rows, np.asarray(freq_arr_hz, dtype=np.float64), ref_freq_hz)
    return np.asarray(alpha.reshape(spec.shape[:-1]))


class PixelFit(NamedTuple):
    """One pixel's Stokes I fit within a chunk (see `_iter_pixel_fits`)."""

    y: int
    """y pixel"""
    x: int
    """x pixel"""
    e_spec: NDArray[np.float64]
    """The pixel's error spectrum, for the noise floor."""
    fit: FitResult | None
    """The fit, or None if the pixel was skipped (too few channels / low SNR)."""


def _iter_pixel_fits(
    i_block: NDArray[np.float64],
    err_block: NDArray[np.float64] | None,
    err_1d: NDArray[np.float64] | None,
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
    n_freq = i_block.shape[0]
    for y, x in pixels:
        i_spec = i_block[:, y, x]
        e_spec = _pixel_stokes_i_error(err_block, err_1d, n_freq, y, x)
        fit = fit_stokes_i_model(
            freq_arr_hz=freq_arr_hz,
            ref_freq_hz=ref_freq_for_pixel(ref_freq_hz, y, x),
            stokes_i_arr=i_spec,
            stokes_i_error_arr=e_spec,
            options=fit_options,
        )
        yield PixelFit(y, x, e_spec, fit)


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


def _fit_block_compiled(
    out: NDArray[np.float64],
    planes: BlockPlanes,
    i_block: NDArray[np.float64],
    err_block: NDArray[np.float64] | None,
    err_1d: NDArray[np.float64] | None,
    freq_arr_hz: NDArray[np.float64],
    ref_freq_hz: RefFreqHz,
    fit_options: StokesIFitOptions,
) -> tuple[int, list[tuple[int, int]]]:
    """Fit every pixel of a block with `fit_stokes_i_spectra`, into `out`.

    Args:
        out (NDArray[np.float64]): Output planes (see `_block_planes`).
        planes (BlockPlanes): Where each output goes in `out`.
        i_block (NDArray[np.float64]): Stokes I (n_freq, cy, cx).
        err_block (NDArray[np.float64] | None): Errors, as `i_block`.
        err_1d (NDArray[np.float64] | None): One error spectrum for every pixel.
        freq_arr_hz (NDArray[np.float64]): Channel frequencies.
        ref_freq_hz (RefFreqHz): Reference frequency, or one per pixel.
        fit_options (StokesIFitOptions): Fit options.

    Returns:
        tuple[int, list[tuple[int, int]]]: Pixels with an unusable model, and
            the (y, x) of those left to the Python fit.
    """
    n_freq, cy, cx = i_block.shape
    if err_block is not None:
        err_rows = err_block.reshape(n_freq, -1).T
    else:
        # One error spectrum for every pixel, without copying it
        err_1d = np.zeros(n_freq) if err_1d is None else err_1d
        err_rows = np.broadcast_to(err_1d, (cy * cx, n_freq))
    fit = fit_stokes_i_spectra(
        StokesISpectra(
            i_rows=np.ascontiguousarray(
                i_block.reshape(n_freq, -1).T, dtype=np.float64
            ),
            err_rows=np.asarray(err_rows, dtype=np.float64),
            freq_arr_hz=np.asarray(freq_arr_hz, dtype=np.float64),
            ref_freq_hz=np.broadcast_to(
                np.asarray(ref_freq_hz, dtype=np.float64), (cy, cx)
            )
            .reshape(-1)
            .copy(),
        ),
        SpectraFitOptions(
            fit_order=fit_options.fit_order,
            model=ModelSettings(
                log_model=fit_options.fit_function == "log",
                f_scale=fit_options.f_scale
                if fit_options.robust_loss == "cauchy"
                else 0.0,
            ),
            snr_cut=-np.inf if fit_options.snr_cut is None else fit_options.snr_cut,
            model_floor_sigma=fit_options.model_floor_sigma,
        ),
    )
    out[planes.model] = fit.model.T.reshape(n_freq, cy, cx)
    out[planes.alpha] = fit.alpha.reshape(cy, cx)
    out[planes.order] = fit.order.reshape(cy, cx)
    out[planes.coeff] = fit.coeff.T.reshape(-1, cy, cx)
    out[planes.coeff_error] = fit.coeff_error.T.reshape(-1, cy, cx)
    needs_scipy = np.flatnonzero(fit.status == FitStatus.NEEDS_SCIPY)
    return (
        int(np.count_nonzero(fit.status == FitStatus.UNUSABLE)),
        [divmod(int(pixel), cx) for pixel in needs_scipy],
    )


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
    err_block = arrays[1] if n_data > 1 else None
    n_freq, cy, cx = i_block.shape
    planes = _block_planes(
        n_freq, abs(fit_options.fit_order) + 1, fit_options.compute_model_error
    )
    out = np.full((planes.n_out, cy, cx), np.nan, dtype=np.float64)
    # The model errors need each fit's covariance, so every pixel takes the
    # per-pixel path. Otherwise only those the compiled fit leaves to scipy do.
    if fit_options.compute_model_error:
        n_rejected, pixels = 0, list(np.ndindex(cy, cx))
    else:
        n_rejected, pixels = _fit_block_compiled(
            out,
            planes,
            i_block,
            err_block,
            err_1d,
            freq_arr_hz,
            ref_freq_hz,
            fit_options,
        )
    # The 1D fitter logs per fit and per failure. At cube scale that floods, so
    # quiet it to at least ERROR whatever the caller's log_level.
    with quiet_logs(max(log_level, logging.ERROR)):
        for y, x, e_spec, fit in _iter_pixel_fits(
            i_block, err_block, err_1d, freq_arr_hz, ref_freq_hz, fit_options, pixels
        ):
            if fit is None:
                continue
            pixel_ref_hz = ref_freq_for_pixel(ref_freq_hz, y, x)
            model = fit.stokes_i_model_func(
                freq_arr_hz / pixel_ref_hz, *np.asarray(fit.popt)
            )
            noise_floor = model_noise_floor(e_spec, fit_options.model_floor_sigma)
            # The whole model divides Q/U, including channels Stokes I had
            # flagged and Q/U did not, so all of it has to be usable.
            if not model_is_usable(model, noise_floor):
                n_rejected += 1
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
