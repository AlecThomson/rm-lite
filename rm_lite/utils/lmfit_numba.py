"""Levenberg-Marquardt fit of the Stokes I models, compiled with numba"""

from __future__ import annotations

from typing import NamedTuple

import numba
import numpy as np
from numpy.typing import NDArray


class ModelSettings(NamedTuple):
    """Stokes I model and loss, as the compiled fit takes them"""

    log_model: bool
    """Fit `power_law` rather than `polynomial`"""
    f_scale: float
    """Cauchy loss scale in sigma; 0 for a linear loss"""


class _RobustCost(NamedTuple):
    """A Stokes I fit's cost at a set of parameters (see `_robust_cost`)"""

    cost: float
    """Cost under the loss"""
    sigma_resid: NDArray[np.float64]
    """Residuals in sigma"""
    scaled_resid: NDArray[np.float64]
    """Residuals scaled for the loss"""
    jac: NDArray[np.float64]
    """Jacobian scaled for the loss"""


class LMFit(NamedTuple):
    """Result of `lm_fit`"""

    popt: NDArray[np.float64]
    """Parameters"""
    pcov: NDArray[np.float64]
    """Their covariance, NaN if not converged"""
    converged: bool
    """Whether the fit converged; if not, scipy's fit is the one to use"""


@numba.njit(cache=True, nogil=True, error_model="numpy")  # type: ignore[misc]
def _robust_cost(
    params: NDArray[np.float64],
    basis: NDArray[np.float64],
    stokes_i_arr: NDArray[np.float64],
    sigma_arr: NDArray[np.float64],
    settings: ModelSettings,
) -> _RobustCost:
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
        _RobustCost: Cost, residuals, and the scaled residuals and Jacobian.
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
    return _RobustCost(cost, sigma_resid, scaled_resid, jac)


@numba.njit(cache=True, nogil=True, error_model="numpy")  # type: ignore[misc]
def lm_fit(
    x_arr: NDArray[np.float64],
    stokes_i_arr: NDArray[np.float64],
    sigma_arr: NDArray[np.float64],
    initial_guess: NDArray[np.float64],
    settings: ModelSettings,
) -> LMFit:
    """Levenberg-Marquardt fit of a Stokes I model, compiled so threads can run it.

    Same optimum as `static_fit`'s `curve_fit` call: residuals divided by
    `sigma_arr`, the Cauchy loss with scipy's robust scaling, and the covariance
    as curve_fit's (pseudo-inverse of the loss-scaled Jacobian, absolute sigma).
    It returns not converged, for scipy to fit, when: `p0` starts or ends on its
    >= 0 bound, there are fewer channels than parameters, the cost or curvature
    is not finite, the damping runs away, 100 steps do not converge, more than
    10% of channels sit past `f_scale`, or the covariance is not finite.

    Args:
        x_arr (NDArray[np.float64]): freq / ref_freq.
        stokes_i_arr (NDArray[np.float64]): Stokes I.
        sigma_arr (NDArray[np.float64]): 1-sigma errors; ones for an unweighted fit.
        initial_guess (NDArray[np.float64]): Starting parameters.
        settings (ModelSettings): Model and loss.

    Returns:
        LMFit: popt, pcov, and whether it converged.
    """
    n_param = initial_guess.size
    params = initial_guess.copy()
    pcov = np.full((n_param, n_param), np.nan)
    if x_arr.size < n_param or params[0] <= 0:
        return LMFit(params, pcov, False)
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
            return LMFit(params, pcov, False)
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
                return LMFit(params, pcov, False)
            trial = params - (np.linalg.pinv(damped) @ grad) / norm
            trial[0] = max(trial[0], 0.0)
            trial_cost = _robust_cost(trial, basis, stokes_i_arr, sigma_arr, settings)
            if np.isfinite(trial_cost.cost) and trial_cost.cost <= current.cost:
                break
            damping *= 10
            if damping > 1e16:
                return LMFit(params, pcov, False)
        params, current = trial, trial_cost
        damping = max(damping / 10, 1e-15)
    if not converged or params[0] <= 0:
        return LMFit(params, pcov, False)
    # Many channels past f_scale: the loss is far from quadratic and has other
    # minima, where solvers part ways. Leave those to scipy.
    f_scale = settings.f_scale
    if f_scale > 0 and np.mean(np.abs(current.sigma_resid) > f_scale) > 0.1:
        return LMFit(params, pcov, False)
    pcov = _curve_fit_pcov(current.jac)
    return LMFit(params, pcov, bool(np.all(np.isfinite(pcov))))


@numba.njit(cache=True, nogil=True, error_model="numpy")  # type: ignore[misc]
def _curve_fit_pcov(jac: NDArray[np.float64]) -> NDArray[np.float64]:
    """Covariance as `curve_fit` makes it: the Moore-Penrose inverse of J^T J,
    dropping tiny singular values.

    Args:
        jac (NDArray[np.float64]): Jacobian, scaled for the loss.

    Returns:
        NDArray[np.float64]: Parameter covariance.
    """
    _, sing, vt = np.linalg.svd(jac, full_matrices=False)
    keep = sing > np.finfo(np.float64).eps * max(jac.shape) * sing[0]
    vt = vt[keep]
    pcov: NDArray[np.float64] = (vt.T / sing[keep] ** 2) @ vt
    return pcov
