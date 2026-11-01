"""Tests for the compiled Stokes I fit, against scipy's curve_fit and the
per-pixel Python fit."""

from __future__ import annotations

import logging
from collections.abc import Callable
from importlib import resources
from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np
import pytest
from numpy.typing import ArrayLike
from rm_lite.utils import fitting
from rm_lite.utils.fitting import (
    FitStatus,
    ModelSettings,
    SpectraFitOptions,
    StokesIFitOptions,
    StokesISpectra,
    dynamic_fit,
    fit_stokes_i_model,
    fit_stokes_i_spectra,
    lm_fit,
    static_fit,
)
from scipy import stats

pytestmark = [
    pytest.mark.filterwarnings("ignore::scipy.optimize.OptimizeWarning"),
    # Uncompiled (NUMBA_DISABLE_JIT), NumPy warns where a trial step overflows
    # (and then multiplies inf by 0), which compiled code does silently and
    # lm_fit then rejects
    pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning"),
    pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning"),
]


def random_spectrum(
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """A curved power law with noise (sometimes flat), sometimes with outlying
    channels"""
    n_chan = int(rng.choice([36, 96, 288]))
    freq_arr_hz = np.linspace(0.8e9, 1.088e9, n_chan)
    ref_freq_hz = float(np.mean(freq_arr_hz))
    x_arr = freq_arr_hz / ref_freq_hz
    flux = 10 ** rng.uniform(-3, 0)
    alpha, beta = rng.normal(-0.7, 0.3), rng.normal(0, 0.5)
    if rng.random() < 0.2:
        alpha = beta = 0.0
    error_arr = np.full(n_chan, flux / 10 ** rng.uniform(0.5, 2.5))
    stokes_i_arr = flux * x_arr ** (alpha + beta * np.log10(x_arr))
    stokes_i_arr += rng.normal(size=n_chan) * error_arr
    if rng.random() < 0.2:
        bad = rng.random(n_chan) < 0.03
        stokes_i_arr[bad] += rng.normal(0, 20, bad.sum()) * error_arr[bad]
    return freq_arr_hz, ref_freq_hz, stokes_i_arr, error_arr


ScipyFitsOnly: TypeAlias = Callable[[pytest.MonkeyPatch], None]


def assert_within_sigma(
    got: ArrayLike, expected: ArrayLike, pcov: ArrayLike, err_msg: str = ""
) -> None:
    """Parameters within 1e-2 of each reference parameter's error"""
    np.testing.assert_array_less(
        np.abs(np.asarray(got) - np.asarray(expected)),
        1e-2 * np.sqrt(np.diag(pcov)) + 1e-300,
        err_msg=err_msg,
    )


@pytest.mark.parametrize("fit_function", ["log", "linear"])
@pytest.mark.parametrize("robust_loss", ["cauchy", "linear"])
def test_compiled_fit_matches_scipy(
    fit_function: Literal["log", "linear"],
    robust_loss: Literal["cauchy", "linear"],
    monkeypatch: pytest.MonkeyPatch,
    scipy_fits_only: ScipyFitsOnly,
) -> None:
    """Same optimum and errors as curve_fit, to well within each parameter's
    uncertainty (or scipy's own fit, where the compiled one leaves it)."""
    rng = np.random.default_rng([ord(c) for c in fit_function + robust_loss])
    for spec in range(100):
        freq_arr_hz, ref_freq_hz, stokes_i_arr, error_arr = random_spectrum(rng)
        args = (freq_arr_hz, ref_freq_hz, stokes_i_arr, error_arr)
        # Above order 3 scipy's own polynomial fit strays from the optimum (see
        # the next test)
        order = int(rng.integers(0, 6 if fit_function == "log" else 4))
        compiled = static_fit(*args, order, fit_function, robust_loss)
        with monkeypatch.context() as patch:
            scipy_fits_only(patch)
            reference = static_fit(*args, order, fit_function, robust_loss)
        msg = f"spectrum {spec}"
        assert_within_sigma(compiled.popt, reference.popt, reference.pcov, msg)
        np.testing.assert_allclose(
            np.sqrt(np.diag(compiled.pcov)),
            np.sqrt(np.diag(reference.pcov)),
            rtol=1e-3,
            atol=1e-300,
            err_msg=msg,
        )
        assert compiled.aic == pytest.approx(reference.aic, abs=1e-3), msg


def test_compiled_polynomial_fit_is_the_least_squares_solution() -> None:
    """With a linear loss the polynomial fit is linear least squares, so its
    optimum is exact, up to order 5 where curve_fit misses it by up to 0.6 sigma"""
    rng = np.random.default_rng(13)
    n_unbounded = n_compiled = 0
    for spec in range(100):
        freq_arr_hz, ref_freq_hz, stokes_i_arr, error_arr = random_spectrum(rng)
        x_arr = freq_arr_hz / ref_freq_hz
        order = int(rng.integers(0, 6))
        design = np.vander(x_arr, order + 1, increasing=True) / error_arr[:, None]
        norm = np.linalg.norm(design, axis=0)
        exact = np.linalg.lstsq(design / norm, stokes_i_arr / error_arr, rcond=None)[0]
        if exact[0] <= 0:
            continue  # beyond the c0 >= 0 bound, so scipy's to fit
        n_unbounded += 1
        popt, pcov, converged = lm_fit(
            x_arr,
            stokes_i_arr,
            error_arr,
            np.r_[max(stokes_i_arr.mean(), 0.0), np.zeros(order)],
            ModelSettings(log_model=False, f_scale=0.0),
        )
        if converged:
            n_compiled += 1
            np.testing.assert_array_less(
                np.abs(popt - exact / norm),
                1e-4 * np.sqrt(np.diag(pcov)),
                err_msg=f"spectrum {spec}",
            )
    # The rest are high orders whose path crosses the bound on the way to an
    # optimum inside it, which scipy's bounded solver handles
    assert n_compiled >= 0.85 * n_unbounded


def test_compiled_fit_handles_the_default_fit() -> None:
    """The default fit (log, order 2, Cauchy) of a curved power law is
    compiled, not left to scipy, for nearly every spectrum."""
    rng = np.random.default_rng(3)
    n_spectra, n_compiled = 200, 0
    for _ in range(n_spectra):
        freq_arr_hz, ref_freq_hz, stokes_i_arr, error_arr = random_spectrum(rng)
        n_compiled += lm_fit(
            freq_arr_hz / ref_freq_hz,
            stokes_i_arr,
            error_arr,
            np.array([max(stokes_i_arr.mean(), 0.0), 0.0, 0.0]),
            ModelSettings(log_model=True, f_scale=3.0),
        )[2]
    assert n_compiled >= 0.98 * n_spectra


def test_dynamic_fit_picks_the_same_order(
    monkeypatch: pytest.MonkeyPatch, scipy_fits_only: ScipyFitsOnly
) -> None:
    rng = np.random.default_rng(7)
    for spec in range(40):
        args = random_spectrum(rng)
        compiled = dynamic_fit(*args, fit_order=3)
        with monkeypatch.context() as patch:
            scipy_fits_only(patch)
            reference = dynamic_fit(*args, fit_order=3)
        assert np.size(compiled.popt) == np.size(reference.popt), f"spectrum {spec}"
        assert_within_sigma(
            compiled.popt, reference.popt, reference.pcov, f"spectrum {spec}"
        )


def test_dynamic_orders_stay_compiled() -> None:
    """A dynamic-order fit stays compiled even where a low order does not
    converge (a flat model of a curved spectrum leaves many channels past
    f_scale), as long as that order loses"""
    rng = np.random.default_rng(17)
    n_freq, n_spec = 96, 100
    freq_arr_hz = np.linspace(0.8e9, 1.088e9, n_freq)
    x_arr = freq_arr_hz / np.mean(freq_arr_hz)
    flux = 10 ** rng.uniform(-2, 0, (n_spec, 1))
    alpha, beta = rng.normal(-0.7, 0.3, (n_spec, 1)), rng.normal(0, 0.5, (n_spec, 1))
    error = flux / 10 ** rng.uniform(1, 2.5, (n_spec, 1))
    i_rows = flux * x_arr ** (alpha + beta * np.log10(x_arr))
    i_rows += rng.normal(size=i_rows.shape) * error
    fit = fit_stokes_i_spectra(
        StokesISpectra(
            i_rows=i_rows,
            err_rows=np.broadcast_to(error, i_rows.shape),
            freq_arr_hz=freq_arr_hz,
            ref_freq_hz=np.full(n_spec, np.mean(freq_arr_hz)),
        ),
        SpectraFitOptions(
            fit_order=-3,
            model=ModelSettings(log_model=True, f_scale=3.0),
            snr_cut=-np.inf,
            model_floor_sigma=1.0,
        ),
    )
    assert np.mean(fit.status == FitStatus.FITTED) >= 0.95


def test_compiled_fit_leaves_a_bounded_fit_to_scipy() -> None:
    """A negative spectrum puts p0 on its >= 0 bound: scipy's to handle"""
    x_arr = np.linspace(0.8, 1.2, 50)
    popt, _, converged = lm_fit(
        x_arr,
        -np.ones(50),
        np.full(50, 0.1),
        np.zeros(3),
        ModelSettings(log_model=True, f_scale=3.0),
    )
    assert not converged
    assert popt[0] == 0.0


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("name", ["bad_fit", "overflow"])
def test_real_spectra_fit_as_scipy(
    name: str, monkeypatch: pytest.MonkeyPatch, scipy_fits_only: ScipyFitsOnly
) -> None:
    """Real Stokes I spectra that once broke the fit"""
    data_path = Path(str(resources.files("rm_lite.data.tests")))
    stokes_i_arr = np.load(data_path / f"stokes_i_arr_{name}.npy")
    error_arr = np.load(data_path / "stokes_i_error_arr_bad.npy")
    freq_arr_hz = np.linspace(1116.0237779633926, 3116.97610232475, len(stokes_i_arr))
    args = (
        freq_arr_hz * 1e6,
        float(np.mean(freq_arr_hz)) * 1e6,
        stokes_i_arr,
        error_arr,
    )
    options = StokesIFitOptions(fit_order=-5, snr_cut=None)
    compiled = fit_stokes_i_model(*args, options=options)
    with monkeypatch.context() as patch:
        scipy_fits_only(patch)
        reference = fit_stokes_i_model(*args, options=options)
    assert compiled is not None
    assert reference is not None
    assert np.size(compiled.popt) == np.size(reference.popt)
    assert_within_sigma(compiled.popt, reference.popt, reference.pcov)


def test_model_functions_take_parameter_planes() -> None:
    """Coefficient planes broadcast against a frequency axis to give a cube"""
    rng = np.random.default_rng(5)
    x_arr = np.linspace(0.8, 1.2, 12)[:, None, None]
    flux, alpha, beta = rng.uniform(0.1, 1, (4, 5)), *rng.normal(size=(2, 4, 5))
    np.testing.assert_allclose(
        fitting.power_law(2)(x_arr, flux, alpha, beta),
        flux * x_arr ** (alpha + beta * np.log10(x_arr)),
    )
    np.testing.assert_allclose(
        fitting.polynomial(2)(x_arr, flux, alpha, beta),
        flux + alpha * x_arr + beta * x_arr**2,
    )


@pytest.mark.parametrize("fit_function", ["log", "linear"])
def test_model_samples_match_one_at_a_time(
    fit_function: Literal["log", "linear"], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Models and alphas for all samples at once are those of each sample"""
    rng = np.random.default_rng(11)
    freq_arr_hz, ref_freq_hz, stokes_i_arr, error_arr = random_spectrum(rng)
    fit = static_fit(freq_arr_hz, ref_freq_hz, stokes_i_arr, error_arr, 2, fit_function)
    x_arr = freq_arr_hz / ref_freq_hz
    params = rng.multivariate_normal(np.asarray(fit.popt), np.asarray(fit.pcov), 200)

    class FixedDraws:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def rvs(self, _size: int) -> np.ndarray:
            return params

    monkeypatch.setattr(stats, "multivariate_normal", FixedDraws)
    samples = fitting.draw_model_samples(fit, x_arr, 200)
    np.testing.assert_array_equal(
        samples, [fit.stokes_i_model_func(x_arr, *p) for p in params]
    )
    np.testing.assert_array_equal(
        fitting._alpha_at_ref(samples, freq_arr_hz, ref_freq_hz),
        [fitting._alpha_at_ref(s, freq_arr_hz, ref_freq_hz) for s in samples],
    )


@pytest.mark.parametrize(
    "options",
    [
        StokesIFitOptions(),
        StokesIFitOptions(fit_order=-3),
        StokesIFitOptions(fit_function="linear", robust_loss="linear"),
        StokesIFitOptions(fit_order=1, snr_cut=None, model_floor_sigma=0.0),
    ],
    ids=["default", "dynamic", "linear", "no-cut"],
)
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_block_fit_matches_per_pixel_fit(
    options: StokesIFitOptions,
    dtype: type,
    monkeypatch: pytest.MonkeyPatch,
    scipy_fits_only: ScipyFitsOnly,
) -> None:
    """The compiled block fit gives what fitting each pixel in Python does"""
    rng = np.random.default_rng(5)
    n_freq, cy, cx = 96, 6, 7
    freq_arr_hz = np.linspace(744e6, 1032e6, n_freq)
    x_arr = (freq_arr_hz / np.median(freq_arr_hz))[:, None, None]
    flux = 10 ** rng.uniform(-3, 0, (cy, cx))
    alpha, beta = rng.normal(-0.7, 0.3, (cy, cx)), rng.normal(0, 0.5, (cy, cx))
    i_block = flux * x_arr ** (alpha + beta * np.log10(x_arr))
    i_block += rng.normal(0, 1e-3, i_block.shape)
    i_block[rng.random(i_block.shape) < 0.02] = np.nan  # flagged channels
    i_block[:, 0, 0] = np.nan  # a blanked pixel
    i_block[rng.random(i_block.shape) < 0.01] += 0.05  # outliers
    i_block[:, 0, 1] = rng.normal(0, 1e-3, n_freq)  # noise only
    i_block[:, 1, 0] = 0.1 - 0.09 * freq_arr_hz / 900e6  # dips below zero
    i_block[::3, 1, 2] += 0.05  # a third of channels out: lm_fit leaves to scipy
    err_block = np.full(i_block.shape, 1e-3)
    err_block[:, 1, 1] = 0.0  # no errors given
    kwargs: dict[str, Any] = {
        "freq_arr_hz": freq_arr_hz,
        "ref_freq_hz": float(np.median(freq_arr_hz)),
        "err_1d": None,
        "fit_options": options,
        "log_level": logging.ERROR,
        "has_ref_block": True,
    }
    # A per-pixel reference frequency, as rmsynth_3d passes it
    blocks: tuple[np.ndarray, ...] = (
        i_block.astype(dtype),
        err_block.astype(dtype),
        rng.uniform(800e6, 950e6, (cy, cx)),
    )
    compiled = fitting._fit_stokes_i_block(*blocks, **kwargs)
    with monkeypatch.context() as patch:
        # Every pixel through the Python path, which still runs lm_fit first
        scipy_fits_only(patch)
        patch.setattr(fitting, "lm_fit", lm_fit)
        reference = fitting._fit_stokes_i_block(*blocks, **kwargs)
    np.testing.assert_array_equal(np.isnan(compiled), np.isnan(reference))
    assert np.isfinite(compiled).any()
    planes = fitting._block_planes(n_freq, abs(options.fit_order) + 1, False)
    np.testing.assert_array_equal(compiled[planes.order], reference[planes.order])
    for plane, rtol in (
        (planes.model, 1e-5),
        (planes.alpha, 1e-4),
        (planes.coeff, 1e-4),
        (planes.coeff_error, 1e-3),
    ):
        np.testing.assert_allclose(
            compiled[plane], reference[plane], rtol=rtol, atol=1e-6, equal_nan=True
        )
