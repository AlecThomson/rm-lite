"""Tests for the compiled Stokes I fit (`lm_fit`), against scipy's curve_fit."""

from __future__ import annotations

import logging
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest
from rm_lite.utils import fitting
from rm_lite.utils.fitting import (
    StokesIFitOptions,
    dynamic_fit,
    fit_stokes_i_model,
    lm_fit,
    static_fit,
)
from scipy import stats

pytestmark = pytest.mark.filterwarnings("ignore::scipy.optimize.OptimizeWarning")


def random_spectrum(
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """A curved power law with noise, sometimes with outlying channels"""
    n_chan = int(rng.choice([36, 96, 288]))
    freq_arr_hz = np.linspace(0.8e9, 1.088e9, n_chan)
    ref_freq_hz = float(np.mean(freq_arr_hz))
    x_arr = freq_arr_hz / ref_freq_hz
    flux = 10 ** rng.uniform(-3, 0)
    alpha, beta = rng.normal(-0.7, 0.3), rng.normal(0, 0.5)
    error_arr = np.full(n_chan, flux / 10 ** rng.uniform(0.5, 2.5))
    stokes_i_arr = flux * x_arr ** (alpha + beta * np.log10(x_arr))
    stokes_i_arr += rng.normal(size=n_chan) * error_arr
    if rng.random() < 0.2:
        bad = rng.random(n_chan) < 0.03
        stokes_i_arr[bad] += rng.normal(0, 20, bad.sum()) * error_arr[bad]
    return freq_arr_hz, ref_freq_hz, stokes_i_arr, error_arr


def scipy_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make `static_fit` skip the compiled fit and use curve_fit"""
    monkeypatch.setattr(fitting, "lm_fit", lambda *args: (args[3], None, False))


@pytest.mark.parametrize("fit_function", ["log", "linear"])
@pytest.mark.parametrize("robust_loss", ["cauchy", "linear"])
def test_compiled_fit_matches_scipy(
    fit_function: Literal["log", "linear"],
    robust_loss: Literal["cauchy", "linear"],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same optimum and errors as curve_fit, to well within each parameter's
    uncertainty (or scipy's own fit, where the compiled one leaves it)."""
    rng = np.random.default_rng(len(fit_function) + len(robust_loss))
    for _ in range(100):
        freq_arr_hz, ref_freq_hz, stokes_i_arr, error_arr = random_spectrum(rng)
        args = (freq_arr_hz, ref_freq_hz, stokes_i_arr, error_arr)
        order = int(rng.integers(0, 4))
        compiled = static_fit(*args, order, fit_function, robust_loss)
        with monkeypatch.context() as patch:
            scipy_only(patch)
            reference = static_fit(*args, order, fit_function, robust_loss)
        sigma = np.sqrt(np.diag(reference.pcov))
        np.testing.assert_array_less(
            np.abs(np.asarray(compiled.popt) - np.asarray(reference.popt)),
            1e-2 * sigma + 1e-300,
        )
        np.testing.assert_allclose(
            np.sqrt(np.diag(compiled.pcov)), sigma, rtol=1e-3, atol=1e-300
        )
        assert compiled.aic == pytest.approx(reference.aic, abs=1e-3)


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
            True,
            3.0,
        )[2]
    assert n_compiled >= 0.98 * n_spectra


def test_dynamic_fit_picks_the_same_order(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(7)
    for _ in range(40):
        args = random_spectrum(rng)
        compiled = dynamic_fit(*args, fit_order=3)
        with monkeypatch.context() as patch:
            scipy_only(patch)
            reference = dynamic_fit(*args, fit_order=3)
        assert np.size(compiled.popt) == np.size(reference.popt)


def test_compiled_fit_leaves_a_bounded_fit_to_scipy() -> None:
    """A negative spectrum puts p0 on its >= 0 bound: scipy's to handle"""
    x_arr = np.linspace(0.8, 1.2, 50)
    popt, _, converged = lm_fit(
        x_arr, -np.ones(50), np.full(50, 0.1), np.zeros(3), True, 3.0
    )
    assert not converged
    assert popt[0] == 0.0


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("name", ["bad_fit", "overflow"])
def test_real_spectra_fit_as_scipy(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
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
        scipy_only(patch)
        reference = fit_stokes_i_model(*args, options=options)
    assert compiled is not None
    assert reference is not None
    assert np.size(compiled.popt) == np.size(reference.popt)
    sigma = np.sqrt(np.diag(reference.pcov))
    np.testing.assert_array_less(
        np.abs(np.asarray(compiled.popt) - np.asarray(reference.popt)),
        1e-2 * sigma + 1e-300,
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


def scipy_spectra(i_rows: np.ndarray, *args: object) -> tuple[np.ndarray, ...]:
    """A `fit_stokes_i_spectra` that leaves every spectrum to the per-pixel path"""
    n_spec, n_freq = i_rows.shape
    n_coeff = abs(int(args[3])) + 1  # type: ignore[call-overload]
    coeff = np.full((n_spec, n_coeff), np.nan)
    model = np.full((n_spec, n_freq), np.nan)
    return model, model[:, 0], model[:, 0], coeff, coeff, np.full(n_spec, 3)


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
    options: StokesIFitOptions, dtype: type, monkeypatch: pytest.MonkeyPatch
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
    err_block = np.full(i_block.shape, 1e-3)
    kwargs: dict[str, Any] = {
        "freq_arr_hz": freq_arr_hz,
        # A per-pixel reference frequency
        "ref_freq_hz": rng.uniform(800e6, 950e6, (cy, cx)),
        "err_1d": None,
        "fit_options": options,
        "log_level": logging.ERROR,
    }
    blocks: tuple[np.ndarray, ...] = (i_block.astype(dtype), err_block.astype(dtype))
    compiled = fitting._fit_stokes_i_block(*blocks, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(fitting, "fit_stokes_i_spectra", scipy_spectra)
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
