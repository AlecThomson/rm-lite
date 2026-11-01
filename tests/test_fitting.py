"""Tests for the compiled Stokes I fit (`lm_fit`), against scipy's curve_fit."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Literal

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
