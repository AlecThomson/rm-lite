"""Tests for bad-channel robustness in the Stokes I fit."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import pytest
from astropy.stats import akaike_info_criterion_lsq
from numpy.typing import NDArray
from rm_lite.utils.fitting import (
    RobustLoss,
    StokesIFitOptions,
    fit_stokes_i_model,
    power_law,
    static_fit,
    stokes_i_snr,
)
from scipy import optimize

if TYPE_CHECKING:
    from collections.abc import Callable


class Spectrum(NamedTuple):
    """A noisy power-law Stokes I spectrum and the truth behind it."""

    freq_arr_hz: NDArray[np.float64]
    ref_freq_hz: float
    truth: NDArray[np.float64]
    stokes_i_arr: NDArray[np.float64]
    error_arr: NDArray[np.float64]
    n_chan: int
    noise: float
    alpha: float
    bad_chan: int
    """The channel the contamination tests spoil."""


@pytest.fixture
def spectrum() -> Spectrum:
    """A clean power-law spectrum with flat per-channel noise."""
    n_chan, noise, alpha = 144, 0.01, -0.8
    freq_arr_hz = np.linspace(0.8e9, 1.088e9, n_chan)
    ref_freq_hz = float(np.mean(freq_arr_hz))
    truth = power_law(1)(freq_arr_hz / ref_freq_hz, 1.0, alpha)
    rng = np.random.default_rng(20260908)
    return Spectrum(
        freq_arr_hz=freq_arr_hz,
        ref_freq_hz=ref_freq_hz,
        truth=truth,
        stokes_i_arr=truth + rng.normal(0, noise, n_chan),
        error_arr=np.full(n_chan, noise),
        n_chan=n_chan,
        noise=noise,
        alpha=alpha,
        bad_chan=n_chan // 3,
    )


@pytest.fixture
def fit_error(spectrum: Spectrum) -> Callable[..., float]:
    """Worst fractional deviation of the fitted model from the truth."""

    def measure(
        stokes_i_arr: NDArray[np.float64],
        stokes_i_error_arr: NDArray[np.float64],
        **option_kwargs: Any,
    ) -> float:
        options = StokesIFitOptions(snr_cut=None, **option_kwargs)
        fit = fit_stokes_i_model(
            freq_arr_hz=spectrum.freq_arr_hz,
            ref_freq_hz=spectrum.ref_freq_hz,
            stokes_i_arr=stokes_i_arr,
            stokes_i_error_arr=stokes_i_error_arr,
            options=options,
        )
        assert fit is not None
        model = fit.stokes_i_model_func(
            spectrum.freq_arr_hz / spectrum.ref_freq_hz, *np.asarray(fit.popt)
        )
        return float(np.abs(model - spectrum.truth).max() / spectrum.truth.max())

    return measure


@pytest.mark.parametrize("amplitude", [5.0, 20.0, 100.0])
def test_robust_loss_shrugs_off_a_flux_outlier(
    spectrum: Spectrum, fit_error: Callable[..., float], amplitude: float
) -> None:
    """One boosted channel wrecks a plain fit but not a robust one."""
    contaminated = spectrum.stokes_i_arr.copy()
    contaminated[spectrum.bad_chan] *= amplitude

    clean = fit_error(spectrum.stokes_i_arr, spectrum.error_arr)
    robust = fit_error(contaminated, spectrum.error_arr)
    plain = fit_error(contaminated, spectrum.error_arr, robust_loss="linear")

    assert robust < 5 * max(clean, 1e-3)
    assert robust < 0.01
    assert plain > 10 * robust


def test_robust_loss_shrugs_off_a_contiguous_bad_band(
    spectrum: Spectrum, fit_error: Callable[..., float]
) -> None:
    """A run of bad channels (an RFI band) is no harder than a single one."""
    contaminated = spectrum.stokes_i_arr.copy()
    contaminated[60:68] *= 3.0

    assert fit_error(contaminated, spectrum.error_arr) < 0.01
    assert fit_error(contaminated, spectrum.error_arr, robust_loss="linear") > 0.1


def test_robust_loss_holds_to_twenty_percent_contamination(
    spectrum: Spectrum, fit_error: Callable[..., float]
) -> None:
    """Well past where a fixed trim fraction would break down."""
    contaminated = spectrum.stokes_i_arr.copy()
    rng = np.random.default_rng(4)
    spoiled = rng.choice(spectrum.n_chan, spectrum.n_chan // 5, replace=False)
    contaminated[spoiled] *= 4.0

    assert fit_error(contaminated, spectrum.error_arr) < 0.01
    assert fit_error(contaminated, spectrum.error_arr, robust_loss="linear") > 0.1


def test_robust_loss_costs_nothing_on_clean_data(
    spectrum: Spectrum, fit_error: Callable[..., float]
) -> None:
    """Robustness is the default, so it must not degrade an uncontaminated fit."""
    robust = fit_error(spectrum.stokes_i_arr, spectrum.error_arr)
    plain = fit_error(spectrum.stokes_i_arr, spectrum.error_arr, robust_loss="linear")
    assert robust < 0.01
    assert robust < 2 * plain


def test_robust_loss_beats_plain_least_squares(
    spectrum: Spectrum, fit_error: Callable[..., float]
) -> None:
    contaminated = spectrum.stokes_i_arr.copy()
    contaminated[spectrum.bad_chan] *= 20.0

    robust = fit_error(contaminated, spectrum.error_arr)
    plain = fit_error(contaminated, spectrum.error_arr, robust_loss="linear")
    assert robust < 0.02
    assert robust < plain / 10


def test_an_over_trusted_channel_does_not_bend_the_fit(
    spectrum: Spectrum, fit_error: Callable[..., float]
) -> None:
    """An error 1000x too small makes plain least squares follow that channel."""
    # It leaves the channel a *small* residual, so spotting outliers in the
    # residuals does not catch it; the loss drops far-out channels to zero weight.
    contaminated = spectrum.stokes_i_arr.copy()
    contaminated[spectrum.bad_chan] += 5 * spectrum.noise
    bad_error = spectrum.error_arr.copy()
    bad_error[spectrum.bad_chan] = spectrum.noise / 1000

    assert fit_error(contaminated, bad_error) < 0.01
    assert fit_error(contaminated, bad_error, robust_loss="linear") > 0.02


def test_one_zero_error_channel_keeps_the_rest_weighted(spectrum: Spectrum) -> None:
    """A single zero error drops that channel, not the whole spectrum."""
    # It used to unweight everything: curve_fit raised and the retry dropped every
    # weight. The noise varies so that is measurable.
    varying_error = spectrum.noise * (1 + 3 * np.linspace(0, 1, spectrum.n_chan) ** 2)
    with_zero = varying_error.copy()
    with_zero[spectrum.bad_chan] = 0.0

    options = StokesIFitOptions(snr_cut=None)
    reference = fit_stokes_i_model(
        spectrum.freq_arr_hz,
        spectrum.ref_freq_hz,
        spectrum.stokes_i_arr,
        varying_error,
        options,
    )
    with_gap = fit_stokes_i_model(
        spectrum.freq_arr_hz,
        spectrum.ref_freq_hz,
        spectrum.stokes_i_arr,
        with_zero,
        options,
    )
    assert reference is not None
    assert with_gap is not None
    # Same weighting minus one channel, so the parameter errors barely move.
    ref_err = np.sqrt(np.diag(np.asarray(reference.pcov)))
    gap_err = np.sqrt(np.diag(np.asarray(with_gap.pcov)))
    np.testing.assert_allclose(gap_err, ref_err, rtol=0.1)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_errors_that_cannot_weight_a_fit_are_dropped(
    spectrum: Spectrum, bad: float
) -> None:
    error_arr = spectrum.error_arr.copy()
    error_arr[spectrum.bad_chan] = bad
    fit = fit_stokes_i_model(
        spectrum.freq_arr_hz,
        spectrum.ref_freq_hz,
        spectrum.stokes_i_arr,
        error_arr,
        StokesIFitOptions(snr_cut=None),
    )
    assert fit is not None
    model = fit.stokes_i_model_func(
        spectrum.freq_arr_hz / spectrum.ref_freq_hz, *np.asarray(fit.popt)
    )
    assert float(np.abs(model - spectrum.truth).max() / spectrum.truth.max()) < 0.01


def test_aic_still_picks_a_sloped_model_through_an_outlier(spectrum: Spectrum) -> None:
    """An outlier must not collapse `fit_order < 0` onto a flat model."""
    # It used to: the unweighted AIC scored every order alike, so the fewest
    # params won.
    contaminated = spectrum.stokes_i_arr.copy()
    contaminated[spectrum.bad_chan] *= 20.0

    fit = fit_stokes_i_model(
        freq_arr_hz=spectrum.freq_arr_hz,
        ref_freq_hz=spectrum.ref_freq_hz,
        stokes_i_arr=contaminated,
        stokes_i_error_arr=spectrum.error_arr,
        options=StokesIFitOptions(fit_order=-5, snr_cut=None),
    )
    assert fit is not None
    popt = np.asarray(fit.popt)
    assert popt.size >= 2, "collapsed to a flat model"
    # The recovered spectral index is the truth, not the outlier's pull.
    assert popt[1] == pytest.approx(spectrum.alpha, abs=0.05)


def test_aic_without_an_error_is_the_plain_least_squares_one(
    spectrum: Spectrum,
) -> None:
    """Unweighted residuals and unit weights, so the score is unchanged."""
    fit = fit_stokes_i_model(
        freq_arr_hz=spectrum.freq_arr_hz,
        ref_freq_hz=spectrum.ref_freq_hz,
        stokes_i_arr=spectrum.stokes_i_arr,
        stokes_i_error_arr=np.zeros(spectrum.n_chan),
        options=StokesIFitOptions(fit_order=2, snr_cut=None, robust_loss="linear"),
    )
    assert fit is not None
    model = fit.stokes_i_model_func(
        spectrum.freq_arr_hz / spectrum.ref_freq_hz, *np.asarray(fit.popt)
    )
    ssr = float(np.sum((spectrum.stokes_i_arr - model) ** 2))
    expected = float(
        akaike_info_criterion_lsq(ssr=ssr, n_params=3, n_samples=spectrum.n_chan)
    )
    assert fit.aic == pytest.approx(expected)


def test_snr_ignores_a_single_over_estimated_error_channel(spectrum: Spectrum) -> None:
    """One inflated error channel must not quietly skip the pixel."""
    # It used to take the SNR from 1200 to 1.4.
    inflated = spectrum.error_arr.copy()
    inflated[5] = spectrum.noise * 1e4
    clean_snr = stokes_i_snr(spectrum.stokes_i_arr, spectrum.error_arr)
    assert stokes_i_snr(spectrum.stokes_i_arr, inflated) == pytest.approx(
        clean_snr, rel=0.05
    )


@pytest.mark.parametrize("spike", [-1e3, 1e3])
def test_snr_ignores_a_single_flux_spike(spectrum: Spectrum, spike: float) -> None:
    """A large negative spike used to make the SNR negative outright."""
    clean_snr = stokes_i_snr(spectrum.stokes_i_arr, spectrum.error_arr)
    spiked = spectrum.stokes_i_arr.copy()
    spiked[5] = spike
    assert stokes_i_snr(spiked, spectrum.error_arr) == pytest.approx(
        clean_snr, rel=0.05
    )


def test_snr_is_inf_without_a_usable_error(spectrum: Spectrum) -> None:
    """So an SNR cut is a no-op rather than rejecting everything."""
    assert stokes_i_snr(spectrum.stokes_i_arr, np.zeros(spectrum.n_chan)) == np.inf
    assert (
        stokes_i_snr(spectrum.stokes_i_arr, np.full(spectrum.n_chan, np.nan)) == np.inf
    )
    assert stokes_i_snr(np.array([]), np.array([])) == np.inf


def test_snr_only_counts_usable_channels(spectrum: Spectrum) -> None:
    """Channels the fit will not see should not set the SNR either."""
    holed_error = spectrum.error_arr.copy()
    holed_error[:20] = np.nan
    # sqrt(n) over the remaining channels, so a smaller but sane SNR.
    expected = stokes_i_snr(spectrum.stokes_i_arr[20:], spectrum.error_arr[20:])
    assert stokes_i_snr(spectrum.stokes_i_arr, holed_error) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"robust_loss": "arctan"}, "robust_loss must be 'cauchy' or 'linear'"),
        ({"f_scale": 0.0}, "f_scale must be positive"),
        ({"f_scale": -1.0}, "f_scale must be positive"),
        ({"model_floor_sigma": -1.0}, "model_floor_sigma must be non-negative"),
    ],
)
def test_options_reject_nonsense(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        StokesIFitOptions(**kwargs)  # type: ignore[arg-type]


def test_option_defaults_are_robust() -> None:
    """A cube gets robustness without asking, which is the point."""
    options = StokesIFitOptions()
    assert options.robust_loss == "cauchy"
    assert options.f_scale == 3.0


def test_no_error_falls_back_to_plain_least_squares(spectrum: Spectrum) -> None:
    """`f_scale` counts sigma, so with no error there is nothing to count."""
    # Left as a robust fit it would be an absolute flux cut instead, and the
    # same spectrum in mJy and Jy would fit differently.
    rng = np.random.default_rng(7)
    stokes_i_arr = spectrum.truth + rng.normal(0, spectrum.noise, spectrum.n_chan)

    losses: tuple[RobustLoss, RobustLoss] = ("cauchy", "linear")
    fits = [
        fit_stokes_i_model(
            freq_arr_hz=spectrum.freq_arr_hz,
            ref_freq_hz=spectrum.ref_freq_hz,
            stokes_i_arr=stokes_i_arr,
            stokes_i_error_arr=np.zeros(spectrum.n_chan),
            options=StokesIFitOptions(snr_cut=None, robust_loss=loss),
        )
        for loss in losses
    ]
    assert fits[0] is not None
    assert fits[1] is not None
    np.testing.assert_allclose(np.asarray(fits[0].popt), np.asarray(fits[1].popt))


def test_a_failed_fit_is_unscoreable(
    spectrum: Spectrum, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A flat model nothing converged to scores inf rather than competing."""
    # Otherwise it competes on AIC once the weighting has shrunk its residuals.

    def always_fails(*_args: object, **_kwargs: object) -> None:
        msg = "curve_fit forced to fail"
        raise RuntimeError(msg)

    monkeypatch.setattr(optimize, "curve_fit", always_fails)
    fit = static_fit(
        spectrum.freq_arr_hz,
        spectrum.ref_freq_hz,
        spectrum.stokes_i_arr,
        spectrum.error_arr,
        2,
        "log",
    )
    assert fit.aic == np.inf
    # Still a usable flat model at the mean, not an exception.
    model = fit.stokes_i_model_func(
        spectrum.freq_arr_hz / spectrum.ref_freq_hz, *np.asarray(fit.popt)
    )
    np.testing.assert_allclose(model, np.mean(spectrum.stokes_i_arr))
