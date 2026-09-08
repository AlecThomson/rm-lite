"""Tests for bad-channel robustness in the Stokes I fit.

`robust_loss` covers a channel bad in flux or in error, since either way it ends
up far from the model in sigma. Errors that cannot weight a fit at all (zero,
negative, non-finite) are dropped before it.
"""

from __future__ import annotations

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

N_CHAN = 144
NOISE = 0.01
ALPHA = -0.8
BAD_CHAN = N_CHAN // 3


def _band() -> tuple[NDArray[np.float64], float]:
    freq_arr_hz = np.linspace(0.8e9, 1.088e9, N_CHAN)
    return freq_arr_hz, float(np.mean(freq_arr_hz))


def _clean_spectrum() -> tuple[
    NDArray[np.float64],
    float,
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Frequencies, reference, truth, a noisy realisation of it, and its error."""
    freq_arr_hz, ref_freq_hz = _band()
    truth = power_law(1)(freq_arr_hz / ref_freq_hz, 1.0, ALPHA)
    rng = np.random.default_rng(20260908)
    stokes_i_arr = truth + rng.normal(0, NOISE, N_CHAN)
    return freq_arr_hz, ref_freq_hz, truth, stokes_i_arr, np.full(N_CHAN, NOISE)


def _fit_error(
    stokes_i_arr: NDArray[np.float64],
    stokes_i_error_arr: NDArray[np.float64],
    **option_kwargs: object,
) -> float:
    """Worst fractional deviation of the fitted model from the truth."""
    freq_arr_hz, ref_freq_hz, truth, _, _ = _clean_spectrum()
    options = StokesIFitOptions(snr_cut=None, **option_kwargs)  # type: ignore[arg-type]
    fit = fit_stokes_i_model(
        freq_arr_hz=freq_arr_hz,
        ref_freq_hz=ref_freq_hz,
        stokes_i_arr=stokes_i_arr,
        stokes_i_error_arr=stokes_i_error_arr,
        options=options,
    )
    assert fit is not None
    model = fit.stokes_i_model_func(freq_arr_hz / ref_freq_hz, *np.asarray(fit.popt))
    return float(np.abs(model - truth).max() / truth.max())


# --------------------------------------------------------------- flux outliers


@pytest.mark.parametrize("amplitude", [5.0, 20.0, 100.0])
def test_robust_loss_shrugs_off_a_flux_outlier(amplitude: float) -> None:
    """One boosted channel wrecks a plain fit but not a robust one."""
    _, _, _, stokes_i_arr, stokes_i_error_arr = _clean_spectrum()
    contaminated = stokes_i_arr.copy()
    contaminated[BAD_CHAN] *= amplitude

    clean = _fit_error(stokes_i_arr, stokes_i_error_arr)
    robust = _fit_error(contaminated, stokes_i_error_arr)
    plain = _fit_error(contaminated, stokes_i_error_arr, robust_loss="linear")

    assert robust < 5 * max(clean, 1e-3)
    assert robust < 0.01
    assert plain > 10 * robust


def test_robust_loss_shrugs_off_a_contiguous_bad_band() -> None:
    """A run of bad channels (an RFI band) is no harder than a single one."""
    _, _, _, stokes_i_arr, stokes_i_error_arr = _clean_spectrum()
    contaminated = stokes_i_arr.copy()
    contaminated[60:68] *= 3.0

    assert _fit_error(contaminated, stokes_i_error_arr) < 0.01
    assert _fit_error(contaminated, stokes_i_error_arr, robust_loss="linear") > 0.1


def test_robust_loss_holds_to_twenty_percent_contamination() -> None:
    """Well past where a fixed trim fraction would break down."""
    _, _, _, stokes_i_arr, stokes_i_error_arr = _clean_spectrum()
    contaminated = stokes_i_arr.copy()
    rng = np.random.default_rng(4)
    contaminated[rng.choice(N_CHAN, N_CHAN // 5, replace=False)] *= 4.0

    assert _fit_error(contaminated, stokes_i_error_arr) < 0.01
    assert _fit_error(contaminated, stokes_i_error_arr, robust_loss="linear") > 0.1


def test_robust_loss_costs_nothing_on_clean_data() -> None:
    """Robustness is the default, so it must not degrade an uncontaminated fit."""
    _, _, _, stokes_i_arr, stokes_i_error_arr = _clean_spectrum()
    robust = _fit_error(stokes_i_arr, stokes_i_error_arr)
    plain = _fit_error(stokes_i_arr, stokes_i_error_arr, robust_loss="linear")
    assert robust < 0.01
    assert robust < 2 * plain


def test_robust_loss_beats_plain_least_squares() -> None:
    _, _, _, stokes_i_arr, stokes_i_error_arr = _clean_spectrum()
    contaminated = stokes_i_arr.copy()
    contaminated[BAD_CHAN] *= 20.0

    robust = _fit_error(contaminated, stokes_i_error_arr)
    plain = _fit_error(contaminated, stokes_i_error_arr, robust_loss="linear")
    assert robust < 0.02
    assert robust < plain / 10


# ---------------------------------------------------------------- bad errors


def test_an_over_trusted_channel_does_not_bend_the_fit() -> None:
    """An error 1000x too small makes plain least squares follow that channel.

    Note it leaves the channel a *small* residual, so this is not caught by
    spotting outliers in the residuals; the loss handles it by dropping far-out
    channels to zero weight.
    """
    _, _, _, stokes_i_arr, stokes_i_error_arr = _clean_spectrum()
    contaminated = stokes_i_arr.copy()
    contaminated[BAD_CHAN] += 5 * NOISE
    bad_error = stokes_i_error_arr.copy()
    bad_error[BAD_CHAN] = NOISE / 1000

    assert _fit_error(contaminated, bad_error) < 0.01
    assert _fit_error(contaminated, bad_error, robust_loss="linear") > 0.02


def test_one_zero_error_channel_keeps_the_rest_weighted() -> None:
    """A single zero error used to unweight the whole spectrum: `curve_fit`
    raised on it and the retry dropped every weight. Now just that channel goes."""
    freq_arr_hz, ref_freq_hz, _, stokes_i_arr, _ = _clean_spectrum()
    # A band whose noise varies, so dropping the weights is measurable.
    varying_error = NOISE * (1 + 3 * np.linspace(0, 1, N_CHAN) ** 2)
    with_zero = varying_error.copy()
    with_zero[BAD_CHAN] = 0.0

    options = StokesIFitOptions(snr_cut=None)
    reference = fit_stokes_i_model(
        freq_arr_hz, ref_freq_hz, stokes_i_arr, varying_error, options
    )
    with_gap = fit_stokes_i_model(
        freq_arr_hz, ref_freq_hz, stokes_i_arr, with_zero, options
    )
    assert reference is not None
    assert with_gap is not None
    # Same weighting minus one channel, so the parameter errors barely move.
    ref_err = np.sqrt(np.diag(np.asarray(reference.pcov)))
    gap_err = np.sqrt(np.diag(np.asarray(with_gap.pcov)))
    np.testing.assert_allclose(gap_err, ref_err, rtol=0.1)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_errors_that_cannot_weight_a_fit_are_dropped(bad: float) -> None:
    freq_arr_hz, ref_freq_hz, truth, stokes_i_arr, stokes_i_error_arr = (
        _clean_spectrum()
    )
    error_arr = stokes_i_error_arr.copy()
    error_arr[BAD_CHAN] = bad
    fit = fit_stokes_i_model(
        freq_arr_hz,
        ref_freq_hz,
        stokes_i_arr,
        error_arr,
        StokesIFitOptions(snr_cut=None),
    )
    assert fit is not None
    model = fit.stokes_i_model_func(freq_arr_hz / ref_freq_hz, *np.asarray(fit.popt))
    assert float(np.abs(model - truth).max() / truth.max()) < 0.01


# ------------------------------------------------------------------ robust AIC


def test_aic_still_picks_a_sloped_model_through_an_outlier() -> None:
    """An outlier used to collapse `fit_order < 0` onto a flat model, because the
    unweighted AIC scored every order alike and the fewest params won."""
    freq_arr_hz, ref_freq_hz, _, stokes_i_arr, stokes_i_error_arr = _clean_spectrum()
    contaminated = stokes_i_arr.copy()
    contaminated[48] *= 20.0

    fit = fit_stokes_i_model(
        freq_arr_hz=freq_arr_hz,
        ref_freq_hz=ref_freq_hz,
        stokes_i_arr=contaminated,
        stokes_i_error_arr=stokes_i_error_arr,
        options=StokesIFitOptions(fit_order=-5, snr_cut=None),
    )
    assert fit is not None
    popt = np.asarray(fit.popt)
    assert popt.size >= 2, "collapsed to a flat model"
    # The recovered spectral index is the truth, not the outlier's pull.
    assert popt[1] == pytest.approx(ALPHA, abs=0.05)


def test_aic_without_an_error_is_the_plain_least_squares_one() -> None:
    """Unweighted residuals and unit weights, so the score is unchanged."""
    freq_arr_hz, ref_freq_hz, _, stokes_i_arr, _ = _clean_spectrum()
    fit = fit_stokes_i_model(
        freq_arr_hz=freq_arr_hz,
        ref_freq_hz=ref_freq_hz,
        stokes_i_arr=stokes_i_arr,
        stokes_i_error_arr=np.zeros(N_CHAN),
        options=StokesIFitOptions(fit_order=2, snr_cut=None, robust_loss="linear"),
    )
    assert fit is not None
    model = fit.stokes_i_model_func(freq_arr_hz / ref_freq_hz, *np.asarray(fit.popt))
    ssr = float(np.sum((stokes_i_arr - model) ** 2))
    expected = float(akaike_info_criterion_lsq(ssr=ssr, n_params=3, n_samples=N_CHAN))
    assert fit.aic == pytest.approx(expected)


# ------------------------------------------------------------------ robust SNR


def test_snr_ignores_a_single_over_estimated_error_channel() -> None:
    """One inflated error channel used to take the SNR from 1200 to 1.4, quietly
    skipping the pixel."""
    _, _, _, stokes_i_arr, stokes_i_error_arr = _clean_spectrum()
    inflated = stokes_i_error_arr.copy()
    inflated[5] = NOISE * 1e4
    clean_snr = stokes_i_snr(stokes_i_arr, stokes_i_error_arr)
    assert stokes_i_snr(stokes_i_arr, inflated) == pytest.approx(clean_snr, rel=0.05)


def test_snr_ignores_a_single_flux_spike() -> None:
    """A large negative spike used to make the SNR negative outright."""
    _, _, _, stokes_i_arr, stokes_i_error_arr = _clean_spectrum()
    clean_snr = stokes_i_snr(stokes_i_arr, stokes_i_error_arr)
    for spike in (-1e3, 1e3):
        spiked = stokes_i_arr.copy()
        spiked[5] = spike
        assert stokes_i_snr(spiked, stokes_i_error_arr) == pytest.approx(
            clean_snr, rel=0.05
        )


def test_snr_is_inf_without_a_usable_error() -> None:
    """So an SNR cut is a no-op rather than rejecting everything."""
    _, _, _, stokes_i_arr, _ = _clean_spectrum()
    assert stokes_i_snr(stokes_i_arr, np.zeros(N_CHAN)) == np.inf
    assert stokes_i_snr(stokes_i_arr, np.full(N_CHAN, np.nan)) == np.inf
    assert stokes_i_snr(np.array([]), np.array([])) == np.inf


def test_snr_only_counts_usable_channels() -> None:
    """Channels the fit will not see should not set the SNR either."""
    _, _, _, stokes_i_arr, stokes_i_error_arr = _clean_spectrum()
    holed_error = stokes_i_error_arr.copy()
    holed_error[:20] = np.nan
    # sqrt(n) over the remaining channels, so a smaller but sane SNR.
    expected = stokes_i_snr(stokes_i_arr[20:], stokes_i_error_arr[20:])
    assert stokes_i_snr(stokes_i_arr, holed_error) == pytest.approx(expected)


# ------------------------------------------------------------------- options


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"robust_loss": "arctan"}, "robust_loss must be 'cauchy' or 'linear'"),
        ({"f_scale": 0.0}, "f_scale must be positive"),
        ({"f_scale": -1.0}, "f_scale must be positive"),
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


# ------------------------------------------------- the unweighted (no error) path


def test_no_error_falls_back_to_plain_least_squares() -> None:
    """`f_scale` counts sigma, so with no error there is nothing to count.

    Left as a robust fit it would be an absolute flux cut instead, and the same
    spectrum in mJy and Jy would fit differently. Robustness needs an error.
    """
    freq_arr_hz, ref_freq_hz = _band()
    x_arr = freq_arr_hz / ref_freq_hz
    truth = power_law(1)(x_arr, 1.0, ALPHA)
    rng = np.random.default_rng(7)
    stokes_i_arr = truth + rng.normal(0, NOISE, N_CHAN)

    losses: tuple[RobustLoss, RobustLoss] = ("cauchy", "linear")
    fits = [
        fit_stokes_i_model(
            freq_arr_hz=freq_arr_hz,
            ref_freq_hz=ref_freq_hz,
            stokes_i_arr=stokes_i_arr,
            stokes_i_error_arr=np.zeros(N_CHAN),
            options=StokesIFitOptions(snr_cut=None, robust_loss=loss),
        )
        for loss in losses
    ]
    assert fits[0] is not None
    assert fits[1] is not None
    np.testing.assert_allclose(np.asarray(fits[0].popt), np.asarray(fits[1].popt))


def test_a_failed_fit_is_unscoreable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A flat model nothing converged to is not a candidate, so it scores inf
    rather than competing on AIC once the weighting has shrunk its residuals."""

    def _always_fails(*_args: object, **_kwargs: object) -> None:
        msg = "curve_fit forced to fail"
        raise RuntimeError(msg)

    monkeypatch.setattr(optimize, "curve_fit", _always_fails)
    _, _, _, stokes_i_arr, stokes_i_error_arr = _clean_spectrum()
    freq_arr_hz, ref_freq_hz = _band()
    fit = static_fit(
        freq_arr_hz, ref_freq_hz, stokes_i_arr, stokes_i_error_arr, 2, "log"
    )
    assert fit.aic == np.inf
    # Still a usable flat model at the mean, not an exception.
    model = fit.stokes_i_model_func(freq_arr_hz / ref_freq_hz, *np.asarray(fit.popt))
    np.testing.assert_allclose(model, np.mean(stokes_i_arr))
