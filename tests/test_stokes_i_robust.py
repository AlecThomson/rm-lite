"""Tests for bad-channel robustness in the Stokes I fit.

Bad channels arrive two ways and are handled in two places, so both are covered
here: a wrong flux is discounted by `StokesIFitOptions.robust_loss`, a wrong
error is masked by `usable_error_mask`. The split matters because least squares
bends the model onto an over-trusted channel, leaving it with a *small*
residual, so a bounded-influence loss still gets pulled by it. Only the
redescending default (cauchy) survives that on its own.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from rm_lite.utils.fitting import (
    RobustLoss,
    StokesIFitOptions,
    fit_stokes_i_model,
    polynomial,
    power_law,
    robust_weights,
    stokes_i_snr,
    usable_error_mask,
)

N_CHAN = 144
NOISE = 0.01
ALPHA = -0.8
BAD_CHAN = N_CHAN // 3


def _band() -> tuple[NDArray[np.float64], float]:
    freq_arr_hz = np.linspace(0.8e9, 1.088e9, N_CHAN)
    return freq_arr_hz, float(np.mean(freq_arr_hz))


def _clean_spectrum() -> tuple[
    NDArray[np.float64], float, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
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
    """One boosted channel wrecks a plain fit but not a robust one.

    Plain least squares is off by 6% for a x5 channel and 30% for x20; the
    default loss holds the model to its clean-data accuracy.
    """
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


@pytest.mark.parametrize("robust_loss", ["cauchy", "soft_l1", "huber"])
def test_every_robust_loss_beats_plain_least_squares(robust_loss: RobustLoss) -> None:
    _, _, _, stokes_i_arr, stokes_i_error_arr = _clean_spectrum()
    contaminated = stokes_i_arr.copy()
    contaminated[BAD_CHAN] *= 20.0

    robust = _fit_error(contaminated, stokes_i_error_arr, robust_loss=robust_loss)
    plain = _fit_error(contaminated, stokes_i_error_arr, robust_loss="linear")
    assert robust < 0.02
    assert robust < plain / 10


# ---------------------------------------------------------------- bad errors


def _over_trusted_spectrum() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """A mildly offset channel whose error says to trust it 1000x too much."""
    _, _, _, stokes_i_arr, stokes_i_error_arr = _clean_spectrum()
    contaminated = stokes_i_arr.copy()
    contaminated[BAD_CHAN] += 5 * NOISE
    bad_error = stokes_i_error_arr.copy()
    bad_error[BAD_CHAN] = NOISE / 1000
    return contaminated, bad_error


@pytest.mark.parametrize("robust_loss", ["linear", "huber", "soft_l1"])
def test_an_over_trusted_channel_needs_the_error_mask(
    robust_loss: RobustLoss,
) -> None:
    """A too-small error is not a case the loss can be relied on to see.

    Least squares bends the model onto the over-trusted channel, so the bad
    channel ends up with a *small* residual and the good channels carry the
    error. A loss whose influence stays bounded but non-zero (huber, soft_l1)
    still gets pulled, which is why `usable_error_mask` exists rather than
    leaving this to the loss.
    """
    contaminated, bad_error = _over_trusted_spectrum()
    masked = _fit_error(contaminated, bad_error, robust_loss=robust_loss)
    unmasked = _fit_error(
        contaminated, bad_error, robust_loss=robust_loss, error_outlier_factor=None
    )
    assert masked < 0.01
    assert unmasked > 10 * masked


def test_cauchy_alone_also_survives_an_over_trusted_channel() -> None:
    """The default loss redescends, so it is the one that does not need the mask.

    Worth pinning: it is why the default pairing is safe even where the error
    map is wrong in a way the mask's median test happens to miss.
    """
    contaminated, bad_error = _over_trusted_spectrum()
    assert (
        _fit_error(
            contaminated, bad_error, robust_loss="cauchy", error_outlier_factor=None
        )
        < 0.01
    )


def test_one_zero_error_channel_keeps_the_rest_weighted() -> None:
    """A single zero error used to discard the weighting for the whole spectrum.

    `curve_fit` raises "Residuals are not finite in the initial point" on a zero
    sigma, and the old code caught that and refitted every channel unweighted.
    The channel is dropped now, so the remaining errors still weight the fit.
    """
    freq_arr_hz, ref_freq_hz, _, stokes_i_arr, stokes_i_error_arr = _clean_spectrum()
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


def test_usable_error_mask_drops_both_tails() -> None:
    """Errors far either side of the band median are untrustworthy."""
    error_arr = np.full(10, 0.01)
    error_arr[2] = 0.01 / 1000  # over-trusted
    error_arr[5] = 0.01 * 1000  # wrecks an rms SNR
    mask = usable_error_mask(error_arr, error_outlier_factor=10.0)
    assert not mask[2]
    assert not mask[5]
    assert mask.sum() == 8


def test_usable_error_mask_drops_unusable_values() -> None:
    error_arr = np.array([0.01, 0.0, -0.01, np.nan, np.inf, 0.01])
    mask = usable_error_mask(error_arr)
    np.testing.assert_array_equal(mask, [True, False, False, False, False, True])


def test_usable_error_mask_keeps_a_heteroscedastic_band() -> None:
    """Noisier band edges are real, not bad channels."""
    error_arr = NOISE * (1 + 3 * np.linspace(-1, 1, N_CHAN) ** 2)
    assert usable_error_mask(error_arr, error_outlier_factor=10.0).all()


def test_usable_error_mask_is_all_true_without_a_usable_error() -> None:
    """All-zero errors are how the callers say "no error given"."""
    assert usable_error_mask(np.zeros(10)).all()
    assert usable_error_mask(np.full(10, np.nan)).all()


def test_usable_error_mask_factor_none_keeps_every_positive_error() -> None:
    error_arr = np.array([0.01, 1e-9, 1e9, 0.0])
    np.testing.assert_array_equal(
        usable_error_mask(error_arr, error_outlier_factor=None),
        [True, True, True, False],
    )


def test_usable_error_mask_falls_back_when_nothing_survives() -> None:
    """A bimodal error map leaves the finite errors rather than nothing."""
    error_arr = np.array([1e-6, 1e-6, 1.0, 1.0])
    mask = usable_error_mask(error_arr, error_outlier_factor=1.5)
    assert mask.all()


# ------------------------------------------------------------- robust weights


@pytest.mark.parametrize(
    ("robust_loss", "expected"),
    [
        ("linear", 1.0),
        ("cauchy", 1.0 / 2.0),
        ("soft_l1", 1.0 / np.sqrt(2.0)),
        ("huber", 1.0),
    ],
)
def test_robust_weights_at_one_f_scale(robust_loss: RobustLoss, expected: float) -> None:
    """At exactly `f_scale` the losses take their documented values."""
    weight = robust_weights(np.array([3.0]), robust_loss, f_scale=3.0)
    np.testing.assert_allclose(weight, [expected])


@pytest.mark.parametrize("robust_loss", ["cauchy", "soft_l1", "huber"])
def test_robust_weights_decrease_with_residual(robust_loss: RobustLoss) -> None:
    scaled = np.array([0.0, 1.0, 3.0, 10.0, 100.0])
    weights = robust_weights(scaled, robust_loss, f_scale=3.0)
    np.testing.assert_allclose(weights[0], 1.0)
    assert np.all(np.diff(weights) <= 0)
    assert weights[-1] < 0.2


def test_linear_weights_are_all_one() -> None:
    weights = robust_weights(np.array([0.0, 5.0, 500.0]), "linear", f_scale=3.0)
    np.testing.assert_allclose(weights, 1.0)


@pytest.mark.parametrize("robust_loss", ["cauchy", "soft_l1", "huber"])
def test_robust_weights_match_what_scipy_minimised(robust_loss: RobustLoss) -> None:
    """The fitted parameters satisfy the M-estimator's stationarity condition.

    Minimising `sum rho((r/sigma)**2)` is stationary where
    `sum w * r * J / sigma**2 == 0` with `w = rho'`, so if `robust_weights`
    really is scipy's `rho'` the fit it returned zeroes that sum. Checked on a
    linear model (order 1), whose Jacobian columns are `[1, x]`, with a positive
    intercept so `static_fit`'s lower bound on the first term stays inactive.
    """
    freq_arr_hz, ref_freq_hz, _, _, stokes_i_error_arr = _clean_spectrum()
    x_arr = freq_arr_hz / ref_freq_hz
    rng = np.random.default_rng(11)
    stokes_i_arr = polynomial(1)(x_arr, 2.0, -0.5) + rng.normal(0, NOISE, N_CHAN)
    stokes_i_arr[BAD_CHAN] += 30 * NOISE

    fit = fit_stokes_i_model(
        freq_arr_hz=freq_arr_hz,
        ref_freq_hz=ref_freq_hz,
        stokes_i_arr=stokes_i_arr,
        stokes_i_error_arr=stokes_i_error_arr,
        options=StokesIFitOptions(
            fit_order=1,
            fit_function="linear",
            snr_cut=None,
            robust_loss=robust_loss,
            f_scale=3.0,
        ),
    )
    assert fit is not None
    resid = stokes_i_arr - fit.stokes_i_model_func(x_arr, *np.asarray(fit.popt))
    scaled = resid / stokes_i_error_arr
    weights = robust_weights(scaled, robust_loss, f_scale=3.0)
    jacobian = np.stack([np.ones_like(x_arr), x_arr])
    gradient = jacobian @ (weights * resid / stokes_i_error_arr**2)
    # Scale by the same sum with unit weights, so this is a relative statement.
    normalisation = np.abs(jacobian @ (np.abs(resid) / stokes_i_error_arr**2)).max()
    np.testing.assert_allclose(gradient / normalisation, 0.0, atol=2e-3)


# ------------------------------------------------------------------ robust AIC


def test_aic_still_picks_a_sloped_model_through_an_outlier() -> None:
    """An outlier used to collapse `fit_order < 0` onto a flat model.

    The AIC was a plain sum of squares, which the bad channel dominated equally
    at every order; the differences vanished and Occam's razor took the fewest
    parameters. Scoring the fit the way it was weighted keeps the slope.
    """
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
    from astropy.stats import akaike_info_criterion_lsq

    expected = float(
        akaike_info_criterion_lsq(ssr=ssr, n_params=3, n_samples=N_CHAN)
    )
    assert fit.aic == pytest.approx(expected)


# ------------------------------------------------------------------ robust SNR


def test_snr_ignores_a_single_over_estimated_error_channel() -> None:
    """One inflated error channel used to take the SNR from 1200 to 1.4, which
    silently skipped the pixel and flattened its model."""
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
        ({"robust_loss": "arctan"}, "robust_loss must be one of"),
        ({"f_scale": 0.0}, "f_scale must be positive"),
        ({"f_scale": -1.0}, "f_scale must be positive"),
        ({"error_outlier_factor": 1.0}, "error_outlier_factor must be greater than 1"),
        ({"error_outlier_factor": 0.5}, "error_outlier_factor must be greater than 1"),
    ],
)
def test_options_reject_nonsense(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        StokesIFitOptions(**kwargs)  # type: ignore[arg-type]


def test_option_defaults_are_robust() -> None:
    """The defaults are the point: a cube gets robustness without asking."""
    options = StokesIFitOptions()
    assert options.robust_loss == "cauchy"
    assert options.f_scale == 3.0
    assert options.error_outlier_factor == 10.0


# ------------------------------------------------- the unweighted (no error) path


@pytest.mark.parametrize("flux_scale", [1.0, 1e2, 1e4, 1e6])
def test_robust_fit_without_an_error_is_scale_free(flux_scale: float) -> None:
    """`f_scale` is a residual in sigma, so with no error it needs a stand-in.

    Left alone it would become an absolute flux cut: the same spectrum in mJy
    and in Jy would be fitted differently, and at large flux scales every
    channel would be discounted at once. `_self_scaled_sigma` takes the spread
    from the data, so the answer must not depend on the units.
    """
    freq_arr_hz, ref_freq_hz = _band()
    x_arr = freq_arr_hz / ref_freq_hz
    truth = flux_scale * power_law(1)(x_arr, 1.0, ALPHA)
    rng = np.random.default_rng(7)
    stokes_i_arr = truth + rng.normal(0, 0.01 * flux_scale, N_CHAN)
    stokes_i_arr[BAD_CHAN] *= 20.0
    no_error = np.zeros(N_CHAN)

    def error_of(robust_loss: RobustLoss) -> float:
        fit = fit_stokes_i_model(
            freq_arr_hz=freq_arr_hz,
            ref_freq_hz=ref_freq_hz,
            stokes_i_arr=stokes_i_arr,
            stokes_i_error_arr=no_error,
            options=StokesIFitOptions(snr_cut=None, robust_loss=robust_loss),
        )
        assert fit is not None
        model = fit.stokes_i_model_func(x_arr, *np.asarray(fit.popt))
        return float(np.abs(model - truth).max() / truth.max())

    assert error_of("cauchy") < 0.01
    assert error_of("linear") > 0.1


def test_unweighted_fit_covariance_tracks_the_flux_scale() -> None:
    """With a stand-in sigma, `curve_fit` rescales pcov by the fit's residuals.

    The old path passed `sigma=None` with `absolute_sigma=True`, which reports
    the covariance of a unit-residual fit: the same parameter errors whatever
    the flux scale, and useless to `compute_model_error`.
    """
    freq_arr_hz, ref_freq_hz = _band()
    x_arr = freq_arr_hz / ref_freq_hz
    errors = []
    for flux_scale in (1.0, 1e3):
        rng = np.random.default_rng(7)
        truth = flux_scale * power_law(1)(x_arr, 1.0, ALPHA)
        stokes_i_arr = truth + rng.normal(0, 0.01 * flux_scale, N_CHAN)
        fit = fit_stokes_i_model(
            freq_arr_hz=freq_arr_hz,
            ref_freq_hz=ref_freq_hz,
            stokes_i_arr=stokes_i_arr,
            stokes_i_error_arr=np.zeros(N_CHAN),
            options=StokesIFitOptions(fit_order=1, snr_cut=None),
        )
        assert fit is not None
        errors.append(float(np.sqrt(np.diag(np.asarray(fit.pcov)))[0]))
    # The flux term's error scales with the flux, as an error should.
    assert errors[1] / errors[0] == pytest.approx(1e3, rel=0.2)


def test_a_noiseless_spectrum_is_not_scaled_by_its_own_round_off() -> None:
    """With no error and no noise there is no spread for `f_scale` to use.

    The residual MAD of a spectrum the model fits exactly is ~1e-16, and scaling
    by that would have the loss discount channels over floating-point noise. The
    fit must fall back to plain least squares and stay exact.
    """
    freq_arr_hz, ref_freq_hz = _band()
    x_arr = freq_arr_hz / ref_freq_hz
    truth = 2.0 * power_law(1)(x_arr, 1.0, -1.4)  # exactly representable, no noise

    fit = fit_stokes_i_model(
        freq_arr_hz=freq_arr_hz,
        ref_freq_hz=ref_freq_hz,
        stokes_i_arr=truth,
        stokes_i_error_arr=np.zeros(N_CHAN),
        options=StokesIFitOptions(fit_order=1, snr_cut=None),
    )
    assert fit is not None
    popt = np.asarray(fit.popt)
    assert popt[0] == pytest.approx(2.0, rel=1e-6)
    assert popt[1] == pytest.approx(-1.4, rel=1e-6)


def test_a_failed_fit_is_unscoreable(monkeypatch: pytest.MonkeyPatch) -> None:
    """The flat model a total failure falls back to must not win on AIC.

    `dynamic_fit` compares orders by AIC, and a flat model's residuals are all
    large, so the robust weighting shrinks its SSR. It still scores far worse
    than a real fit, but it is not a candidate at all, so it goes in as inf the
    way `flat_fit_result` does.
    """
    from rm_lite.utils import fitting

    def _always_fails(*_args: object, **_kwargs: object) -> None:
        msg = "curve_fit forced to fail"
        raise RuntimeError(msg)

    monkeypatch.setattr(fitting.optimize, "curve_fit", _always_fails)
    _, _, _, stokes_i_arr, stokes_i_error_arr = _clean_spectrum()
    freq_arr_hz, ref_freq_hz = _band()
    fit = fitting.static_fit(
        freq_arr_hz, ref_freq_hz, stokes_i_arr, stokes_i_error_arr, 2, "log"
    )
    assert fit.aic == np.inf
    # Still a usable flat model at the mean, not an exception.
    model = fit.stokes_i_model_func(freq_arr_hz / ref_freq_hz, *np.asarray(fit.popt))
    np.testing.assert_allclose(model, np.mean(stokes_i_arr))
