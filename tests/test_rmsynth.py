"""Tests for the RM synthesis and related tools"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
from numpy.typing import NDArray
from rm_lite.tools_1d.rmsynth import run_rmsynth
from rm_lite.utils.fitting import power_law
from rm_lite.utils.logging import logger
from rm_lite.utils.synthesis import (
    FWHM,
    calc_faraday_peaks,
    freq_to_lambda2,
    get_fwhm_rmsf,
    lambda2_to_freq,
    make_phi_arr,
    rmsynth_nufft,
)

# Seeded for reproducibility: an unseeded generator draws a fresh random RM
# each run, which occasionally lands on a near-Nyquist/aliased Faraday depth
# where the global moment integration legitimately picks up distant structure
# and the moment assertions below fail intermittently.
RNG = np.random.default_rng(1234)


class MockData(NamedTuple):
    freqs: NDArray[np.float64]
    lsq: NDArray[np.float64]
    stokes_i: NDArray[np.float64]
    stokes_q: NDArray[np.float64]
    stokes_u: NDArray[np.float64]


class MockModel(NamedTuple):
    flux: float
    frac_pol: float
    rm: float
    pa_0: float
    fwhm: float


@pytest.fixture
def test_data_path() -> Path:
    """Fixture to provide the path to the test data directory."""
    return Path(resources.files("rm_lite.data.tests"))  # type: ignore[arg-type]


@pytest.fixture
def racs_model() -> MockModel:
    fwhm = 49.57
    rm = RNG.uniform(-1000, 1000)
    pa = RNG.uniform(0, 180)
    frac_pol = RNG.uniform(0.5, 0.7)
    flux = RNG.uniform(1, 10)

    return MockModel(flux, frac_pol, rm, pa, fwhm)


@pytest.fixture
def racs_data(racs_model):
    freqs = (np.arange(744, 1032, 1) * 1e6).astype(np.float64)
    lsq = freq_to_lambda2(freqs)
    stokes_i = np.ones_like(freqs) * racs_model.flux
    stokes_q = (
        stokes_i
        * racs_model.frac_pol
        * np.cos(2 * racs_model.rm * lsq + 2 * racs_model.pa_0)
    )
    stokes_u = (
        stokes_i
        * racs_model.frac_pol
        * np.sin(2 * racs_model.rm * lsq + 2 * racs_model.pa_0)
    )
    return MockData(freqs, lsq, stokes_i, stokes_q, stokes_u)


def test_get_fwhm_rmsf(racs_data, racs_model):
    assert np.allclose(racs_data.lsq, freq_to_lambda2(lambda2_to_freq(racs_data.lsq)))
    fwhm: FWHM = get_fwhm_rmsf(racs_data.lsq)
    assert np.isclose(fwhm.fwhm_rmsf_radm2, racs_model.fwhm, atol=1)
    assert np.isclose(
        fwhm.d_lambda_sq_max_m2, np.nanmax(np.abs(np.diff(racs_data.lsq)))
    )
    assert np.isclose(
        fwhm.lambda_sq_range_m2,
        np.nanmax(racs_data.lsq) - np.nanmin(racs_data.lsq),
    )


def test_rmsynth_nufft(racs_data: MockData, racs_model: MockModel):
    phis = make_phi_arr(
        phi_max_radm2=1000,
        d_phi_radm2=1,
    )
    fdf_dirty = rmsynth_nufft(
        complex_pol_arr=racs_data.stokes_q + 1j * racs_data.stokes_u,
        lambda_sq_arr_m2=racs_data.lsq,
        phi_arr_radm2=phis,
        weight_arr=np.ones_like(racs_data.stokes_q),
        lam_sq_0_m2=float(np.mean(racs_data.lsq)),
    )

    peak_rm = phis[np.argmax(np.abs(fdf_dirty))]
    assert np.isclose(peak_rm, racs_model.rm, atol=1)


@pytest.mark.filterwarnings(
    "ignore: Covariance of the parameters could not be estimated"
)
def test_run_rmsynth(racs_data: MockData, racs_model: MockModel):
    complex_data = racs_data.stokes_q + 1j * racs_data.stokes_u
    complex_error = np.ones_like(racs_data.stokes_q) + 1j * np.ones_like(
        racs_data.stokes_u
    )
    complex_error *= 1e-3

    rmsyth_results = run_rmsynth(
        freq_arr_hz=racs_data.freqs,
        complex_pol_arr=complex_data,
        complex_pol_error=complex_error,
        stokes_i_arr=racs_data.stokes_i,
        stokes_i_error_arr=np.ones_like(racs_data.stokes_i) * 1e-3,
    )

    fdf_parameters = rmsyth_results.fdf_parameters
    logger.info(fdf_parameters)

    assert np.isclose(
        fdf_parameters["peak_rm_fit"][0],
        racs_model.rm,
        atol=1,
    )

    assert np.isclose(
        fdf_parameters["frac_pol"].to_numpy()[0],
        racs_model.frac_pol,
        atol=0.1,
    )

    assert fdf_parameters["moment_threshold_snr"][0] == 5.0

    # The synthetic data are noiseless, so the default 5-sigma moment cut sits
    # far below the RMSF sidelobes and the moments pick them up. Cut at half
    # the fitted peak instead so only the main lobe survives: mom1 recovers
    # the RM, mom0 is of order the peak polarised flux, and mom2 is below the
    # RMSF width.
    half_peak_snr = float(
        0.5 * fdf_parameters["peak_pi_fit"][0] / fdf_parameters["fdf_error_noise"][0]
    )
    fdf_parameters = run_rmsynth(
        freq_arr_hz=racs_data.freqs,
        complex_pol_arr=complex_data,
        complex_pol_error=complex_error,
        stokes_i_arr=racs_data.stokes_i,
        stokes_i_error_arr=np.ones_like(racs_data.stokes_i) * 1e-3,
        moment_threshold_snr=half_peak_snr,
    ).fdf_parameters
    expected_pi = racs_model.flux * racs_model.frac_pol
    assert np.isclose(fdf_parameters["mom1_radm2"][0], racs_model.rm, atol=1)
    assert 0 < fdf_parameters["mom0"][0] < 2 * expected_pi
    assert 0 < fdf_parameters["mom2_radm2"][0] < racs_model.fwhm
    # Bias correction shrinks mom0, and only slightly at this SNR
    assert 0 < fdf_parameters["mom0_debias"][0] <= fdf_parameters["mom0"][0]
    assert np.isclose(
        fdf_parameters["mom0_debias"][0], fdf_parameters["mom0"][0], rtol=0.01
    )


def test_peak_finders_agree(racs_data: MockData, racs_model: MockModel):
    """The cube peak finder must land on the same peak as the 1D Gaussian fit.

    `get_fdf_parameters` fits a Gaussian to the main lobe, `calc_faraday_peaks`
    interpolates a parabola through the brightest three samples; both then share
    `calc_peak_stats` for the angles and errors, so on a well-sampled RMSF the
    two must agree.
    """
    complex_data = racs_data.stokes_q + 1j * racs_data.stokes_u
    complex_error = np.full_like(racs_data.stokes_q, 1e-3 + 1e-3j, dtype=np.complex128)

    results = run_rmsynth(
        freq_arr_hz=racs_data.freqs,
        complex_pol_arr=complex_data,
        complex_pol_error=complex_error,
    )
    params = results.fdf_parameters
    fdf_arr = results.fdf_arrs["fdf_dirty_complex_arr"].to_numpy().astype(complex)
    phi_arr_radm2 = results.fdf_arrs["phi_arr_radm2"].to_numpy()

    peaks = calc_faraday_peaks(
        fdf_arr,
        phi_arr_radm2,
        params["fwhm_rmsf_radm2"][0],
        fdf_error=params["fdf_error_noise"][0],
        lam_sq_0_m2=params["lam_sq_0_m2"][0],
        lambda_sq_arr_m2=racs_data.lsq,
    )

    assert np.isclose(float(peaks.peak_rm_radm2), racs_model.rm, atol=1)
    assert np.isclose(float(peaks.peak_rm_radm2), params["peak_rm_fit"][0], atol=1)
    assert np.isclose(float(peaks.peak_pi), params["peak_pi_fit"][0], rtol=0.05)
    assert np.isclose(float(peaks.peak_pi_error), params["peak_pi_error"][0])
    # Angles wrap at 180 deg, so compare the wrapped difference.
    for peak_deg, column in (
        (float(peaks.peak_pa_deg), "peak_pa_fit_deg"),
        (float(peaks.peak_pa0_deg), "peak_pa0_fit_deg"),
    ):
        assert abs((peak_deg - params[column][0] + 90) % 180 - 90) < 2.0
    # The mock spectra use pa_0 as an angle in radians, so wrap it into degrees
    # before comparing with the recovered intrinsic angle.
    expected_pa0_deg = np.degrees(racs_model.pa_0) % 180
    assert abs((float(peaks.peak_pa0_deg) - expected_pa0_deg + 90) % 180 - 90) < 2.0
    for field, column in (
        (peaks.peak_rm_error_radm2, "peak_rm_fit_error"),
        (peaks.peak_pa_error_deg, "peak_pa_fit_deg_error"),
        (peaks.peak_pa0_error_deg, "peak_pa0_fit_deg_error"),
    ):
        assert np.isclose(float(field), params[column][0], rtol=0.05)


def test_2d_synth(racs_data: MockData, racs_model: MockModel):
    stokes_q = racs_data.stokes_q
    stokes_u = racs_data.stokes_u
    complex_pol_arr = stokes_q + 1j * stokes_u
    complesx_pol_2d = np.tile(complex_pol_arr, (10, 1))

    phis = make_phi_arr(
        phi_max_radm2=1000,
        d_phi_radm2=1,
    )
    weights = np.ones_like(stokes_q)
    lambda_0_m2 = float(np.mean(racs_data.lsq))

    with pytest.raises(
        ValueError,
        match=r"Data depth does not match lambda\^2 vector \((\d+) vs (\d+)\)\.",
    ):
        dirty_fdf = rmsynth_nufft(
            complex_pol_arr=complesx_pol_2d,
            lambda_sq_arr_m2=racs_data.lsq,
            phi_arr_radm2=phis,
            weight_arr=weights,
            lam_sq_0_m2=lambda_0_m2,
        )

    dirty_fdf = rmsynth_nufft(
        complex_pol_arr=complesx_pol_2d.T,
        lambda_sq_arr_m2=racs_data.lsq,
        phi_arr_radm2=phis,
        weight_arr=weights,
        lam_sq_0_m2=lambda_0_m2,
    )

    peak_rms = phis[:, np.newaxis][np.argmax(np.abs(dirty_fdf), axis=0)].squeeze()
    peak_pis = np.max(np.abs(dirty_fdf), axis=0)
    assert np.allclose(peak_rms, racs_model.rm, atol=1)
    assert np.allclose(peak_pis, racs_model.frac_pol * racs_model.flux, atol=0.1)


@pytest.mark.filterwarnings(
    "ignore: Covariance of the parameters could not be estimated"
)
@pytest.mark.filterwarnings("ignore: invalid value encountered in std_dev")
def test_real_data_bad_fit(test_data_path):
    # The following data from K. Rose caused the fit to the Stokes I spectrum to fail
    complex_spectrum = np.load(test_data_path / "complex_spectrum_bad_fit.npy")
    complex_noise = np.load(test_data_path / "complex_noise_bad.npy")
    stokes_i_arr = np.load(test_data_path / "stokes_i_arr_bad_fit.npy")
    stokes_i_error_arr = np.load(test_data_path / "stokes_i_error_arr_bad.npy")
    freq_hz = np.linspace(1116.0237779633926, 3116.97610232475, len(complex_spectrum))
    _ = run_rmsynth(
        freq_arr_hz=freq_hz,
        complex_pol_arr=complex_spectrum,
        complex_pol_error=complex_noise,
        do_fit_rmsf=True,
        stokes_i_arr=stokes_i_arr,
        stokes_i_error_arr=stokes_i_error_arr,
        fit_order=-5,
    )


@pytest.mark.filterwarnings(
    "ignore: Covariance of the parameters could not be estimated"
)
@pytest.mark.filterwarnings("ignore: invalid value encountered")
def test_real_data_bad_peak(test_data_path):
    # The following data from K. Rose caused the fit to the FDF to fail
    complex_spectrum = np.load(test_data_path / "complex_spectrum_bad_peak.npy")
    complex_noise = np.load(test_data_path / "complex_noise_bad.npy")
    freq_hz = np.linspace(1116.0237779633926, 3116.97610232475, len(complex_spectrum))
    _ = run_rmsynth(
        freq_arr_hz=freq_hz,
        complex_pol_arr=complex_spectrum,
        complex_pol_error=complex_noise,
        do_fit_rmsf=True,
        n_samples=100,
    )


@pytest.mark.filterwarnings(
    "ignore: Covariance of the parameters could not be estimated"
)
@pytest.mark.filterwarnings("ignore: invalid value encountered")
@pytest.mark.filterwarnings("ignore: overflow")
def test_real_data_bad_overflow(test_data_path):
    # The following data from K. Rose caused the fit to the FDF to fail
    complex_spectrum = np.load(test_data_path / "complex_spectrum_overflow.npy")
    complex_noise = np.load(test_data_path / "complex_noise_bad.npy")
    stokes_i_arr = np.load(test_data_path / "stokes_i_arr_overflow.npy")
    stokes_i_error_arr = np.load(test_data_path / "stokes_i_error_arr_bad.npy")
    freq_hz = np.linspace(1116.0237779633926, 3116.97610232475, len(complex_spectrum))
    _ = run_rmsynth(
        freq_arr_hz=freq_hz,
        complex_pol_arr=complex_spectrum,
        do_fit_rmsf=True,
        n_samples=50,
        complex_pol_error=complex_noise,
        stokes_i_arr=stokes_i_arr,
        stokes_i_error_arr=stokes_i_error_arr,
        fit_order=3,
        fit_function="log",
    )


@pytest.mark.filterwarnings(
    "ignore: Covariance of the parameters could not be estimated"
)
@pytest.mark.filterwarnings("ignore: invalid value encountered")
@pytest.mark.filterwarnings("ignore: divide by zero")
def test_real_data_bad_zero(test_data_path):
    # The following data from K. Rose caused the fit to the FDF to fail
    complex_spectrum = np.load(test_data_path / "complex_spectrum_zero_div.npy")
    complex_noise = np.load(test_data_path / "complex_noise_bad.npy")
    freq_hz = np.linspace(1116.0237779633926, 3116.97610232475, len(complex_spectrum))
    _ = run_rmsynth(
        freq_arr_hz=freq_hz,
        complex_pol_arr=complex_spectrum,
        do_fit_rmsf=True,
        n_samples=50,
        complex_pol_error=complex_noise,
    )


def test_stokes_i_terms_describe_the_fitted_model():
    """The 1D `stokes_i_terms` frame is the whole Stokes I model: evaluating it at
    the frequencies reproduces the model array the fit used."""
    freq_arr_hz = (np.arange(744, 1032, 1) * 1e6).astype(np.float64)
    lsq = freq_to_lambda2(freq_arr_hz)
    ref_freq_hz = float(lambda2_to_freq(float(np.mean(lsq))))
    flux, alpha = 3.4, -0.9
    stokes_i = flux * (freq_arr_hz / ref_freq_hz) ** alpha
    complex_pol = 0.4 * stokes_i * np.exp(2j * (60.0 * lsq + 0.3))

    results = run_rmsynth(
        freq_arr_hz=freq_arr_hz,
        complex_pol_arr=complex_pol,
        complex_pol_error=((1 + 1j) * np.full_like(freq_arr_hz, 1e-3)).astype(
            np.complex128
        ),
        stokes_i_arr=stokes_i,
        stokes_i_error_arr=np.full_like(freq_arr_hz, 1e-3),
        fit_order=1,
    )
    terms = results.stokes_i_terms
    assert terms["term_name"].to_list() == ["flux", "alpha"]
    assert terms["fit_function"].to_list() == ["log", "log"]
    # A pure power law, so the fitted flux and index are the input ones.
    np.testing.assert_allclose(terms["term_value"].to_numpy(), [flux, alpha], rtol=1e-4)
    assert (terms["term_error"].to_numpy() >= 0).all()

    # ref_freq_hz is the FDF's own reference frequency, and the terms rebuild the
    # model the fractional spectra were divided by.
    term_ref_freq = float(terms["ref_freq_hz"][0])
    assert np.isclose(term_ref_freq, ref_freq_hz, rtol=1e-6)
    rebuilt = power_law(len(terms) - 1)(
        freq_arr_hz / term_ref_freq, *terms["term_value"].to_numpy()
    )
    np.testing.assert_allclose(
        rebuilt, results.stokes_i_arrs["stokes_i_model_arr"].to_numpy(), rtol=1e-3
    )


def test_stokes_i_terms_empty_for_a_supplied_model():
    """Nothing was fitted when the model is handed in, so there are no terms."""
    freq_arr_hz = (np.arange(744, 1032, 4) * 1e6).astype(np.float64)
    lsq = freq_to_lambda2(freq_arr_hz)
    stokes_i_model = np.full_like(freq_arr_hz, 2.0)
    complex_pol = 0.4 * stokes_i_model * np.exp(2j * (30.0 * lsq))

    results = run_rmsynth(
        freq_arr_hz=freq_arr_hz,
        complex_pol_arr=complex_pol,
        complex_pol_error=((1 + 1j) * np.full_like(freq_arr_hz, 1e-3)).astype(
            np.complex128
        ),
        stokes_i_model_arr=stokes_i_model,
        stokes_i_model_error=np.full_like(freq_arr_hz, 1e-3),
    )
    assert results.stokes_i_terms.is_empty()
    assert results.stokes_i_terms.columns == [
        "term_name",
        "term_value",
        "term_error",
        "ref_freq_hz",
        "fit_function",
    ]
