"""Tests for the RM synthesis and related tools"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
from numpy.typing import NDArray
from rm_lite.tools_1d.rmsynth import run_rmsynth
from rm_lite.utils.logging import logger
from rm_lite.utils.synthesis import (
    FWHM,
    freq_to_lambda2,
    get_fwhm_rmsf,
    lambda2_to_freq,
    make_phi_arr,
    rmsynth_nufft,
)

RNG = np.random.default_rng()


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
    freqs = np.arange(744, 1032, 1) * 1e6
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
        # atol=fdf_parameters["peak_rm_fit_error"][0],
        atol=1,
    )

    assert np.isclose(
        fdf_parameters["frac_pol"].to_numpy()[0],
        racs_model.frac_pol,
        # atol=fdf_parameters["frac_pol_error"].to_numpy()[0],
        atol=0.1,
    )


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
    stokes_i_error_arr = np.load(test_data_path / "stokes_i_error_arr_bad_fit.npy")
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
@pytest.mark.filterwarnings("ignore: invalid value encountered in std_dev")
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
