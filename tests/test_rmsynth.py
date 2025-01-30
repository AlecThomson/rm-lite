"""Tests for the RM synthesis and related tools"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest
from numpy.typing import NDArray
from rm_lite.tools_1d.rmsynth import run_rmsynth
from rm_lite.utils.logging import logger
from rm_lite.utils.synthesis import (
    FWHM,
    arange,
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


def test_arange():
    paras_minimal_working_example = {
        "arange simple": {
            "start": 0,
            "stop": 7,
            "step": 1,
            "include_start": True,
            "include_stop": False,
            "res_exp": np.array([0, 1, 2, 3, 4, 5, 6]),
        },
        "stop not on grid": {
            "start": 0,
            "stop": 6.5,
            "step": 1,
            "include_start": True,
            "include_stop": False,
            "res_exp": np.array([0, 1, 2, 3, 4, 5, 6]),
        },
        "arange failing example: stop excl": {
            "start": 1,
            "stop": 1.3,
            "step": 0.1,
            "include_start": True,
            "include_stop": False,
            "res_exp": np.array([1.0, 1.1, 1.2]),
        },
        "arange failing example: stop incl": {
            "start": 1,
            "stop": 1.3,
            "step": 0.1,
            "include_start": True,
            "include_stop": True,
            "res_exp": np.array([1.0, 1.1, 1.2, 1.3]),
        },
        "arange failing example: stop excl + start excl": {
            "start": 1,
            "stop": 1.3,
            "step": 0.1,
            "include_start": False,
            "include_stop": False,
            "res_exp": np.array([1.1, 1.2]),
        },
        "arange failing example: stop incl + start excl": {
            "start": 1,
            "stop": 1.3,
            "step": 0.1,
            "include_start": False,
            "include_stop": True,
            "res_exp": np.array([1.1, 1.2, 1.3]),
        },
    }
    for desc, paras in paras_minimal_working_example.items():
        start, stop, step, include_start, include_stop, res_exp = paras.values()
        res = arange(
            start, stop, step, include_start=include_start, include_stop=include_stop
        )
        assert np.allclose(res, res_exp), (
            f"Unexpected result in {desc}: {res=}, {res_exp=}"
        )


def test_run_rmsynth(racs_data: MockData, racs_model: MockModel):
    complex_data = racs_data.stokes_q + 1j * racs_data.stokes_u
    complex_error = np.ones_like(racs_data.stokes_q) + 1j * np.ones_like(
        racs_data.stokes_u
    )
    complex_error *= 1e-3

    fdf_parameters, fdf_arrs, rmsf_arrs = run_rmsynth(
        freq_arr_hz=racs_data.freqs,
        complex_pol_arr=complex_data,
        complex_pol_error=complex_error,
        stokes_i_arr=racs_data.stokes_i,
        stokes_i_error_arr=np.ones_like(racs_data.stokes_i) * 1e-3,
    )

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
