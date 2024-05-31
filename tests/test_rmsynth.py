#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the RM synthesis and related tools"""

from typing import NamedTuple

import numpy as np
import pytest

from rm_lite.utils.synthesis import (
    FWHMRMSF,
    freq_to_lambda2,
    lambda2_to_freq,
    make_phi_array,
    get_fwhm_rmsf,
    rmsynth_nufft,
)


class MockData(NamedTuple):
    freqs: np.ndarray
    lsq: np.ndarray
    stokes_i: np.ndarray
    stokes_q: np.ndarray
    stokes_u: np.ndarray


class MockModel(NamedTuple):
    flux: float
    frac_pol: float
    rm: float
    pa_0: float
    fwhm: float


@pytest.fixture
def racs_model() -> MockModel:
    fwhm = 49.57
    rm = np.random.uniform(-1000, 1000)
    pa = np.random.uniform(0, 180)
    frac_pol = np.random.uniform(0.5, 0.7)
    flux = np.random.uniform(1, 10)

    return MockModel(flux, frac_pol, rm, pa, fwhm)


@pytest.fixture
def racs_data(racs_model):
    freqs = np.arange(744, 1032, 8) * 1e6
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
    fwhm: FWHMRMSF = get_fwhm_rmsf(racs_data.lsq, super_resolution=False)
    assert np.isclose(fwhm.fwhm_rmsf_radm2, racs_model.fwhm, atol=0.1)
    assert np.isclose(
        fwhm.d_lambda_sq_max_m2, np.nanmax(np.abs(np.diff(racs_data.lsq)))
    )
    assert np.isclose(
        fwhm.lambda_sq_range_m2,
        np.nanmax(racs_data.lsq) - np.nanmin(racs_data.lsq),
    )


def test_rmsynth_nufft(racs_data: MockData, racs_model: MockModel):
    phis = make_phi_array(
        phi_max_radm2=1000,
        d_phi_radm2=1,
    )
    fdf_dirty = rmsynth_nufft(
        stokes_q_array=racs_data.stokes_q,
        stokes_u_array=racs_data.stokes_u,
        lambda_sq_arr_m2=racs_data.lsq,
        phi_arr_radm2=phis,
        weight_array=np.ones_like(racs_data.stokes_q),
        lam_sq_0_m2=np.mean(racs_data.lsq),
    )

    peak_rm = phis[np.argmax(np.abs(fdf_dirty))]
    assert np.isclose(peak_rm, racs_model.rm, atol=1)
