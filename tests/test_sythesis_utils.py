#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the synthesis utilities"""

import numpy as np

from rm_lite.utils.synthesis import (
    freq_to_lambda2,
    lambda2_to_freq,
    make_phi_arr,
)


def test_freq_lsq_freq():
    freqs = np.random.uniform(1e6, 10e9, 1000)

    assert np.allclose(lambda2_to_freq(freq_to_lambda2(freqs)), freqs)


def test_lsq_freq_lsq():
    lambda2s = np.random.uniform(1e-6, 1, 1000)

    assert np.allclose(freq_to_lambda2(lambda2_to_freq(lambda2s)), lambda2s)


def test_phi_arr():
    for max_val in [100, 1000, 10000]:
        for step_val in [1, 10, 100]:
            phi_one = make_phi_arr(max_val, step_val)
            assert len(phi_one) == max_val // step_val * 2 + 1
            assert np.isclose(np.max(phi_one), max_val)
            assert np.isclose(np.min(phi_one), -max_val)
            # Check that middle pixel is 0
            assert phi_one[len(phi_one) // 2] == 0
