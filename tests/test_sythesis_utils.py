#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the synthesis utilities"""

import numpy as np

from rm_lite.utils.synthesis import freq_to_lambda2, lambda2_to_freq


def test_conversion():
    freqs = np.random.uniform(1e6, 10e9, 1000)

    assert np.allclose(lambda2_to_freq(freq_to_lambda2(freqs)), freqs)
