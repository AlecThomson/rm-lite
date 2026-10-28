"""Tests for the Hogbom minor loop and its RMSF shift."""

from __future__ import annotations

import numpy as np
import pytest
from rm_lite.utils.clean import shift_rmsf


@pytest.mark.parametrize(("fdf_len", "n_phi_pad"), [(101, 50), (64, 64), (7, 3)])
def test_shift_rmsf_matches_roll_and_clip(fdf_len: int, n_phi_pad: int) -> None:
    """Every shift, wrapping or not, is the RMSF rolled to the channel and clipped."""
    rng = np.random.default_rng(fdf_len)
    double_len = fdf_len + 2 * n_phi_pad
    rmsf = rng.normal(size=double_len) + 1j * rng.normal(size=double_len)
    max_rmsf_index = int(np.argmax(np.abs(rmsf)))
    for fdf_index in range(fdf_len):
        rolled = np.roll(rmsf, fdf_index + n_phi_pad - max_rmsf_index)
        shifted = shift_rmsf(rmsf, fdf_index, n_phi_pad, max_rmsf_index)
        np.testing.assert_array_equal(shifted, rolled[n_phi_pad : n_phi_pad + fdf_len])
        assert not np.shares_memory(shifted, rmsf)
