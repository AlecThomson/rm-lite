"""Interferometric weighting: natural/uniform_lsq/briggs from the local lambda^2
sampling density (a grid-origin-independent gridded density)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from rm_lite.utils.synthesis import (
    FDFOptions,
    _lambda_sq_density,
    briggs_weight,
    freq_to_lambda2,
    natural_weight,
    uniform_lsq_weight,
)

# Uniform-in-frequency band (real channelised data): dense in lambda^2 at the
# high-frequency (small lambda^2) end, sparse at the low-frequency end.
FREQ_HZ = np.linspace(700e6, 1800e6, 300)
LAMBDA_SQ = freq_to_lambda2(FREQ_HZ)
NO_ERROR = np.zeros_like(FREQ_HZ)
CELL_M2 = float(np.sqrt(3.0) / 300.0)  # lambda^2 gridding cell


def _rmsf_fwhm(weight_arr: NDArray[np.float64]) -> float:
    """RMSF main-lobe FWHM (rad/m^2) by direct evaluation of the weighted DFT."""
    weight = weight_arr / np.nansum(weight_arr)
    phi = np.linspace(-500.0, 500.0, 40001)
    centred = LAMBDA_SQ - LAMBDA_SQ.mean()
    rmsf = np.abs(weight @ np.exp(-2j * np.outer(centred, phi)))
    rmsf /= rmsf.max()
    above = rmsf >= 0.5
    return float(phi[above][-1] - phi[above][0])


def test_natural_equals_variance() -> None:
    # natural weighting *is* inverse-variance weighting.
    error = np.linspace(0.5, 2.0, FREQ_HZ.size)
    np.testing.assert_allclose(natural_weight(error), 1.0 / error**2)
    # No noise -> all ones.
    np.testing.assert_array_equal(natural_weight(NO_ERROR), np.ones_like(FREQ_HZ))


def test_uniform_lsq_narrows_rmsf() -> None:
    # Interferometric uniform (density-compensated) narrows the main lobe vs
    # equal-per-channel weighting.
    fwhm_channel = _rmsf_fwhm(np.ones_like(FREQ_HZ))
    fwhm_lsq = _rmsf_fwhm(uniform_lsq_weight(LAMBDA_SQ, NO_ERROR, CELL_M2))
    assert fwhm_lsq < fwhm_channel


def test_briggs_interpolates_natural_and_uniform_lsq() -> None:
    fwhm_natural = _rmsf_fwhm(natural_weight(NO_ERROR))
    fwhm_lsq = _rmsf_fwhm(uniform_lsq_weight(LAMBDA_SQ, NO_ERROR, CELL_M2))

    fwhm_high = _rmsf_fwhm(briggs_weight(LAMBDA_SQ, NO_ERROR, 5.0, CELL_M2))
    fwhm_low = _rmsf_fwhm(briggs_weight(LAMBDA_SQ, NO_ERROR, -5.0, CELL_M2))

    assert fwhm_high == pytest.approx(fwhm_natural, rel=1e-2)
    assert fwhm_low == pytest.approx(fwhm_lsq, rel=1e-2)

    # Monotonic in robust: lowering robust narrows the RMSF towards uniform.
    fwhms = [
        _rmsf_fwhm(briggs_weight(LAMBDA_SQ, NO_ERROR, r, CELL_M2))
        for r in (5.0, 1.0, 0.0, -1.0, -5.0)
    ]
    assert fwhms == sorted(fwhms, reverse=True)


def test_uniform_lsq_gap_edge_not_overweighted() -> None:
    # Regression for the gap bug: a channel bordering a large lambda^2 gap must be
    # weighted by its LOCAL sampling density (few neighbours in its own cell), not
    # by the gap size. The old spacing-based weighting up-weighted the gap edges
    # ~5x the interior; the density weighting keeps them ~1.7x (an edge cell holds
    # about half the neighbours of an interior cell).
    cluster1 = np.linspace(0.030, 0.050, 50)
    cluster2 = np.linspace(0.150, 0.170, 50)  # big gap 0.05 -> 0.15
    lam2 = np.concatenate([cluster1, cluster2])
    err = np.zeros_like(lam2)
    cell = 0.002  # ~5 channels per cell within a cluster

    weight = uniform_lsq_weight(lam2, err, cell)
    density = _lambda_sq_density(lam2, cell)

    # weight is exactly natural / local density
    np.testing.assert_allclose(weight, 1.0 / density)
    # gap-edge cells (indices 49, 50) hold fewer channels than an interior cell
    assert density[49] < density[25]
    assert density[50] < density[25]
    # so the gap edges are NOT over-weighted (old spacing approach gave ~4.9x)
    assert weight.max() / np.median(weight) < 2.5


def test_uniform_lsq_robust_to_gaps() -> None:
    # End-to-end: a wide flagged gap must not collapse the sensitivity.
    freq = np.linspace(700e6, 1800e6, 300)
    freq = freq[~((freq > 800e6) & (freq < 1300e6))]  # drop a wide chunk
    lam2 = freq_to_lambda2(freq)
    err = np.zeros_like(freq)

    weight = uniform_lsq_weight(lam2, err, CELL_M2)
    weight = weight / weight.sum()
    efficiency = float(weight.sum() ** 2 / np.sum(weight**2) / len(weight))

    assert efficiency > 0.3, "sensitivity stays usable across the gap"


def test_fdf_options_validation() -> None:
    with pytest.raises(ValueError, match="weight_type must be one of"):
        FDFOptions(weight_type="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="requires a `robust`"):
        FDFOptions(weight_type="briggs")
    FDFOptions(weight_type="briggs", robust=0.0)  # ok
