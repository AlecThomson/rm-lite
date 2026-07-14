"""Interferometric weighting: natural/uniform_lsq/briggs from per-cell occupancy
on a virtual lambda^2 grid (inverse local density, no smoothing)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from rm_lite.utils.synthesis import (
    FDFOptions,
    briggs_weight,
    compute_rmsynth_params,
    freq_to_lambda2,
    natural_weight,
    uniform_lsq_weight,
)

# Uniform-in-frequency band (real channelised data): dense in lambda^2 at the
# high-frequency (small lambda^2) end, sparse at the low-frequency end.
FREQ_HZ = np.linspace(700e6, 1800e6, 300)
LAMBDA_SQ = freq_to_lambda2(FREQ_HZ)
ONES = np.ones_like(FREQ_HZ)
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
    np.testing.assert_array_equal(
        natural_weight(np.zeros_like(FREQ_HZ)), np.ones_like(FREQ_HZ)
    )


def test_uniform_lsq_narrows_rmsf() -> None:
    # Interferometric uniform (density-compensated) narrows the main lobe vs
    # equal-per-channel weighting.
    fwhm_channel = _rmsf_fwhm(np.ones_like(FREQ_HZ))
    fwhm_lsq = _rmsf_fwhm(uniform_lsq_weight(LAMBDA_SQ, ONES, CELL_M2))
    assert fwhm_lsq < fwhm_channel


def test_briggs_interpolates_natural_and_uniform_lsq() -> None:
    fwhm_natural = _rmsf_fwhm(ONES)
    fwhm_lsq = _rmsf_fwhm(uniform_lsq_weight(LAMBDA_SQ, ONES, CELL_M2))

    fwhm_high = _rmsf_fwhm(briggs_weight(LAMBDA_SQ, ONES, 5.0, CELL_M2))
    fwhm_low = _rmsf_fwhm(briggs_weight(LAMBDA_SQ, ONES, -5.0, CELL_M2))

    assert fwhm_high == pytest.approx(fwhm_natural, rel=1e-2)
    assert fwhm_low == pytest.approx(fwhm_lsq, rel=1e-2)

    # Monotonic in robust: lowering robust narrows the RMSF towards uniform.
    fwhms = [
        _rmsf_fwhm(briggs_weight(LAMBDA_SQ, ONES, r, CELL_M2))
        for r in (5.0, 1.0, 0.0, -1.0, -5.0)
    ]
    assert fwhms == sorted(fwhms, reverse=True)


def test_uniform_lsq_noise_independent() -> None:
    # uniform_lsq weights each channel by the lambda^2 interval it samples, with
    # the noise cancelling: two equally-sampled clusters contribute the same total
    # weight even with a 10x noise difference.
    lam2 = np.concatenate(
        [np.linspace(0.010, 0.0119, 20), np.linspace(0.050, 0.0519, 20)]
    )
    error = np.concatenate([np.full(20, 0.1), np.full(20, 1.0)])  # 10x noise gap
    weight = uniform_lsq_weight(lam2, natural_weight(error), 0.01)
    assert weight[:20].sum() == pytest.approx(weight[20:].sum(), rel=1e-6)


def test_uniform_lsq_uniform_within_cell() -> None:
    # Core grid guarantee: channels sharing a virtual cell get identical weight
    # (for equal natural weight), so no single channel jumps within a cell. Jumps
    # only happen between cells; that is genuine sampling density, not aliasing.
    lam2 = np.sort(freq_to_lambda2(np.linspace(700e6, 1800e6, 100)))
    weight = uniform_lsq_weight(lam2, np.ones_like(lam2), CELL_M2)
    cell_idx = np.floor((lam2 - lam2.min()) / CELL_M2).astype(int)
    for c in np.unique(cell_idx):
        in_cell = weight[cell_idx == c]
        np.testing.assert_allclose(in_cell, in_cell[0], rtol=1e-12)


def test_uniform_lsq_equal_per_cell() -> None:
    # Defining property of uniform weighting: each occupied cell contributes equal
    # total weight (flat noise), regardless of how many channels fall in it.
    lam2 = np.sort(freq_to_lambda2(np.linspace(700e6, 1800e6, 200)))
    weight = uniform_lsq_weight(lam2, np.ones_like(lam2), CELL_M2)
    cell_idx = np.floor((lam2 - lam2.min()) / CELL_M2).astype(int)
    totals = np.array([weight[cell_idx == c].sum() for c in np.unique(cell_idx)])
    np.testing.assert_allclose(totals, totals[0], rtol=1e-12)


def test_uniform_lsq_gap_bounded_and_local() -> None:
    # Punching an interior gap only affects the cells at the gap edges (occupancy
    # is per cell): a lone-channel cell is the weight ceiling (cell_m2 for flat
    # noise), and channels far from the gap keep their cell weight unchanged.
    freq = np.linspace(700e6, 1800e6, 300)
    keep = ~((freq > 1000e6) & (freq < 1300e6))  # punch a wide interior gap
    lam2 = freq_to_lambda2(freq)

    weight_full = uniform_lsq_weight(lam2, np.ones_like(lam2), CELL_M2)
    weight_gap = uniform_lsq_weight(lam2[keep], np.ones_like(lam2[keep]), CELL_M2)

    assert weight_gap.max() <= CELL_M2  # lone-channel cell is the ceiling
    far = np.r_[np.arange(30), np.arange(len(weight_gap) - 30, len(weight_gap))]
    np.testing.assert_allclose(weight_gap[far], weight_full[keep][far], rtol=1e-12)


def test_uniform_lsq_robust_to_gaps() -> None:
    # End-to-end: a wide flagged gap must not collapse the sensitivity.
    freq = np.linspace(700e6, 1800e6, 300)
    freq = freq[~((freq > 800e6) & (freq < 1300e6))]  # drop a wide chunk
    lam2 = freq_to_lambda2(freq)

    weight = uniform_lsq_weight(lam2, np.ones_like(freq), CELL_M2)
    weight = weight / weight.sum()
    efficiency = float(weight.sum() ** 2 / np.sum(weight**2) / len(weight))

    assert efficiency > 0.3, "sensitivity stays usable across the gap"


def test_flagged_channels_zeroed_neighbours_not_spiked() -> None:
    # NaN-flagged channels get zero natural weight and drop out of the occupancy,
    # so a channel bordering the flagged block is up-weighted only modestly
    # (its cell just has fewer channels), never a runaway spike.
    pol = np.ones_like(FREQ_HZ, dtype=np.complex128)
    pol[100:200] = np.nan
    pol_error = (0.1 + 0.1j) * np.ones_like(FREQ_HZ, dtype=np.complex128)
    options = FDFOptions(weight_type="uniform_lsq", n_samples=10.0)

    params = compute_rmsynth_params(FREQ_HZ, pol, pol_error, options)

    assert (params.weight_arr[100:200] == 0).all()
    good = params.weight_arr[params.weight_arr > 0]
    interior = np.median(good)
    # neighbours of the flagged block are up-weighted only modestly (bounded, not
    # a runaway spike)
    assert params.weight_arr[99] < 5.0 * interior
    assert params.weight_arr[200] < 5.0 * interior


def test_fdf_options_validation() -> None:
    with pytest.raises(ValueError, match="weight_type must be one of"):
        FDFOptions(weight_type="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="requires a `robust`"):
        FDFOptions(weight_type="briggs")
    FDFOptions(weight_type="briggs", robust=0.0)  # ok
