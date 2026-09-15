"""Interferometric weighting: natural/uniform_lsq/briggs from per-cell occupancy
on a virtual lambda^2 grid (inverse local density, no smoothing)."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest
from numpy.typing import NDArray
from rm_lite.utils.synthesis import (
    FDFOptions,
    WeightType,
    _lambda_sq_density,
    briggs_weight,
    compute_rmsynth_params,
    error_from_weight,
    freq_to_lambda2,
    natural_weight,
    uniform_lsq_weight,
    weighted_lam_sq_0,
)


class Band(NamedTuple):
    """A uniform-in-frequency band and the lambda^2 grid it samples."""

    freq_hz: NDArray[np.float64]
    lambda_sq: NDArray[np.float64]
    cell_m2: float
    """lambda^2 gridding cell."""


@pytest.fixture(scope="module")
def band() -> Band:
    """Real channelised data: dense in lambda^2 at the high-frequency end."""
    freq_hz = np.linspace(700e6, 1800e6, 300)
    return Band(freq_hz, freq_to_lambda2(freq_hz), float(np.sqrt(3.0) / 300.0))


@pytest.fixture
def flat_weight(band: Band) -> NDArray[np.float64]:
    """Equal weight per channel, the reference every grid weighting is read against."""
    return np.ones_like(band.freq_hz)


@pytest.fixture
def varying_error(band: Band) -> NDArray[np.float64]:
    """Per-channel RMS rising toward the low-frequency end, as real data does.

    Constant noise weights lambda^2 evenly, which makes the weighted and
    unweighted means equal, so the lambda^2_0 tests would pass without the fix.
    """
    return np.linspace(2e-3, 0.8e-3, band.freq_hz.size)


def rmsf_fwhm(weight_arr: NDArray[np.float64], lambda_sq: NDArray[np.float64]) -> float:
    """RMSF main-lobe FWHM (rad/m^2) by direct evaluation of the weighted DFT."""
    weight = weight_arr / np.nansum(weight_arr)
    phi = np.linspace(-500.0, 500.0, 40001)
    centred = lambda_sq - lambda_sq.mean()
    rmsf = np.abs(weight @ np.exp(-2j * np.outer(centred, phi)))
    rmsf /= rmsf.max()
    above = rmsf >= 0.5
    return float(phi[above][-1] - phi[above][0])


def test_natural_equals_variance(band: Band) -> None:
    # natural weighting *is* inverse-variance weighting.
    error = np.linspace(0.5, 2.0, band.freq_hz.size)
    np.testing.assert_allclose(natural_weight(error), 1.0 / error**2)
    # No noise -> all ones.
    np.testing.assert_array_equal(
        natural_weight(np.zeros_like(band.freq_hz)), np.ones_like(band.freq_hz)
    )


def test_natural_weight_drops_unusable_errors(
    varying_error: NDArray[np.float64],
) -> None:
    """A zero or blank error carries no information, so it weights zero."""
    error = varying_error
    error[7] = 0.0
    error[9] = np.nan
    error[11] = np.inf

    weight = natural_weight(error)

    assert np.array_equal(weight[[7, 9, 11]], np.zeros(3))
    assert np.isfinite(weight).all(), "an unusable error leaked an inf or nan"
    # Every other channel is untouched
    kept = np.setdiff1d(np.arange(error.size), [7, 9, 11])
    np.testing.assert_allclose(weight[kept], 1.0 / error[kept] ** 2)


def test_natural_weight_keeps_lam_sq_0_weighted(
    band: Band, varying_error: NDArray[np.float64]
) -> None:
    """A zero-error channel must not cost lambda^2_0 its weighting."""
    error = varying_error
    error[7] = 0.0

    lam_sq_0 = weighted_lam_sq_0(natural_weight(error), band.lambda_sq)

    # An infinite weight makes weighted_lam_sq_0 NaN (nansum keeps infinities),
    # and compute_rmsynth_params then falls back to the unweighted mean. That
    # moves the FDF's phase reference and rotates the polarisation angle.
    assert np.isfinite(lam_sq_0)
    unweighted = float(np.nanmean(band.lambda_sq))
    assert not np.isclose(lam_sq_0, unweighted), (
        "lambda^2_0 fell back to the unweighted mean"
    )
    # It is the weighted reference of the channels that survived
    expected = weighted_lam_sq_0(
        natural_weight(np.delete(error, 7)), np.delete(band.lambda_sq, 7)
    )
    np.testing.assert_allclose(lam_sq_0, expected)


def test_natural_weight_rejects_negative_error(
    varying_error: NDArray[np.float64],
) -> None:
    """A negative noise is a caller bug, not missing data."""
    # 1/error**2 would silently turn a small negative into a huge weight.
    error = varying_error
    error[5] = -6.1e-5

    with pytest.raises(ValueError, match="negative"):
        natural_weight(error)


def test_natural_weight_round_trips_error_from_weight() -> None:
    """Both directions agree that zero weight means infinite error."""
    weight = np.array([4.0, 0.0, 1.0, 0.25])

    error = error_from_weight(weight)
    assert np.array_equal(np.asarray(error).real, [0.5, np.inf, 1.0, 2.0])
    np.testing.assert_allclose(natural_weight(np.asarray(error).real), weight)


def test_uniform_lsq_narrows_rmsf(band: Band, flat_weight: NDArray[np.float64]) -> None:
    # Interferometric uniform (density-compensated) narrows the main lobe vs
    # equal-per-channel weighting.
    fwhm_channel = rmsf_fwhm(flat_weight, band.lambda_sq)
    fwhm_lsq = rmsf_fwhm(
        uniform_lsq_weight(band.lambda_sq, flat_weight, band.cell_m2), band.lambda_sq
    )
    assert fwhm_lsq < fwhm_channel


def test_briggs_interpolates_natural_and_uniform_lsq(
    band: Band, flat_weight: NDArray[np.float64]
) -> None:
    fwhm_natural = rmsf_fwhm(flat_weight, band.lambda_sq)
    fwhm_lsq = rmsf_fwhm(
        uniform_lsq_weight(band.lambda_sq, flat_weight, band.cell_m2), band.lambda_sq
    )

    fwhm_high = rmsf_fwhm(
        briggs_weight(band.lambda_sq, flat_weight, 5.0, band.cell_m2), band.lambda_sq
    )
    fwhm_low = rmsf_fwhm(
        briggs_weight(band.lambda_sq, flat_weight, -5.0, band.cell_m2), band.lambda_sq
    )

    assert fwhm_high == pytest.approx(fwhm_natural, rel=1e-2)
    assert fwhm_low == pytest.approx(fwhm_lsq, rel=1e-2)

    # Monotonic in robust: lowering robust narrows the RMSF towards uniform.
    fwhms = [
        rmsf_fwhm(
            briggs_weight(band.lambda_sq, flat_weight, r, band.cell_m2), band.lambda_sq
        )
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


def test_uniform_lsq_uniform_within_cell(band: Band) -> None:
    # Core grid guarantee: channels sharing a virtual cell get identical weight
    # (for equal natural weight), so no single channel jumps within a cell. Jumps
    # only happen between cells; that is genuine sampling density, not aliasing.
    lam2 = np.sort(freq_to_lambda2(np.linspace(700e6, 1800e6, 100)))
    weight = uniform_lsq_weight(lam2, np.ones_like(lam2), band.cell_m2)
    cell_idx = np.floor((lam2 - lam2.min()) / band.cell_m2).astype(int)
    for c in np.unique(cell_idx):
        in_cell = weight[cell_idx == c]
        np.testing.assert_allclose(in_cell, in_cell[0], rtol=1e-12)


def test_uniform_lsq_equal_per_cell(band: Band) -> None:
    # Defining property of uniform weighting: each occupied cell contributes equal
    # total weight (flat noise), regardless of how many channels fall in it.
    lam2 = np.sort(freq_to_lambda2(np.linspace(700e6, 1800e6, 200)))
    weight = uniform_lsq_weight(lam2, np.ones_like(lam2), band.cell_m2)
    cell_idx = np.floor((lam2 - lam2.min()) / band.cell_m2).astype(int)
    totals = np.array([weight[cell_idx == c].sum() for c in np.unique(cell_idx)])
    np.testing.assert_allclose(totals, totals[0], rtol=1e-12)


def test_uniform_lsq_gap_bounded_and_local(band: Band) -> None:
    # Punching an interior gap only affects the cells at the gap edges (occupancy
    # is per cell): a lone-channel cell is the weight ceiling (cell_m2 for flat
    # noise), and channels far from the gap keep their cell weight unchanged.
    freq = np.linspace(700e6, 1800e6, 300)
    keep = ~((freq > 1000e6) & (freq < 1300e6))  # punch a wide interior gap
    lam2 = freq_to_lambda2(freq)

    weight_full = uniform_lsq_weight(lam2, np.ones_like(lam2), band.cell_m2)
    weight_gap = uniform_lsq_weight(lam2[keep], np.ones_like(lam2[keep]), band.cell_m2)

    assert weight_gap.max() <= band.cell_m2  # lone-channel cell is the ceiling
    far = np.r_[np.arange(30), np.arange(len(weight_gap) - 30, len(weight_gap))]
    np.testing.assert_allclose(weight_gap[far], weight_full[keep][far], rtol=1e-12)


def test_uniform_lsq_robust_to_gaps(band: Band) -> None:
    # End-to-end: a wide flagged gap must not collapse the sensitivity.
    freq = np.linspace(700e6, 1800e6, 300)
    freq = freq[~((freq > 800e6) & (freq < 1300e6))]  # drop a wide chunk
    lam2 = freq_to_lambda2(freq)

    weight = uniform_lsq_weight(lam2, np.ones_like(freq), band.cell_m2)
    weight = weight / weight.sum()
    efficiency = float(weight.sum() ** 2 / np.sum(weight**2) / len(weight))

    assert efficiency > 0.3, "sensitivity stays usable across the gap"


def test_flagged_channels_zeroed_neighbours_not_spiked(band: Band) -> None:
    # NaN-flagged channels get zero natural weight and drop out of the occupancy,
    # so a channel bordering the flagged block is up-weighted only modestly
    # (its cell just has fewer channels), never a runaway spike.
    pol = np.ones_like(band.freq_hz, dtype=np.complex128)
    pol[100:200] = np.nan
    pol_error = (0.1 + 0.1j) * np.ones_like(band.freq_hz, dtype=np.complex128)
    options = FDFOptions(weight_type="uniform_lsq", n_samples=10.0)

    params = compute_rmsynth_params(band.freq_hz, pol, pol_error, options)

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


def test_lambda_sq_density_is_per_pixel(
    band: Band, flat_weight: NDArray[np.float64]
) -> None:
    """Each pixel gets its own lambda^2 occupancy, from its own flagging."""
    # A per-pixel weight array comes with lambda^2 broadcast to (n_freq, 1, 1);
    # indexing that with a 3D boolean selection used to raise IndexError, so
    # uniform_lsq and briggs could not be used with per-pixel weights at all.
    ny, nx = 3, 4
    n_freq = flat_weight.size
    weight_3d = np.broadcast_to(flat_weight[:, None, None], (n_freq, ny, nx)).copy()
    # One pixel keeps only the top half of the band, so its cells are its own.
    weight_3d[: n_freq // 2, 0, 0] = 0.0

    density_1d = _lambda_sq_density(band.lambda_sq, flat_weight, band.cell_m2)
    density_3d = _lambda_sq_density(
        band.lambda_sq[:, None, None], weight_3d, band.cell_m2
    )

    assert density_3d.shape == weight_3d.shape
    # An untouched pixel matches the per-channel answer exactly: pixels do not
    # pool into each other's cells.
    np.testing.assert_allclose(density_3d[:, 1, 1], density_1d)
    # The half-flagged pixel is zero where it is flagged, and its own density
    # elsewhere, computed from its own channels, not its neighbours'.
    assert (density_3d[: n_freq // 2, 0, 0] == 0).all()
    flagged = flat_weight.copy()
    flagged[: n_freq // 2] = 0.0
    np.testing.assert_allclose(
        density_3d[:, 0, 0], _lambda_sq_density(band.lambda_sq, flagged, band.cell_m2)
    )


@pytest.mark.parametrize("weight_type", ["uniform_lsq", "briggs"])
def test_grid_weighting_is_per_pixel(
    band: Band, flat_weight: NDArray[np.float64], weight_type: WeightType
) -> None:
    """uniform_lsq and briggs give each pixel the weights its own band earns."""
    ny, nx = 2, 3
    n_freq = flat_weight.size
    weight_3d = np.broadcast_to(flat_weight[:, None, None], (n_freq, ny, nx)).copy()
    weight_3d[:, 1, 2] *= 4.0  # a quieter pixel, same band shape

    def weights(
        lambda_sq: NDArray[np.float64], natural: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if weight_type == "briggs":
            return briggs_weight(lambda_sq, natural, 0.0, band.cell_m2)
        return uniform_lsq_weight(lambda_sq, natural, band.cell_m2)

    weight_1d = weights(band.lambda_sq, flat_weight)
    weight_pixel = weights(band.lambda_sq[:, None, None], weight_3d)

    assert weight_pixel.shape == weight_3d.shape
    # Scaling one pixel's noise scales its weights and nothing else, and the
    # normalised weighting (all that the FDF sees) is unchanged.
    for pixel in ((0, 0), (1, 2)):
        column = weight_pixel[(slice(None), *pixel)]
        np.testing.assert_allclose(
            column / column.sum(), weight_1d / weight_1d.sum(), rtol=1e-12
        )
