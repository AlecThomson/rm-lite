"""Tests for the dask-chunked 3D RM-synthesis and RM-CLEAN tools."""

from __future__ import annotations

import logging
import time
import warnings
from typing import Any, NamedTuple

import dask.array as da
import numpy as np
import pytest
import rm_lite.tools_3d.rmclean as rmclean3d_mod
import rm_lite.tools_3d.rmsynth as rmsynth3d_mod
import zarr
from astropy.io import fits
from astropy.io.fits import Header
from dask.base import compute
from numpy.typing import NDArray
from rm_lite.tools_1d.rmsynth import run_rmsynth
from rm_lite.tools_3d.rmclean import (
    RMClean3DResults,
    run_rmclean,
    run_rmclean_from_synth,
)
from rm_lite.tools_3d.rmsynth import (
    _match_chunks_to_fdf,
    rmsynth_3d,
    rmsynth_3d_from_fits,
)
from rm_lite.utils import fitting as fitting_mod
from rm_lite.utils.clean import RMCleanOptions, RMSynthArrays, rmclean
from rm_lite.utils.dask_io import (
    channel_chunk_size,
    estimate_channel_noise_mad,
    freq_arr_hz_from_header,
    read_fits_cube_channel_chunks,
    read_fits_cube_dask,
    spatial_chunk_size,
    write_zarr_group,
)
from rm_lite.utils.synthesis import (
    FDFOptions,
    WeightType,
    calc_faraday_moments,
    calc_faraday_peaks,
    compute_theoretical_noise,
    derotate_to,
    freq_to_lambda2,
    lambda2_to_freq,
)

RNG = np.random.default_rng(2025)

D_PHI_RADM2 = 1.0
MASK_THRESHOLD = 0.15
CLEAN_THRESHOLD = 0.03


class SyntheticCube(NamedTuple):
    freq_arr_hz: NDArray[np.float64]
    stokes_q: NDArray[np.float64]
    stokes_u: NDArray[np.float64]
    rm_map: NDArray[np.float64]
    pa_map: NDArray[np.float64]


@pytest.fixture
def synthetic_cube() -> SyntheticCube:
    """A small Stokes Q/U cube with a different RM/PA/noise draw per pixel."""
    freq_arr_hz = (np.arange(744, 1032, 3) * 1e6).astype(np.float64)
    lambda_sq_arr_m2 = freq_to_lambda2(freq_arr_hz)
    ny, nx = 7, 9

    rm_map = RNG.uniform(-150, 150, size=(ny, nx))
    pa_map = RNG.uniform(0, 180, size=(ny, nx))
    frac_pol = 0.7

    stokes_q = np.zeros((freq_arr_hz.size, ny, nx))
    stokes_u = np.zeros((freq_arr_hz.size, ny, nx))
    for j in range(ny):
        for i in range(nx):
            angle = 2 * rm_map[j, i] * lambda_sq_arr_m2 + 2 * np.deg2rad(pa_map[j, i])
            stokes_q[:, j, i] = frac_pol * np.cos(angle)
            stokes_u[:, j, i] = frac_pol * np.sin(angle)

    # Small per-pixel noise draw. No flagged channels here deliberately: 1D
    # per-pixel processing derives lam_sq_0_m2/weight_arr from that pixel's
    # own flagging, while the 3D orchestration uses one global lam_sq_0_m2
    # and per-channel weight_arr shared across every pixel (matching classic
    # RM-Tools 3D convention). A global flag would make the two legitimately
    # diverge, which isn't what this test is checking.
    stokes_q += RNG.normal(0, 0.01, stokes_q.shape)
    stokes_u += RNG.normal(0, 0.01, stokes_u.shape)

    return SyntheticCube(freq_arr_hz, stokes_q, stokes_u, rm_map, pa_map)


def _require_cube(cube: da.Array | None) -> da.Array:
    """Assert an optional result cube is populated, for the type checker."""
    assert cube is not None
    return cube


def _chunked(array: NDArray[np.float64], cy: int, cx: int) -> da.Array:
    return da.from_array(array, chunks=(-1, cy, cx))


@pytest.mark.filterwarnings("ignore: All channels masked")
def test_rmsynth_3d_matches_1d_per_pixel(synthetic_cube: SyntheticCube):
    """Chunked 3D dirty FDF must match rm_lite.tools_1d.rmsynth per pixel."""
    ny, nx = synthetic_cube.rm_map.shape
    # Deliberately uneven chunking (7, 9 not multiples of 3, 4) to force
    # size-1 edge blocks through the pipeline.
    q_dask = _chunked(synthetic_cube.stokes_q, 3, 4)
    u_dask = _chunked(synthetic_cube.stokes_u, 3, 4)

    result = rmsynth_3d(
        q_dask, u_dask, synthetic_cube.freq_arr_hz, d_phi_radm2=D_PHI_RADM2
    )
    fdf_cube = result.fdf_dirty_cube.compute()

    for j in range(ny):
        for i in range(nx):
            complex_pol = (
                synthetic_cube.stokes_q[:, j, i] + 1j * synthetic_cube.stokes_u[:, j, i]
            )
            ref = run_rmsynth(
                freq_arr_hz=synthetic_cube.freq_arr_hz,
                complex_pol_arr=complex_pol,
                complex_pol_error=np.ones_like(complex_pol),
                d_phi_radm2=D_PHI_RADM2,
                weight_type="uniform",
            )
            ref_fdf = ref.fdf_arrs["fdf_dirty_complex_arr"].to_numpy().astype(complex)
            np.testing.assert_allclose(fdf_cube[:, j, i], ref_fdf, atol=1e-8)


@pytest.mark.filterwarnings("ignore: All channels masked")
def test_rmclean_3d_matches_per_pixel_rmclean(synthetic_cube: SyntheticCube):
    """Chunked 3D RM-CLEAN must match calling utils.clean.rmclean per pixel."""
    ny, nx = synthetic_cube.rm_map.shape
    q_dask = _chunked(synthetic_cube.stokes_q, 3, 4)
    u_dask = _chunked(synthetic_cube.stokes_u, 3, 4)

    synth = rmsynth_3d(
        q_dask, u_dask, synthetic_cube.freq_arr_hz, d_phi_radm2=D_PHI_RADM2
    )
    clean = run_rmclean(
        synth.fdf_dirty_cube,
        synth.rmsf_arr,
        synth.phi_arr_radm2,
        synth.phi_double_arr_radm2,
        synth.fwhm_rmsf_radm2,
        mask=MASK_THRESHOLD,
        threshold=CLEAN_THRESHOLD,
    )
    clean_cube, model_cube, resid_cube, iter_map = compute(
        clean.clean_fdf_cube,
        clean.model_fdf_cube,
        clean.resid_fdf_cube,
        clean.iter_count_map,
    )

    dirty_cube = synth.fdf_dirty_cube.compute()

    for j in range(ny):
        for i in range(nx):
            ref = rmclean(
                RMSynthArrays(
                    dirty_fdf_arr=dirty_cube[:, j, i],
                    phi_arr_radm2=synth.phi_arr_radm2,
                    rmsf_arr=synth.rmsf_arr,
                    phi_double_arr_radm2=synth.phi_double_arr_radm2,
                    fwhm_rmsf_arr=np.array(synth.fwhm_rmsf_radm2),
                ),
                RMCleanOptions(mask=MASK_THRESHOLD, threshold=CLEAN_THRESHOLD),
            )
            np.testing.assert_allclose(
                clean_cube[:, j, i], ref.clean_fdf_arr, atol=1e-8
            )
            np.testing.assert_allclose(
                model_cube[:, j, i], ref.model_fdf_arr, atol=1e-8
            )
            np.testing.assert_allclose(
                resid_cube[:, j, i], ref.resid_fdf_arr, atol=1e-8
            )
            assert iter_map[j, i] == ref.clean_iter_arr

    # Regression guard for the old attempt's bug: the iteration-count map
    # must not silently come back all-zero.
    assert (iter_map > 0).any()


@pytest.mark.filterwarnings("ignore: All channels masked")
def test_rmclean_3d_block_runs_once_per_chunk(
    synthetic_cube: SyntheticCube, monkeypatch
):
    """The expensive per-pixel Hogbom loop must run once per chunk, not once per output."""
    call_count = 0
    original = rmclean3d_mod._rmclean_on_block

    def counting_wrapper(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(rmclean3d_mod, "_rmclean_on_block", counting_wrapper)

    q_dask = _chunked(synthetic_cube.stokes_q, 3, 4)
    u_dask = _chunked(synthetic_cube.stokes_u, 3, 4)
    synth = rmsynth_3d(
        q_dask, u_dask, synthetic_cube.freq_arr_hz, d_phi_radm2=D_PHI_RADM2
    )
    clean = rmclean3d_mod.run_rmclean(
        synth.fdf_dirty_cube,
        synth.rmsf_arr,
        synth.phi_arr_radm2,
        synth.phi_double_arr_radm2,
        synth.fwhm_rmsf_radm2,
        mask=MASK_THRESHOLD,
        threshold=CLEAN_THRESHOLD,
    )
    compute(
        clean.clean_fdf_cube,
        clean.model_fdf_cube,
        clean.resid_fdf_cube,
        clean.iter_count_map,
        scheduler="synchronous",
    )

    assert (
        call_count
        == synth.fdf_dirty_cube.numblocks[1] * synth.fdf_dirty_cube.numblocks[2]
    )


def _counting(monkeypatch, mod, name: str) -> dict[str, int]:
    """Patch `mod.name` with a wrapper counting real invocations."""
    counter = {"calls": 0}
    original = getattr(mod, name)

    def wrapper(*args, **kwargs):
        counter["calls"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(mod, name, wrapper)
    return counter


@pytest.mark.filterwarnings("ignore: All channels masked")
def test_mixed_batch_shares_upstream_work(monkeypatch):
    """A batch mixing clean-derived and synthesis-derived outputs must not refit.

    `run_rmclean` takes its input cubes apart with `to_delayed`, whose default
    `optimize_graph=True` can fuse an upstream per-block task into the block task
    RM-CLEAN consumes and drop its key. Anything else in the same `dask.compute`
    built on the same `RMSynth3DResults` then no longer shares that task: the
    Stokes I fit and the NUFFT run a second time per chunk. Both are more
    expensive than the CLEAN loop they feed.

    Whether the fusion happens depends on the dask version -- dask 2025.1.0 fuses
    here, 2026.6.0 does not -- so this pins the sharing rather than any one
    dask's optimiser.
    """
    # Coarse Faraday depths on purpose: a deep phi axis makes `_match_chunks_to_fdf`
    # rechunk, and the rechunk layer is itself a fusion barrier that would hide
    # what this test is checking.
    phi_max_radm2 = 20.0
    d_phi_radm2 = 2.0

    freq_arr_hz = (np.arange(744, 1032, 3) * 1e6).astype(np.float64)
    ny, nx = 6, 8
    ref_freq_hz = float(np.median(freq_arr_hz))
    stokes_i = RNG.uniform(1.0, 3.0, size=(ny, nx))[None] * (
        (freq_arr_hz / ref_freq_hz)[:, None, None] ** -0.8
    )
    rm_map = RNG.uniform(-100, 100, size=(ny, nx))
    angle = 2 * rm_map[None] * freq_to_lambda2(freq_arr_hz)[:, None, None]
    stokes_q = 0.6 * stokes_i * np.cos(angle)
    stokes_u = 0.6 * stokes_i * np.sin(angle)

    fit_calls = _counting(monkeypatch, fitting_mod, "_fit_stokes_i_block")
    nufft_calls = _counting(monkeypatch, rmsynth3d_mod, "rmsynth_nufft")
    clean_calls = _counting(monkeypatch, rmclean3d_mod, "_rmclean_on_block")

    synth = rmsynth3d_mod.rmsynth_3d(
        _chunked(stokes_q, 3, 4),
        _chunked(stokes_u, 3, 4),
        freq_arr_hz,
        stokes_i=_chunked(stokes_i, 3, 4),
        stokes_i_error=np.full(freq_arr_hz.size, 1e-3),
        phi_max_radm2=phi_max_radm2,
        d_phi_radm2=d_phi_radm2,
    )
    clean = rmclean3d_mod.run_rmclean(
        synth.fdf_dirty_cube,
        synth.rmsf_arr,
        synth.phi_arr_radm2,
        synth.phi_double_arr_radm2,
        synth.fwhm_rmsf_radm2,
        mask=MASK_THRESHOLD,
        threshold=CLEAN_THRESHOLD,
    )
    n_chunks = synth.fdf_dirty_cube.numblocks[1] * synth.fdf_dirty_cube.numblocks[2]
    assert synth.fdf_dirty_cube.chunks[1] == (3, 3), "rechunked, test premise gone"

    # `map_blocks` probes the block function once with a zero-size block while the
    # graph is built, to infer its meta. Count from after that, not from zero.
    for counter in (fit_calls, nufft_calls, clean_calls):
        counter["calls"] = 0

    compute(
        clean.mom0_map,
        clean.mom1_map,
        clean.mom2_map,
        synth.fdf_dirty_cube,
        _require_cube(synth.stokes_i_alpha_map),
        _require_cube(synth.stokes_i_ref_flux_map),
        _require_cube(synth.stokes_i_model_order_map),
        _require_cube(synth.stokes_i_coeff_cube),
        _require_cube(synth.stokes_i_coeff_error_cube),
        scheduler="synchronous",
    )

    assert fit_calls["calls"] == n_chunks
    assert nufft_calls["calls"] == n_chunks
    assert clean_calls["calls"] == n_chunks


@pytest.mark.filterwarnings("ignore: All channels masked")
def test_write_zarr_group_shares_computation_across_arrays(
    synthetic_cube: SyntheticCube, monkeypatch, tmp_path
):
    """write_zarr_group must not recompute a shared upstream graph once per array.

    rmclean_3d's four outputs all come from one per-chunk dask.delayed call;
    writing them with a naive per-array `to_zarr()` loop (call to_zarr, which
    defaults to compute=True, once per array) would rerun that delayed call
    once per array instead of once per chunk.
    """
    call_count = 0
    original = rmclean3d_mod._rmclean_on_block

    def counting_wrapper(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(rmclean3d_mod, "_rmclean_on_block", counting_wrapper)

    q_dask = _chunked(synthetic_cube.stokes_q, 3, 4)
    u_dask = _chunked(synthetic_cube.stokes_u, 3, 4)
    synth = rmsynth_3d(
        q_dask, u_dask, synthetic_cube.freq_arr_hz, d_phi_radm2=D_PHI_RADM2
    )
    clean = rmclean3d_mod.run_rmclean(
        synth.fdf_dirty_cube,
        synth.rmsf_arr,
        synth.phi_arr_radm2,
        synth.phi_double_arr_radm2,
        synth.fwhm_rmsf_radm2,
        mask=MASK_THRESHOLD,
        threshold=CLEAN_THRESHOLD,
    )

    write_zarr_group(
        tmp_path / "shared_compute.zarr",
        {
            "fdf_clean": clean.clean_fdf_cube,
            "fdf_model": clean.model_fdf_cube,
            "fdf_resid": clean.resid_fdf_cube,
            "iter_count": clean.iter_count_map,
        },
    )

    n_chunks = synth.fdf_dirty_cube.numblocks[1] * synth.fdf_dirty_cube.numblocks[2]
    assert call_count == n_chunks


@pytest.mark.filterwarnings("ignore: All channels masked")
def test_zarr_round_trip(synthetic_cube: SyntheticCube, tmp_path):
    """All six output cubes write and read back correctly via zarr."""
    q_dask = _chunked(synthetic_cube.stokes_q, 3, 4)
    u_dask = _chunked(synthetic_cube.stokes_u, 3, 4)
    synth = rmsynth_3d(
        q_dask,
        u_dask,
        synthetic_cube.freq_arr_hz,
        d_phi_radm2=D_PHI_RADM2,
        per_pixel_rmsf=True,
    )
    clean = run_rmclean(
        synth.fdf_dirty_cube,
        synth.rmsf_arr,
        synth.phi_arr_radm2,
        synth.phi_double_arr_radm2,
        synth.fwhm_rmsf_radm2,
        mask=MASK_THRESHOLD,
        threshold=CLEAN_THRESHOLD,
    )

    store = tmp_path / "rmlite_test.zarr"
    arrays = {
        "fdf_dirty": synth.fdf_dirty_cube,
        "rmsf": synth.rmsf_cube,
        "fdf_clean": clean.clean_fdf_cube,
        "fdf_model": clean.model_fdf_cube,
        "fdf_resid": clean.resid_fdf_cube,
        "iter_count": clean.iter_count_map,
    }
    write_zarr_group(store, arrays)

    group = zarr.open(store)
    for name, array in arrays.items():
        # atol accounts for finufft's non-bit-reproducible threaded summation:
        # the lazy graph is genuinely recomputed for this comparison (once by
        # to_zarr, once by .compute() here), and repeat NUFFT calls can differ
        # at the ~1e-15 level.
        np.testing.assert_allclose(group[name][:], array.compute(), atol=1e-10)


@pytest.mark.filterwarnings("ignore: All channels masked")
def test_rmclean_3d_multiscale_smoke(synthetic_cube: SyntheticCube):
    """Multiscale RM-CLEAN threads through the delayed/dask 3D path unbroken."""
    q_dask = _chunked(synthetic_cube.stokes_q, 3, 4)
    u_dask = _chunked(synthetic_cube.stokes_u, 3, 4)

    synth = rmsynth_3d(
        q_dask, u_dask, synthetic_cube.freq_arr_hz, d_phi_radm2=D_PHI_RADM2
    )
    clean = run_rmclean(
        synth.fdf_dirty_cube,
        synth.rmsf_arr,
        synth.phi_arr_radm2,
        synth.phi_double_arr_radm2,
        synth.fwhm_rmsf_radm2,
        mask=MASK_THRESHOLD,
        threshold=CLEAN_THRESHOLD,
        multiscale=True,
    )
    clean_cube, model_cube, iter_map = compute(
        clean.clean_fdf_cube, clean.model_fdf_cube, clean.iter_count_map
    )

    ny, nx = synthetic_cube.rm_map.shape
    assert clean_cube.shape[1:] == (ny, nx)
    assert np.isfinite(clean_cube).all()
    assert np.isfinite(model_cube).all()
    assert (iter_map > 0).any()


def test_freq_arr_hz_from_header_reads_spectral_wcs():
    header = Header()
    header["NAXIS"] = 3
    header["NAXIS1"] = 10
    header["NAXIS2"] = 10
    header["NAXIS3"] = 4
    header["CTYPE3"] = "FREQ"
    header["CRVAL3"] = 1.0e9
    header["CDELT3"] = 1.0e6
    header["CRPIX3"] = 1
    header["CUNIT3"] = "Hz"

    freq_arr_hz = freq_arr_hz_from_header(header, n_freq=4)

    np.testing.assert_allclose(freq_arr_hz, [1.000e9, 1.001e9, 1.002e9, 1.003e9])


def test_spatial_chunk_size_gives_full_width_bands_within_target():
    cy, cx = spatial_chunk_size(
        n_freq=100, ny=1000, nx=1000, itemsize=16, target_chunk_mb=32
    )
    # Full image width, so each channel's slice of a block is contiguous on disk.
    assert cx == 1000
    assert cy * cx * 100 * 16 <= 32 * 1024**2
    # Clipped to image dims when the target budget exceeds the whole image.
    assert spatial_chunk_size(
        n_freq=10, ny=5, nx=5, itemsize=16, target_chunk_mb=1024
    ) == (5, 5)
    # Never below one row, even when a single band overshoots the target.
    assert spatial_chunk_size(
        n_freq=100, ny=50, nx=1000, itemsize=16, target_chunk_mb=1e-6
    ) == (1, 1000)


def test_channel_chunk_size_respects_target_and_bounds():
    n_chan = channel_chunk_size(
        n_freq=100, ny=500, nx=500, itemsize=4, target_chunk_mb=4
    )
    assert n_chan * 500 * 500 * 4 <= 4 * 1024**2
    # Clipped to the spectral axis, and never below one channel.
    assert channel_chunk_size(n_freq=7, ny=4, nx=4, itemsize=4, target_chunk_mb=99) == 7
    assert (
        channel_chunk_size(n_freq=7, ny=400, nx=400, itemsize=4, target_chunk_mb=1e-6)
        == 1
    )


def test_estimate_channel_noise_mad_recovers_true_noise():
    rng = np.random.default_rng(99)
    n_freq, ny, nx = 20, 64, 64
    true_noise = np.linspace(0.05, 0.5, n_freq)
    stokes_q = rng.normal(0, 1, (n_freq, ny, nx)) * true_noise[:, None, None]
    stokes_u = rng.normal(0, 1, (n_freq, ny, nx)) * true_noise[:, None, None]

    q_dask = _chunked(stokes_q, 16, 16)
    u_dask = _chunked(stokes_u, 16, 16)

    noise = estimate_channel_noise_mad(q_dask, u_dask)

    assert isinstance(noise, np.ndarray)
    assert noise.shape == (n_freq,)
    np.testing.assert_allclose(noise, true_noise, rtol=0.1)


def test_estimate_channel_noise_mad_is_spatial_chunk_invariant():
    rng = np.random.default_rng(7)
    n_freq, ny, nx = 12, 32, 32
    stokes_q = rng.normal(0, 1, (n_freq, ny, nx))
    stokes_u = rng.normal(0, 1, (n_freq, ny, nx))

    fine = estimate_channel_noise_mad(
        _chunked(stokes_q, 8, 8), _chunked(stokes_u, 8, 8)
    )
    coarse = estimate_channel_noise_mad(
        _chunked(stokes_q, -1, -1), _chunked(stokes_u, -1, -1)
    )
    np.testing.assert_array_equal(fine, coarse)


def test_estimate_channel_noise_mad_shape_mismatch_raises():
    stokes_q = _chunked(np.zeros((5, 4, 4)), 2, 2)
    stokes_u = _chunked(np.zeros((5, 3, 3)), 2, 2)
    with pytest.raises(ValueError, match="same shape"):
        estimate_channel_noise_mad(stokes_q, stokes_u)


@pytest.mark.filterwarnings("ignore: All channels masked")
def test_rmclean_3d_moment_maps(synthetic_cube: SyntheticCube):
    """3D RM-CLEAN exposes Faraday moment maps matching a direct call."""
    q_dask = _chunked(synthetic_cube.stokes_q, 3, 4)
    u_dask = _chunked(synthetic_cube.stokes_u, 3, 4)

    synth = rmsynth_3d(
        q_dask, u_dask, synthetic_cube.freq_arr_hz, d_phi_radm2=D_PHI_RADM2
    )
    moment_threshold = 5.0 * synth.theoretical_noise.fdf_error_noise
    clean = run_rmclean(
        synth.fdf_dirty_cube,
        synth.rmsf_arr,
        synth.phi_arr_radm2,
        synth.phi_double_arr_radm2,
        synth.fwhm_rmsf_radm2,
        mask=MASK_THRESHOLD,
        threshold=CLEAN_THRESHOLD,
        moment_threshold=moment_threshold,
    )

    assert isinstance(clean.mom0_map, da.Array)
    assert clean.mom0_map.shape == synthetic_cube.rm_map.shape

    ref = calc_faraday_moments(
        clean.clean_fdf_cube,
        synth.phi_arr_radm2,
        synth.fwhm_rmsf_radm2,
        threshold=moment_threshold,
    )
    m0, m1, m2, r0, r1, r2 = compute(
        clean.mom0_map, clean.mom1_map, clean.mom2_map, ref.mom0, ref.mom1, ref.mom2
    )
    np.testing.assert_allclose(m0, r0, equal_nan=True)
    np.testing.assert_allclose(m1, r1, equal_nan=True)
    np.testing.assert_allclose(m2, r2, equal_nan=True)


@pytest.mark.filterwarnings("ignore: All channels masked")
def test_rmclean_3d_from_synth_moment_maps(synthetic_cube: SyntheticCube):
    """The from_synth convenience wrapper scales the moment threshold from SNR."""
    q_dask = _chunked(synthetic_cube.stokes_q, 3, 4)
    u_dask = _chunked(synthetic_cube.stokes_u, 3, 4)

    synth = rmsynth_3d(
        q_dask, u_dask, synthetic_cube.freq_arr_hz, d_phi_radm2=D_PHI_RADM2
    )
    clean = run_rmclean_from_synth(synth, moment_threshold_snr=5.0)

    assert isinstance(clean.mom1_map, da.Array)
    assert clean.mom1_map.shape == synthetic_cube.rm_map.shape
    # Every pixel holds a strong polarised source, so the moment maps must be
    # populated (positive flux, finite mean Faraday depth), not all-NaN.
    mom0, mom1 = compute(clean.mom0_map, clean.mom1_map)
    assert (mom0 > 0).any()
    assert np.isfinite(mom1).any()


@pytest.mark.filterwarnings("ignore: All channels masked")
def test_rmclean_3d_peak_maps(synthetic_cube: SyntheticCube):
    """The peak maps come off the clean result, and match `calc_faraday_peaks`."""
    q_dask = _chunked(synthetic_cube.stokes_q, 3, 4)
    u_dask = _chunked(synthetic_cube.stokes_u, 3, 4)

    synth = rmsynth_3d(
        q_dask, u_dask, synthetic_cube.freq_arr_hz, d_phi_radm2=D_PHI_RADM2
    )
    clean = run_rmclean(
        synth.fdf_dirty_cube,
        synth.rmsf_arr,
        synth.phi_arr_radm2,
        synth.phi_double_arr_radm2,
        synth.fwhm_rmsf_radm2,
        mask=MASK_THRESHOLD,
        threshold=CLEAN_THRESHOLD,
        fdf_noise=synth.theoretical_noise.fdf_error_noise,
        lam_sq_0_m2=synth.lam_sq_0_map,
        lambda_sq_arr_m2=synth.lambda_sq_arr_m2,
    )

    assert isinstance(clean.peak_rm_map, da.Array)
    assert clean.peak_rm_map.shape == synthetic_cube.rm_map.shape

    ref = calc_faraday_peaks(
        clean.clean_fdf_cube,
        synth.phi_arr_radm2,
        synth.fwhm_rmsf_radm2,
        fdf_error=synth.theoretical_noise.fdf_error_noise,
        lam_sq_0_m2=synth.lam_sq_0_map,
        lambda_sq_arr_m2=synth.lambda_sq_arr_m2,
    )
    maps = (
        clean.peak_pi_map,
        clean.peak_pi_debias_map,
        clean.peak_pi_error_map,
        clean.peak_rm_map,
        clean.peak_rm_error_map,
        clean.peak_pa_map,
        clean.peak_pa_error_map,
        clean.peak_pa0_map,
        clean.peak_pa0_error_map,
    )
    for peak_map, ref_map in zip(compute(*maps), compute(*ref), strict=True):
        np.testing.assert_allclose(peak_map, ref_map, equal_nan=True)


@pytest.mark.filterwarnings("ignore: All channels masked")
def test_rmclean_3d_from_synth_peak_maps(synthetic_cube: SyntheticCube):
    """Every pixel holds a bright source, so the peak maps must recover it."""
    q_dask = _chunked(synthetic_cube.stokes_q, 3, 4)
    u_dask = _chunked(synthetic_cube.stokes_u, 3, 4)

    synth = rmsynth_3d(
        q_dask, u_dask, synthetic_cube.freq_arr_hz, d_phi_radm2=D_PHI_RADM2
    )
    clean = run_rmclean_from_synth(synth)

    peak_pi, peak_rm, peak_pa0, peak_rm_error = compute(
        clean.peak_pi_map,
        clean.peak_rm_map,
        clean.peak_pa0_map,
        clean.peak_rm_error_map,
    )
    assert np.isfinite(peak_pi).all()
    np.testing.assert_allclose(peak_rm, synthetic_cube.rm_map, atol=1.0)
    np.testing.assert_allclose(peak_pi, 0.7, rtol=0.05)
    # The intrinsic angle wraps at 180 deg, so compare the wrapped difference.
    pa0_offset = (peak_pa0 - synthetic_cube.pa_map + 90) % 180 - 90
    assert np.abs(pa0_offset).max() < 2.0
    # The depth error scales as the RMSF width over the SNR, so a detection
    # pins the peak down to better than the RMSF itself.
    assert (peak_rm_error > 0).all()
    assert (peak_rm_error < synth.fwhm_rmsf_radm2).all()


@pytest.mark.filterwarnings("ignore: All channels masked")
@pytest.mark.parametrize("weight_type", ["variance", "uniform"])
def test_per_pixel_weight_matches_the_shared_one(
    synthetic_cube: SyntheticCube, weight_type: WeightType
):
    """A per-pixel weight cube that is uniform must reduce to the per-channel one.

    Three ways it did not. `compute_theoretical_noise` sums over every element it
    is handed, so a 3D weight ran the sums over ny*nx times as many terms and
    reported a noise sqrt(ny*nx) too small -- which `run_rmclean_from_synth`
    scales its CLEAN mask and threshold from, so CLEAN saw a threshold orders of
    magnitude too low. It also came back as a lazy dask scalar, which
    `run_rmclean_from_synth` cannot format or compare. And the shared RMSF took a
    spatial mean of the whole cube where one pixel is what it documents.
    """
    q_dask = _chunked(synthetic_cube.stokes_q, 3, 4)
    u_dask = _chunked(synthetic_cube.stokes_u, 3, 4)
    n_freq = synthetic_cube.freq_arr_hz.size
    ny, nx = synthetic_cube.rm_map.shape
    weight_1d = 1.0 / np.linspace(1e-3, 3e-3, n_freq) ** 2
    weight_3d = np.broadcast_to(weight_1d[:, None, None], (n_freq, ny, nx)).copy()

    def run(weight_arr):
        synth = rmsynth_3d(
            q_dask,
            u_dask,
            synthetic_cube.freq_arr_hz,
            weight_arr=weight_arr,
            weight_type=weight_type,
            d_phi_radm2=D_PHI_RADM2,
        )
        clean = run_rmclean_from_synth(synth)
        return synth, clean

    shared_synth, shared_clean = run(weight_1d)
    pixel_synth, pixel_clean = run(weight_3d)

    # A per-pixel weight gives a per-pixel noise map; a uniform one must put the
    # per-channel answer in every pixel of it.
    pixel_noise = np.asarray(pixel_synth.theoretical_noise.fdf_error_noise)
    assert pixel_noise.shape == (ny, nx)
    np.testing.assert_allclose(
        pixel_noise,
        np.full((ny, nx), shared_synth.theoretical_noise.fdf_error_noise),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        pixel_synth.rmsf_arr, shared_synth.rmsf_arr, rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(
        pixel_synth.lam_sq_0_m2, shared_synth.lam_sq_0_m2, rtol=1e-12
    )

    shared_fdf, pixel_fdf, shared_iters, pixel_iters = compute(
        shared_clean.clean_fdf_cube,
        pixel_clean.clean_fdf_cube,
        shared_clean.iter_count_map,
        pixel_clean.iter_count_map,
    )
    np.testing.assert_allclose(pixel_fdf, shared_fdf, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(pixel_iters, shared_iters)


def _varying_noise_cube(ny: int = 6, nx: int = 6, span: float = 100.0):
    """Q/U with the noise rising across the field, as a mosaic edge does."""
    n_freq = 32
    freq_arr_hz = np.linspace(0.9e9, 1.4e9, n_freq)
    lambda_sq = freq_to_lambda2(freq_arr_hz)
    angle = 2 * (
        RNG.uniform(-60, 60, (ny, nx))[None] * lambda_sq[:, None, None]
        + RNG.uniform(0, np.pi, (ny, nx))[None]
    )
    sigma_map = np.geomspace(1e-4, 1e-4 * span, ny * nx).reshape(ny, nx)
    sigma = np.broadcast_to(sigma_map[None], (n_freq, ny, nx)).copy()
    stokes_q = 0.6 * np.cos(angle) + RNG.normal(0, 1, sigma.shape) * sigma
    stokes_u = 0.6 * np.sin(angle) + RNG.normal(0, 1, sigma.shape) * sigma
    return freq_arr_hz, stokes_q, stokes_u, sigma


@pytest.mark.filterwarnings("ignore: All channels masked")
def test_noise_map_is_right_pixel_by_pixel():
    """Spatially varying noise must give each pixel its own theoretical noise.

    `compute_theoretical_noise` used to sum over the spatial axes too, so one
    scalar stood for the whole image and it was sqrt(ny*nx) below even the
    quietest pixel. CLEAN scales its mask and threshold from this, so the number
    has to be each pixel's own.
    """
    freq_arr_hz, stokes_q, stokes_u, sigma = _varying_noise_cube()
    weight_arr = 1.0 / sigma**2

    synth = rmsynth_3d(
        _chunked(stokes_q, 3, 3),
        _chunked(stokes_u, 3, 3),
        freq_arr_hz,
        weight_arr=weight_arr,
        d_phi_radm2=D_PHI_RADM2,
    )
    noise_map = np.asarray(synth.theoretical_noise.fdf_error_noise)
    ny, nx = sigma.shape[1:]
    assert noise_map.shape == (ny, nx)

    # Each pixel against the same formula run on that pixel's own spectrum.
    for pixel in ((0, 0), (0, nx - 1), (ny - 1, nx - 1)):
        column = sigma[(slice(None), *pixel)]
        expected = compute_theoretical_noise(
            (column + 1j * column).astype(np.complex128),
            1.0 / column**2,
        ).fdf_error_noise
        np.testing.assert_allclose(noise_map[pixel], expected, rtol=1e-12)

    # And the map really does span the field's dynamic range, not one value.
    assert noise_map.max() / noise_map.min() > 50


@pytest.mark.filterwarnings("ignore: All channels masked")
def test_clean_thresholds_follow_the_noise_map():
    """A noisy pixel must not be cleaned to a quiet pixel's threshold.

    With one scalar threshold for the image, the noisiest pixels get a cut far
    below their own noise and CLEAN grinds into it.
    """
    freq_arr_hz, stokes_q, stokes_u, sigma = _varying_noise_cube()
    synth = rmsynth_3d(
        _chunked(stokes_q, 3, 3),
        _chunked(stokes_u, 3, 3),
        freq_arr_hz,
        weight_arr=1.0 / sigma**2,
        d_phi_radm2=D_PHI_RADM2,
    )
    noise_map = np.asarray(synth.theoretical_noise.fdf_error_noise)
    per_pixel = run_rmclean_from_synth(synth)

    # The same run, but told the whole image has the quietest pixel's noise:
    # that is what a single scalar amounts to.
    flat = synth._replace(
        theoretical_noise=synth.theoretical_noise._replace(
            fdf_error_noise=float(noise_map.min())
        )
    )
    uniform = run_rmclean_from_synth(flat)

    per_pixel_iters, uniform_iters = compute(
        per_pixel.iter_count_map, uniform.iter_count_map
    )
    noisiest = np.unravel_index(np.argmax(noise_map), noise_map.shape)
    assert per_pixel_iters[noisiest] < uniform_iters[noisiest]
    quietest = np.unravel_index(np.argmin(noise_map), noise_map.shape)
    assert per_pixel_iters[quietest] == uniform_iters[quietest]


@pytest.mark.filterwarnings("ignore: All channels masked")
def test_shared_rmsf_kept_when_pixels_only_differ_in_scale():
    """Noise varying across the image but not with frequency still shares an RMSF.

    The RMSF is normalised by the weight sum, so pixels whose weights are scalar
    multiples of each other have the same one. Keeping the shared spectrum there
    is what stops per-pixel noise costing a whole extra cube.
    """
    freq_arr_hz, stokes_q, stokes_u, sigma = _varying_noise_cube(ny=4, nx=4)
    synth = rmsynth_3d(
        _chunked(stokes_q, 2, 2),
        _chunked(stokes_u, 2, 2),
        freq_arr_hz,
        weight_arr=1.0 / sigma**2,
        d_phi_radm2=D_PHI_RADM2,
    )
    assert synth.rmsf_cube is None

    per_pixel = rmsynth_3d(
        _chunked(stokes_q, 2, 2),
        _chunked(stokes_u, 2, 2),
        freq_arr_hz,
        weight_arr=1.0 / sigma**2,
        d_phi_radm2=D_PHI_RADM2,
        per_pixel_rmsf=True,
    )
    cube = _require_cube(per_pixel.rmsf_cube).compute()
    for pixel in ((0, 0), (3, 3)):
        np.testing.assert_allclose(
            cube[(slice(None), *pixel)], synth.rmsf_arr, rtol=1e-10, atol=1e-12
        )


@pytest.mark.filterwarnings("ignore: All channels masked")
def test_per_pixel_rmsf_forced_when_spectra_differ(caplog):
    """When the noise spectrum itself varies per pixel, no one RMSF fits.

    Nothing asked for the cube here; it is switched on because a shared spectrum
    would silently be wrong for most of the image.
    """
    freq_arr_hz, stokes_q, stokes_u, _ = _varying_noise_cube(ny=4, nx=4)
    ny, nx = 4, 4
    # Each pixel's band tilts differently, so no rescaling maps one onto another.
    tilt = np.linspace(-1.0, 1.0, ny * nx).reshape(ny, nx)
    sigma = 1e-3 * (freq_arr_hz[:, None, None] / freq_arr_hz[0]) ** tilt[None]

    with caplog.at_level(logging.INFO, logger="rm-lite"):
        synth = rmsynth_3d(
            _chunked(stokes_q, 2, 2),
            _chunked(stokes_u, 2, 2),
            freq_arr_hz,
            weight_arr=1.0 / sigma**2,
            d_phi_radm2=D_PHI_RADM2,
        )

    assert synth.rmsf_cube is not None
    assert "do not share one" in caplog.text
    cube = _require_cube(synth.rmsf_cube).compute()
    corner_to_corner = np.max(np.abs(cube[:, 0, 0] - cube[:, -1, -1])) / np.max(
        np.abs(cube[:, 0, 0])
    )
    assert corner_to_corner > 0.1


def _mosaic_cube(ny: int = 8, nx: int = 8, edge: float = 0.78):
    """A linmos-like cube: the primary beam shrinks with frequency, so pixels
    near the field edge lose the top of the band and the number of contributing
    channels varies across the image."""
    n_freq = 96
    freq_arr_hz = np.linspace(744e6, 1032e6, n_freq)
    lambda_sq = freq_to_lambda2(freq_arr_hz)
    yy, xx = np.mgrid[0:ny, 0:nx]
    radius = np.sqrt((yy - (ny - 1) / 2) ** 2 + (xx - (nx - 1) / 2) ** 2)
    radius = edge * radius / radius.max()
    # Primary beam radius goes as 1/nu.
    inside = radius[None] <= (freq_arr_hz[0] / freq_arr_hz)[:, None, None]
    angle = 2 * (
        RNG.uniform(-60, 60, (ny, nx))[None] * lambda_sq[:, None, None]
        + RNG.uniform(0, np.pi, (ny, nx))[None]
    )
    stokes_q = np.where(inside, 0.6 * np.cos(angle), np.nan)
    stokes_u = np.where(inside, 0.6 * np.sin(angle), np.nan)
    # linmos blanks the noise cube the same way it blanks the data.
    weight_arr = np.where(inside, 1.0 / 1e-3**2, np.nan)
    return freq_arr_hz, stokes_q, stokes_u, weight_arr, inside


@pytest.mark.filterwarnings("ignore: All channels masked")
def test_mosaic_edge_channels_are_handled():
    """A frequency-dependent primary beam blanks the band per pixel.

    The blanks arrive as NaN in both the data and the mosaicked noise cube.
    Summing those through made every partially blanked channel NaN, and a NaN
    channel weight drops that channel from `lam_sq_0_m2` altogether -- silently
    reweighting the cube onto whichever channels happen to be complete.
    """
    freq_arr_hz, stokes_q, stokes_u, weight_arr, inside = _mosaic_cube()
    n_freq = freq_arr_hz.size
    n_chan = inside.sum(axis=0)
    assert n_chan.min() < n_chan.max() == n_freq, "premise: the edge loses channels"

    synth = rmsynth_3d(
        _chunked(stokes_q, 4, 4),
        _chunked(stokes_u, 4, 4),
        freq_arr_hz,
        weight_arr=weight_arr,
        phi_max_radm2=200.0,
        d_phi_radm2=2.0,
    )

    # lam_sq_0_m2 must weight every channel by how many pixels actually kept it.
    lambda_sq = freq_to_lambda2(freq_arr_hz)
    channel_weight = inside.sum(axis=(1, 2))
    expected = float(np.sum(channel_weight * lambda_sq) / np.sum(channel_weight))
    np.testing.assert_allclose(synth.lam_sq_0_m2, expected, rtol=1e-12)

    # Pixels no longer weight the channels alike, so they cannot share an RMSF:
    # the per-pixel cube is switched on without being asked for.
    cube = _require_cube(synth.rmsf_cube).compute()
    ny, nx = n_chan.shape
    centre = cube[:, ny // 2, nx // 2]
    corner = cube[:, 0, 0]
    assert np.max(np.abs(centre - corner)) / np.max(np.abs(centre)) > 0.1

    # And an edge pixel's noise reflects the channels it actually has.
    noise_map = np.asarray(synth.theoretical_noise.fdf_error_noise)
    assert noise_map.shape == (ny, nx)
    quiet, loud = np.argmax(n_chan), np.argmin(n_chan)
    np.testing.assert_allclose(
        noise_map.ravel()[loud] / noise_map.ravel()[quiet],
        np.sqrt(n_chan.ravel()[quiet] / n_chan.ravel()[loud]),
        rtol=1e-10,
    )


@pytest.mark.filterwarnings("ignore: All channels masked")
@pytest.mark.parametrize("mode", ["auto", "per_pixel", 0.1])
def test_lam_sq_0_modes_keep_one_reference(mode):
    """Whatever picks the reference, the phase and flux references are the same.

    `stokes_i_ref_freq_hz` is derived from the reference lambda^2 in one place,
    so a caller cannot end up with an FDF derotated to one frequency and Stokes
    I terms defined at another.
    """
    freq_arr_hz, stokes_q, stokes_u, weight_arr, _ = _mosaic_cube(ny=4, nx=4)
    ref = float(np.median(freq_arr_hz))
    stokes_i = np.where(
        np.isfinite(stokes_q),
        (freq_arr_hz / ref)[:, None, None] ** -0.8,
        np.nan,
    )
    synth = rmsynth_3d(
        _chunked(stokes_q, 2, 2),
        _chunked(stokes_u, 2, 2),
        freq_arr_hz,
        weight_arr=weight_arr,
        lam_sq_0_m2=mode,
        stokes_i=_chunked(stokes_i, 2, 2),
        stokes_i_error=np.full(freq_arr_hz.size, 1e-3),
        phi_max_radm2=200.0,
        d_phi_radm2=2.0,
    )

    lam_sq_0_map = synth.lam_sq_0_map.compute()
    assert lam_sq_0_map.shape == stokes_q.shape[1:]
    ref_freq = synth.stokes_i_ref_freq_hz
    ref_freq_arr = np.asarray(
        ref_freq.compute() if isinstance(ref_freq, da.Array) else ref_freq
    )
    # The one invariant: the flux reference is the phase reference, in Hz.
    np.testing.assert_allclose(
        np.broadcast_to(ref_freq_arr, lam_sq_0_map.shape),
        lambda2_to_freq(lam_sq_0_map),
        rtol=1e-12,
    )

    if mode == "per_pixel":
        # Edge pixels lost the top of the band, so their reference moved.
        assert lam_sq_0_map.max() > lam_sq_0_map.min()
        assert isinstance(ref_freq, da.Array)
        # A per-pixel reference means a per-pixel RMSF, whether or not it was
        # asked for: CLEAN needs the RMSF at the FDF's own reference.
        assert synth.rmsf_cube is not None
    else:
        expected = synth.lam_sq_0_m2
        np.testing.assert_allclose(lam_sq_0_map, expected, rtol=1e-12)
        assert not isinstance(ref_freq, da.Array)
        if not isinstance(mode, str):
            assert synth.lam_sq_0_m2 == pytest.approx(mode)


@pytest.mark.filterwarnings("ignore: All channels masked")
def test_per_pixel_reference_is_an_exact_derotation():
    """Choosing a reference only ever moves phase, never amplitude.

    B&dB eq. 25 is a shift theorem, so the per-pixel FDF is the shared-reference
    FDF moved to the map, exactly. Checked without a Stokes I model, since a
    per-pixel reference frequency also moves the flux the FDF is scaled by.
    """
    freq_arr_hz, stokes_q, stokes_u, weight_arr, _ = _mosaic_cube(ny=4, nx=4)
    common: dict[str, Any] = {
        "freq_arr_hz": freq_arr_hz,
        "weight_arr": weight_arr,
        "phi_max_radm2": 200.0,
        "d_phi_radm2": 2.0,
    }
    shared = rmsynth_3d(
        _chunked(stokes_q, 2, 2), _chunked(stokes_u, 2, 2), lam_sq_0_m2="auto", **common
    )
    per_pixel = rmsynth_3d(
        _chunked(stokes_q, 2, 2),
        _chunked(stokes_u, 2, 2),
        lam_sq_0_m2="per_pixel",
        **common,
    )

    shared_fdf, pixel_fdf, lam_sq_0_map = compute(
        shared.fdf_dirty_cube, per_pixel.fdf_dirty_cube, per_pixel.lam_sq_0_map
    )
    # Amplitudes are untouched bar the last bit of the phase multiply.
    np.testing.assert_allclose(np.abs(shared_fdf), np.abs(pixel_fdf), rtol=1e-14)
    np.testing.assert_allclose(
        derotate_to(shared_fdf, shared.phi_arr_radm2, shared.lam_sq_0_m2, lam_sq_0_map),
        pixel_fdf,
        rtol=1e-12,
        atol=1e-14,
    )
    # And it inverts: back to the shared reference is where it started.
    np.testing.assert_allclose(
        derotate_to(pixel_fdf, shared.phi_arr_radm2, lam_sq_0_map, shared.lam_sq_0_m2),
        shared_fdf,
        rtol=1e-12,
        atol=1e-14,
    )


def test_lam_sq_0_option_is_validated():
    """A typo in the mode is caught when the options are built, not at compute."""
    bad_values: list[Any] = ["nope", -1.0, 0.0, np.nan]
    for bad in bad_values:
        with pytest.raises(ValueError, match="lam_sq_0_m2"):
            FDFOptions(lam_sq_0_m2=bad)


def _write_cube(path, data: NDArray[np.float64]) -> None:
    """Write a cube to FITS in FITS-native big-endian float32."""
    fits.PrimaryHDU(data.astype(">f4")).writeto(path, overwrite=True)


def _memmap_read(path, y0: int, y1: int) -> NDArray[np.float32]:
    """The pre-fix reader: squeeze a memmap and slice out a spatial block."""
    with fits.open(path, memmap=True) as hdul:
        return np.array(np.squeeze(hdul[0].data)[:, y0:y1, :])


@pytest.mark.parametrize("degenerate_axis", [False, True])
def test_read_fits_cube_dask_matches_memmap_read(tmp_path, degenerate_axis: bool):
    """New section-based reader is bit-identical to the old memmap path."""
    n_freq, ny, nx = 6, 13, 5
    cube = RNG.normal(0, 1, (n_freq, ny, nx))
    path = tmp_path / "cube.fits"
    _write_cube(path, cube[np.newaxis] if degenerate_axis else cube)

    # A band height of 4 leaves a short final band (13 % 4 == 1).
    target_chunk_mb = n_freq * 4 * nx * 4 / 1024**2
    arr, _ = read_fits_cube_dask(path, target_chunk_mb=target_chunk_mb)

    assert arr.chunks[1] == (4, 4, 4, 1)
    np.testing.assert_array_equal(arr.compute(), _memmap_read(path, 0, ny))


def test_read_fits_cube_dask_single_row_bands(tmp_path):
    """A one-row band (cy == 1) survives rather than being squeezed away."""
    cube = RNG.normal(0, 1, (4, 3, 5))
    path = tmp_path / "cube.fits"
    _write_cube(path, cube[np.newaxis])

    arr, _ = read_fits_cube_dask(path, target_chunk_mb=1e-9)

    assert arr.chunks == ((4,), (1, 1, 1), (5,))
    np.testing.assert_array_equal(arr.compute(), _memmap_read(path, 0, 3))


def test_read_fits_cube_dask_has_no_astype_layer(tmp_path):
    """Blocks come back native-endian, so dask inserts no astype layer.

    FITS is big-endian on disk; declaring that dtype made `da.concatenate`
    graft an `astype` onto every block, which then absorbed the whole cost of
    materialising the read and showed up as the culprit in worker kills.
    """
    path = tmp_path / "cube.fits"
    _write_cube(path, RNG.normal(0, 1, (4, 8, 5)))

    arr, _ = read_fits_cube_dask(path, target_chunk_mb=1e-9)

    assert arr.dtype.byteorder in ("=", "|")
    assert arr.dtype == np.float32
    prefixes = {str(key[0]).split("-")[0] for key in arr.__dask_graph__()}
    assert not any("astype" in prefix for prefix in prefixes), prefixes


@pytest.mark.parametrize("reader", [read_fits_cube_dask, read_fits_cube_channel_chunks])
def test_readers_emit_one_graph_layer_whatever_the_block_count(tmp_path, reader):
    """The whole block grid is one layer, not a layer per block plus a concatenate."""
    path = tmp_path / "cube.fits"
    _write_cube(path, RNG.normal(0, 1, (8, 64, 4)))

    one_block, _ = reader(path, target_chunk_mb=1e3)
    many_blocks, _ = reader(path, target_chunk_mb=1e-9)

    assert many_blocks.npartitions > one_block.npartitions
    assert len(one_block.dask.layers) == 1
    assert len(many_blocks.dask.layers) == 1


def _rmclean_graph(
    path, rows: int, n_freq: int, nx: int
) -> tuple[float, RMClean3DResults]:
    """Build the RM-CLEAN graph over a cube read in `rows`-tall bands, and time it.

    A single `map_blocks` stands in for RM-synthesis, so the layer count above
    the reader is what `rmsynth_3d` contributes and the chunk count is exactly
    the reader's.
    """
    n_phi = 5
    target_chunk_mb = rows * n_freq * nx * 4 / 1024**2
    cube, _ = read_fits_cube_dask(path, target_chunk_mb=target_chunk_mb)
    fdf_dirty_cube = da.map_blocks(
        lambda block: block.astype(np.complex128)[:n_phi],
        cube,
        chunks=((n_phi,), cube.chunks[1], cube.chunks[2]),
        dtype=np.complex128,
    )
    tick = time.perf_counter()
    results = run_rmclean(
        fdf_dirty_cube,
        np.zeros(2 * n_phi - 1, dtype=np.complex128),
        np.linspace(-6, 6, n_phi),
        np.linspace(-12, 12, 2 * n_phi - 1),
        fwhm_rmsf_radm2=1.0,
        mask=MASK_THRESHOLD,
        threshold=CLEAN_THRESHOLD,
    )
    return time.perf_counter() - tick, results


def test_rmclean_graph_build_stays_linear_in_chunk_count(tmp_path):
    """Quadrupling the chunk count must not blow up graph building.

    Both `read_fits_cube_dask` and `run_rmclean` used to work a layer at a
    time, so every `HighLevelGraph.from_collections` walked an upstream layer
    count that itself grew with the chunk count. Graph building alone then went
    quadratic in `target_chunk_mb`, and on a real cube it took minutes and
    several GB before a single block had been read.
    """
    n_freq, ny, nx = 4, 8192, 8
    path = tmp_path / "cube.fits"
    _write_cube(path, np.zeros((n_freq, ny, nx)))

    def best_of(rows: int) -> float:
        # Min over repeats: a loaded runner inflates individual builds, but
        # nothing makes one spuriously fast.
        return min(_rmclean_graph(path, rows, n_freq, nx)[0] for _ in range(5))

    coarse_time = best_of(16)
    fine_time = best_of(4)

    _, coarse = _rmclean_graph(path, 16, n_freq, nx)
    _, fine = _rmclean_graph(path, 4, n_freq, nx)
    assert fine.clean_fdf_cube.npartitions == 4 * coarse.clean_fdf_cube.npartitions
    # Layers stay flat; only the number of tasks in them grows.
    assert len(fine.clean_fdf_cube.dask.layers) == len(
        coarse.clean_fdf_cube.dask.layers
    )
    # 4x the chunks is at worst ~4x the tasks to build, so ~5x is generous
    # headroom. The per-layer version came in around 15x.
    assert fine_time < 5 * coarse_time, (coarse_time, fine_time)


def test_read_fits_cube_channel_chunks_matches_spatial_read(tmp_path):
    """Channel-chunked reader returns the same cube, chunked along frequency."""
    n_freq, ny, nx = 7, 6, 5
    path = tmp_path / "cube.fits"
    _write_cube(path, RNG.normal(0, 1, (n_freq, ny, nx))[np.newaxis])

    spatial, _ = read_fits_cube_dask(path)
    channels, _ = read_fits_cube_channel_chunks(
        path, target_chunk_mb=2 * ny * nx * 4 / 1024**2
    )

    assert channels.chunks == ((2, 2, 2, 1), (ny,), (nx,))
    np.testing.assert_array_equal(channels.compute(), spatial.compute())


# (n_freq, ny, nx) of the squeezed cube used by the degenerate-axis matrix.
DEGEN_NF, DEGEN_NY, DEGEN_NX = 13, 17, 11

DEGENERATE_SHAPES = {
    "plain-3d": (DEGEN_NF, DEGEN_NY, DEGEN_NX),
    "leading": (1, DEGEN_NF, DEGEN_NY, DEGEN_NX),
    # The ASKAP/CASA dummy-Stokes layout.
    "middle": (DEGEN_NF, 1, DEGEN_NY, DEGEN_NX),
    "trailing": (DEGEN_NF, DEGEN_NY, DEGEN_NX, 1),
    "several": (1, DEGEN_NF, 1, DEGEN_NY, DEGEN_NX, 1),
}


@pytest.mark.parametrize(
    "reader",
    [read_fits_cube_dask, read_fits_cube_channel_chunks],
    ids=["spatial", "channel"],
)
@pytest.mark.parametrize(
    "target_chunk_mb", [1e-9, 256], ids=["many-chunks", "one-chunk"]
)
@pytest.mark.parametrize(
    "disk_shape", list(DEGENERATE_SHAPES.values()), ids=list(DEGENERATE_SHAPES)
)
def test_readers_drop_degenerate_axes(tmp_path, reader, target_chunk_mb, disk_shape):
    """Both readers return the squeezed cube whatever the degenerate-axis layout.

    A single chunk covering the whole array is the case that regressed: an
    `int` index only squeezes its axis when the request doesn't span the full
    array, so a whole-array read came back un-squeezed.
    """
    expected = np.arange(DEGEN_NF * DEGEN_NY * DEGEN_NX, dtype=">f4").reshape(
        DEGEN_NF, DEGEN_NY, DEGEN_NX
    )
    path = tmp_path / "cube.fits"
    fits.PrimaryHDU(expected.reshape(disk_shape)).writeto(path)

    arr, _ = reader(path, target_chunk_mb=target_chunk_mb)

    assert arr.shape == (DEGEN_NF, DEGEN_NY, DEGEN_NX)
    # Both halves of the matrix have to bite: one block for the big target,
    # many for the small one.
    assert (int(np.prod(arr.numblocks)) == 1) == (target_chunk_mb == 256)
    np.testing.assert_array_equal(arr.compute(), expected)


@pytest.mark.parametrize("dtype", [">f4", ">f8", ">i2"])
def test_readers_agree_byte_for_byte_across_dtypes(tmp_path, dtype: str):
    """Each on-disk dtype survives the read natively, and both readers agree."""
    expected = (
        np.arange(DEGEN_NF * DEGEN_NY * DEGEN_NX)
        .reshape(DEGEN_NF, DEGEN_NY, DEGEN_NX)
        .astype(dtype)
    )
    path = tmp_path / "cube.fits"
    fits.PrimaryHDU(expected.reshape(DEGEN_NF, 1, DEGEN_NY, DEGEN_NX)).writeto(path)

    spatial, _ = read_fits_cube_dask(path, target_chunk_mb=1e-9)
    channels, _ = read_fits_cube_channel_chunks(path, target_chunk_mb=1e-9)

    assert spatial.dtype == np.dtype(dtype).newbyteorder("=")
    np.testing.assert_array_equal(spatial.compute(), expected)
    np.testing.assert_array_equal(channels.compute(), spatial.compute())


def test_rmsynth_3d_from_fits_on_a_dummy_stokes_axis(
    tmp_path, synthetic_cube: SyntheticCube
):
    """End-to-end on the ASKAP (n_freq, 1, ny, nx) layout with noise weighting.

    The composition is where the degenerate-axis bug actually surfaced: the
    default `weight_type="variance"` pulls in the channel-chunked reader too,
    which asks for full y and full x on every block.
    """
    freq_arr_hz = synthetic_cube.freq_arr_hz
    header = Header()
    header["CTYPE3"] = "FREQ"
    header["CRVAL3"] = freq_arr_hz[0]
    header["CDELT3"] = freq_arr_hz[1] - freq_arr_hz[0]
    header["CRPIX3"] = 1
    header["CUNIT3"] = "Hz"

    for name, data in (("q", synthetic_cube.stokes_q), ("u", synthetic_cube.stokes_u)):
        fits.PrimaryHDU(data[:, np.newaxis].astype(">f4"), header=header).writeto(
            tmp_path / f"{name}.fits"
        )

    synth = rmsynth_3d_from_fits(
        tmp_path / "q.fits",
        tmp_path / "u.fits",
        d_phi_radm2=D_PHI_RADM2,
        phi_max_radm2=250.0,
    )
    fdf_cube = synth.fdf_dirty_cube.compute()

    peak_rm = synth.phi_arr_radm2[np.abs(fdf_cube).argmax(axis=0)]
    np.testing.assert_allclose(
        peak_rm, synthetic_cube.rm_map, atol=2 * synth.fwhm_rmsf_radm2
    )


def test_weight_as_stokes_i_error_is_quiet_outside_the_beam(tmp_path):
    """A zeroed linmos weight must blank pixels silently, not warn per chunk.

    `noise_files_are_weight` inverts the weight cube, and a primary-beam weight
    is 0 over most of the field. Infinite error there is intended -- the pixel
    drops out of the Stokes I fit and its maps come back NaN -- so the only
    thing to check is that numpy stays quiet about the division while the
    pixels inside the cutoff still recover their spectral index.
    """
    freq_arr_hz = (np.arange(744, 1032, 6) * 1e6).astype(np.float64)
    ny = nx = 12
    alpha = -0.9
    cutoff = 3.0
    ref_freq_hz = float(np.median(freq_arr_hz))

    yy, xx = np.mgrid[0:ny, 0:nx]
    radius = np.hypot(yy - (ny - 1) / 2, xx - (nx - 1) / 2)
    inside = radius <= cutoff
    taper = np.where(inside, np.exp(-(radius**2) / (2 * cutoff**2)), 0.0)
    weight = np.repeat(taper[np.newaxis], freq_arr_hz.size, axis=0)

    spectrum = (freq_arr_hz / ref_freq_hz)[:, np.newaxis, np.newaxis] ** alpha
    stokes_i = np.broadcast_to(2.0 * spectrum, weight.shape).copy()
    rm_map = RNG.uniform(-100, 100, (ny, nx))
    angle = 2 * rm_map[np.newaxis] * freq_to_lambda2(freq_arr_hz)[:, None, None]
    stokes_q = 0.5 * np.cos(angle) * stokes_i
    stokes_u = 0.5 * np.sin(angle) * stokes_i

    header = Header()
    header["CTYPE3"] = "FREQ"
    header["CRVAL3"] = freq_arr_hz[0]
    header["CDELT3"] = freq_arr_hz[1] - freq_arr_hz[0]
    header["CRPIX3"] = 1
    header["CUNIT3"] = "Hz"
    for name, data in (
        ("q", stokes_q),
        ("u", stokes_u),
        ("i", stokes_i),
        ("weight", weight),
    ):
        fits.PrimaryHDU(data.astype(">f4"), header=header).writeto(
            tmp_path / f"{name}.fits"
        )

    synth = rmsynth_3d_from_fits(
        tmp_path / "q.fits",
        tmp_path / "u.fits",
        stokes_i_file=tmp_path / "i.fits",
        stokes_i_error_file=tmp_path / "weight.fits",
        noise_files_are_weight=True,
        d_phi_radm2=D_PHI_RADM2,
        phi_max_radm2=150.0,
        fit_order=1,
        stokes_i_snr_cut=None,
    )
    alpha_map = _require_cube(synth.stokes_i_alpha_map)

    # Both blanked divisions are on this path: inverting the zero weight, and
    # dividing Q+iU by the NaN model the fit leaves outside the cutoff.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        computed = alpha_map.compute()
        fdf_cube = synth.fdf_dirty_cube.compute()
    blanked = [
        str(warning.message)
        for warning in caught
        if issubclass(warning.category, RuntimeWarning)
        and (
            "divide by zero" in str(warning.message)
            or "invalid value encountered in divide" in str(warning.message)
        )
    ]
    assert blanked == []

    assert np.all(np.isnan(computed[~inside]))
    np.testing.assert_allclose(computed[inside], alpha, atol=1e-5)
    assert np.all(np.isnan(fdf_cube[:, ~inside]))
    assert np.all(np.isfinite(fdf_cube[:, inside]))


def test_channel_noise_from_channel_chunks_matches_whole_cube(tmp_path):
    """Per-channel noise off a channel-chunked read matches a whole-cube reduction."""
    n_freq, ny, nx = 8, 16, 16
    stokes_q = RNG.normal(0, 1, (n_freq, ny, nx))
    stokes_u = RNG.normal(0, 1, (n_freq, ny, nx))
    q_path, u_path = tmp_path / "q.fits", tmp_path / "u.fits"
    _write_cube(q_path, stokes_q)
    _write_cube(u_path, stokes_u)

    q_chunked, _ = read_fits_cube_channel_chunks(
        q_path, target_chunk_mb=2 * ny * nx * 4 / 1024**2
    )
    u_chunked, _ = read_fits_cube_channel_chunks(
        u_path, target_chunk_mb=2 * ny * nx * 4 / 1024**2
    )
    chunked_noise = estimate_channel_noise_mad(q_chunked, u_chunked)

    whole_cube_noise = estimate_channel_noise_mad(
        _chunked(stokes_q, -1, -1),
        _chunked(stokes_u, -1, -1),
    )
    np.testing.assert_allclose(chunked_noise, whole_cube_noise, rtol=1e-6)


def test_channel_noise_warns_on_spatially_chunked_input(caplog):
    """Spatially chunked input still works, but says it is gathering the cube."""
    stokes_q = RNG.normal(0, 1, (4, 8, 8))
    stokes_u = RNG.normal(0, 1, (4, 8, 8))

    with caplog.at_level("WARNING"):
        estimate_channel_noise_mad(_chunked(stokes_q, 4, 4), _chunked(stokes_u, 4, 4))

    assert "read_fits_cube_channel_chunks" in caplog.text


def test_rmsynth_3d_output_chunks_stay_within_the_input_chunk_budget():
    """Spatial chunks shrink so an FDF chunk costs what the input chunk costs.

    The Faraday-depth axis is longer than the frequency axis and complex128
    rather than float32, so keeping the caller's spatial chunking would make
    every output chunk a large multiple of the input chunk they sized.
    """
    n_freq, ny, nx = 128, 64, 32
    freq_arr_hz = np.linspace(8.0e8, 1.0e9, n_freq)
    stokes_q = RNG.normal(0, 1, (n_freq, ny, nx))
    stokes_u = RNG.normal(0, 1, (n_freq, ny, nx))

    q_dask = _chunked(stokes_q, 32, nx)
    u_dask = _chunked(stokes_u, 32, nx)
    input_chunk_bytes = np.prod(q_dask.chunksize) * q_dask.dtype.itemsize

    synth = rmsynth_3d(
        q_dask,
        u_dask,
        freq_arr_hz,
        phi_max_radm2=100.0,
        d_phi_radm2=2.0,
        per_pixel_rmsf=True,
    )

    assert synth.rmsf_cube is not None
    for cube in (synth.fdf_dirty_cube, synth.rmsf_cube):
        assert np.prod(cube.chunksize) * cube.dtype.itemsize <= input_chunk_bytes
    # Shrunk along y only, so each block is still one contiguous read.
    assert synth.fdf_dirty_cube.chunksize[2] == nx
    # And the result is unchanged by the rechunking.
    np.testing.assert_allclose(
        synth.fdf_dirty_cube.compute(),
        rmsynth_3d(
            _chunked(stokes_q, 1, nx),
            _chunked(stokes_u, 1, nx),
            freq_arr_hz,
            phi_max_radm2=100.0,
            d_phi_radm2=2.0,
        ).fdf_dirty_cube.compute(),
    )


def test_rmsynth_3d_says_so_when_one_row_overshoots_the_budget(caplog):
    """One FDF row is the chunking floor, so an overshoot is logged, not silent."""
    wide = da.zeros((100, 8, 4000), chunks=(-1, 8, 4000), dtype=np.float32)
    narrow = da.zeros((100, 64, 32), chunks=(-1, 64, 32), dtype=np.float32)

    with caplog.at_level("WARNING"):
        _match_chunks_to_fdf(narrow, narrow, n_phi_double=1000)
    assert "One row of FDF output" not in caplog.text

    with caplog.at_level("WARNING"):
        _match_chunks_to_fdf(wide, wide, n_phi_double=1000)
    assert "One row of FDF output" in caplog.text


def test_rmsynth_3d_keeps_chunks_when_the_fdf_is_no_bigger():
    """Coarse chunks survive when the output is no larger than the input."""
    n_freq, ny, nx = 400, 8, 8
    freq_arr_hz = np.linspace(8.0e8, 1.4e9, n_freq)
    q_dask = _chunked(RNG.normal(0, 1, (n_freq, ny, nx)), 4, nx)
    u_dask = _chunked(RNG.normal(0, 1, (n_freq, ny, nx)), 4, nx)

    synth = rmsynth_3d(q_dask, u_dask, freq_arr_hz, d_phi_radm2=200.0)

    assert synth.fdf_dirty_cube.chunksize[1] == 4


def test_write_zarr_group_rechunks_irregular_arrays(tmp_path, caplog):
    """Irregular dask chunks must be rechunked, not silently written wrong.

    A zarr array has one chunk size per axis, so writing an array whose chunks
    differ mid-axis against `chunks[0]` would misplace every later chunk.
    `dask.array.Array.to_zarr` guards against this; so must we, since we build
    the zarr arrays ourselves.
    """
    data = np.arange(60.0).reshape(10, 6)
    irregular = da.from_array(data, chunks=(10, 6)).rechunk(((3, 5, 2), (6,)))
    regular = da.from_array(data, chunks=(4, 6))

    store = tmp_path / "irregular.zarr"
    with caplog.at_level("WARNING", logger="rm-lite"):
        write_zarr_group(store, {"odd": irregular, "even": regular})

    assert "irregular chunk sizes" in caplog.text
    assert np.array_equal(np.asarray(zarr.open_array(store, path="odd")), data)
    assert np.array_equal(np.asarray(zarr.open_array(store, path="even")), data)


def test_shared_rmsf_is_what_the_per_pixel_cube_holds(synthetic_cube: SyntheticCube):
    """Every pixel of the per-pixel cube is the shared RMSF, which is the whole
    reason the cube is not returned by default."""
    q_dask = _chunked(synthetic_cube.stokes_q, 3, 4)
    u_dask = _chunked(synthetic_cube.stokes_u, 3, 4)

    default = rmsynth_3d(
        q_dask, u_dask, synthetic_cube.freq_arr_hz, d_phi_radm2=D_PHI_RADM2
    )
    assert default.rmsf_cube is None
    assert default.rmsf_arr.shape == default.phi_double_arr_radm2.shape

    per_pixel = rmsynth_3d(
        q_dask,
        u_dask,
        synthetic_cube.freq_arr_hz,
        d_phi_radm2=D_PHI_RADM2,
        per_pixel_rmsf=True,
    )
    cube = _require_cube(per_pixel.rmsf_cube).compute()
    ny, nx = synthetic_cube.rm_map.shape
    assert cube.shape == (default.phi_double_arr_radm2.size, ny, nx)
    # atol, not exact: finufft splits a batched multithreaded type-3 differently
    # from a single transform.
    for j in range(ny):
        for i in range(nx):
            np.testing.assert_allclose(cube[:, j, i], default.rmsf_arr, atol=1e-12)


@pytest.mark.filterwarnings("ignore: All channels masked")
def test_rmclean_agrees_between_shared_and_per_pixel_rmsf(
    synthetic_cube: SyntheticCube,
):
    """Handing RM-CLEAN the one shared spectrum must clean exactly as handing it a
    cube of copies of that spectrum."""
    q_dask = _chunked(synthetic_cube.stokes_q, 3, 4)
    u_dask = _chunked(synthetic_cube.stokes_u, 3, 4)
    synth = rmsynth_3d(
        q_dask,
        u_dask,
        synthetic_cube.freq_arr_hz,
        d_phi_radm2=D_PHI_RADM2,
        per_pixel_rmsf=True,
    )
    common = (
        synth.phi_arr_radm2,
        synth.phi_double_arr_radm2,
        synth.fwhm_rmsf_radm2,
    )
    shared = run_rmclean(
        synth.fdf_dirty_cube,
        synth.rmsf_arr,
        *common,
        mask=MASK_THRESHOLD,
        threshold=CLEAN_THRESHOLD,
    )
    per_pixel = run_rmclean(
        synth.fdf_dirty_cube,
        _require_cube(synth.rmsf_cube),
        *common,
        mask=MASK_THRESHOLD,
        threshold=CLEAN_THRESHOLD,
    )
    shared_fdf, per_pixel_fdf = compute(shared.clean_fdf_cube, per_pixel.clean_fdf_cube)
    np.testing.assert_allclose(shared_fdf, per_pixel_fdf, atol=1e-10)


def test_rmclean_rejects_an_rmsf_it_cannot_use(synthetic_cube: SyntheticCube):
    """Only a shared spectrum or a matching per-pixel cube make sense. The rest are
    caught before the graph is built, not part-way through a CLEAN run."""
    q_dask = _chunked(synthetic_cube.stokes_q, 3, 4)
    u_dask = _chunked(synthetic_cube.stokes_u, 3, 4)
    synth = rmsynth_3d(
        q_dask,
        u_dask,
        synthetic_cube.freq_arr_hz,
        d_phi_radm2=D_PHI_RADM2,
        per_pixel_rmsf=True,
    )
    rmsf_cube = _require_cube(synth.rmsf_cube)

    def clean_with(rmsf):
        return run_rmclean(
            synth.fdf_dirty_cube,
            rmsf,
            synth.phi_arr_radm2,
            synth.phi_double_arr_radm2,
            synth.fwhm_rmsf_radm2,
            mask=MASK_THRESHOLD,
            threshold=CLEAN_THRESHOLD,
        )

    # Neither one spectrum nor a cube.
    with pytest.raises(ValueError, match="must be 1D"):
        clean_with(synth.rmsf_arr[:, np.newaxis])

    # A computed cube: the right shape, but the blocks have to stay lazy.
    with pytest.raises(TypeError, match="must be a dask array"):
        clean_with(rmsf_cube.compute())

    # Chunked differently from the FDF, so block N of each would be different
    # pixels.
    with pytest.raises(ValueError, match="identical .*spatial chunking"):
        clean_with(rmsf_cube.rechunk({1: 1, 2: 1}))

    # Split along Faraday depth. The CLEAN tasks index the RMSF by the FDF's
    # block index, so this would silently hand each block the first depth chunk
    # of the RMSF rather than the whole spectrum.
    with pytest.raises(ValueError, match="per-pixel rmsf must be chunked spatially"):
        clean_with(rmsf_cube.rechunk({0: 1}))
