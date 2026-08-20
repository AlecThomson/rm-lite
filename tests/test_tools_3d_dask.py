"""Tests for the dask-chunked 3D RM-synthesis and RM-CLEAN tools."""

from __future__ import annotations

from typing import NamedTuple

import dask.array as da
import numpy as np
import pytest
import rm_lite.tools_3d.rmclean as rmclean3d_mod
import zarr
from astropy.io import fits
from astropy.io.fits import Header
from dask.base import compute
from numpy.typing import NDArray
from rm_lite.tools_1d.rmsynth import run_rmsynth
from rm_lite.tools_3d.rmclean import run_rmclean, run_rmclean_from_synth
from rm_lite.tools_3d.rmsynth import rmsynth_3d
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
from rm_lite.utils.synthesis import calc_faraday_moments, freq_to_lambda2

RNG = np.random.default_rng(2025)

D_PHI_RADM2 = 1.0
MASK_THRESHOLD = 0.15
CLEAN_THRESHOLD = 0.03


class SyntheticCube(NamedTuple):
    freq_arr_hz: NDArray[np.float64]
    stokes_q: NDArray[np.float64]
    stokes_u: NDArray[np.float64]
    rm_map: NDArray[np.float64]


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

    return SyntheticCube(freq_arr_hz, stokes_q, stokes_u, rm_map)


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
        synth.rmsf_cube,
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
    rmsf_cube = synth.rmsf_cube.compute()

    for j in range(ny):
        for i in range(nx):
            ref = rmclean(
                RMSynthArrays(
                    dirty_fdf_arr=dirty_cube[:, j, i],
                    phi_arr_radm2=synth.phi_arr_radm2,
                    rmsf_arr=rmsf_cube[:, j, i],
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
    original = rmclean3d_mod._clean_block

    def counting_wrapper(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(rmclean3d_mod, "_clean_block", counting_wrapper)

    q_dask = _chunked(synthetic_cube.stokes_q, 3, 4)
    u_dask = _chunked(synthetic_cube.stokes_u, 3, 4)
    synth = rmsynth_3d(
        q_dask, u_dask, synthetic_cube.freq_arr_hz, d_phi_radm2=D_PHI_RADM2
    )
    clean = rmclean3d_mod.run_rmclean(
        synth.fdf_dirty_cube,
        synth.rmsf_cube,
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
    original = rmclean3d_mod._clean_block

    def counting_wrapper(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(rmclean3d_mod, "_clean_block", counting_wrapper)

    q_dask = _chunked(synthetic_cube.stokes_q, 3, 4)
    u_dask = _chunked(synthetic_cube.stokes_u, 3, 4)
    synth = rmsynth_3d(
        q_dask, u_dask, synthetic_cube.freq_arr_hz, d_phi_radm2=D_PHI_RADM2
    )
    clean = rmclean3d_mod.run_rmclean(
        synth.fdf_dirty_cube,
        synth.rmsf_cube,
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
        q_dask, u_dask, synthetic_cube.freq_arr_hz, d_phi_radm2=D_PHI_RADM2
    )
    clean = run_rmclean(
        synth.fdf_dirty_cube,
        synth.rmsf_cube,
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
        synth.rmsf_cube,
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
        synth.rmsf_cube,
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
        q_dask, u_dask, freq_arr_hz, phi_max_radm2=100.0, d_phi_radm2=2.0
    )

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


def test_rmsynth_3d_keeps_chunks_when_the_fdf_is_no_bigger():
    """Coarse chunks survive when the output is no larger than the input."""
    n_freq, ny, nx = 400, 8, 8
    freq_arr_hz = np.linspace(8.0e8, 1.4e9, n_freq)
    q_dask = _chunked(RNG.normal(0, 1, (n_freq, ny, nx)), 4, nx)
    u_dask = _chunked(RNG.normal(0, 1, (n_freq, ny, nx)), 4, nx)

    synth = rmsynth_3d(q_dask, u_dask, freq_arr_hz, d_phi_radm2=200.0)

    assert synth.fdf_dirty_cube.chunksize[1] == 4
