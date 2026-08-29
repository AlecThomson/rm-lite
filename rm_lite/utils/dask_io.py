"""Dask-backed I/O helpers for chunked 3D RM-synthesis/CLEAN."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import astropy.units as u
import dask
import dask.array as da
import numpy as np
import zarr
from astropy.io import fits
from astropy.io.fits import Header
from astropy.stats import mad_std
from astropy.wcs import WCS
from dask.base import compute, tokenize
from dask.diagnostics import ProgressBar
from numpy.typing import NDArray

from rm_lite.utils.logging import logger

DEFAULT_TARGET_CHUNK_MB = 256


def spatial_chunk_size(
    n_freq: int,
    ny: int,
    nx: int,
    itemsize: int,
    target_chunk_mb: float = DEFAULT_TARGET_CHUNK_MB,
) -> tuple[int, int]:
    """Pick a full-width y-band spatial chunk for a target chunk memory footprint.

    The frequency/Faraday-depth axis is never chunked, so a chunk's memory
    footprint is `n_freq * cy * nx * itemsize`. Bands span the full image
    width rather than being square tiles, so each channel's slice of a block
    is one contiguous run of bytes on disk; square tiles scatter it into `cy`
    short runs, which makes the read either slow (many seeks) or unbounded in
    memory (page-cache over-fetch).

    One band is the floor, so a very wide cube can overshoot
    `target_chunk_mb`: `n_freq * nx * itemsize` bytes is the smallest block
    this chunking can produce.

    Args:
        n_freq (int): Size of the (unchunked) spectral axis.
        ny (int): Full image height in pixels.
        nx (int): Full image width in pixels.
        itemsize (int): Size in bytes of one array element.
        target_chunk_mb (float, optional): Target memory footprint of a
            single chunk, in MB. Defaults to 256.

    Returns:
        tuple[int, int]: Spatial chunk size (cy, nx). The x size is always the
        full image width.
    """
    target_chunk_bytes = target_chunk_mb * 1024**2
    rows_per_chunk = target_chunk_bytes / (n_freq * nx * itemsize)
    cy = max(1, int(np.floor(rows_per_chunk)))
    return min(cy, ny), nx


def channel_chunk_size(
    n_freq: int,
    ny: int,
    nx: int,
    itemsize: int,
    target_chunk_mb: float = DEFAULT_TARGET_CHUNK_MB,
) -> int:
    """Pick a channel-chunk size (whole image planes) for a target chunk footprint.

    The complement of `spatial_chunk_size`: whole spatial planes, chunked
    along frequency. This is the right chunking for per-channel reductions
    like `estimate_channel_noise_mad`, which need every pixel of a channel in
    one block.

    Args:
        n_freq (int): Full size of the spectral axis.
        ny (int): Full image height in pixels.
        nx (int): Full image width in pixels.
        itemsize (int): Size in bytes of one array element.
        target_chunk_mb (float, optional): Target memory footprint of a
            single chunk, in MB. Defaults to 256.

    Returns:
        int: Number of channels per chunk.
    """
    target_chunk_bytes = target_chunk_mb * 1024**2
    channels_per_chunk = target_chunk_bytes / (ny * nx * itemsize)
    return min(max(1, int(np.floor(channels_per_chunk))), n_freq)


def _chunk_bounds(size: int, chunk: int) -> list[tuple[int, int]]:
    return [(start, min(start + chunk, size)) for start in range(0, size, chunk)]


def _section_index(
    disk_shape: tuple[int, ...],
    freq_bounds: tuple[int, int],
    y_bounds: tuple[int, int],
) -> tuple[slice, ...]:
    """Index into a HDU's on-disk shape, keeping degenerate length-1 axes.

    Degenerate axes get `slice(0, 1)` rather than an `int`, and are dropped by
    the reshape in `_read_fits_block`. An `int` looks like it squeezes the
    axis, but astropy only does that when the request doesn't span the whole
    array; a full-extent read comes back un-squeezed instead. The three
    non-degenerate axes get the real (freq, y, x) slices, in that order.
    """
    data_slices = iter((slice(*freq_bounds), slice(*y_bounds), slice(None)))
    return tuple(slice(0, 1) if size == 1 else next(data_slices) for size in disk_shape)


def _read_fits_block(
    path: str | Path,
    freq_bounds: tuple[int, int],
    y_bounds: tuple[int, int],
) -> NDArray[Any]:
    """Read one (freq, y) block of a cube, always shaped (n_freq, cy, nx)."""
    # Reopens the file per block rather than closing over one shared memmap:
    # dask.array.from_array unconditionally does `x = x.copy()` on anything
    # array-like (including a memmap-backed ndarray), and its default
    # tokenizing hashes the full buffer. Both silently force a full-cube
    # read into memory. Passing only cheap primitives (path, int bounds) as
    # the task's arguments sidesteps both.
    #
    # `memmap=False` + `.section` reads exactly the block's bytes and returns
    # a real array. A memmap slice would instead return a lazy view that both
    # keeps the whole-file mapping alive and faults in far more than the
    # block when something finally touches it.
    with fits.open(path, memmap=False) as hdul:
        block = hdul[0].section[_section_index(hdul[0].shape, freq_bounds, y_bounds)]
    # Guaranteed 3D whatever the on-disk axis count, so callers never depend on
    # astropy's index-dependent squeeze semantics. Free view when the block is
    # already that shape.
    block = block.reshape(
        freq_bounds[1] - freq_bounds[0], y_bounds[1] - y_bounds[0], -1
    )
    # Native byte order: FITS is big-endian on disk, and leaving the block
    # that way makes dask insert an `astype` layer on top of every read.
    # Swapped in place rather than through `astype`, which allocates a second
    # copy of the block: `copy=False` only avoids that when no conversion is
    # needed, and a byte-order change is one. The block was just read from disk
    # and nothing else holds it, so rewriting its buffer is safe. Measured on a
    # 119 MiB block, this takes the read task's peak from 238 MiB to 119 MiB.
    if block.dtype.isnative:
        return block
    return block.byteswap(inplace=True).view(block.dtype.newbyteorder("="))


def _cube_meta(path: str | Path) -> tuple[tuple[int, int, int], np.dtype[Any], Header]:
    """Cube shape (freq, y, x), native-order dtype and header, without reading data."""
    with fits.open(path, memmap=False) as hdul:
        header = hdul[0].header.copy()
        disk_shape = tuple(hdul[0].shape)
        shape = tuple(size for size in disk_shape if size != 1)
        if len(shape) != 3:
            msg = (
                "Expected a 3D (freq, y, x) cube after squeezing degenerate axes, "
                f"got shape {disk_shape} from {path}."
            )
            raise ValueError(msg)
        # One-element read: the only public way to get an HDU's dtype without
        # touching the data array.
        probe = hdul[0].section[_section_index(disk_shape, (0, 1), (0, 1))]
        dtype = probe.dtype.newbyteorder("=")
    n_freq, ny, nx = shape
    return (n_freq, ny, nx), dtype, header


def _read_fits_cube_in_one_layer(
    path: str | Path,
    dtype: np.dtype[Any],
    nx: int,
    freq_bounds: Sequence[tuple[int, int]],
    y_bounds: Sequence[tuple[int, int]],
) -> da.Array:
    """A dask array over a (freq, y) grid of `_read_fits_block` calls, in one layer.

    One graph layer for the whole grid, not a `dask.array.from_delayed` per
    block plus a concatenate. Every downstream `HighLevelGraph.from_collections`
    walks each upstream layer and re-wraps it, so a layer per block makes graph
    building quadratic in the block count for any consumer that also works
    block-by-block (`rm_lite.tools_3d.rmclean.run_rmclean`).
    """
    name = f"read-fits-block-{tokenize(str(path), dtype, nx, freq_bounds, y_bounds)}"
    layer = {
        (name, i, j, 0): (_read_fits_block, path, f_bounds, y_bnds)
        for i, f_bounds in enumerate(freq_bounds)
        for j, y_bnds in enumerate(y_bounds)
    }
    chunks = (
        tuple(stop - start for start, stop in freq_bounds),
        tuple(stop - start for start, stop in y_bounds),
        (nx,),
    )
    return da.Array(layer, name, chunks, dtype=dtype)


def read_fits_cube_dask(
    path: str | Path,
    target_chunk_mb: float = DEFAULT_TARGET_CHUNK_MB,
) -> tuple[da.Array, Header]:
    """Lazily read a Stokes FITS cube as a spatially chunked dask array.

    Each block is a full-width band of `cy` image rows across all channels,
    read by its own task via `astropy.io.fits`' `.section`, so only that
    block's bytes are read and resident. Actual reads from disk are deferred
    until a block is computed.

    Degenerate length-1 axes (e.g. a dummy Stokes axis, common in ASKAP/EMU
    cutout cubes) are dropped.

    Peak memory during processing is a few times `target_chunk_mb`, not a hard
    cap: a block is copied when it is materialised, and again when it is
    written out. `rm_lite.tools_3d.rmsynth.rmsynth_3d` shrinks these chunks
    further where its complex128 output would otherwise outgrow them.

    That multiplier is per task, so the process peak is roughly the multiplier
    times the number of scheduler threads. Budget for that, or run the
    synchronous scheduler if you want the single-task number.

    Args:
        path (str | Path): Path to the FITS cube. Assumed axis order
            (freq, y, x) once degenerate axes are dropped, i.e. the
            frequency axis is first in numpy order.
        target_chunk_mb (float, optional): Target chunk memory footprint in
            MB, see `spatial_chunk_size`. Defaults to 256.

    Returns:
        tuple[da.Array, Header]: Lazy dask array and the FITS header.
    """
    (n_freq, ny, nx), dtype, header = _cube_meta(path)

    cy, _ = spatial_chunk_size(
        n_freq=n_freq,
        ny=ny,
        nx=nx,
        itemsize=dtype.itemsize,
        target_chunk_mb=target_chunk_mb,
    )

    cube = _read_fits_cube_in_one_layer(
        path=path,
        dtype=dtype,
        nx=nx,
        freq_bounds=[(0, n_freq)],
        y_bounds=_chunk_bounds(ny, cy),
    )

    return cube, header


def read_fits_cube_channel_chunks(
    path: str | Path,
    target_chunk_mb: float = DEFAULT_TARGET_CHUNK_MB,
) -> tuple[da.Array, Header]:
    """Lazily read a Stokes FITS cube chunked along frequency, whole image planes.

    Same reader as `read_fits_cube_dask` but with the chunking transposed:
    each block is `k` consecutive whole channel planes, one contiguous read.
    Use this for per-channel reductions (`estimate_channel_noise_mad`), which
    need every pixel of a channel in one block. Not for RM-synthesis, which
    needs every channel of a pixel instead.

    Args:
        path (str | Path): Path to the FITS cube, see `read_fits_cube_dask`.
        target_chunk_mb (float, optional): Target chunk memory footprint in
            MB, see `channel_chunk_size`. Defaults to 256.

    Returns:
        tuple[da.Array, Header]: Lazy dask array and the FITS header.
    """
    (n_freq, ny, nx), dtype, header = _cube_meta(path)

    n_chan = channel_chunk_size(
        n_freq=n_freq,
        ny=ny,
        nx=nx,
        itemsize=dtype.itemsize,
        target_chunk_mb=target_chunk_mb,
    )

    cube = _read_fits_cube_in_one_layer(
        path=path,
        dtype=dtype,
        nx=nx,
        freq_bounds=_chunk_bounds(n_freq, n_chan),
        y_bounds=[(0, ny)],
    )

    return cube, header


def freq_arr_hz_from_header(header: Header, n_freq: int) -> NDArray[np.float64]:
    """Derive the frequency array in Hz from a FITS header's spectral WCS.

    Args:
        header (Header): FITS header containing a spectral axis.
        n_freq (int): Number of channels along the spectral axis.

    Returns:
        NDArray[np.float64]: Frequency array in Hz.
    """
    spectral_wcs = WCS(header).spectral
    # Uses the low-level WCS API (`pixel_to_world_values`) rather than the
    # high-level `pixel_to_world`: the latter builds a `SpectralCoord`, which
    # requires a known observer reference frame from `SPECSYS` and raises
    # `NotImplementedError` if that header keyword is absent, common in
    # ASKAP/RACS cutouts. The plain pixel-to-frequency transform needed here
    # doesn't depend on an observer frame at all.
    freq_values = spectral_wcs.pixel_to_world_values(np.arange(n_freq))
    freq_unit = u.Unit(spectral_wcs.world_axis_units[0])
    freq_quantity = freq_values * freq_unit
    return freq_quantity.to(u.Hz, equivalencies=u.spectral()).value


def complex_pol_dask(stokes_q: da.Array, stokes_u: da.Array) -> da.Array:
    """Combine chunked Stokes Q and U dask arrays into a complex Q + iU array.

    Args:
        stokes_q (da.Array): Stokes Q dask array.
        stokes_u (da.Array): Stokes U dask array.

    Returns:
        da.Array: Complex Q + iU dask array, same chunks as the inputs.
    """
    return stokes_q + 1j * stokes_u


def _regular_chunks(name: str, array: da.Array) -> da.Array:
    """Rechunk to uniform chunks if needed, as `dask.array.Array.to_zarr` does."""
    # Regular == every chunk on an axis equal, bar a smaller final one. Matches
    # `dask.array.core._check_regular_chunks`, which is private.
    if all(
        len(axis) == 1 or (len(set(axis[:-1])) == 1 and axis[-1] <= axis[0])
        for axis in array.chunks
    ):
        return array
    logger.warning(
        f"Array {name!r} has irregular chunk sizes; rechunking to uniform chunks "
        "so it can be written safely. Rechunk it yourself to avoid this."
    )
    return array.rechunk(tuple(max(axis) for axis in array.chunks))


def write_zarr_group(
    store: str | Path,
    arrays: Mapping[str, da.Array],
    overwrite: bool = True,
) -> None:
    """Write a set of dask arrays lazily/incrementally to a shared zarr store.

    Written chunk-by-chunk, so no array is ever fully materialised. One
    `dask.array.store` for the whole set with fusion off, rather than a
    `to_zarr` per array, so arrays sharing an upstream task (the four outputs of
    `run_rmclean` come from one task per chunk) compute it once: fusion inlines
    that task into each consumer branch, and a per-array `to_zarr` bakes the
    copy in when its `Delayed` is built. The lost task
    boundary is worth far less than redoing a spatial chunk of RM-CLEAN.

    Args:
        store (str | Path): Path to the zarr store (a group containing one
            array per key in `arrays`).
        arrays (Mapping[str, da.Array]): Name -> dask array to write.
        overwrite (bool, optional): Overwrite existing arrays. Defaults to True.
    """
    names = list(arrays)
    # `to_zarr` rechunks an irregularly chunked array before writing, because a
    # zarr array has one chunk size per axis and the leading dask chunk would
    # silently misplace the rest. Same guard here, since we are making the zarr
    # arrays ourselves.
    sources = [_regular_chunks(name, arrays[name]) for name in names]
    # Same creation arguments `dask.array.to_zarr` builds, so the on-disk layout
    # is unchanged; only the store call below is batched.
    sinks = [
        zarr.create(
            shape=array.shape,
            # An empty dask array has chunk size 0, which zarr rejects.
            chunks=tuple(max(chunk[0], 1) for chunk in array.chunks),
            dtype=array.dtype,
            store=str(store),
            path=name,
            overwrite=overwrite,
        )
        for name, array in zip(names, sources, strict=True)
    ]
    tick = time.time()
    with dask.config.set({"optimization.fuse.active": False}), ProgressBar():
        compute(da.store(sources, sinks, lock=False, compute=False))
    tock = time.time()
    logger.info(f"Wrote {names} to {store} in {tock - tick:.3g} seconds.")


def mad_std_on_chan_block(block: NDArray[np.float64]) -> NDArray[np.float64]:
    # One plane at a time rather than one `mad_std(..., axis=1)` over the whole
    # block: `nanmedian` copies its input, so the vectorised form peaked at
    # ~2x the block where this peaks at ~2x a single plane, for the same
    # result and the same wall-clock.
    return np.array([mad_std(plane, ignore_nan=True) for plane in block])


def da_channel_mad(cube: da.Array) -> da.Array:
    """Lazy per-channel MAD std over every spatial pixel of a cube.

    Returns one value per channel, shape (n_freq,). Needs every pixel of a
    channel in one block (a robust median can't be combined incrementally
    across separate spatial chunks), so a spatially chunked cube has to be
    gathered first. Not computed -- so several cubes can be reduced in one
    `compute` call.
    """
    if len(cube.chunks[1]) > 1 or len(cube.chunks[2]) > 1:
        # The rechunk is all-to-all: every input block feeds every output
        # channel plane, so dask pins the whole cube for the duration. There
        # is no bounded way to do this from spatially chunked blocks; read
        # the cube with `read_fits_cube_channel_chunks` instead.
        logger.warning(
            "Estimating per-channel noise from a spatially chunked cube forces "
            f"a gather of the whole {cube.nbytes / 1024**3:.2g} GiB cube into one "
            "task. Read the cube with `read_fits_cube_channel_chunks` (or use "
            "`rmsynth_3d_from_fits`, which does) to keep this bounded."
        )
        cube = cube.rechunk({1: -1, 2: -1})
    return da.map_blocks(
        mad_std_on_chan_block,
        cube,
        dtype=np.float64,
        drop_axis=(1, 2),  # each block collapses to one scalar per channel
        chunks=(cube.chunks[0],),
    )


def estimate_single_stokes_channel_noise(stokes_i: da.Array) -> NDArray[np.float64]:
    """Robust per-channel noise from a single Stokes I cube.

    Same MAD-based per-channel estimator as `estimate_channel_noise_mad`, but
    for one cube (no Q/U combination). Useful as the `stokes_i_error` fed to
    `rm_lite.tools_3d.rmsynth.rmsynth_3d`'s per-pixel Stokes I fit when the I
    cube carries no separate error cube.

    Args:
        stokes_i (da.Array): Stokes I dask array, shape (n_freq, ny, nx).
            Chunk it along frequency (`read_fits_cube_channel_chunks`), see
            `estimate_channel_noise_mad`.

    Returns:
        NDArray[np.float64]: Per-channel noise estimate, shape (n_freq,). A
        plain numpy array, not lazy.
    """
    tick = time.time()
    (noise,) = compute(da_channel_mad(stokes_i))
    tock = time.time()
    logger.info(f"Per-channel noise estimation completed in {tock - tick:.3g} seconds.")
    return np.asarray(noise)


def estimate_channel_noise_mad(
    stokes_q: da.Array,
    stokes_u: da.Array,
) -> NDArray[np.float64]:
    """Robust per-channel noise from Stokes Q/U cubes, for auto-masking/thresholding.

    Computes `astropy.stats.mad_std` over every spatial pixel in each channel
    plane, then combines the Q and U estimates the same way
    `rm_lite.utils.synthesis.compute_rmsynth_params` combines a per-channel
    complex error into `real_qu_error` (`abs(real + imag) / 2`).

    Needs each channel's full spatial plane in one block: a robust statistic
    like the median can't be combined incrementally across separate spatial
    chunks the way a sum or mean can. Pass cubes chunked along frequency
    (`read_fits_cube_channel_chunks`), not spatially chunked ones as used for
    RM-synth/CLEAN -- gathering those costs one whole cube in a single task,
    and is warned about rather than done silently.

    The per-channel noise this returns can be turned into a weight array
    (`weight_arr = 1 / noise**2`) for `rm_lite.tools_3d.rmsynth.rmsynth_3d`,
    and from there `rm_lite.utils.synthesis.compute_theoretical_noise` gives
    the FDF-domain noise used to set `rm_lite.tools_3d.rmclean.run_rmclean`'s
    `mask`/`threshold` (mirroring the 1D `run_rmclean_from_synth` auto-mask/
    auto-threshold convention).

    Args:
        stokes_q (da.Array): Stokes Q dask array, shape (n_freq, ny, nx),
            chunked along frequency.
        stokes_u (da.Array): Stokes U dask array, same shape as `stokes_q`.

    Returns:
        NDArray[np.float64]: Per-channel noise estimate, shape (n_freq,). A
        plain numpy array, not lazy; computed once, explicitly, here.
    """
    if stokes_q.shape != stokes_u.shape:
        msg = f"Stokes Q and U must have the same shape. Got {stokes_q.shape} and {stokes_u.shape}."
        raise ValueError(msg)

    tick = time.time()
    # One compute so the Q and U reductions share scheduling.
    q_noise, u_noise = compute(da_channel_mad(stokes_q), da_channel_mad(stokes_u))
    tock = time.time()
    logger.info(f"Per-channel noise estimation completed in {tock - tick:.3g} seconds.")
    return np.abs(q_noise + u_noise) / 2.0
