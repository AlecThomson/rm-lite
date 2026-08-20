"""Dask-backed I/O helpers for chunked 3D RM-synthesis/CLEAN."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import astropy.units as u
import dask.array as da
import numpy as np
from astropy.io import fits
from astropy.io.fits import Header
from astropy.stats import mad_std
from astropy.wcs import WCS
from dask import delayed
from dask.base import compute
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
) -> tuple[int | slice, ...]:
    """Index into a HDU's on-disk shape, dropping degenerate length-1 axes.

    An `int` drops an axis the way `np.squeeze` would; the three surviving
    axes get the real (freq, y, x) slices, in that order.
    """
    data_slices = iter((slice(*freq_bounds), slice(*y_bounds), slice(None)))
    return tuple(0 if size == 1 else next(data_slices) for size in disk_shape)


def _read_fits_block(
    path: str | Path,
    freq_bounds: tuple[int, int],
    y_bounds: tuple[int, int],
) -> NDArray[Any]:
    # Reopens the file per block rather than closing over one shared memmap:
    # dask.array.from_array unconditionally does `x = x.copy()` on anything
    # array-like (including a memmap-backed ndarray), and its default
    # tokenizing hashes the full buffer. Both silently force a full-cube
    # read into memory. Passing only cheap primitives (path, int bounds) to
    # a `dask.delayed` call sidesteps both.
    #
    # `memmap=False` + `.section` reads exactly the block's bytes and returns
    # a real array. A memmap slice would instead return a lazy view that both
    # keeps the whole-file mapping alive and faults in far more than the
    # block when something finally touches it.
    with fits.open(path, memmap=False) as hdul:
        block = hdul[0].section[_section_index(hdul[0].shape, freq_bounds, y_bounds)]
    # Native byte order: FITS is big-endian on disk, and leaving the block
    # that way makes dask insert an `astype` layer on top of every read.
    return block.astype(block.dtype.newbyteorder("="), copy=False)


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


def read_fits_cube_dask(
    path: str | Path,
    target_chunk_mb: float = DEFAULT_TARGET_CHUNK_MB,
) -> tuple[da.Array, Header]:
    """Lazily read a Stokes FITS cube as a spatially chunked dask array.

    Each block is a full-width band of `cy` image rows across all channels,
    read by its own `dask.delayed` task via `astropy.io.fits`' `.section`, so
    only that block's bytes are read and resident. Actual reads from disk are
    deferred until a block is computed.

    Degenerate length-1 axes (e.g. a dummy Stokes axis, common in ASKAP/EMU
    cutout cubes) are dropped.

    Peak memory during processing is a few times `target_chunk_mb`, not a hard
    cap: a block is copied when it is materialised, and again when it is
    written out. `rm_lite.tools_3d.rmsynth.rmsynth_3d` shrinks these chunks
    further where its complex128 output would otherwise outgrow them.

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

    blocks = [
        da.from_delayed(
            delayed(_read_fits_block, pure=True)(path, (0, n_freq), y_bounds),
            shape=(n_freq, y_bounds[1] - y_bounds[0], nx),
            dtype=dtype,
        )
        for y_bounds in _chunk_bounds(ny, cy)
    ]

    return da.concatenate(blocks, axis=1), header


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

    blocks = [
        da.from_delayed(
            delayed(_read_fits_block, pure=True)(path, freq_bounds, (0, ny)),
            shape=(freq_bounds[1] - freq_bounds[0], ny, nx),
            dtype=dtype,
        )
        for freq_bounds in _chunk_bounds(n_freq, n_chan)
    ]

    return da.concatenate(blocks, axis=0), header


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


def write_zarr_group(
    store: str | Path,
    arrays: Mapping[str, da.Array],
    overwrite: bool = True,
) -> None:
    """Write a set of dask arrays lazily/incrementally to a shared zarr store.

    Each array is written chunk-by-chunk via `dask.array.Array.to_zarr`; the
    full array is never materialised in memory before writing. All arrays are
    written in a single `dask.compute()` call rather than one `to_zarr()` call
    per array: if two arrays share upstream graph nodes (e.g. the four outputs
    of `rm_lite.tools_3d.rmclean.run_rmclean`, which all come from one per-chunk
    `dask.delayed` call), computing them separately would silently redo that
    shared work once per array.

    Args:
        store (str | Path): Path to the zarr store (a group containing one
            array per key in `arrays`).
        arrays (Mapping[str, da.Array]): Name -> dask array to write.
        overwrite (bool, optional): Overwrite existing arrays. Defaults to True.
    """
    writes = [
        array.to_zarr(store, component=name, overwrite=overwrite, compute=False)
        for name, array in arrays.items()
    ]
    tick = time.time()
    with ProgressBar():
        compute(*writes)
    tock = time.time()
    logger.info(f"Wrote {list(arrays)} to {store} in {tock - tick:.3g} seconds.")


def _channel_mad_std_block(block: NDArray[np.float64]) -> NDArray[np.float64]:
    n_freq_block = block.shape[0]
    return mad_std(block.reshape(n_freq_block, -1), axis=1, ignore_nan=True)


def _channel_mad_lazy(cube: da.Array) -> da.Array:
    """Lazy per-channel MAD std over every spatial pixel of a cube.

    Needs every pixel of a channel in one block (a robust median can't be
    combined incrementally across separate spatial chunks), so a spatially
    chunked cube has to be gathered first. Not computed -- so several cubes
    can be reduced in one `compute` call.
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
        _channel_mad_std_block, cube, drop_axis=(1, 2), dtype=np.float64
    )


def estimate_stokes_i_channel_noise(stokes_i: da.Array) -> NDArray[np.float64]:
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
    (noise,) = compute(_channel_mad_lazy(stokes_i))
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
    q_noise, u_noise = compute(_channel_mad_lazy(stokes_q), _channel_mad_lazy(stokes_u))
    tock = time.time()
    logger.info(f"Per-channel noise estimation completed in {tock - tick:.3g} seconds.")
    return np.abs(q_noise + u_noise) / 2.0
