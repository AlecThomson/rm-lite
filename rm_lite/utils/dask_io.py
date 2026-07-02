"""Dask-backed I/O helpers for chunked 3D RM-synthesis/CLEAN.

Chunking convention used throughout tools_3d: the spectral/Faraday-depth axis
(axis 0) is always kept whole in a single chunk, since every per-pixel
RM-synthesis/CLEAN call needs the full spectrum for that pixel. Only the two
spatial axes are chunked, sized from a target per-chunk memory footprint so
memory use is bounded by chunk size rather than cube size.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import dask.array as da
import numpy as np
from astropy.io import fits
from astropy.io.fits import Header

DEFAULT_TARGET_CHUNK_BYTES = 256 * 1024**2


def spatial_chunk_size(
    n_freq: int,
    ny: int,
    nx: int,
    itemsize: int,
    target_chunk_bytes: int = DEFAULT_TARGET_CHUNK_BYTES,
) -> tuple[int, int]:
    """Pick a square spatial chunk size for a fixed target chunk memory footprint.

    The frequency/Faraday-depth axis is never chunked, so a chunk's memory
    footprint is `n_freq * cy * cx * itemsize`. `cy` and `cx` are solved for
    that footprint to equal `target_chunk_bytes`, then clipped to the image
    dimensions.

    Args:
        n_freq (int): Size of the (unchunked) spectral axis.
        ny (int): Full image height in pixels.
        nx (int): Full image width in pixels.
        itemsize (int): Size in bytes of one array element.
        target_chunk_bytes (int, optional): Target memory footprint of a
            single chunk, in bytes. Defaults to 256 MiB.

    Returns:
        tuple[int, int]: Spatial chunk size (cy, cx).
    """
    pixels_per_chunk = target_chunk_bytes / (n_freq * itemsize)
    side = max(1, int(np.floor(np.sqrt(pixels_per_chunk))))
    return min(side, ny), min(side, nx)


def read_fits_cube_dask(
    path: str | Path,
    target_chunk_bytes: int = DEFAULT_TARGET_CHUNK_BYTES,
) -> tuple[da.Array, Header]:
    """Lazily read a Stokes FITS cube as a chunked dask array.

    The FITS file is opened with `memmap=True` so the underlying numpy array
    is a memmap; wrapping it in `dask.array.from_array` defers actual reads
    from disk until a chunk is computed.

    Args:
        path (str | Path): Path to the FITS cube. Assumed axis order
            (freq, y, x), i.e. the frequency axis is first in numpy order.
        target_chunk_bytes (int, optional): Target chunk memory footprint in
            bytes, see `spatial_chunk_size`. Defaults to 256 MiB.

    Returns:
        tuple[da.Array, Header]: Lazy dask array and the FITS header.
    """
    hdul = fits.open(path, memmap=True)
    data = hdul[0].data
    header = hdul[0].header

    n_freq, ny, nx = data.shape
    cy, cx = spatial_chunk_size(
        n_freq=n_freq,
        ny=ny,
        nx=nx,
        itemsize=data.itemsize,
        target_chunk_bytes=target_chunk_bytes,
    )
    return da.from_array(data, chunks=(-1, cy, cx)), header


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
    full array is never materialised in memory before writing.

    Args:
        store (str | Path): Path to the zarr store (a group containing one
            array per key in `arrays`).
        arrays (Mapping[str, da.Array]): Name -> dask array to write.
        overwrite (bool, optional): Overwrite existing arrays. Defaults to True.
    """
    for name, array in arrays.items():
        array.to_zarr(store, component=name, overwrite=overwrite)
