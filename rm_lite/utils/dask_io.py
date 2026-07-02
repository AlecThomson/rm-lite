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

import astropy.units as u
import dask.array as da
import numpy as np
from astropy.io import fits
from astropy.io.fits import Header
from astropy.stats import mad_std
from astropy.wcs import WCS
from dask.base import compute
from numpy.typing import NDArray

DEFAULT_TARGET_CHUNK_MB = 256


def spatial_chunk_size(
    n_freq: int,
    ny: int,
    nx: int,
    itemsize: int,
    target_chunk_mb: float = DEFAULT_TARGET_CHUNK_MB,
) -> tuple[int, int]:
    """Pick a square spatial chunk size for a fixed target chunk memory footprint.

    The frequency/Faraday-depth axis is never chunked, so a chunk's memory
    footprint is `n_freq * cy * cx * itemsize`. `cy` and `cx` are solved for
    that footprint to equal `target_chunk_mb`, then clipped to the image
    dimensions.

    Args:
        n_freq (int): Size of the (unchunked) spectral axis.
        ny (int): Full image height in pixels.
        nx (int): Full image width in pixels.
        itemsize (int): Size in bytes of one array element.
        target_chunk_mb (float, optional): Target memory footprint of a
            single chunk, in MB. Defaults to 256.

    Returns:
        tuple[int, int]: Spatial chunk size (cy, cx).
    """
    target_chunk_bytes = target_chunk_mb * 1024**2
    pixels_per_chunk = target_chunk_bytes / (n_freq * itemsize)
    side = max(1, int(np.floor(np.sqrt(pixels_per_chunk))))
    return min(side, ny), min(side, nx)


def read_fits_cube_dask(
    path: str | Path,
    target_chunk_mb: float = DEFAULT_TARGET_CHUNK_MB,
) -> tuple[da.Array, Header]:
    """Lazily read a Stokes FITS cube as a chunked dask array.

    The FITS file is opened with `memmap=True` so the underlying numpy array
    is a memmap; wrapping it in `dask.array.from_array` defers actual reads
    from disk until a chunk is computed.

    Degenerate length-1 axes (e.g. a dummy Stokes axis, common in ASKAP/EMU
    cutout cubes) are squeezed out first. `np.squeeze` on a memmap returns a
    view, not a copy, so this stays lazy.

    Args:
        path (str | Path): Path to the FITS cube. Assumed axis order
            (freq, y, x) once degenerate axes are squeezed out, i.e. the
            frequency axis is first in numpy order.
        target_chunk_mb (float, optional): Target chunk memory footprint in
            MB, see `spatial_chunk_size`. Defaults to 256.

    Returns:
        tuple[da.Array, Header]: Lazy dask array and the FITS header.
    """
    hdul = fits.open(path, memmap=True)
    data = np.squeeze(hdul[0].data)
    header = hdul[0].header

    if data.ndim != 3:
        msg = (
            "Expected a 3D (freq, y, x) cube after squeezing degenerate axes, "
            f"got shape {data.shape} from {path}."
        )
        raise ValueError(msg)

    n_freq, ny, nx = data.shape
    cy, cx = spatial_chunk_size(
        n_freq=n_freq,
        ny=ny,
        nx=nx,
        itemsize=data.itemsize,
        target_chunk_mb=target_chunk_mb,
    )
    return da.from_array(data, chunks=(-1, cy, cx)), header


def freq_arr_hz_from_header(header: Header, n_freq: int) -> NDArray[np.float64]:
    """Derive the frequency array in Hz from a FITS header's spectral WCS.

    Args:
        header (Header): FITS header containing a spectral axis.
        n_freq (int): Number of channels along the spectral axis.

    Returns:
        NDArray[np.float64]: Frequency array in Hz.
    """
    spectral_wcs = WCS(header).spectral
    freq_quantity = spectral_wcs.pixel_to_world(np.arange(n_freq))
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
    of `rm_lite.tools_3d.rmclean.rmclean_3d`, which all come from one per-chunk
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
    compute(*writes)


def _channel_mad_std_block(block: NDArray[np.float64]) -> NDArray[np.float64]:
    n_freq_block = block.shape[0]
    return mad_std(block.reshape(n_freq_block, -1), axis=1, ignore_nan=True)


def estimate_channel_noise_mad(
    stokes_q: da.Array,
    stokes_u: da.Array,
) -> NDArray[np.float64]:
    """Robust per-channel noise from Stokes Q/U cubes, for auto-masking/thresholding.

    Computes `astropy.stats.mad_std` over every spatial pixel in each channel
    plane, then combines the Q and U estimates the same way
    `rm_lite.utils.synthesis.compute_rmsynth_params` combines a per-channel
    complex error into `real_qu_error` (`abs(real + imag) / 2`).

    Unlike the RM-synth/CLEAN chunking convention, this rechunks the spatial
    axes to a single block per channel: a robust statistic like the median
    can't be combined incrementally across separate spatial chunks the way a
    sum or mean can, so each channel's full spatial plane has to be brought
    together to compute it -- the same "per channel, one full image plane at a
    time" access pattern classic per-channel RMS estimation uses.

    The per-channel noise this returns can be turned into a weight array
    (`weight_arr = 1 / noise**2`) for `rm_lite.tools_3d.rmsynth.rmsynth_3d`,
    and from there `rm_lite.utils.synthesis.compute_theoretical_noise` gives
    the FDF-domain noise used to set `rm_lite.tools_3d.rmclean.rmclean_3d`'s
    `mask`/`threshold` (mirroring the 1D `run_rmclean_from_synth` auto-mask/
    auto-threshold convention).

    Args:
        stokes_q (da.Array): Stokes Q dask array, shape (n_freq, ny, nx).
        stokes_u (da.Array): Stokes U dask array, same shape as `stokes_q`.

    Returns:
        NDArray[np.float64]: Per-channel noise estimate, shape (n_freq,). A
        plain numpy array, not lazy -- computed once, explicitly, here.
    """
    if stokes_q.shape != stokes_u.shape:
        msg = f"Stokes Q and U must have the same shape. Got {stokes_q.shape} and {stokes_u.shape}."
        raise ValueError(msg)

    q_full_spatial = stokes_q.rechunk({1: -1, 2: -1})
    u_full_spatial = stokes_u.rechunk({1: -1, 2: -1})

    q_noise, u_noise = compute(
        da.map_blocks(
            _channel_mad_std_block, q_full_spatial, drop_axis=(1, 2), dtype=np.float64
        ),
        da.map_blocks(
            _channel_mad_std_block, u_full_spatial, drop_axis=(1, 2), dtype=np.float64
        ),
    )
    return np.abs(q_noise + u_noise) / 2.0
