"""Measures `fits_cube_to_zarr` peak against the `target_chunk_mb` it was sized for.

Sizes the store the way `rmsynth_3d_from_fits` does, so the number is the one a
real run pays.

Prints the `tracemalloc` peak in kB. RSS is no good here: glibc keeps freed
buffers, so it climbs with the number of chunks.
"""

from __future__ import annotations

import logging
import sys
import tempfile
import tracemalloc
from pathlib import Path

import dask
import numpy as np
from rm_lite.tools_3d.rmsynth import fdf_spatial_chunk
from rm_lite.utils.dask_io import fits_cube_to_zarr, zarr_store_layout

logging.disable(logging.CRITICAL)


def main() -> None:
    fits_file = Path(sys.argv[1])
    target_chunk_mb = float(sys.argv[2])
    n_freq, ny, nx = (int(arg) for arg in sys.argv[3:6])

    # Synchronous, so the peak is one write task rather than however many a
    # threaded scheduler happened to overlap.
    dask.config.set(scheduler="synchronous")

    budget = fdf_spatial_chunk(
        2 * n_freq, np.dtype(np.complex64), target_chunk_mb, ny, nx
    )
    spatial_chunk, shard_rows = zarr_store_layout(
        n_freq=n_freq,
        ny=ny,
        nx=nx,
        itemsize=4,
        chunk_budget=budget,
        target_chunk_mb=target_chunk_mb,
    )

    tracemalloc.start()
    with tempfile.TemporaryDirectory() as tmpdir:
        fits_cube_to_zarr(
            fits_file,
            f"{tmpdir}/cube.zarr",
            spatial_chunk=spatial_chunk,
            shard_rows=shard_rows,
        )
    peak_bytes = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    print(peak_bytes // 1024)


if __name__ == "__main__":
    main()
