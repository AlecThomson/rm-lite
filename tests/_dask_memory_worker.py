"""Subprocess worker for the memory-scaling test.

Run in a fresh process per configuration so `resource.getrusage(...).ru_maxrss`
(monotonic, process-lifetime peak RSS) gives a clean per-run peak instead of
carrying over allocations from a previous configuration in the same process.

Writes output via `write_zarr_group` (lazy, chunk-by-chunk) rather than
`.compute()`, since `.compute()` always assembles the full result in memory
regardless of chunk size -- the property under test is that *processing*
memory (and the write path) is bounded by chunk size, not cube size.
"""

from __future__ import annotations

import logging
import resource
import sys
import tempfile

import dask.array as da
import numpy as np
from rm_lite.tools_3d.rmsynth import rmsynth_3d
from rm_lite.utils.dask_io import write_zarr_group

logging.disable(logging.CRITICAL)


def main() -> None:
    side = int(sys.argv[1])
    n_freq = int(sys.argv[2])
    chunk_arg = sys.argv[3]
    d_phi_radm2 = float(sys.argv[4])

    rng = np.random.default_rng(0)
    freqs = np.linspace(700e6, 1000e6, n_freq)
    stokes_q = rng.normal(0, 1, (n_freq, side, side))
    stokes_u = rng.normal(0, 1, (n_freq, side, side))

    chunk_size = side if chunk_arg == "full" else int(chunk_arg)
    q_dask = da.from_array(stokes_q, chunks=(-1, chunk_size, chunk_size))
    u_dask = da.from_array(stokes_u, chunks=(-1, chunk_size, chunk_size))

    synth = rmsynth_3d(q_dask, u_dask, freqs, d_phi_radm2=d_phi_radm2)

    with tempfile.TemporaryDirectory() as tmpdir:
        write_zarr_group(
            f"{tmpdir}/out.zarr",
            {"fdf_dirty": synth.fdf_dirty_cube, "rmsf": synth.rmsf_cube},
        )

    print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


if __name__ == "__main__":
    main()
