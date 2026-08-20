"""Subprocess worker for the FITS-path memory-scaling test.

Companion to `_dask_memory_worker.py`, but driving `rmsynth_3d_from_fits` on
real FITS files on disk with a noise-based `weight_type`, so the FITS reader
and the per-channel noise estimator are both exercised. Those are the two
places the array-based worker never touches, and both were previously
unbounded: the reader returned lazy memmap views (so the whole cube faulted in
downstream) and the noise estimator gathered the whole cube into one task.

Reports the `tracemalloc` peak, i.e. the largest amount of data live at once,
rather than peak RSS. RSS measures the allocator's high-water mark, which on
glibc keeps freed block buffers in the heap instead of returning them: the
same run that peaks at 39 MB of live data reports 239 MB of RSS on macOS and
~1 GB on Linux, climbing with the number of chunks read rather than with the
working set. That made an RSS bound untestable, since it grew with how finely
the cube was chunked, i.e. the opposite of the property under test.

The cubes are written by the caller, not here, so nothing but the pipeline
itself allocates in this process. Prints the peak in kB.
"""

from __future__ import annotations

import logging
import sys
import tempfile
import tracemalloc
from pathlib import Path

import dask
from rm_lite.tools_3d.rmsynth import rmsynth_3d_from_fits
from rm_lite.utils.dask_io import write_zarr_group

logging.disable(logging.CRITICAL)


def main() -> None:
    q_path = Path(sys.argv[1])
    u_path = Path(sys.argv[2])
    target_chunk_mb = float(sys.argv[3])
    d_phi_radm2 = float(sys.argv[4])
    phi_max_radm2 = float(sys.argv[5])

    # Synchronous: peak live data is roughly (threads x per-task footprint), so
    # a threaded scheduler makes the peak depend on how tasks happen to
    # overlap. The thread multiplier is a documented property of
    # `read_fits_cube_dask`, not something this scaling test measures.
    dask.config.set(scheduler="synchronous")

    # Started before the synthesis call, not just the write: building the graph
    # also runs the per-channel noise estimate, which is half of what this
    # measures.
    tracemalloc.start()
    with tempfile.TemporaryDirectory() as tmpdir:
        synth = rmsynth_3d_from_fits(
            q_path,
            u_path,
            d_phi_radm2=d_phi_radm2,
            phi_max_radm2=phi_max_radm2,
            weight_type="variance",
            target_chunk_mb=target_chunk_mb,
        )
        # FDF only: the RMSF cube is the same spectrum in every pixel, so
        # writing it doubles the runtime and tests nothing extra here. Written
        # lazily via `write_zarr_group` rather than `.compute()`, which would
        # assemble the whole result in memory whatever the chunking.
        write_zarr_group(f"{tmpdir}/out.zarr", {"fdf_dirty": synth.fdf_dirty_cube})
    peak_bytes = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    print(peak_bytes // 1024)


if __name__ == "__main__":
    main()
