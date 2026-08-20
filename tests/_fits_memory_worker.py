"""Subprocess worker for the FITS-path memory-scaling test.

Companion to `_dask_memory_worker.py`, but driving `rmsynth_3d_from_fits` on
real FITS files on disk with a noise-based `weight_type`, so the FITS reader
and the per-channel noise estimator are both exercised. Those are the two
places the array-based worker never touches, and both were previously
unbounded: the reader returned lazy memmap views (so the whole cube faulted in
downstream) and the noise estimator gathered the whole cube into one task.

The cubes are written by the caller, not here, so nothing but the pipeline
itself allocates in this process. Prints the computation-phase peak RSS delta
in kB, see `_dask_memory_worker` for why a delta rather than an absolute.
"""

from __future__ import annotations

import logging
import resource
import sys
import tempfile
from pathlib import Path

import dask
from rm_lite.tools_3d.rmsynth import rmsynth_3d_from_fits
from rm_lite.utils.dask_io import write_zarr_group

logging.disable(logging.CRITICAL)


def _current_rss_kb() -> int:
    """Current resident set size in kB."""
    try:
        with Path("/proc/self/status").open("r") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError:
        pass
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        rss //= 1024  # macOS reports bytes
    return rss


def main() -> None:
    q_path = Path(sys.argv[1])
    u_path = Path(sys.argv[2])
    target_chunk_mb = float(sys.argv[3])
    d_phi_radm2 = float(sys.argv[4])
    phi_max_radm2 = float(sys.argv[5])

    # Fixed thread count: peak RSS is roughly (threads x per-task footprint),
    # so leaving it to the host's core count makes the number machine-dependent.
    dask.config.set(scheduler="threads", num_workers=2)

    pre_compute_rss = _current_rss_kb()

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
        # writing it doubles the runtime and tests nothing extra here.
        write_zarr_group(f"{tmpdir}/out.zarr", {"fdf_dirty": synth.fdf_dirty_cube})

    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        peak_rss //= 1024

    print(max(0, peak_rss - pre_compute_rss))


if __name__ == "__main__":
    main()
