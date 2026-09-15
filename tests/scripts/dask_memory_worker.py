"""Subprocess worker for the memory-scaling test.

Run in a fresh process per configuration so one run's allocations cannot carry
into the next.

Writes output via `write_zarr_group` (lazy, chunk-by-chunk) rather than
`.compute()`, since `.compute()` always assembles the full result in memory
regardless of chunk size. The property under test is that *processing* memory
(and the write path) is bounded by chunk size, not cube size.

Prints the compute phase's own peak RSS, above the live RSS just before
`write_zarr_group`, so interpreter and input-cube overhead (identical across
chunkings, and varying by Python version) drops out.

Measuring that needs the kernel's resettable peak, `VmHWM`, not
`resource.getrusage(...).ru_maxrss`. `ru_maxrss` is a process-lifetime peak with
no way to reset it, so it also carries whatever setup transiently reached; when
that exceeds the compute peak, both chunkings report the same
"peak minus current" number and the comparison is meaningless. That made this
test flaky (~2 runs in 3 on CI) until it moved to `VmHWM` plus a `clear_refs`
reset. `ru_maxrss` remains the fallback off Linux, where the test is not
expected to be tight anyway.
"""

from __future__ import annotations

import contextlib
import logging
import resource
import sys
import tempfile
from pathlib import Path

import dask.array as da
import numpy as np
from rm_lite.tools_3d.rmsynth import rmsynth_3d
from rm_lite.utils.dask_io import write_zarr_group

logging.disable(logging.CRITICAL)


def _status_kb(field: str) -> int | None:
    """A /proc/self/status memory field in kB, or None if unreadable."""
    try:
        with Path("/proc/self/status").open("r") as fh:
            for line in fh:
                if line.startswith(field):
                    return int(line.split()[1])  # already in kB
    except OSError:
        pass
    return None


def _getrusage_peak_kb() -> int:
    """Process-lifetime peak RSS in kB, for platforms without /proc."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        peak //= 1024  # macOS reports bytes; normalise to kB
    return peak


def _current_rss_kb() -> int:
    """Live resident set size in kB."""
    rss = _status_kb("VmRSS:")
    return rss if rss is not None else _getrusage_peak_kb()


def _reset_peak_rss() -> None:
    """Reset the kernel's peak-RSS watermark (`VmHWM`); Linux 4.0+, else a no-op."""
    with contextlib.suppress(OSError):
        Path("/proc/self/clear_refs").write_text("5")


def _peak_rss_kb() -> int:
    """Peak RSS in kB since the last `_reset_peak_rss`, else process lifetime."""
    hwm = _status_kb("VmHWM:")
    return hwm if hwm is not None else _getrusage_peak_kb()


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

    # rmsynth_3d builds the dask graph lazily; no heavy allocation yet.
    # per_pixel_rmsf: the RMSF cube is the largest thing the graph produces, so
    # keep measuring the worst case rather than the default that omits it.
    synth = rmsynth_3d(
        q_dask, u_dask, freqs, d_phi_radm2=d_phi_radm2, per_pixel_rmsf=True
    )

    # Reset the peak watermark, then snapshot live RSS, so what follows measures
    # the compute phase alone: no setup transient above it, and no interpreter
    # or input-cube baseline below it.
    _reset_peak_rss()
    pre_compute_rss = _current_rss_kb()

    with tempfile.TemporaryDirectory() as tmpdir:
        write_zarr_group(
            f"{tmpdir}/out.zarr",
            {"fdf_dirty": synth.fdf_dirty_cube, "rmsf": synth.rmsf_cube},
        )

    print(max(0, _peak_rss_kb() - pre_compute_rss))


if __name__ == "__main__":
    main()
