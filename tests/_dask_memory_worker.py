"""Subprocess worker for the memory-scaling test.

Run in a fresh process per configuration so `resource.getrusage(...).ru_maxrss`
(monotonic, process-lifetime peak RSS) gives a clean per-run peak instead of
carrying over allocations from a previous configuration in the same process.

Writes output via `write_zarr_group` (lazy, chunk-by-chunk) rather than
`.compute()`, since `.compute()` always assembles the full result in memory
regardless of chunk size. The property under test is that *processing* memory
(and the write path) is bounded by chunk size, not cube size.

The worker prints the *computation-phase RSS delta* (peak RSS minus the RSS
snapshot taken just before write_zarr_group is called) so that Python
interpreter baseline overhead — which varies across Python versions and does
not depend on chunk size — is excluded from the comparison.
"""

from __future__ import annotations

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


def _current_rss_kb() -> int:
    """Return the *current* resident set size in KB.

    Reads /proc/self/status on Linux for an accurate live value.
    Falls back to resource.getrusage on other platforms (note: ru_maxrss is
    monotonic/peak there, so the delta approach is less precise, but the test
    is only expected to be tight on Linux CI).
    """
    try:
        with Path("/proc/self/status").open("r") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])  # already in kB
    except OSError:
        pass
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        rss //= 1024  # macOS reports bytes; normalise to kB
    return rss


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
    synth = rmsynth_3d(q_dask, u_dask, freqs, d_phi_radm2=d_phi_radm2)

    # Snapshot current RSS before triggering computation.  Subtracting this
    # removes Python-interpreter and input-data overhead that is identical
    # across chunk configurations, isolating computation-phase memory.
    pre_compute_rss = _current_rss_kb()

    with tempfile.TemporaryDirectory() as tmpdir:
        write_zarr_group(
            f"{tmpdir}/out.zarr",
            {"fdf_dirty": synth.fdf_dirty_cube, "rmsf": synth.rmsf_cube},
        )

    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        peak_rss //= 1024  # bytes → kB

    # ru_maxrss is monotonic; pre_compute_rss is the live value just before
    # the heavy work, so the delta is the computation-phase peak contribution.
    print(max(0, peak_rss - pre_compute_rss))


if __name__ == "__main__":
    main()
