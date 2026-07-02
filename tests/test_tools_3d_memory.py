"""Peak-memory scaling test for the dask-chunked 3D RM-synthesis pipeline.

Runs the pipeline in a fresh subprocess per configuration so
`resource.getrusage(...).ru_maxrss` reports a clean per-run peak, and writes
output via `write_zarr_group` rather than `.compute()`, since `.compute()`
always assembles the full result in memory regardless of chunk size -- see
`tests/_dask_memory_worker.py` for the worker itself.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

WORKER = pathlib.Path(__file__).parent / "_dask_memory_worker.py"

CUBE_SIDE = 300
N_FREQ = 40
D_PHI_RADM2 = 10.0


def _peak_rss(chunk_arg: str) -> int:
    result = subprocess.run(
        [
            sys.executable,
            str(WORKER),
            str(CUBE_SIDE),
            str(N_FREQ),
            chunk_arg,
            str(D_PHI_RADM2),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(result.stdout.strip().splitlines()[-1])


def test_memory_scales_with_chunk_size_not_cube_size():
    small_chunk_peak_rss = _peak_rss("32")
    full_block_peak_rss = _peak_rss("full")

    assert small_chunk_peak_rss < full_block_peak_rss * 0.7, (
        f"small-chunk peak RSS ({small_chunk_peak_rss}) should be well below "
        f"the single-block peak RSS ({full_block_peak_rss}) for the same cube"
    )
