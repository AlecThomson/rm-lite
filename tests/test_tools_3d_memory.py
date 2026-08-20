"""Peak-memory scaling test for the dask-chunked 3D RM-synthesis pipeline.

Runs the pipeline in a fresh subprocess per configuration so
`resource.getrusage(...).ru_maxrss` reports a clean per-run peak, and writes
output via `write_zarr_group` rather than `.compute()`, since `.compute()`
always assembles the full result in memory regardless of chunk size. See
`tests/_dask_memory_worker.py` for the worker itself.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import numpy as np
import pytest
from astropy.io import fits

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
    small_chunk_compute_rss = _peak_rss("32")
    full_block_compute_rss = _peak_rss("full")

    assert small_chunk_compute_rss < full_block_compute_rss * 0.7, (
        f"small-chunk computation-phase RSS delta ({small_chunk_compute_rss} kB) "
        f"should be well below the single-block computation-phase RSS delta "
        f"({full_block_compute_rss} kB) for the same cube"
    )


FITS_WORKER = pathlib.Path(__file__).parent / "_fits_memory_worker.py"

FITS_N_FREQ = 48
# 4x the pixels between the two cubes, at one fixed target_chunk_mb.
FITS_SIDES = (512, 1024)
FITS_TARGET_CHUNK_MB = 4.0
# Pinned so the Faraday-depth axis stays a sane length for this narrow
# synthetic band; left to the defaults it lands near 10x n_freq.
FITS_PHI_MAX_RADM2 = 200.0
# Peak RSS does grow a little with cube size at a fixed chunk target: 4x the
# pixels is 4x the blocks, so the graph and the zarr metadata grow with it.
# Measured 1.3-1.5x across this 4x size step, so the bar is half the cube's
# own growth: clear of the real number, and well under the ~3-4x a cube-sized
# read would give. A flat 1.5x bar sat right on the true value and flaked.
MAX_RSS_GROWTH_FRACTION = 0.5


def _cube_mb(side: int) -> float:
    return FITS_N_FREQ * side**2 * 4 / 1024**2


@pytest.fixture(scope="module")
def qu_fits_cubes(tmp_path_factory) -> dict[int, tuple[pathlib.Path, pathlib.Path]]:
    """Q/U FITS cube pairs at two sizes, with a degenerate leading Stokes axis."""
    tmpdir = tmp_path_factory.mktemp("fits_memory")
    rng = np.random.default_rng(0)
    header = fits.Header()
    header["CTYPE3"] = "FREQ"
    header["CRVAL3"] = 8.0e8
    header["CDELT3"] = 1.0e6
    header["CRPIX3"] = 1
    header["CUNIT3"] = "Hz"
    cubes = {}
    for side in FITS_SIDES:
        paths = []
        for stokes in ("q", "u"):
            path = tmpdir / f"{stokes}_{side}.fits"
            data = rng.normal(0, 1, (1, FITS_N_FREQ, side, side))
            fits.PrimaryHDU(data.astype(">f4"), header=header).writeto(path)
            paths.append(path)
        cubes[side] = (paths[0], paths[1])
    return cubes


def _fits_peak_rss_mb(paths: tuple[pathlib.Path, pathlib.Path]) -> float:
    result = subprocess.run(
        [
            sys.executable,
            str(FITS_WORKER),
            str(paths[0]),
            str(paths[1]),
            str(FITS_TARGET_CHUNK_MB),
            str(D_PHI_RADM2),
            str(FITS_PHI_MAX_RADM2),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(result.stdout.strip().splitlines()[-1]) / 1024


def test_fits_path_memory_scales_with_chunk_size_not_cube_size(qu_fits_cubes):
    """`rmsynth_3d_from_fits` peak memory is flat in cube size at a fixed chunk target.

    Guards both FITS-path blowups at once. The reader used to hand dask lazy
    memmap views, so the whole cube faulted in when something downstream
    touched a block, and the per-channel noise estimator (reached here via
    `weight_type="variance"`) gathered the whole cube into a single task. Both
    made peak memory a function of cube size with `target_chunk_mb` inert, and
    both are invisible to `_dask_memory_worker`, which never reads a FITS file.
    """
    small_side, large_side = FITS_SIDES
    small = _fits_peak_rss_mb(qu_fits_cubes[small_side])
    large = _fits_peak_rss_mb(qu_fits_cubes[large_side])

    cube_growth = _cube_mb(large_side) / _cube_mb(small_side)
    assert large < MAX_RSS_GROWTH_FRACTION * cube_growth * small, (
        f"peak RSS should grow far slower than cube size: {small:.0f} MB on a "
        f"{_cube_mb(small_side):.0f} MB cube vs {large:.0f} MB on a "
        f"{_cube_mb(large_side):.0f} MB cube, {large / small:.2g}x for a "
        f"{cube_growth:.0f}x cube, at the same {FITS_TARGET_CHUNK_MB} MB "
        "chunk target"
    )
    assert large < _cube_mb(large_side), (
        f"peak RSS ({large:.0f} MB) exceeds one whole cube "
        f"({_cube_mb(large_side):.0f} MB) at a {FITS_TARGET_CHUNK_MB} MB chunk target"
    )
