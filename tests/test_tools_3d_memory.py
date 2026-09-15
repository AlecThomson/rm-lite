"""Peak-memory scaling tests for the dask-chunked 3D RM-synthesis pipeline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from rm_lite.tools_3d.rmsynth import target_chunk_mb_for_worker

if TYPE_CHECKING:
    from collections.abc import Callable

# Peak is a fixed cost plus a multiple of the target, so only the slope between
# two targets is the multiple. One target alone measures the interpreter.
TARGETS_MB = (16.0, 64.0)

N_FREQ = 48


@pytest.fixture(scope="module")
def peak_kb(pytestconfig: pytest.Config) -> Callable[..., int]:
    """Run a worker script in a fresh process and read its reported peak, in kB."""
    scripts = pytestconfig.rootpath / "tests" / "scripts"

    def run(script: str, *args: object) -> int:
        worker = scripts / f"{script}.py"
        result = subprocess.run(
            [sys.executable, str(worker), *(str(a) for a in args)],
            capture_output=True,
            text=True,
            check=True,
        )
        return int(result.stdout.strip().splitlines()[-1])

    return run


@pytest.fixture(scope="module")
def fits_cubes(
    tmp_path_factory: pytest.TempPathFactory, fits_cube: Callable[..., Path]
) -> Callable[..., list[Path]]:
    """Q/U FITS cube pairs at a given size, with a degenerate leading Stokes axis."""
    tmpdir = tmp_path_factory.mktemp("memory_cubes")
    built: dict[tuple[int, int, tuple[str, ...]], list[Path]] = {}

    def make(side: int, *, seed: int = 0, stokes: tuple[str, ...] = ("q", "u")):
        key = (side, seed, stokes)
        if key in built:
            return built[key]
        freq_arr_hz = 8.0e8 + 1.0e6 * np.arange(N_FREQ)
        rng = np.random.default_rng(seed)
        built[key] = [
            fits_cube(
                tmpdir / f"{stoke}_{side}_{seed}.fits",
                rng.normal(0, 1, (1, N_FREQ, side, side)),
                freq_arr_hz,
            )
            for stoke in stokes
        ]
        return built[key]

    return make


def test_memory_scales_with_chunk_size_not_cube_size(
    peak_kb: Callable[..., int],
) -> None:
    """Chunked computation must peak well below a single-block one, same cube."""
    side, n_freq, d_phi_radm2 = 300, 40, 10.0
    small_chunk = peak_kb("dask_memory_worker", side, n_freq, "32", d_phi_radm2)
    full_block = peak_kb("dask_memory_worker", side, n_freq, "full", d_phi_radm2)

    assert small_chunk < full_block * 0.7, (
        f"small-chunk computation-phase RSS delta ({small_chunk} kB) should be "
        f"well below the single-block one ({full_block} kB) for the same cube"
    )


@pytest.mark.parametrize(
    "options", ["clean,maps", "clean,maps,per_pixel_rmsf", "clean,maps,debias"]
)
def test_peak_memory_per_target_stays_within_the_budgeted_factor(
    peak_kb: Callable[..., int], fits_cubes: Callable[..., list[Path]], options: str
) -> None:
    """Peak grows by no more than the multiple `target_chunk_mb_for_worker` assumes."""
    budgeted = 1024.0 / target_chunk_mb_for_worker(
        1024.0,
        per_pixel_rmsf="per_pixel_rmsf" in options,
        debias="debias" in options,
    )
    # Small enough to RM-CLEAN once per target, twice per option.
    q_path, u_path = fits_cubes(256, seed=1)
    small_target, large_target = TARGETS_MB
    # Fine and wide, so the Faraday axis dominates the fixed cost.
    d_phi_radm2, phi_max_radm2 = 2.0, 400.0
    peaks = [
        peak_kb(
            "budget_memory_worker",
            q_path,
            u_path,
            target,
            d_phi_radm2,
            phi_max_radm2,
            options,
        )
        / 1024
        for target in (small_target, large_target)
    ]

    slope = (peaks[1] - peaks[0]) / (large_target - small_target)
    assert slope <= budgeted, (
        f"{options} costs {slope:.2f} MB of peak per MB of target, over the "
        f"{budgeted} budgeted for it"
    )


def test_zarr_conversion_peak_stays_inside_the_target(
    peak_kb: Callable[..., int], fits_cubes: Callable[..., list[Path]]
) -> None:
    """A conversion costs about one target, as `zarr_store_layout` assumes."""
    # Wide enough that the shard is set by the target rather than capped by the
    # cube, which needs more pixels than three times the largest target holds.
    side = 640
    (path,) = fits_cubes(side, seed=2, stokes=("q",))
    small_target, large_target = TARGETS_MB
    peaks = [
        peak_kb("zarr_convert_memory_worker", path, target, N_FREQ, side, side) / 1024
        for target in (small_target, large_target)
    ]

    slope = (peaks[1] - peaks[0]) / (large_target - small_target)
    assert slope <= 1.2, (
        f"a conversion costs {slope:.2f} MB of peak per MB of target, over the "
        "1.2 a third-of-the-target shard budgets for"
    )


def test_fits_path_memory_scales_with_chunk_size_not_cube_size(
    peak_kb: Callable[..., int], fits_cubes: Callable[..., list[Path]]
) -> None:
    """Peak memory is near-flat in cube size at a fixed target."""
    # 4x the pixels between the two cubes, at one fixed target_chunk_mb.
    small_side, large_side = 512, 1024
    target_chunk_mb = 4.0
    # Pinned so the Faraday-depth axis stays a sane length for this narrow
    # synthetic band; left to the defaults it lands near 10x n_freq.
    phi_max_radm2 = 200.0
    small, large = (
        peak_kb(
            "fits_memory_worker",
            *fits_cubes(side),
            target_chunk_mb,
            10.0,
            phi_max_radm2,
        )
        / 1024
        for side in (small_side, large_side)
    )

    def cube_mb(side: int) -> float:
        return N_FREQ * side**2 * 4 / 1024**2

    cube_growth = cube_mb(large_side) / cube_mb(small_side)
    # Peak does grow a little with cube size: 4x the blocks means a bigger graph.
    # Measured 1.15x for a 4x cube, so half the cube's own growth is a fair bar.
    assert large < 0.5 * cube_growth * small, (
        f"peak live data should grow far slower than cube size: {small:.0f} MB "
        f"on a {cube_mb(small_side):.0f} MB cube vs {large:.0f} MB on a "
        f"{cube_mb(large_side):.0f} MB cube, {large / small:.2g}x for a "
        f"{cube_growth:.0f}x cube, at the same {target_chunk_mb} MB chunk target"
    )
    # An absolute bound too, since a ratio alone passes if both sizes blow up.
    # Measured 8.5x and 9.75x the target; a whole-cube read per block gives 18x.
    assert large < 20 * target_chunk_mb, (
        f"peak live data ({large:.0f} MB) should be set by the "
        f"{target_chunk_mb} MB chunk target, not the "
        f"{cube_mb(large_side):.0f} MB cube: that is "
        f"{large / target_chunk_mb:.0f}x the target"
    )
