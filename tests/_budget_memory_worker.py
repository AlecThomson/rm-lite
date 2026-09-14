"""Subprocess worker measuring peak memory for one pipeline configuration.

Companion to `_fits_memory_worker.py`, which pins that peak scales with
`target_chunk_mb` rather than cube size. This one answers the other question:
*how many times* the target a given set of options costs, so
`rm_lite.tools_3d.rmsynth.chunk_target_for_budget` can invert it.

Configurations rather than isolated stages, because a configuration is what a
caller chooses and what the budget function is asked about. What a worker needs
to survive is the largest live set the run reaches, wherever that happens.

Options, additive: `convert` converts the inputs to zarr first, `clean` runs
RM-CLEAN, `per_pixel_rmsf` gives every pixel its own RMSF, `maps` computes the
Faraday moment and peak maps, `cubes` writes the FDF cube, `debias` computes the
debiased FDF. With neither `maps` nor `cubes`, the FDF cube is written, since
something has to be computed or the lazy graph never runs.

Reports the `tracemalloc` peak for the reason `_fits_memory_worker` gives: RSS
on glibc keeps freed block buffers and climbs with the number of chunks, i.e.
the opposite of the property under test. Prints the peak in kB.
"""

from __future__ import annotations

import logging
import sys
import tempfile
import tracemalloc
from pathlib import Path

import dask
from rm_lite.tools_3d.rmclean import run_rmclean_from_synth
from rm_lite.tools_3d.rmsynth import rmsynth_3d_from_fits
from rm_lite.utils.dask_io import write_zarr_group
from rm_lite.utils.synthesis import debias_fdf

logging.disable(logging.CRITICAL)


def main() -> None:
    q_path = Path(sys.argv[1])
    u_path = Path(sys.argv[2])
    target_chunk_mb = float(sys.argv[3])
    d_phi_radm2 = float(sys.argv[4])
    phi_max_radm2 = float(sys.argv[5])
    options = set(sys.argv[6].split(",")) - {""}

    # Synchronous, so the peak is one task's footprint rather than however many
    # tasks a threaded scheduler happened to overlap. The thread multiplier is
    # applied by the budget function, not measured here.
    dask.config.set(scheduler="synchronous")

    tracemalloc.start()
    with tempfile.TemporaryDirectory() as tmpdir:
        synth = rmsynth_3d_from_fits(
            q_path,
            u_path,
            d_phi_radm2=d_phi_radm2,
            phi_max_radm2=phi_max_radm2,
            weight_type="variance",
            target_chunk_mb=target_chunk_mb,
            convert_to_zarr="convert" in options,
            per_pixel_rmsf="per_pixel_rmsf" in options,
        )

        targets = {}
        fdf_cube = synth.fdf_dirty_cube
        if "clean" in options:
            clean = run_rmclean_from_synth(synth)
            fdf_cube = clean.clean_fdf_cube
            if "maps" in options:
                targets["mom0"] = clean.mom0_map
                targets["peak_pi"] = clean.peak_pi_map
        if "debias" in options:
            targets["debias"] = debias_fdf(
                synth.fdf_dirty_cube, synth.phi_arr_radm2, synth.lam_sq_0_m2
            )
        if "cubes" in options or not targets:
            targets["fdf"] = fdf_cube

        # Written rather than computed: `.compute()` would assemble the whole
        # result in this process whatever the chunking, which is not what the
        # pipeline does and would swamp the measurement.
        write_zarr_group(f"{tmpdir}/out.zarr", targets)

    peak_bytes = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    print(peak_bytes // 1024)


if __name__ == "__main__":
    main()
