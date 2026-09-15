"""Measures how many times `target_chunk_mb` one pipeline configuration peaks at.

Options are additive: `convert` (inputs to zarr first), `clean`,
`per_pixel_rmsf`, `maps` (moment and peak maps), `cubes` (write the FDF cube),
`debias`. With neither `maps` nor `cubes` the FDF cube is written anyway, or the
lazy graph never runs.

Prints the `tracemalloc` peak in kB. RSS is no good here: glibc keeps freed
buffers, so it climbs with the number of chunks, the opposite of what is under
test.
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

        # `.compute()` would assemble the whole
        # result in this process whatever the chunking, which is not what the
        # pipeline does and would swamp the measurement.
        write_zarr_group(f"{tmpdir}/out.zarr", targets)

    peak_bytes = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    print(peak_bytes // 1024)


if __name__ == "__main__":
    main()
