"""RM-CLEAN on chunked 3D FDF/RMSF cubes via dask.

`rm_lite.utils.clean.rmclean` runs a genuinely per-pixel, data-dependent
Hogbom loop (`_rmclean_nd` loops in Python over every pixel in the block it is
given) -- it cannot be vectorized into a single array op. The dask-level
parallelism here comes from calling that per-pixel loop once per spatial
chunk via `dask.array.map_blocks`.

`rmclean` returns four outputs (clean FDF, model FDF, residual FDF, iteration
count) from one Hogbom loop per block. A naive per-output `map_blocks` would
rerun that loop four times per chunk. Instead, each block is run through
`dask.delayed` exactly once; the four outputs are split out of the same
delayed call with `dask.delayed` attribute access (each access is a distinct
delayed node depending on the same shared computation, so the scheduler runs
the block once and reuses the result for all four), then reassembled into
full cubes with `dask.array.from_delayed` + `dask.array.block`. This also
avoids the in-place-shared-array-mutation pattern that silently returns all
zeros for the iteration-count map under multiprocessing -- every output here
flows through the task graph instead of a mutated shared array.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import dask.array as da
import numpy as np
from dask.delayed import delayed
from numpy.typing import NDArray

from rm_lite.tools_3d.rmsynth import RMSynth3DResults
from rm_lite.utils.clean import rmclean
from rm_lite.utils.logging import logger, quiet_logs


class RMClean3DResults(NamedTuple):
    """Results of chunked 3D RM-CLEAN."""

    clean_fdf_cube: da.Array
    """Cleaned FDF cube, lazy dask array of shape (n_phi, ny, nx)."""
    model_fdf_cube: da.Array
    """Clean-component (model) FDF cube, same shape as `clean_fdf_cube`."""
    resid_fdf_cube: da.Array
    """Residual FDF cube, same shape as `clean_fdf_cube`."""
    iter_count_map: da.Array
    """Per-pixel CLEAN iteration count, lazy dask array of shape (ny, nx)."""


class _RMCleanBlockResult(NamedTuple):
    clean_fdf: NDArray[np.complex128]
    model_fdf: NDArray[np.complex128]
    resid_fdf: NDArray[np.complex128]
    iter_count: NDArray[np.int64]


def _clean_block(
    dirty_fdf_block: NDArray[np.complex128],
    rmsf_block: NDArray[np.complex128],
    phi_arr_radm2: NDArray[np.float64],
    phi_double_arr_radm2: NDArray[np.float64],
    fwhm_rmsf_radm2: float,
    mask: float,
    threshold: float,
    max_iter: int,
    gain: float,
    log_level: int,
) -> _RMCleanBlockResult:
    with quiet_logs(log_level):
        result = rmclean(
            dirty_fdf_arr=dirty_fdf_block,
            phi_arr_radm2=phi_arr_radm2,
            rmsf_arr=rmsf_block,
            phi_double_arr_radm2=phi_double_arr_radm2,
            fwhm_rmsf_arr=np.array(fwhm_rmsf_radm2),
            mask=mask,
            threshold=threshold,
            max_iter=max_iter,
            gain=gain,
        )
    return _RMCleanBlockResult(
        clean_fdf=result.clean_fdf_arr,
        model_fdf=result.model_fdf_arr,
        resid_fdf=result.resid_fdf_arr,
        # RMCleanResults.clean_iter_arr is annotated NDArray[np.int16] but is
        # actually built with dtype=int (int64) in _rmclean_nd.
        iter_count=result.clean_iter_arr,  # type: ignore[arg-type]
    )


def rmclean_3d(
    fdf_dirty_cube: da.Array,
    rmsf_cube: da.Array,
    phi_arr_radm2: NDArray[np.float64],
    phi_double_arr_radm2: NDArray[np.float64],
    fwhm_rmsf_radm2: float,
    mask: float,
    threshold: float,
    max_iter: int = 1000,
    gain: float = 0.1,
    log_level: int = logging.ERROR,
) -> RMClean3DResults:
    """Run RM-CLEAN on chunked dirty FDF and RMSF cubes.

    Args:
        fdf_dirty_cube (da.Array): Dirty FDF cube, shape (n_phi, ny, nx),
            chunked spatially only (as produced by `rm_lite.tools_3d.rmsynth.rmsynth_3d`).
        rmsf_cube (da.Array): RMSF cube, shape (n_phi_double, ny, nx), with the
            same spatial chunking as `fdf_dirty_cube`.
        phi_arr_radm2 (NDArray[np.float64]): Faraday depth values in rad/m^2.
        phi_double_arr_radm2 (NDArray[np.float64]): Double-length Faraday depth
            values in rad/m^2, for the RMSF.
        fwhm_rmsf_radm2 (float): RMSF FWHM, shared by every pixel (3D RM-CLEAN
            here does not support a per-pixel FWHM map).
        mask (float): Masking threshold -- pixels below this value are not cleaned.
        threshold (float): Cleaning threshold -- stop when all pixels are below this value.
        max_iter (int, optional): Maximum CLEAN iterations. Defaults to 1000.
        gain (float, optional): CLEAN loop gain. Defaults to 0.1.
        log_level (int, optional): Log level applied to `rm_lite`'s logger while
            each chunk runs. `rmclean`'s Hogbom loop logs at INFO and WARNING
            per pixel (e.g. "Starting minor loop...", "All channels masked...
            performed N iterations") -- these are routine per-pixel loop
            termination conditions, not anomalies, and at cube scale they're
            just noise, so this defaults to ERROR (silencing both). Pass
            `logging.WARNING` or `logging.INFO` to restore progressively more
            per-pixel verbosity, e.g. while debugging a specific chunk.
            Defaults to `logging.ERROR`.

    Returns:
        RMClean3DResults: Lazy clean/model/residual FDF cubes and iteration-count map.
    """
    if fdf_dirty_cube.chunks[1:] != rmsf_cube.chunks[1:]:
        msg = "fdf_dirty_cube and rmsf_cube must have identical spatial chunking."
        raise ValueError(msg)

    n_phi = fdf_dirty_cube.shape[0]
    spatial_chunks = fdf_dirty_cube.chunks[1:]
    numblocks = fdf_dirty_cube.numblocks

    dirty_delayed = fdf_dirty_cube.to_delayed()
    rmsf_delayed = rmsf_cube.to_delayed()

    clean_blocks = np.empty(numblocks, dtype=object)
    model_blocks = np.empty(numblocks, dtype=object)
    resid_blocks = np.empty(numblocks, dtype=object)
    iter_blocks = np.empty(numblocks[1:], dtype=object)

    for idx in np.ndindex(numblocks):
        _, iy, ix = idx
        cy = spatial_chunks[0][iy]
        cx = spatial_chunks[1][ix]

        block_result = delayed(_clean_block, pure=True)(
            dirty_delayed[idx],
            rmsf_delayed[idx],
            phi_arr_radm2,
            phi_double_arr_radm2,
            fwhm_rmsf_radm2,
            mask,
            threshold,
            max_iter,
            gain,
            log_level,
        )

        clean_blocks[idx] = da.from_delayed(
            block_result.clean_fdf, shape=(n_phi, cy, cx), dtype=np.complex128
        )
        model_blocks[idx] = da.from_delayed(
            block_result.model_fdf, shape=(n_phi, cy, cx), dtype=np.complex128
        )
        resid_blocks[idx] = da.from_delayed(
            block_result.resid_fdf, shape=(n_phi, cy, cx), dtype=np.complex128
        )
        iter_blocks[idx[1:]] = da.from_delayed(
            block_result.iter_count, shape=(cy, cx), dtype=np.int64
        )

    return RMClean3DResults(
        clean_fdf_cube=da.block(clean_blocks.tolist()),
        model_fdf_cube=da.block(model_blocks.tolist()),
        resid_fdf_cube=da.block(resid_blocks.tolist()),
        iter_count_map=da.block(iter_blocks.tolist()),
    )


def rmclean_3d_from_synth(
    rm_synth_3d_results: RMSynth3DResults,
    auto_mask: float = 7,
    auto_threshold: float = 1,
    max_iter: int = 1000,
    gain: float = 0.1,
    log_level: int = logging.ERROR,
) -> RMClean3DResults:
    """Run RM-CLEAN on the results of `rm_lite.tools_3d.rmsynth.rmsynth_3d`.

    Convenience wrapper that unpacks an `RMSynth3DResults` into `rmclean_3d`,
    mirroring `rm_lite.tools_1d.rmclean.run_rmclean_from_synth`. `mask`/
    `threshold` are scaled from `rm_synth_3d_results.theoretical_noise`, the
    same way the 1D version scales from its per-pixel theoretical noise --
    except 3D RM-synthesis only carries a per-channel (not per-pixel) noise
    estimate (see `rm_lite.utils.dask_io.estimate_channel_noise_mad`), so the
    resulting `mask`/`threshold` are uniform across the cube rather than
    per-pixel.

    Args:
        rm_synth_3d_results (RMSynth3DResults): Results from `rmsynth_3d`.
        auto_mask (float, optional): Masking threshold in SNR, scaled by the
            theoretical FDF noise. Defaults to 7.
        auto_threshold (float, optional): Cleaning threshold in SNR, scaled by
            the theoretical FDF noise. Defaults to 1.
        max_iter (int, optional): Maximum CLEAN iterations. Defaults to 1000.
        gain (float, optional): CLEAN loop gain. Defaults to 0.1.
        log_level (int, optional): See `rmclean_3d`. Defaults to `logging.ERROR`.

    Returns:
        RMClean3DResults: Lazy clean/model/residual FDF cubes and iteration-count map.
    """
    fdf_error_noise = rm_synth_3d_results.theoretical_noise.fdf_error_noise
    mask = auto_mask * fdf_error_noise
    threshold = auto_threshold * fdf_error_noise

    logger.info(
        f"Theoretical FDF noise: {fdf_error_noise:0.3g}. "
        f"Auto mask: {mask:0.3g}, auto threshold: {threshold:0.3g}."
    )

    return rmclean_3d(
        fdf_dirty_cube=rm_synth_3d_results.fdf_dirty_cube,
        rmsf_cube=rm_synth_3d_results.rmsf_cube,
        phi_arr_radm2=rm_synth_3d_results.phi_arr_radm2,
        phi_double_arr_radm2=rm_synth_3d_results.phi_double_arr_radm2,
        fwhm_rmsf_radm2=rm_synth_3d_results.fwhm_rmsf_radm2,
        mask=mask,
        threshold=threshold,
        max_iter=max_iter,
        gain=gain,
        log_level=log_level,
    )
