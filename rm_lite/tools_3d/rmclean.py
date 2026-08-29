"""RM-CLEAN on chunked 3D FDF/RMSF cubes via dask."""

from __future__ import annotations

import logging
import operator
from dataclasses import replace
from typing import Any, Literal, NamedTuple

import dask.array as da
import numpy as np
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph
from numpy.typing import NDArray

from rm_lite.tools_3d.rmsynth import RMSynth3DResults
from rm_lite.utils.clean import (
    MultiscaleOptions,
    RMCleanOptions,
    RMSynthArrays,
    SelectionType,
    rmclean,
)
from rm_lite.utils.logging import logger, quiet_logs
from rm_lite.utils.synthesis import calc_faraday_moments


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
    mom0_map: da.Array
    """Zeroth Faraday moment (total polarised intensity) of the clean FDF,
    lazy dask array of shape (ny, nx). See `calc_faraday_moments`."""
    mom1_map: da.Array
    """First Faraday moment (mean Faraday depth, rad/m^2), shape (ny, nx)."""
    mom2_map: da.Array
    """Second Faraday moment (Faraday depth dispersion, rad/m^2), shape (ny, nx)."""


class _RMCleanBlockResult(NamedTuple):
    clean_fdf: NDArray[np.complex128]
    model_fdf: NDArray[np.complex128]
    resid_fdf: NDArray[np.complex128]
    iter_count: NDArray[np.int64]


_PER_PIXEL_CLEAN_FIELDS = ("mask", "threshold", "fdf_noise")
"""`RMCleanOptions` fields that may be a per-pixel map instead of a scalar."""


def _spatially_chunked_like(
    value: NDArray[np.float64] | da.Array, fdf_dirty_cube: da.Array, field: str
) -> da.Array:
    """A per-pixel CLEAN parameter chunked to match the FDF's spatial blocks.

    Block N of the map is then block N of the cube, so a CLEAN task can pull its
    own pixels' values by key. Kept lazy: a per-pixel noise map derived from a
    weight cube is itself lazy, and forcing it here would read the whole cube
    while the graph is still being built.
    """
    array = value if isinstance(value, da.Array) else da.from_array(value)
    if array.shape != fdf_dirty_cube.shape[1:]:
        msg = (
            f"A per-pixel CLEAN {field} must match the FDF's spatial shape "
            f"{fdf_dirty_cube.shape[1:]}, got {array.shape}."
        )
        raise ValueError(msg)
    return array.rechunk(fdf_dirty_cube.chunks[1:])


def _describe(value: float | NDArray[np.float64] | da.Array) -> str:
    """One log-friendly string for a scalar or for a per-pixel map."""
    if np.ndim(value) == 0:
        return f"{float(value):0.3g}"
    values = np.asarray(value)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "all non-finite"
    return f"{finite.min():0.3g} to {finite.max():0.3g} per pixel"


def _clean_block(
    dirty_fdf_block: NDArray[np.complex128],
    rmsf_block: NDArray[np.complex128],
    phi_arr_radm2: NDArray[np.float64],
    phi_double_arr_radm2: NDArray[np.float64],
    fwhm_rmsf_radm2: float,
    clean_options: RMCleanOptions,
    log_level: int,
    multiscale_options: MultiscaleOptions | None = None,
    *option_blocks: NDArray[np.float64] | None,
) -> _RMCleanBlockResult:
    """CLEAN one spatial chunk. `rmsf_block` is either the block's own RMSF cube
    or the 1D RMSF every pixel shares, which is broadcast to the block here since
    `rmclean` wants the RMSF and FDF on the same axes. `option_blocks` carries
    this block's slice of any per-pixel mask/threshold/noise, in
    `_PER_PIXEL_CLEAN_FIELDS` order, each None where that one is a scalar."""
    per_pixel: dict[str, Any] = {
        field: block
        for field, block in zip(_PER_PIXEL_CLEAN_FIELDS, option_blocks, strict=False)
        if block is not None
    }
    if per_pixel:
        clean_options = replace(clean_options, **per_pixel)
    if rmsf_block.ndim == 1:
        rmsf_block = np.broadcast_to(
            rmsf_block[:, np.newaxis, np.newaxis],
            (rmsf_block.shape[0], *dirty_fdf_block.shape[1:]),
        )
    with quiet_logs(log_level):
        result = rmclean(
            RMSynthArrays(
                dirty_fdf_arr=dirty_fdf_block,
                phi_arr_radm2=phi_arr_radm2,
                rmsf_arr=rmsf_block,
                phi_double_arr_radm2=phi_double_arr_radm2,
                fwhm_rmsf_arr=np.array(fwhm_rmsf_radm2),
            ),
            clean_options,
            multiscale_options=multiscale_options,
        )
    return _RMCleanBlockResult(
        clean_fdf=result.clean_fdf_arr,
        model_fdf=result.model_fdf_arr,
        resid_fdf=result.resid_fdf_arr,
        iter_count=result.clean_iter_arr,
    )


def _build_clean_output_arrays(
    fdf_dirty_cube: da.Array,
    rmsf: NDArray[np.complex128] | da.Array,
    rmsf_cube: da.Array | None,
    phi_arr_radm2: NDArray[np.float64],
    phi_double_arr_radm2: NDArray[np.float64],
    fwhm_rmsf_radm2: float,
    clean_options: RMCleanOptions,
    multiscale_options: MultiscaleOptions | None,
    log_level: int,
) -> tuple[da.Array, da.Array, da.Array, da.Array]:
    """The four `_clean_block` outputs, from one pass over one graph.

    A `dask.array.from_delayed` per output per block would walk (and re-wrap)
    every upstream layer 4 * n_chunks times, which is quadratic in the chunk
    count once the upstream layer count grows with it too. Here the upstream
    graph is traversed once, by the single `HighLevelGraph.from_collections`
    call that hangs the CLEAN layer off it.

    Keys are referenced by name rather than going through
    `dask.array.Array.to_delayed`, which keeps the input graph's own keys: an
    upstream per-block task (the Stokes I fit, the NUFFT) must not be fused
    into the block task RM-CLEAN consumes, or anything else built on the same
    `RMSynth3DResults` stops sharing that work and recomputes it in the same
    `dask.compute`.
    """
    numblocks = fdf_dirty_cube.numblocks
    fdf_chunks = fdf_dirty_cube.chunks
    spatial_chunks = fdf_chunks[1:]

    token = tokenize(
        fdf_dirty_cube.name,
        rmsf_cube.name if rmsf_cube is not None else rmsf,
        phi_arr_radm2,
        phi_double_arr_radm2,
        fwhm_rmsf_radm2,
        clean_options,
        multiscale_options,
        log_level,
    )
    block_name = f"rmclean-block-{token}"

    layer: dict[Any, Any] = {}
    # A shared RMSF becomes one graph key that every block points at, rather
    # than the same spectrum re-embedded per block or a cube holding ny*nx
    # copies.
    shared_rmsf_key = f"rmclean-rmsf-{token}"
    if rmsf_cube is None:
        layer[shared_rmsf_key] = rmsf

    # A per-pixel mask/threshold/noise is a map over the whole image, so each
    # block reads its own slice of it, by key like the FDF and RMSF do. Scalars
    # stay on `clean_options` and are shared by every block.
    option_arrays = {
        field: _spatially_chunked_like(value, fdf_dirty_cube, field)
        for field in _PER_PIXEL_CLEAN_FIELDS
        if (value := getattr(clean_options, field)) is not None and np.ndim(value) != 0
    }

    for idx in np.ndindex(numblocks):
        layer[(block_name, *idx)] = (
            _clean_block,
            (fdf_dirty_cube.name, *idx),
            (rmsf_cube.name, *idx) if rmsf_cube is not None else shared_rmsf_key,
            phi_arr_radm2,
            phi_double_arr_radm2,
            fwhm_rmsf_radm2,
            clean_options,
            log_level,
            multiscale_options,
            *(
                None
                if (array := option_arrays.get(field)) is None
                else (array.name, *idx[1:])
                for field in _PER_PIXEL_CLEAN_FIELDS
            ),
        )

    dependencies = [fdf_dirty_cube]
    if rmsf_cube is not None:
        dependencies.append(rmsf_cube)
    dependencies.extend(option_arrays.values())
    graph = HighLevelGraph.from_collections(
        block_name, layer, dependencies=dependencies
    )

    layers: dict[str, Any] = dict(graph.layers)
    layer_deps: dict[str, set[str]] = dict(graph.dependencies)
    arrays: list[tuple[str, type, tuple[tuple[int, ...], ...]]] = []
    for field, dtype, chunks in (
        ("clean_fdf", np.complex128, fdf_chunks),
        ("model_fdf", np.complex128, fdf_chunks),
        ("resid_fdf", np.complex128, fdf_chunks),
        ("iter_count", np.int64, spatial_chunks),
    ):
        name = f"rmclean-{field.replace('_', '-')}-{token}"
        field_index = _RMCleanBlockResult._fields.index(field)
        # The 2D iteration-count map drops the leading (single-block) axis.
        layers[name] = {
            (name, *idx[-len(chunks) :]): (
                operator.getitem,
                (block_name, *idx),
                field_index,
            )
            for idx in np.ndindex(numblocks)
        }
        layer_deps[name] = {block_name}
        arrays.append((name, dtype, chunks))

    # One graph shared by all four arrays, so a `dask.compute` over any subset
    # of them runs `_clean_block` once per chunk. Culling at compute time drops
    # the layers an individual array doesn't reach.
    shared_graph = HighLevelGraph(layers, layer_deps)
    clean, model, resid, iter_count = (
        da.Array(shared_graph, name, chunks, dtype=dtype)
        for name, dtype, chunks in arrays
    )
    return clean, model, resid, iter_count


def run_rmclean(
    fdf_dirty_cube: da.Array,
    rmsf: NDArray[np.complex128] | da.Array,
    phi_arr_radm2: NDArray[np.float64],
    phi_double_arr_radm2: NDArray[np.float64],
    fwhm_rmsf_radm2: float,
    mask: float | NDArray[np.float64] | da.Array,
    threshold: float | NDArray[np.float64] | da.Array,
    max_iter: int = 100_000,
    gain: float = 0.1,
    moment_threshold: float | NDArray[np.float64] | da.Array | None = None,
    fdf_noise: float | NDArray[np.float64] | da.Array | None = None,
    log_level: int = logging.ERROR,
    multiscale: bool = False,
    multiscale_scales: NDArray[np.float64] | None = None,
    multiscale_n_scales: int | None = None,
    multiscale_kernel: Literal["tapered_quad", "gaussian"] = "tapered_quad",
    multiscale_max_iter_sub_minor: int = 10_000,
    multiscale_sub_minor_fraction: float = 0.5,
    multiscale_selection: SelectionType = "hybrid",
    multiscale_selection_margin: float = 0.08,
) -> RMClean3DResults:
    """Run RM-CLEAN on chunked dirty FDF and RMSF cubes.

    Args:
        fdf_dirty_cube (da.Array): Dirty FDF cube, shape (n_phi, ny, nx),
            chunked spatially only (as produced by `rm_lite.tools_3d.rmsynth.rmsynth_3d`).
        rmsf (NDArray[np.complex128] | da.Array): Either the RMSF every pixel
            shares, shape (n_phi_double,) (`RMSynth3DResults.rmsf_arr`), or a
            per-pixel RMSF cube, shape (n_phi_double, ny, nx) with the same
            spatial chunking as `fdf_dirty_cube` (`rmsf_cube`, only produced with
            `per_pixel_rmsf=True`).
        phi_arr_radm2 (NDArray[np.float64]): Faraday depth values in rad/m^2.
        phi_double_arr_radm2 (NDArray[np.float64]): Double-length Faraday depth
            values in rad/m^2, for the RMSF.
        fwhm_rmsf_radm2 (float): RMSF FWHM, shared by every pixel (3D RM-CLEAN
            here does not support a per-pixel FWHM map).
        mask (float): Masking threshold. Pixels below this value are not cleaned.
        threshold (float): Cleaning threshold. Stop when all pixels are below this value.
        max_iter (int, optional): Maximum CLEAN iterations. Defaults to 1000.
        gain (float, optional): CLEAN loop gain. Defaults to 0.1.
        moment_threshold (float | None, optional): Amplitude cut (in FDF
            amplitude units) applied to the clean FDF before computing the
            Faraday moment maps, passed to `calc_faraday_moments`. None includes
            all amplitudes (noise-biased). Defaults to None.
        fdf_noise (float | None, optional): Theoretical FDF noise; enables the
            adaptive off-source auto-mask (mask contracts off the RMSF sidelobes of
            bright sources, then relaxes as they subtract). None keeps the
            fixed-mask behaviour. Defaults to None.
        log_level (int, optional): Log level applied to `rm_lite`'s logger while
            each chunk runs. `rmclean`'s Hogbom loop logs at INFO and WARNING
            per pixel (e.g. "Starting minor loop...", "All channels masked...
            performed N iterations"). These are routine per-pixel loop
            termination conditions, not anomalies, and at cube scale they're
            just noise, so this defaults to ERROR (silencing both). Pass
            `logging.WARNING` or `logging.INFO` to restore progressively more
            per-pixel verbosity, e.g. while debugging a specific chunk.
            Defaults to `logging.ERROR`.
        multiscale (bool, optional): Use multiscale RM-CLEAN (recovers
            Faraday-thick structure). Defaults to False.
        multiscale_scales (NDArray[np.float64] | None, optional): Explicit scales
            (RMSF FWHM units); None auto-selects.
        multiscale_n_scales (int | None, optional): Cap on the auto scale count.
        multiscale_kernel ("tapered_quad" | "gaussian", optional): Scale kernel. Defaults to "tapered_quad".
        multiscale_max_iter_sub_minor (int, optional): Max sub-minor iterations. Defaults to 10_000.
        multiscale_sub_minor_fraction (float, optional): Sub-minor re-selection fraction. Defaults to 0.5.
        multiscale_selection ("snr" | "hybrid", optional): Scale-selection strategy. Defaults to "hybrid".
        multiscale_selection_margin (float, optional): Hybrid scale-selection parsimony margin in [0, 1). Among scales within this fraction of the best matched-filter score the smallest is chosen, keeping points on the delta scale. Defaults to 0.08.

    Returns:
        RMClean3DResults: Lazy clean/model/residual FDF cubes and iteration-count map.
    """
    if fdf_dirty_cube.numblocks[0] != 1:
        msg = (
            "fdf_dirty_cube must be chunked spatially only, but its Faraday "
            f"depth axis is split into {fdf_dirty_cube.numblocks[0]} chunks."
        )
        raise ValueError(msg)

    rmsf_cube: da.Array | None = None
    if rmsf.ndim == 3:
        if not isinstance(rmsf, da.Array):
            msg = "A per-pixel rmsf must be a dask array, chunked like fdf_dirty_cube."
            raise TypeError(msg)
        if rmsf.numblocks[0] != 1:
            msg = (
                "A per-pixel rmsf must be chunked spatially only, but its "
                f"Faraday depth axis is split into {rmsf.numblocks[0]} chunks."
            )
            raise ValueError(msg)
        if fdf_dirty_cube.chunks[1:] != rmsf.chunks[1:]:
            msg = (
                "fdf_dirty_cube and a per-pixel rmsf must have identical "
                "spatial chunking."
            )
            raise ValueError(msg)
        rmsf_cube = rmsf
    elif rmsf.ndim != 1:
        msg = f"rmsf must be 1D (shared) or 3D (per-pixel), got {rmsf.ndim}D."
        raise ValueError(msg)

    clean_options = RMCleanOptions(
        mask=mask,
        threshold=threshold,
        max_iter=max_iter,
        gain=gain,
        fdf_noise=fdf_noise,
    )
    multiscale_options = (
        MultiscaleOptions(
            scales=multiscale_scales,
            n_scales=multiscale_n_scales,
            kernel=multiscale_kernel,
            max_iter_sub_minor=multiscale_max_iter_sub_minor,
            sub_minor_fraction=multiscale_sub_minor_fraction,
            selection=multiscale_selection,
            selection_margin=multiscale_selection_margin,
        )
        if multiscale
        else None
    )

    clean, model, resid, iter_count = _build_clean_output_arrays(
        fdf_dirty_cube=fdf_dirty_cube,
        rmsf=rmsf,
        rmsf_cube=rmsf_cube,
        phi_arr_radm2=phi_arr_radm2,
        phi_double_arr_radm2=phi_double_arr_radm2,
        fwhm_rmsf_radm2=fwhm_rmsf_radm2,
        clean_options=clean_options,
        multiscale_options=multiscale_options,
        log_level=log_level,
    )

    moments = calc_faraday_moments(
        clean,
        phi_arr_radm2=phi_arr_radm2,
        fwhm_rmsf_radm2=fwhm_rmsf_radm2,
        threshold=moment_threshold,
    )

    return RMClean3DResults(
        clean_fdf_cube=clean,
        model_fdf_cube=model,
        resid_fdf_cube=resid,
        iter_count_map=iter_count,
        mom0_map=moments.mom0,
        mom1_map=moments.mom1,
        mom2_map=moments.mom2,
    )


def run_rmclean_from_synth(
    rm_synth_3d_results: RMSynth3DResults,
    auto_mask: float = 7,
    auto_threshold: float = 1,
    max_iter: int = 100_000,
    gain: float = 0.1,
    moment_threshold_snr: float = 5.0,
    log_level: int = logging.ERROR,
    multiscale: bool = False,
    multiscale_scales: NDArray[np.float64] | None = None,
    multiscale_n_scales: int | None = None,
    multiscale_kernel: Literal["tapered_quad", "gaussian"] = "tapered_quad",
    multiscale_max_iter_sub_minor: int = 10_000,
    multiscale_sub_minor_fraction: float = 0.5,
    multiscale_selection: SelectionType = "hybrid",
    multiscale_selection_margin: float = 0.08,
) -> RMClean3DResults:
    """Run RM-CLEAN on the results of `rm_lite.tools_3d.rmsynth.rmsynth_3d`.

    Convenience wrapper that unpacks an `RMSynth3DResults` into `run_rmclean`,
    mirroring `rm_lite.tools_1d.rmclean.run_rmclean_from_synth`. `mask` and
    `threshold` are scaled from `rm_synth_3d_results.theoretical_noise`, the
    same way the 1D version scales from its per-pixel theoretical noise. 3D
    RM-synthesis only carries a per-channel (not per-pixel) noise estimate (see
    `rm_lite.utils.dask_io.estimate_channel_noise_mad`), so the resulting `mask`
    and `threshold` are uniform across the cube rather than per-pixel.

    Args:
        rm_synth_3d_results (RMSynth3DResults): Results from `rmsynth_3d`.
        auto_mask (float, optional): Masking threshold in SNR, scaled by the
            theoretical FDF noise. Defaults to 7.
        auto_threshold (float, optional): Cleaning threshold in SNR, scaled by
            the theoretical FDF noise. Defaults to 1.
        max_iter (int, optional): Maximum CLEAN iterations. Defaults to 1000.
        gain (float, optional): CLEAN loop gain. Defaults to 0.1.
        moment_threshold_snr (float, optional): SNR cut (times the theoretical
            FDF noise) applied to the clean FDF before computing the Faraday
            moment maps. Defaults to 5.0.
        log_level (int, optional): See `run_rmclean`. Defaults to `logging.ERROR`.
        multiscale (bool, optional): Use multiscale RM-CLEAN (recovers
            Faraday-thick structure). Defaults to False.
        scales, n_scales, kernel, max_iter_sub_minor, sub_minor_fraction,
            selection, selection_margin: Multiscale options, see `run_rmclean`.

    Returns:
        RMClean3DResults: Lazy clean/model/residual FDF cubes and iteration-count map.
    """
    fdf_error_noise = rm_synth_3d_results.theoretical_noise.fdf_error_noise
    mask = auto_mask * fdf_error_noise
    threshold = auto_threshold * fdf_error_noise
    moment_threshold = moment_threshold_snr * fdf_error_noise

    logger.info(
        f"Theoretical FDF noise: {_describe(fdf_error_noise)}. "
        f"Auto mask: {_describe(mask)}, auto threshold: {_describe(threshold)}."
    )

    return run_rmclean(
        fdf_dirty_cube=rm_synth_3d_results.fdf_dirty_cube,
        rmsf=(
            rm_synth_3d_results.rmsf_arr
            if rm_synth_3d_results.rmsf_cube is None
            else rm_synth_3d_results.rmsf_cube
        ),
        phi_arr_radm2=rm_synth_3d_results.phi_arr_radm2,
        phi_double_arr_radm2=rm_synth_3d_results.phi_double_arr_radm2,
        fwhm_rmsf_radm2=rm_synth_3d_results.fwhm_rmsf_radm2,
        mask=mask,
        threshold=threshold,
        max_iter=max_iter,
        gain=gain,
        moment_threshold=moment_threshold,
        fdf_noise=fdf_error_noise,
        log_level=log_level,
        multiscale=multiscale,
        multiscale_scales=multiscale_scales,
        multiscale_n_scales=multiscale_n_scales,
        multiscale_kernel=multiscale_kernel,
        multiscale_max_iter_sub_minor=multiscale_max_iter_sub_minor,
        multiscale_sub_minor_fraction=multiscale_sub_minor_fraction,
        multiscale_selection=multiscale_selection,
        multiscale_selection_margin=multiscale_selection_margin,
    )
