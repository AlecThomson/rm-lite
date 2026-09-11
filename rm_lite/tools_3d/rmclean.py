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

from rm_lite.tools_3d.rmsynth import (
    PerPixelRMSF,
    RMSynth3DResults,
    rmsf_block_for_clean,
)
from rm_lite.utils.arrays import format_scalar_or_map
from rm_lite.utils.clean import (
    PER_PIXEL_CLEAN_FIELDS,
    MultiscaleOptions,
    RMCleanOptions,
    RMSynthArrays,
    SelectionType,
    rmclean,
)
from rm_lite.utils.logging import logger, quiet_logs
from rm_lite.utils.synthesis import (
    FaradayMoments,
    FaradayPeaks,
    calc_faraday_moments,
    calc_faraday_peaks,
)


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
    mom0_debias_map: da.Array
    """`mom0_map` less the noise's own contribution, shape (ny, nx). NaN
    without `fdf_noise`."""
    mom0_error_map: da.Array
    """1-sigma error on `mom0_map`, shape (ny, nx). NaN without `fdf_noise`."""
    mom1_map: da.Array
    """First Faraday moment (mean Faraday depth, rad/m^2), shape (ny, nx)."""
    mom1_error_map: da.Array
    """1-sigma error on `mom1_map` in rad/m^2, shape (ny, nx). NaN without `fdf_noise`."""
    mom2_map: da.Array
    """Second Faraday moment (Faraday depth dispersion, rad/m^2), shape (ny, nx)."""
    mom2_error_map: da.Array
    """1-sigma error on `mom2_map` in rad/m^2, shape (ny, nx). NaN without `fdf_noise`."""
    pi_lam_sq_0_map: da.Array
    """Polarised intensity at the reference lambda^2, on the `mom0_map` scale,
    shape (ny, nx). `mom0_map` drops the phase, this keeps it, so the ratio is
    the depolarisation there. Pin one `lam_sq_0_m2` for a map: per-pixel puts
    every pixel at a different wavelength."""
    pi_lam_sq_0_debias_map: da.Array
    """`pi_lam_sq_0_map` corrected for polarisation bias, shape (ny, nx). NaN
    without `fdf_noise`."""
    pi_lam_sq_0_error_map: da.Array
    """1-sigma error on `pi_lam_sq_0_map`, shape (ny, nx). NaN without `fdf_noise`."""
    pa_lam_sq_0_map: da.Array
    """Polarisation angle of `pi_lam_sq_0_map` in degrees, shape (ny, nx)."""
    pa_lam_sq_0_error_map: da.Array
    """1-sigma error on `pa_lam_sq_0_map` in degrees, shape (ny, nx). NaN
    without `fdf_noise`."""
    peak_pi_map: da.Array
    """Peak polarised intensity of the clean FDF, shape (ny, nx). See
    `calc_faraday_peaks`. NaN where no peak was found. No detection cut is
    applied, so select on `peak_pi_map / peak_pi_error_map`."""
    peak_pi_debias_map: da.Array
    """Peak polarised intensity corrected for polarisation bias, shape (ny, nx).
    NaN without `fdf_noise`."""
    peak_pi_error_map: da.Array
    """1-sigma error on the peak, i.e. `fdf_noise` as a map, shape (ny, nx).
    Reported even where no peak was found. NaN without `fdf_noise`."""
    peak_rm_map: da.Array
    """Faraday depth of the peak in rad/m^2, shape (ny, nx)."""
    peak_rm_error_map: da.Array
    """1-sigma error on the peak Faraday depth, shape (ny, nx). NaN without `fdf_noise`."""
    peak_pa_map: da.Array
    """Polarisation angle at the peak in degrees, at `lam_sq_0_m2`, shape (ny, nx)."""
    peak_pa_error_map: da.Array
    """1-sigma error on the polarisation angle, shape (ny, nx). NaN without `fdf_noise`."""
    peak_pa0_map: da.Array
    """Derotated (intrinsic) polarisation angle in degrees, shape (ny, nx). NaN
    without `lam_sq_0_m2`."""
    peak_pa0_error_map: da.Array
    """1-sigma error on the intrinsic angle, shape (ny, nx). NaN without
    `lambda_sq_arr_m2`."""


class _RMCleanBlockResult(NamedTuple):
    clean_fdf: NDArray[np.complexfloating]
    model_fdf: NDArray[np.complexfloating]
    resid_fdf: NDArray[np.complexfloating]
    iter_count: NDArray[np.int64]


def _align_option_map_to_fdf(
    value: NDArray[np.float64] | da.Array, fdf_dirty_cube: da.Array, field: str
) -> da.Array:
    """A per-pixel CLEAN parameter rechunked to the FDF's spatial blocks.

    Block N of the map is then block N of the cube, so a task can pull its own
    pixels by key. Kept lazy: forcing it here would read the whole weight cube
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


def faraday_maps_on_block(
    clean_block: NDArray[np.complexfloating],
    *per_pixel_blocks: NDArray[np.float64],
    per_pixel_fields: tuple[str, ...],
    phi_arr_radm2: NDArray[np.float64],
    fwhm_rmsf_radm2: float,
    lambda_sq_arr_m2: NDArray[np.float64] | None,
    fdf_noise: float | None,
    moment_threshold: float | None,
    lam_sq_0_m2: float,
) -> NDArray[np.float64]:
    """Every Faraday moment and peak map for one block, stacked on a leading axis.

    Each map is a reduction along the Faraday axis, which is never chunked, so a
    block holds everything its pixels need. Computing them together keeps the
    graph to one task a block rather than a chain per map, and lets the block be
    dropped once they are all done.

    `per_pixel_fields` names which of the trailing block arguments is which, so
    any of the three can be a map or a single value.
    """
    values: dict[str, Any] = {
        "fdf_noise": fdf_noise,
        "moment_threshold": moment_threshold,
        "lam_sq_0_m2": lam_sq_0_m2,
    }
    values.update(zip(per_pixel_fields, per_pixel_blocks, strict=False))

    moments = calc_faraday_moments(
        clean_block,
        phi_arr_radm2=phi_arr_radm2,
        fwhm_rmsf_radm2=fwhm_rmsf_radm2,
        fdf_error=values["fdf_noise"],
        threshold=values["moment_threshold"],
    )
    peaks = calc_faraday_peaks(
        clean_block,
        phi_arr_radm2=phi_arr_radm2,
        fwhm_rmsf_radm2=fwhm_rmsf_radm2,
        fdf_error=values["fdf_noise"],
        lam_sq_0_m2=values["lam_sq_0_m2"],
        lambda_sq_arr_m2=lambda_sq_arr_m2,
    )
    spatial_shape = clean_block.shape[1:]
    return np.stack(
        [
            np.broadcast_to(np.asarray(value, dtype=np.float64), spatial_shape)
            for value in (*moments, *peaks)
        ]
    )


def faraday_maps(
    clean: da.Array,
    phi_arr_radm2: NDArray[np.float64],
    fwhm_rmsf_radm2: float,
    lam_sq_0_m2: float | NDArray[np.float64] | da.Array,
    lambda_sq_arr_m2: NDArray[np.float64] | None,
    fdf_noise: float | NDArray[np.float64] | da.Array | None,
    moment_threshold: float | NDArray[np.float64] | da.Array | None,
) -> dict[str, da.Array]:
    """The moment and peak maps, from one `map_blocks` over an FDF cube.

    One task a block for all of them, rather than a chain per map. Debiased
    maps are not included: `debias_fdf` needs neighbouring pixels, so it cannot
    run inside a per-block task.
    """
    names = FaradayMoments._fields + FaradayPeaks._fields
    scalars = {
        "fdf_noise": fdf_noise,
        "moment_threshold": moment_threshold,
        "lam_sq_0_m2": lam_sq_0_m2,
    }
    # Anything given per pixel rides along as a block argument, so a task pulls
    # its own pixels rather than the whole map.
    per_pixel = {
        field: _align_option_map_to_fdf(value, clean, field)
        for field, value in scalars.items()
        if value is not None and np.ndim(value) != 0
    }
    for field in per_pixel:
        scalars[field] = None

    stacked = da.map_blocks(
        faraday_maps_on_block,
        clean,
        *per_pixel.values(),
        chunks=((len(names),), *clean.chunks[1:]),
        dtype=np.float64,
        per_pixel_fields=tuple(per_pixel),
        phi_arr_radm2=phi_arr_radm2,
        fwhm_rmsf_radm2=fwhm_rmsf_radm2,
        lambda_sq_arr_m2=lambda_sq_arr_m2,
        **scalars,
    )
    return {name: stacked[index] for index, name in enumerate(names)}


def _rmclean_on_block(
    dirty_fdf_block: NDArray[np.complexfloating],
    rmsf_block: NDArray[np.complexfloating],
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
    `PER_PIXEL_CLEAN_FIELDS` order, each None where that one is a scalar."""
    per_pixel: dict[str, Any] = {
        field: block
        for field, block in zip(PER_PIXEL_CLEAN_FIELDS, option_blocks, strict=False)
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


def _rmsf_then_clean_on_block(
    dirty_fdf_block: NDArray[np.complexfloating],
    pol_block: NDArray[np.complexfloating],
    weight_block: NDArray[np.float64] | None,
    lam_sq_0_block: NDArray[np.float64] | None,
    recipe: PerPixelRMSF,
    *clean_args: Any,
) -> _RMCleanBlockResult:
    """Build this block's RMSF and clean with it, in one task.

    The RMSF never becomes a graph key, so the scheduler cannot build a queue of
    them ahead of the CLEAN tasks that consume them.
    """
    rmsf_block = rmsf_block_for_clean(pol_block, weight_block, lam_sq_0_block, recipe)
    return _rmclean_on_block(dirty_fdf_block, rmsf_block, *clean_args)


def _build_clean_output_arrays(
    fdf_dirty_cube: da.Array,
    rmsf: NDArray[np.complexfloating] | da.Array,
    rmsf_cube: da.Array | None,
    rmsf_recipe: PerPixelRMSF | None,
    phi_arr_radm2: NDArray[np.float64],
    phi_double_arr_radm2: NDArray[np.float64],
    fwhm_rmsf_radm2: float,
    clean_options: RMCleanOptions,
    multiscale_options: MultiscaleOptions | None,
    log_level: int,
) -> tuple[da.Array, da.Array, da.Array, da.Array]:
    """The four `_rmclean_on_block` outputs, from one pass over one graph.

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
        rmsf_recipe.pol_cube.name
        if rmsf_recipe is not None
        else (rmsf_cube.name if rmsf_cube is not None else rmsf),
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
    if rmsf_recipe is None and rmsf_cube is None:
        layer[shared_rmsf_key] = rmsf

    # A per-pixel mask/threshold/noise is a map over the whole image, so each
    # block reads its own slice of it, by key like the FDF and RMSF do. Scalars
    # stay on `clean_options` and are shared by every block.
    option_arrays = {
        field: _align_option_map_to_fdf(value, fdf_dirty_cube, field)
        for field in PER_PIXEL_CLEAN_FIELDS
        if (value := getattr(clean_options, field)) is not None and np.ndim(value) != 0
    }

    for idx in np.ndindex(numblocks):
        clean_args = (
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
                for field in PER_PIXEL_CLEAN_FIELDS
            ),
        )
        if rmsf_recipe is not None:
            layer[(block_name, *idx)] = (
                _rmsf_then_clean_on_block,
                (fdf_dirty_cube.name, *idx),
                (rmsf_recipe.pol_cube.name, *idx),
                None
                if rmsf_recipe.weight_cube is None
                else (rmsf_recipe.weight_cube.name, *idx),
                None
                if rmsf_recipe.lam_sq_0_map is None
                else (rmsf_recipe.lam_sq_0_map.name, *idx[1:]),
                rmsf_recipe,
                *clean_args,
            )
        else:
            layer[(block_name, *idx)] = (
                _rmclean_on_block,
                (fdf_dirty_cube.name, *idx),
                (rmsf_cube.name, *idx) if rmsf_cube is not None else shared_rmsf_key,
                *clean_args,
            )

    dependencies = [fdf_dirty_cube]
    if rmsf_recipe is not None:
        dependencies.append(rmsf_recipe.pol_cube)
        dependencies.extend(
            array
            for array in (rmsf_recipe.weight_cube, rmsf_recipe.lam_sq_0_map)
            if array is not None
        )
    elif rmsf_cube is not None:
        dependencies.append(rmsf_cube)
    dependencies.extend(option_arrays.values())
    graph = HighLevelGraph.from_collections(
        block_name, layer, dependencies=dependencies
    )

    layers: dict[str, Any] = dict(graph.layers)
    layer_deps: dict[str, set[str]] = dict(graph.dependencies)
    arrays: list[tuple[str, Any, tuple[tuple[int, ...], ...]]] = []
    for field, dtype, chunks in (
        ("clean_fdf", fdf_dirty_cube.dtype, fdf_chunks),
        ("model_fdf", fdf_dirty_cube.dtype, fdf_chunks),
        ("resid_fdf", fdf_dirty_cube.dtype, fdf_chunks),
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
    # of them runs `_rmclean_on_block` once per chunk. Culling at compute time drops
    # the layers an individual array doesn't reach.
    shared_graph = HighLevelGraph(layers, layer_deps)
    clean, model, resid, iter_count = (
        da.Array(shared_graph, name, chunks, dtype=dtype)
        for name, dtype, chunks in arrays
    )
    return clean, model, resid, iter_count


def run_rmclean(
    fdf_dirty_cube: da.Array,
    rmsf: NDArray[np.complexfloating] | da.Array,
    phi_arr_radm2: NDArray[np.float64],
    phi_double_arr_radm2: NDArray[np.float64],
    fwhm_rmsf_radm2: float,
    mask: float | NDArray[np.float64] | da.Array,
    threshold: float | NDArray[np.float64] | da.Array,
    max_iter: int = 100_000,
    gain: float = 0.1,
    moment_threshold: float | NDArray[np.float64] | da.Array | None = None,
    fdf_noise: float | NDArray[np.float64] | da.Array | None = None,
    lam_sq_0_m2: float | NDArray[np.float64] | da.Array | None = None,
    lambda_sq_arr_m2: NDArray[np.float64] | None = None,
    per_pixel_rmsf: PerPixelRMSF | None = None,
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
        rmsf (NDArray[np.complexfloating] | da.Array): Either the RMSF every pixel
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
            amplitude units) applied before the Faraday moment maps, passed to
            `calc_faraday_moments`. mom1 and mom2 need it: uncut, mom2 reports
            the grid's width rather than the source's. `mom0_debias_map` does
            not. Peak maps are never cut. Defaults to None.
        fdf_noise (float | None, optional): Theoretical FDF noise; enables the
            adaptive off-source auto-mask (mask contracts off the RMSF sidelobes of
            bright sources, then relaxes as they subtract) and the peak errors
            and debiased peak. None keeps the fixed-mask behaviour. Defaults to None.
        lam_sq_0_m2 (float | NDArray[np.float64] | da.Array | None, optional):
            Reference wavelength^2 the FDF is derotated to, scalar or a per-pixel
            map (`RMSynth3DResults.lam_sq_0_map`). Enables the intrinsic
            polarisation angle map. Defaults to None.
        lambda_sq_arr_m2 (NDArray[np.float64] | None, optional): Channel
            lambda^2 in m^2 (`RMSynth3DResults.lambda_sq_arr_m2`), for the
            intrinsic-angle error map. Defaults to None.
        per_pixel_rmsf (PerPixelRMSF | None, optional): Build each block's RMSF
            in the CLEAN task rather than reading it from a cube
            (`RMSynth3DResults.per_pixel_rmsf`). Takes precedence over a 3D
            `rmsf`. Defaults to None.
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
        RMClean3DResults: Lazy clean/model/residual FDF cubes, iteration-count
            map, and the Faraday moment and peak maps.
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
        rmsf_recipe=per_pixel_rmsf,
        phi_arr_radm2=phi_arr_radm2,
        phi_double_arr_radm2=phi_double_arr_radm2,
        fwhm_rmsf_radm2=fwhm_rmsf_radm2,
        clean_options=clean_options,
        multiscale_options=multiscale_options,
        log_level=log_level,
    )

    maps = faraday_maps(
        clean,
        phi_arr_radm2=phi_arr_radm2,
        fwhm_rmsf_radm2=fwhm_rmsf_radm2,
        lam_sq_0_m2=lam_sq_0_m2,
        lambda_sq_arr_m2=lambda_sq_arr_m2,
        fdf_noise=fdf_noise,
        moment_threshold=moment_threshold,
    )

    return RMClean3DResults(
        clean_fdf_cube=clean,
        model_fdf_cube=model,
        resid_fdf_cube=resid,
        iter_count_map=iter_count,
        mom0_map=maps["mom0"],
        mom0_debias_map=maps["mom0_debias"],
        mom0_error_map=maps["mom0_error"],
        mom1_map=maps["mom1"],
        mom1_error_map=maps["mom1_error"],
        mom2_map=maps["mom2"],
        mom2_error_map=maps["mom2_error"],
        pi_lam_sq_0_map=maps["pi_lam_sq_0"],
        pi_lam_sq_0_debias_map=maps["pi_lam_sq_0_debias"],
        pi_lam_sq_0_error_map=maps["pi_lam_sq_0_error"],
        pa_lam_sq_0_map=maps["pa_lam_sq_0"],
        pa_lam_sq_0_error_map=maps["pa_lam_sq_0_error"],
        peak_pi_map=maps["peak_pi"],
        peak_pi_debias_map=maps["peak_pi_debias"],
        peak_pi_error_map=maps["peak_pi_error"],
        peak_rm_map=maps["peak_rm_radm2"],
        peak_rm_error_map=maps["peak_rm_error_radm2"],
        peak_pa_map=maps["peak_pa_deg"],
        peak_pa_error_map=maps["peak_pa_error_deg"],
        peak_pa0_map=maps["peak_pa0_deg"],
        peak_pa0_error_map=maps["peak_pa0_error_deg"],
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
            FDF noise) applied before the Faraday moment maps, see
            `run_rmclean`. Peak maps are never cut; select on
            `peak_pi_map / peak_pi_error_map`. Defaults to 5.0.
        log_level (int, optional): See `run_rmclean`. Defaults to `logging.ERROR`.
        multiscale (bool, optional): Use multiscale RM-CLEAN (recovers
            Faraday-thick structure). Defaults to False.
        scales, n_scales, kernel, max_iter_sub_minor, sub_minor_fraction,
            selection, selection_margin: Multiscale options, see `run_rmclean`.

    Returns:
        RMClean3DResults: Lazy clean/model/residual FDF cubes, iteration-count
            map, and the Faraday moment and peak maps, each with errors. The peak maps carry the
            errors and intrinsic angle, since the reference wavelength^2 and
            theoretical noise come with the synthesis results.
    """
    fdf_error_noise = rm_synth_3d_results.theoretical_noise.fdf_error_noise
    mask = auto_mask * fdf_error_noise
    threshold = auto_threshold * fdf_error_noise
    moment_threshold = moment_threshold_snr * fdf_error_noise

    logger.info(
        f"Theoretical FDF noise: {format_scalar_or_map(fdf_error_noise)}. "
        f"Auto mask: {format_scalar_or_map(mask)}, auto threshold: {format_scalar_or_map(threshold)}."
    )

    return run_rmclean(
        fdf_dirty_cube=rm_synth_3d_results.fdf_dirty_cube,
        rmsf=rm_synth_3d_results.rmsf_arr,
        per_pixel_rmsf=rm_synth_3d_results.per_pixel_rmsf,
        phi_arr_radm2=rm_synth_3d_results.phi_arr_radm2,
        phi_double_arr_radm2=rm_synth_3d_results.phi_double_arr_radm2,
        fwhm_rmsf_radm2=rm_synth_3d_results.fwhm_rmsf_radm2,
        mask=mask,
        threshold=threshold,
        max_iter=max_iter,
        gain=gain,
        moment_threshold=moment_threshold,
        fdf_noise=fdf_error_noise,
        lam_sq_0_m2=rm_synth_3d_results.lam_sq_0_map,
        lambda_sq_arr_m2=rm_synth_3d_results.lambda_sq_arr_m2,
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
