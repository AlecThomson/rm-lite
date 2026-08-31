"""RM-synthesis on chunked 3D Stokes Q/U cubes via dask."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal, NamedTuple, cast

import dask.array as da
import numpy as np
from dask.base import compute
from numpy.typing import NDArray

from rm_lite.utils.arrays import (
    divide_quiet,
    error_from_weight_cube,
    zero_nonfinite,
)
from rm_lite.utils.dask_io import (
    DEFAULT_TARGET_CHUNK_MB,
    complex_pol_dask,
    estimate_channel_noise_mad,
    estimate_single_stokes_channel_noise,
    freq_arr_hz_from_header,
    read_fits_cube_channel_chunks,
    read_fits_cube_dask,
)
from rm_lite.utils.fitting import (
    StokesIFitOptions,
    alpha_from_model_block,
    coefficient_names,
    fit_stokes_cube,
    ref_flux_from_block,
)
from rm_lite.utils.logging import logger, quiet_logs
from rm_lite.utils.synthesis import (
    FDFOptions,
    LamSq0Mode,
    RMSynthParams,
    TheoreticalNoise,
    WeightType,
    apply_weight_type,
    compute_rmsynth_params,
    compute_theoretical_noise,
    derotate_to,
    error_from_weight,
    get_fwhm_rmsf,
    get_rmsf_nufft,
    lam_sq_0_per_pixel,
    lambda2_to_freq,
    make_double_phi_arr,
    rmsynth_nufft,
)


class RMSynth3DResults(NamedTuple):
    """Results of chunked 3D RM-synthesis."""

    fdf_dirty_cube: da.Array
    """Dirty FDF cube, lazy dask array of shape (n_phi, ny, nx)."""
    rmsf_arr: NDArray[np.complex128]
    """The RMSF every pixel shares, shape (n_phi_double,), built from the
    per-channel weights. A pixel's RMSF depends only on which channels it has
    flagged, and flagging is per-channel rather than per-pixel, so one spectrum
    describes the whole cube. Per-pixel blanking that `weight_arr` does not carry
    is the exception: `per_pixel_rmsf=True` gets the exact cube for that."""
    phi_arr_radm2: NDArray[np.float64]
    """Faraday depth values in rad/m^2."""
    phi_double_arr_radm2: NDArray[np.float64]
    """Double-length Faraday depth values in rad/m^2, for the RMSF."""
    fwhm_rmsf_radm2: float
    """Analytic RMSF FWHM (per-pixel fitting is not performed in 3D)."""
    lam_sq_0_m2: float
    """Reference wavelength^2 in m^2 the FDF is derotated to. With
    `lam_sq_0_m2="per_pixel"` this is the cube-wide value, a common reference to
    move to via `lam_sq_0_map`; the per-pixel references actually used are in
    that map."""
    lam_sq_0_map: da.Array
    """Per-pixel reference wavelength^2, shape (ny, nx), lazy. Constant at
    `lam_sq_0_m2` unless `lam_sq_0_m2="per_pixel"`. Returned in every mode: it is
    what lets an FDF be moved between references afterwards
    (`rm_lite.utils.synthesis.derotate_to`, Brentjens & de Bruyn 2005 eq. 33)."""
    theoretical_noise: TheoreticalNoise
    """Theoretical FDF-domain noise from the per-channel weight array. This is a
    per-channel, not per-pixel, estimate, so it is uniform across the cube. When
    a Stokes I model is used the FDF is rescaled to flux per pixel; this noise
    stays in the Q/U-error domain it was computed in, which the rescaling keeps
    roughly consistent (exactly so for a flat Stokes I spectrum)."""
    stokes_i_model_cube: da.Array | None = None
    """Per-pixel Stokes I model cube, lazy, shape (n_freq, ny, nx). None unless a
    Stokes I cube or model was supplied to `rmsynth_3d`."""
    stokes_i_model_error_cube: da.Array | None = None
    """Per-pixel Stokes I model 1-sigma error cube, shape (n_freq, ny, nx). None
    unless `compute_model_error=True` (computed in the same per-pixel fit pass)."""
    stokes_i_ref_flux_map: da.Array | None = None
    """Stokes I model at the reference frequency (`lambda2_to_freq(lam_sq_0_m2)`),
    shape (ny, nx). This is the factor the fractional FDF was multiplied by to
    reach flux units. A 2D map, like the moment maps. None unless a Stokes I cube
    or model was supplied."""
    stokes_i_alpha_map: da.Array | None = None
    """Stokes I spectral index (d ln I / d ln nu) at the reference frequency,
    shape (ny, nx). A 2D map, like the moment maps. NaN where a pixel was not
    fitted (below the SNR cut). None unless a Stokes I cube or model was
    supplied."""
    stokes_i_alpha_error_map: da.Array | None = None
    """Per-pixel 1-sigma (16th/84th-percentile) uncertainty on
    `stokes_i_alpha_map`, shape (ny, nx), from the same Monte-Carlo over the fit
    covariance as `stokes_i_model_error_cube`. None unless `compute_model_error=True`
    and a Stokes I cube was fitted (a supplied model has no covariance, so no
    alpha error). NaN where a pixel was not fitted."""
    stokes_i_model_order_map: da.Array | None = None
    """Per-pixel fitted polynomial order of the Stokes I model (`len(popt) - 1`),
    shape (ny, nx). With a negative `fit_order` this is the AIC-chosen order per
    pixel; with a fixed order it is uniform on fitted pixels. NaN where a pixel
    was not fitted (below the SNR cut or flat fallback). None unless a Stokes I
    cube was fitted (a supplied model has no fitted order)."""
    stokes_i_coeff_cube: da.Array | None = None
    """Fitted Stokes I model terms, shape (n_coeff, ny, nx) with
    `n_coeff = abs(fit_order) + 1`, in `popt` order and named plane by plane in
    `stokes_i_coeff_names`. Together with `stokes_i_ref_freq_hz` these are the
    whole model, so it can be evaluated at any frequency without the model cube.
    A pixel the AIC gave fewer terms than `n_coeff` has zeros in the rest, since
    a dropped term contributes nothing (`stokes_i_model_order_map` says how many
    were fitted). NaN where a pixel was not fitted. None unless a Stokes I cube
    was fitted (a supplied model has no terms to report)."""
    stokes_i_coeff_error_cube: da.Array | None = None
    """1-sigma marginal error on each term, `sqrt(diag(pcov))`, shaped like
    `stokes_i_coeff_cube`. Marginal, so it ignores the correlations between
    terms, which are strong; use `compute_model_error` to propagate the model
    itself. None on the same terms as `stokes_i_coeff_cube`."""
    stokes_i_coeff_names: tuple[str, ...] | None = None
    """Name of each plane of `stokes_i_coeff_cube`: ("flux", "alpha", "beta", ...)
    for `fit_function="log"`, ("c0", "c1", ...) for "linear". None on the same
    terms as `stokes_i_coeff_cube`."""
    stokes_i_ref_freq_hz: float | da.Array | None = None
    """Frequency the Stokes I model terms are defined at, in Hz: the FDF's own
    reference frequency, `lambda2_to_freq` of the reference lambda^2, so the flux
    and phase references are the same by construction. A per-pixel map when
    `lam_sq_0_m2="per_pixel"`, and then the terms are each pixel's own -- compare
    alpha across the image by re-evaluating to a common frequency first. With `fit_function="log"`
    the model is `flux * 10**(alpha*log10(nu/nu_ref) + beta*log10(nu/nu_ref)**2
    + ...)`; with "linear" it is `sum(c_i * (nu/nu_ref)**i)`. None unless a Stokes
    I cube or model was supplied."""
    rmsf_cube: da.Array | None = None
    """Per-pixel RMSF cube, lazy, shape (n_phi_double, ny, nx). None unless
    `per_pixel_rmsf=True`, since it is `2 * n_phi_double / n_phi` times the FDF
    cube and holds `rmsf_arr` in every pixel whenever flagging is per-channel."""


def _compute_global_params(
    freq_arr_hz: NDArray[np.float64],
    weight_arr: NDArray[np.float64] | da.Array,
    weight_summary: WeightSummary,
    fdf_options: FDFOptions,
) -> tuple[RMSynthParams, TheoreticalNoise]:
    """Compute phi_arr/lam_sq_0_m2/weight_arr and theoretical FDF noise, once for the whole cube.

    `compute_rmsynth_params` is written for a single per-pixel spectrum, but
    its weight-array derivation round-trips exactly from a per-channel error
    spectrum (`weight = 1/error**2`), so a synthetic, fully-finite spectrum
    with `error = 1/sqrt(weight_arr)` reuses it unmodified for a per-channel
    (not per-pixel) weight array shared by every spatial chunk. The same
    reconstructed error feeds `compute_theoretical_noise` for a per-channel
    (not per-pixel) theoretical noise estimate.
    """
    # The globals take the summed channel profile, not the whole array: exact
    # for a weighted mean (the pixel sums factor out) and far cheaper. Per-pixel
    # weights are applied per chunk, in `_weight_arr_for_block`.
    complex_pol_error = error_from_weight(weight_summary.channel_profile)
    complex_pol_arr = np.ones_like(freq_arr_hz, dtype=np.complex128)

    rmsynth_params = compute_rmsynth_params(
        freq_arr_hz=freq_arr_hz,
        complex_pol_arr=complex_pol_arr,
        complex_pol_error=complex_pol_error,
        fdf_options=fdf_options,
    )
    # From the full array, so a per-pixel weight gives a per-pixel noise map,
    # left lazy until something asks for it.
    theoretical_noise = compute_theoretical_noise(
        complex_pol_error=error_from_weight(weight_arr),
        weight_arr=weight_arr,
    )
    return rmsynth_params, theoretical_noise


class WeightSummary(NamedTuple):
    """A weight array reduced to its channel weighting and whether pixels share it."""

    channel_profile: NDArray[np.float64]
    """The cube's aggregate per-channel weights, shape (n_freq,)."""
    pixels_proportional: bool
    """Whether every pixel weights the channels in the same proportions."""


def _summarise_weight(
    weight_arr: NDArray[Any] | da.Array, rtol: float = 1e-8
) -> WeightSummary:
    """The two things the globals need from a weight array, in one pass.

    The channel profile sets the Faraday depth grid and `lam_sq_0_m2`; summing
    over the image is exact for a weighted mean over the cube, since the pixel
    sums factor out.

    Proportionality decides whether one RMSF describes the cube: the RMSF is
    normalised by the weight sum, so pixels whose weights are scalar multiples
    of each other share one exactly. That covers noise varying spatially but not
    with frequency; a per-pixel noise *spectrum* fails it and needs its own RMSF.
    """
    if np.ndim(weight_arr) == 1:
        # Passed through in its own dtype: upcasting a float32 weight here would
        # quietly shift every downstream number for the per-channel path.
        return WeightSummary(np.asarray(weight_arr), True)

    # Blanks are no weight, not poison. A mosaic blanks the top of the band at
    # the field edge; summing that through would make every partially blanked
    # channel NaN, dropping it from `lam_sq_0_m2` altogether.
    weight_arr = zero_nonfinite(weight_arr)
    profile = np.sum(weight_arr, axis=(1, 2))
    pixel_totals = np.sum(weight_arr, axis=0)
    grand_total = np.sum(profile)
    with np.errstate(divide="ignore", invalid="ignore"):
        # What each pixel would hold if every pixel shared the cube's spectrum,
        # rescaled to that pixel's own total.
        expected = profile[:, np.newaxis, np.newaxis] * (pixel_totals / grand_total)
    channel_profile, deviation, scale = compute(
        profile, np.max(np.abs(weight_arr - expected)), np.max(np.abs(weight_arr))
    )
    proportional = bool(np.isfinite(deviation) and deviation <= rtol * scale)
    return WeightSummary(np.asarray(channel_profile), proportional)


def _shared_rmsf(
    rmsynth_params: RMSynthParams,
    nthreads: int,
    log_level: int,
) -> NDArray[np.complex128]:
    """The single RMSF the whole cube shares, from the per-channel weights.

    Every pixel whose flagged channels are the cube's flagged channels has this
    RMSF, and for the noise-based `weight_type`s a channel blank across the cube
    already carries a zero weight here. It is one spectrum, so it is computed up
    front rather than lazily per chunk.
    """
    with quiet_logs(log_level):
        rmsf_result = get_rmsf_nufft(
            lambda_sq_arr_m2=rmsynth_params.lambda_sq_arr_m2,
            phi_arr_radm2=rmsynth_params.phi_arr_radm2,
            weight_arr=rmsynth_params.weight_arr,
            lam_sq_0_m2=rmsynth_params.lam_sq_0_m2,
            do_fit_rmsf=False,
            nthreads=nthreads,
        )
    # RMSFResults.rmsf_cube is annotated NDArray[np.float64] but is complex128 at
    # runtime (built from a finufft complex output).
    return np.asarray(rmsf_result.rmsf_cube, dtype=np.complex128)


def _match_chunks_to_fdf(
    stokes_q: da.Array,
    stokes_u: da.Array,
    n_phi_double: int,
) -> tuple[da.Array, da.Array]:
    """Shrink spatial chunks so an FDF chunk costs what an input chunk costs.

    A chunk's output axis is `n_phi_double` long, not `n_freq`, and complex128
    rather than float32, so a chunk of RMSF is `(n_phi_double / n_freq) * 4` times
    its input chunk, often a factor of tens. That is the sizing case even without
    `per_pixel_rmsf`, since RM-CLEAN broadcasts the shared RMSF to the same shape
    per chunk. Peak memory follows the output, so the caller's input chunking is
    only a memory budget if the spatial chunk shrinks by that same factor here.

    Only ever shrinks: a caller who chunked coarsely on purpose keeps their
    chunks when the FDF is no larger than the input.

    One row is the floor, the same caveat `rm_lite.utils.dask_io.spatial_chunk_size`
    carries for one band, so a wide cube with a deep Faraday-depth axis can
    still overshoot the budget. That is logged when it happens.
    """
    n_freq = stokes_q.shape[0]
    cy = stokes_q.chunksize[1]
    cx = stokes_q.chunksize[2]
    budget_bytes = n_freq * cy * cx * stokes_q.dtype.itemsize
    out_bytes_per_pixel = n_phi_double * np.dtype(np.complex128).itemsize
    new_cy = max(1, int(budget_bytes // (out_bytes_per_pixel * cx)))
    row_bytes = out_bytes_per_pixel * cx
    if row_bytes > budget_bytes:
        logger.warning(
            f"One row of FDF output is {row_bytes / 1024**2:.3g} MiB, "
            f"{row_bytes / budget_bytes:.3g}x the {budget_bytes / 1024**2:.3g} MiB "
            "input chunk it was sized from. One row is the floor, so output "
            "chunks overshoot that budget; narrow the cube in x or coarsen "
            "d_phi_radm2 to bring it back under."
        )
    if new_cy >= cy:
        return stokes_q, stokes_u

    logger.info(
        f"Shrinking spatial chunks from {cy} to {new_cy} rows: {n_phi_double} "
        f"Faraday depths in complex128 against {n_freq} channels in "
        f"{stokes_q.dtype} would otherwise make each output chunk "
        f"{cy / new_cy:.3g}x the input chunk it was sized from."
    )
    return stokes_q.rechunk({1: new_cy}), stokes_u.rechunk({1: new_cy})


def _weight_arr_for_block(
    block: NDArray[np.complex128],
    weight_block: NDArray[np.float64] | None,
    rmsynth_params: RMSynthParams,
    fdf_options: FDFOptions,
) -> NDArray[np.float64]:
    """This chunk's weights, of the requested weight type.

    Applied per chunk rather than cube-wide: the grid weightings cost memory per
    pixel weighted at once, and each pixel's own flagging can shape its weights.
    A per-channel weight array was already weighted globally and is used as-is.
    """
    if weight_block is None:
        return rmsynth_params.weight_arr
    return apply_weight_type(
        lambda_sq_arr_m2=rmsynth_params.lambda_sq_arr_m2,
        real_qu_error=np.asarray(error_from_weight(weight_block).real),
        channel_mask=~np.isfinite(block),
        fdf_options=fdf_options,
        cell_m2=rmsynth_params.cell_m2,
    )


def _lam_sq_0_on_block(
    block: NDArray[np.complex128],
    weight_block: NDArray[np.float64] | None = None,
    *,
    rmsynth_params: RMSynthParams,
    fdf_options: FDFOptions,
) -> NDArray[np.float64]:
    """This chunk's per-pixel reference lambda^2, from its own weights."""
    weight_arr = _weight_arr_for_block(block, weight_block, rmsynth_params, fdf_options)
    if np.ndim(weight_arr) == 1:
        weight_arr = np.broadcast_to(weight_arr[:, np.newaxis, np.newaxis], block.shape)
    # A channel the pixel does not have cannot pull its reference; this matches
    # the weights `rmsynth_nufft` builds the pixel's own RMSF from.
    weight_arr = np.where(np.isfinite(block), weight_arr, 0.0)
    return lam_sq_0_per_pixel(weight_arr, rmsynth_params.lambda_sq_arr_m2)


def _lam_sq_0_map(
    weight_arr: NDArray[np.float64] | da.Array,
    pol_cube: da.Array,
    rmsynth_params: RMSynthParams,
    fdf_options: FDFOptions,
) -> da.Array:
    """Lazy (ny, nx) map of the reference lambda^2 each pixel is derotated to.

    Constant for a shared reference. Returned in every mode: it is what lets an
    FDF be moved between references later (`derotate_to`).
    """
    if fdf_options.lam_sq_0_m2 != "per_pixel":
        return da.full(
            pol_cube.shape[1:],
            rmsynth_params.lam_sq_0_m2,
            chunks=pol_cube.chunks[1:],
            dtype=np.float64,
        )
    return da.map_blocks(
        _lam_sq_0_on_block,
        pol_cube,
        *_weight_arr_map_blocks_args(weight_arr, pol_cube),
        drop_axis=0,
        dtype=np.float64,
        rmsynth_params=rmsynth_params,
        fdf_options=fdf_options,
    )


def _ref_freq_from_lam_sq_0(
    lam_sq_0_map: da.Array, lam_sq_0_m2: float, per_pixel: bool
) -> float | da.Array:
    """The Stokes I reference frequency implied by the reference lambda^2.

    Derived here and nowhere else, so the flux and phase references cannot drift.
    """
    if not per_pixel:
        return float(lambda2_to_freq(lam_sq_0_m2))
    return cast("da.Array", lambda2_to_freq(lam_sq_0_map))


def _rmsynth_on_block(
    block: NDArray[np.complex128],
    weight_block: NDArray[np.float64] | None = None,
    *,
    rmsynth_params: RMSynthParams,
    fdf_options: FDFOptions,
    n_phi: int,
    log_level: int,
    nufft_nthreads: int = 1,
) -> NDArray[np.complex128]:
    _, cy, cx = block.shape
    weight_arr = _weight_arr_for_block(block, weight_block, rmsynth_params, fdf_options)
    with quiet_logs(log_level):
        fdf = rmsynth_nufft(
            complex_pol_arr=block,
            lambda_sq_arr_m2=rmsynth_params.lambda_sq_arr_m2,
            phi_arr_radm2=rmsynth_params.phi_arr_radm2,
            weight_arr=weight_arr,
            lam_sq_0_m2=rmsynth_params.lam_sq_0_m2,
            nthreads=nufft_nthreads,
        )
    # rmsynth_nufft squeezes size-1 spatial axes; restore the block shape.
    return fdf.reshape(n_phi, cy, cx)


def _rmsf_on_block(
    block: NDArray[np.complex128],
    weight_block: NDArray[np.float64] | None = None,
    *,
    rmsynth_params: RMSynthParams,
    fdf_options: FDFOptions,
    n_phi_double: int,
    log_level: int,
    nufft_nthreads: int = 1,
) -> NDArray[np.complex128]:
    _, cy, cx = block.shape
    weight_arr = _weight_arr_for_block(block, weight_block, rmsynth_params, fdf_options)
    with quiet_logs(log_level):
        rmsf_result = get_rmsf_nufft(
            lambda_sq_arr_m2=rmsynth_params.lambda_sq_arr_m2,
            phi_arr_radm2=rmsynth_params.phi_arr_radm2,
            weight_arr=weight_arr,
            lam_sq_0_m2=rmsynth_params.lam_sq_0_m2,
            mask_arr=~np.isfinite(block),
            do_fit_rmsf=False,
            nthreads=nufft_nthreads,
        )
    return rmsf_result.rmsf_cube.reshape(n_phi_double, cy, cx)  # type: ignore[return-value]


def _weight_arr_map_blocks_args(
    weight_arr: NDArray[np.float64] | da.Array,
    target: da.Array,
) -> tuple[da.Array, ...]:
    """Positional map_blocks args for `weight_arr`, if it needs one.

    A 3D weight_arr varies per pixel, so it must be rechunked to match
    `target`'s spatial chunks and passed as a real map_blocks array argument
    -- otherwise every block's task would receive the whole, unsliced array
    instead of its own pixels' weights. This applies whether weight_arr is
    already a dask array or a plain numpy array (map_blocks does not
    auto-chunk a numpy positional argument to match other array arguments,
    so a numpy 3D weight_arr needs wrapping in `da.from_array` first).

    A 1D (per-channel, shared) weight_arr needs no slicing and can stay a
    closed-over kwarg on rmsynth_params, so this returns an empty tuple for
    that case.
    """
    if weight_arr.ndim != 3:
        return ()
    weight_da = (
        weight_arr if isinstance(weight_arr, da.Array) else da.from_array(weight_arr)
    )
    return (weight_da.rechunk({0: -1, 1: target.chunks[1], 2: target.chunks[2]}),)


def rmsynth_3d(
    stokes_q: da.Array,
    stokes_u: da.Array,
    freq_arr_hz: NDArray[np.float64],
    weight_arr: NDArray[np.float64] | da.Array | None = None,
    lam_sq_0_m2: float | LamSq0Mode = "auto",
    phi_max_radm2: float | None = None,
    d_phi_radm2: float | None = None,
    n_samples: float | None = 10.0,
    weight_type: WeightType = "variance",
    robust: float | None = None,
    stokes_i: da.Array | None = None,
    stokes_i_error: NDArray[np.float64] | da.Array | None = None,
    stokes_i_model: da.Array | None = None,
    estimate_stokes_i_noise: bool = False,
    fit_order: int = 2,
    fit_function: Literal["log", "linear"] = "log",
    stokes_i_snr_cut: float | None = 5.0,
    compute_model_error: bool = False,
    n_error_samples: int = 1000,
    per_pixel_rmsf: bool = False,
    nufft_nthreads: int = 1,
    log_level: int = logging.WARNING,
) -> RMSynth3DResults:
    """Run RM-synthesis on chunked Stokes Q/U cubes.

    Given a Stokes I cube or model, Q/U are divided by a per-pixel Stokes I model
    (fitted or supplied) and the FDF is rescaled to flux at the reference
    frequency; otherwise the FDF stays in Q/U flux.

    Args:
        stokes_q (da.Array): Stokes Q cube (n_freq, ny, nx), chunked spatially
            only. Its chunking is taken as the per-chunk memory budget, and
            spatial chunks are shrunk where the complex128 FDF output would
            otherwise outgrow it.
        stokes_u (da.Array): Stokes U cube, same shape/chunks as `stokes_q`.
        freq_arr_hz (NDArray[np.float64]): Frequency array in Hz.
        weight_arr (NDArray[np.float64] | None, optional): Weight array,
            per-channel (n_freq,) if shared by every pixel, or per-channel
            per-pixel (n_freq, ny, nx) if weights vary spatially. Defaults to
            uniform.
        phi_max_radm2 (float | None, optional): Maximum Faraday depth. Defaults to None.
        d_phi_radm2 (float | None, optional): Faraday depth resolution. Defaults to None.
        n_samples (float | None, optional): Samples across the RMSF. Defaults to 10.0.
        weight_type (WeightType, optional): 'variance'/'natural' (1/sigma^2), 'uniform' (equal per channel), 'uniform_lsq' (equal per lambda^2, narrows the RMSF), 'briggs' (robust). Defaults to "variance".
        robust (float | None, optional): Briggs robust parameter, required for weight_type='briggs'. Defaults to None.
        stokes_i (da.Array | None, optional): Stokes I cube to fit per pixel for
            the fractional correction. Ignored if `stokes_i_model` is given.
            Defaults to None (FDF stays in Q/U flux).
        stokes_i_error (NDArray[np.float64] | da.Array | None, optional): Stokes I
            error, per-channel (n_freq,) or per-pixel cube (n_freq, ny, nx), to
            weight the fit, and to measure SNR against for `stokes_i_snr_cut`.
            Defaults to None (unweighted, or estimated if
            `estimate_stokes_i_noise`).
        stokes_i_model (da.Array | None, optional): Pre-computed Stokes I model
            cube, used directly (no fitting). Takes precedence over `stokes_i`.
            Defaults to None.
        estimate_stokes_i_noise (bool, optional): Derive a per-channel error from
            `stokes_i` when no `stokes_i_error` is given. Defaults to False.
        fit_order (int, optional): Stokes I fit order; negative iterates orders and
            picks the best by AIC. Defaults to 2.
        fit_function ("log", "linear", optional): "log" = power law, "linear" =
            polynomial. Defaults to "log".
        stokes_i_snr_cut (float | None, optional): Below this frequency-averaged
            Stokes I SNR a pixel falls back to a flat model (no spectral
            correction, not blanked). None fits every pixel. Fit path only.
            Needs a Stokes I error to measure SNR against, so raises unless one
            of `stokes_i_error` / `estimate_stokes_i_noise` is given.
            Defaults to 5.0.
        compute_model_error (bool, optional): Also compute a per-pixel model error
            cube via Monte-Carlo over the fit covariance, in the same fit pass.
            Logs a warning about the compute coupling when enabled. Defaults to False.
        n_error_samples (int, optional): Monte-Carlo samples per pixel for
            `compute_model_error`. Defaults to 1000.
        per_pixel_rmsf (bool, optional): Also return the per-pixel RMSF cube
            (`rmsf_cube`) alongside the shared `rmsf_arr`. Only worth it when
            pixels within the cube have different channels flagged and the
            per-channel `weight_arr` does not already say so; otherwise every
            pixel of it holds `rmsf_arr` at `2 * n_phi_double / n_phi` times the
            cost of the FDF cube. Defaults to False.
        nufft_nthreads (int, optional): finufft OpenMP threads per chunk. Defaults
            to 1 so dask parallelises across chunks without oversubscribing finufft's
            own threads (the fast config on many chunks). Set to 0 (finufft default,
            all cores) only when computing with few chunks on the synchronous scheduler.
        log_level (int, optional): `rm_lite` logger level while chunks run;
            defaults to WARNING to silence per-chunk noise.

    Returns:
        RMSynth3DResults: Lazy FDF cube, the shared RMSF, and parameters. With a
            Stokes I model, also the model cube, the 2D reference-flux and
            spectral-index maps, and the fitted model terms.
    """
    if stokes_q.shape != stokes_u.shape:
        msg = f"Stokes Q and U must have the same shape. Got {stokes_q.shape} and {stokes_u.shape}."
        raise ValueError(msg)
    if stokes_q.chunks != stokes_u.chunks:
        msg = "Stokes Q and U must have identical chunking."
        raise ValueError(msg)

    n_freq = int(stokes_q.shape[0])
    if weight_arr is None:
        weight_arr = np.ones(n_freq, dtype=np.float64)

    fdf_options = FDFOptions(
        phi_max_radm2=phi_max_radm2,
        d_phi_radm2=d_phi_radm2,
        n_samples=n_samples,
        weight_type=weight_type,
        robust=robust,
        lam_sq_0_m2=lam_sq_0_m2,
    )
    fit_options = StokesIFitOptions(
        fit_order=fit_order,
        fit_function=fit_function,
        snr_cut=stokes_i_snr_cut,
        compute_model_error=compute_model_error,
        n_error_samples=n_error_samples,
    )

    if weight_arr is None:
        weight_arr = np.ones_like(freq_arr_hz)
    weight_summary = _summarise_weight(weight_arr)
    if fdf_options.lam_sq_0_m2 == "per_pixel" and not per_pixel_rmsf:
        logger.info(
            "lam_sq_0_m2='per_pixel' derotates each pixel to its own reference, "
            "so the RMSF differs pixel to pixel too; computing the per-pixel "
            "RMSF cube, which RM-CLEAN needs to match the FDF it is cleaning."
        )
        per_pixel_rmsf = True
    if not weight_summary.pixels_proportional and not per_pixel_rmsf:
        logger.info(
            "Pixels weight the channels differently, so they do not share one "
            "RMSF; computing the per-pixel RMSF cube. Pass per_pixel_rmsf=True "
            "to ask for it explicitly, or use a weight array whose spectrum is "
            "the same in every pixel to keep the shared one."
        )
        per_pixel_rmsf = True
    rmsynth_params, theoretical_noise = _compute_global_params(
        freq_arr_hz=freq_arr_hz,
        weight_arr=weight_arr,
        weight_summary=weight_summary,
        fdf_options=fdf_options,
    )
    n_phi = rmsynth_params.phi_arr_radm2.shape[0]
    phi_double_arr_radm2 = make_double_phi_arr(rmsynth_params.phi_arr_radm2)
    n_phi_double = phi_double_arr_radm2.shape[0]
    fwhm_rmsf_radm2 = get_fwhm_rmsf(rmsynth_params.lambda_sq_arr_m2).fwhm_rmsf_radm2
    rmsf_arr = _shared_rmsf(rmsynth_params, nufft_nthreads, log_level)

    stokes_q, stokes_u = _match_chunks_to_fdf(stokes_q, stokes_u, n_phi_double)

    pol_cube = complex_pol_dask(stokes_q, stokes_u)

    # Optional per-pixel Stokes I fractional-polarization correction. Either a
    # model cube is supplied directly, or one is fitted per pixel; Q/U are then
    # divided by it and the FDF is rescaled to absolute flux by the per-pixel
    # reference-frequency Stokes I flux (see module docstring).
    # One reference, resolved once; the Stokes I frequency is derived from it
    # below and nowhere else, so the phase and flux references cannot drift.
    lam_sq_0_map = _lam_sq_0_map(
        weight_arr=weight_arr,
        pol_cube=pol_cube,
        rmsynth_params=rmsynth_params,
        fdf_options=fdf_options,
    )
    per_pixel_ref = fdf_options.lam_sq_0_m2 == "per_pixel"
    ref_freq_hz = _ref_freq_from_lam_sq_0(
        lam_sq_0_map, rmsynth_params.lam_sq_0_m2, per_pixel_ref
    )
    stokes_i_model_cube: da.Array | None = None
    stokes_i_model_error_cube: da.Array | None = None
    ref_flux_map: da.Array | None = None
    alpha_map: da.Array | None = None
    alpha_error_map: da.Array | None = None
    order_map: da.Array | None = None
    coeff_cube: da.Array | None = None
    coeff_error_cube: da.Array | None = None
    coeff_names: tuple[str, ...] | None = None
    if stokes_i_model is not None:
        stokes_i_model_cube = stokes_i_model.rechunk(stokes_q.chunks)
        alpha_map = da.map_blocks(
            alpha_from_model_block,
            stokes_i_model_cube,
            drop_axis=0,
            dtype=np.float64,
            freq_arr_hz=freq_arr_hz,
            ref_freq_hz=ref_freq_hz,
        )
    elif stokes_i is not None:
        stokes_i = cast(da.Array, stokes_i.rechunk(stokes_q.chunks))
        if stokes_i_error is None and estimate_stokes_i_noise:
            stokes_i_error = estimate_single_stokes_channel_noise(stokes_i)
        if compute_model_error:
            logger.warning(
                "compute_model_error=True: the model-error cube shares one dask "
                "task with the model cube, so computing the FDF also runs the "
                "Monte-Carlo error sampling. Compute the error cube together with "
                "the model/FDF in one pass to avoid recomputing the fit."
            )
        fit_cubes = fit_stokes_cube(
            stokes_i=stokes_i,
            stokes_i_error=stokes_i_error,
            freq_arr_hz=freq_arr_hz,
            ref_freq_hz=ref_freq_hz,
            fit_options=fit_options,
            log_level=log_level,
        )
        stokes_i_model_cube = fit_cubes.model_cube
        alpha_map = fit_cubes.alpha_map
        order_map = fit_cubes.order_map
        coeff_cube = fit_cubes.coeff_cube
        coeff_error_cube = fit_cubes.coeff_error_cube
        coeff_names = coefficient_names(int(coeff_cube.shape[0]), fit_function)
        stokes_i_model_error_cube = fit_cubes.model_error_cube
        alpha_error_map = fit_cubes.alpha_error_map

    if stokes_i_model_cube is not None:
        pol_cube = divide_quiet(pol_cube, stokes_i_model_cube)
        ref_flux_map = da.map_blocks(
            ref_flux_from_block,
            stokes_i_model_cube,
            drop_axis=0,
            dtype=np.float64,
            freq_arr_hz=freq_arr_hz,
            ref_freq_hz=ref_freq_hz,
        )

    fdf_dirty_cube = da.map_blocks(
        _rmsynth_on_block,
        pol_cube,
        *_weight_arr_map_blocks_args(weight_arr, pol_cube),
        chunks=((n_phi,), pol_cube.chunks[1], pol_cube.chunks[2]),
        dtype=np.complex128,
        rmsynth_params=rmsynth_params,
        fdf_options=fdf_options,
        n_phi=n_phi,
        log_level=log_level,
        nufft_nthreads=nufft_nthreads,
    )

    if ref_flux_map is not None:
        # Rescale fractional FDF to absolute polarised flux per pixel.
        fdf_dirty_cube = fdf_dirty_cube * ref_flux_map[np.newaxis, :, :]

    if per_pixel_ref:
        # Synthesised at the cube's reference, then moved to each pixel's own.
        # Exact, and a phase ramp rather than a transform per pixel.
        fdf_dirty_cube = derotate_to(
            fdf_dirty_cube,
            rmsynth_params.phi_arr_radm2,
            rmsynth_params.lam_sq_0_m2,
            lam_sq_0_map,
        )

    rmsf_cube: da.Array | None = None
    if per_pixel_rmsf:
        rmsf_cube = da.map_blocks(
            _rmsf_on_block,
            pol_cube,
            *_weight_arr_map_blocks_args(weight_arr, pol_cube),
            chunks=((n_phi_double,), pol_cube.chunks[1], pol_cube.chunks[2]),
            dtype=np.complex128,
            rmsynth_params=rmsynth_params,
            fdf_options=fdf_options,
            n_phi_double=n_phi_double,
            log_level=log_level,
            nufft_nthreads=nufft_nthreads,
        )
        if per_pixel_ref:
            # Same reference as the FDF, or CLEAN subtracts a rotated response.
            rmsf_cube = derotate_to(
                rmsf_cube,
                phi_double_arr_radm2,
                rmsynth_params.lam_sq_0_m2,
                lam_sq_0_map,
            )

    return RMSynth3DResults(
        fdf_dirty_cube=fdf_dirty_cube,
        rmsf_arr=rmsf_arr,
        phi_arr_radm2=rmsynth_params.phi_arr_radm2,
        phi_double_arr_radm2=phi_double_arr_radm2,
        fwhm_rmsf_radm2=fwhm_rmsf_radm2,
        lam_sq_0_m2=rmsynth_params.lam_sq_0_m2,
        lam_sq_0_map=lam_sq_0_map,
        theoretical_noise=theoretical_noise,
        stokes_i_model_cube=stokes_i_model_cube,
        stokes_i_model_error_cube=stokes_i_model_error_cube,
        stokes_i_ref_flux_map=ref_flux_map,
        stokes_i_alpha_map=alpha_map,
        stokes_i_alpha_error_map=alpha_error_map,
        stokes_i_model_order_map=order_map,
        stokes_i_coeff_cube=coeff_cube,
        stokes_i_coeff_error_cube=coeff_error_cube,
        stokes_i_coeff_names=coeff_names,
        stokes_i_ref_freq_hz=(ref_freq_hz if stokes_i_model_cube is not None else None),
        rmsf_cube=rmsf_cube,
    )


def get_noise_from_error_fits(
    stokes_q_error_file: str | Path,
    stokes_u_error_file: str | Path,
    target_chunk_mb: float = DEFAULT_TARGET_CHUNK_MB,
) -> da.Array:
    """Lazy per-pixel noise cube, the mean of the Q and U error cubes."""
    stokes_q_error, _ = read_fits_cube_dask(
        stokes_q_error_file, target_chunk_mb=target_chunk_mb
    )
    stokes_u_error, _ = read_fits_cube_dask(
        stokes_u_error_file, target_chunk_mb=target_chunk_mb
    )

    return (stokes_q_error + stokes_u_error) / 2


def get_noise_from_fits(
    stokes_q_file: str | Path,
    stokes_u_file: str | Path,
    target_chunk_mb: float = DEFAULT_TARGET_CHUNK_MB,
) -> NDArray[np.float64]:
    """Per-channel noise estimated from the Q and U cubes themselves.

    Computed, not lazy, so it comes back as a plain (n_freq,) array.
    """
    # Per-channel noise needs whole image planes, so it gets its own
    # frequency-chunked read
    q_planes, _ = read_fits_cube_channel_chunks(
        stokes_q_file, target_chunk_mb=target_chunk_mb
    )
    u_planes, _ = read_fits_cube_channel_chunks(
        stokes_u_file, target_chunk_mb=target_chunk_mb
    )
    return estimate_channel_noise_mad(q_planes, u_planes)


def get_weight_arr_from_fits(
    stokes_q_file: str | Path | None = None,
    stokes_u_file: str | Path | None = None,
    stokes_q_error_file: str | Path | None = None,
    stokes_u_error_file: str | Path | None = None,
    target_chunk_mb: float = DEFAULT_TARGET_CHUNK_MB,
    noise_files_are_weight: bool = False,
) -> NDArray[np.float64] | da.Array:
    """The weight array for `rmsynth_3d`, from error cubes or from Q/U themselves.

    Error cubes take priority when given, and yield a lazy per-pixel
    (n_freq, ny, nx) dask weight array; falling back to Q/U gives a computed
    per-channel (n_freq,) one. `noise_files_are_weight` takes the error cubes
    as weights directly, skipping the 1/noise**2 inversion.
    """
    no_stokes_files = stokes_q_file is None and stokes_u_file is None
    no_error_files = stokes_q_error_file is None and stokes_u_error_file is None

    if no_stokes_files and no_error_files:
        msg = "Must provide at least Stokes QU files -OR- Stokes QU error files"
        raise ValueError(msg)

    if not no_error_files:
        # If user supplies error files - they take priority
        if stokes_q_error_file is None or stokes_u_error_file is None:
            msg = f"Must pass both Q and U error file! Got {stokes_q_error_file=} {stokes_u_error_file=}"
            raise ValueError(msg)
        noise_arr = get_noise_from_error_fits(
            stokes_q_error_file,
            stokes_u_error_file,
            target_chunk_mb,
        )
        if noise_files_are_weight:
            logger.warning(
                "Interpreting Q and U error files directly as weights! Will not invert and square"
            )
            return noise_arr
    else:
        # Fall back to noise estimate from QU cubes
        if stokes_q_file is None or stokes_u_file is None:
            msg = f"Must pass both Q and U file! Got {stokes_q_file=} {stokes_u_file=}"
            raise ValueError(msg)
        noise_arr = get_noise_from_fits(stokes_q_file, stokes_u_file, target_chunk_mb)

    return 1.0 / noise_arr**2


def rmsynth_3d_from_fits(
    stokes_q_file: str | Path,
    stokes_u_file: str | Path,
    stokes_q_error_file: str | Path | None = None,
    stokes_u_error_file: str | Path | None = None,
    noise_files_are_weight: bool = False,
    weight_arr: NDArray[np.float64] | da.Array | None = None,
    lam_sq_0_m2: float | LamSq0Mode = "auto",
    phi_max_radm2: float | None = None,
    d_phi_radm2: float | None = None,
    n_samples: float | None = 10.0,
    weight_type: WeightType = "variance",
    robust: float | None = None,
    stokes_i_file: str | Path | None = None,
    stokes_i_error_file: str | Path | None = None,
    stokes_i_model_file: str | Path | None = None,
    estimate_stokes_i_noise: bool = False,
    fit_order: int = 2,
    fit_function: Literal["log", "linear"] = "log",
    stokes_i_snr_cut: float | None = 5.0,
    compute_model_error: bool = False,
    n_error_samples: int = 1000,
    per_pixel_rmsf: bool = False,
    nufft_nthreads: int = 1,
    target_chunk_mb: float = DEFAULT_TARGET_CHUNK_MB,
    log_level: int = logging.WARNING,
) -> RMSynth3DResults:
    """Run RM-synthesis directly on Stokes Q/U FITS cubes on disk.

    Convenience wrapper around `rm_lite.utils.dask_io.read_fits_cube_dask` +
    `rmsynth_3d`, for the common case where Q/U are FITS files rather than
    already-loaded dask arrays. The frequency array is derived from the
    Stokes Q header's spectral WCS, and, if `weight_arr` is not given, so is
    the per-channel weight array (via `estimate_channel_noise_mad`).

    Args:
        stokes_q_file (str | Path): Path to the Stokes Q FITS cube.
        stokes_u_file (str | Path): Path to the Stokes U FITS cube.
        stokes_q_error_file (str | Path None, optional): Path to the Stokes Q error cube.
        stokes_u_error_file (str | Path None, optional): Path to the Stokes U error cube.
        noise_files_are_weight (bool): Interpret 'error' files directly as weights.
        weight_arr (NDArray[np.float64] | None, optional): Per-channel weight
            array. Defaults to an estimate from the cube noise (see
            `rm_lite.utils.dask_io.estimate_channel_noise_mad`).
        phi_max_radm2 (float | None, optional): Maximum Faraday depth. Defaults to None.
        d_phi_radm2 (float | None, optional): Faraday depth resolution. Defaults to None.
        n_samples (float | None, optional): Number of samples across the RMSF. Defaults to 10.0.
        weight_type (WeightType, optional): See `rmsynth_3d`. Defaults to "variance".
        robust (float | None, optional): Briggs robust parameter, required for weight_type='briggs'. Defaults to None.
        stokes_i_file (str | Path | None, optional): Path to a Stokes I FITS cube
            (measurements) to fit per pixel for fractional-polarization
            correction. See `rmsynth_3d`. Defaults to None.
        stokes_i_error_file (str | Path | None, optional): Path to a Stokes I
            error FITS cube used to weight the per-pixel fit. Defaults to None.
        stokes_i_model_file (str | Path | None, optional): Path to a pre-computed
            Stokes I model FITS cube, used directly (no fitting). Takes
            precedence over `stokes_i_file`. Defaults to None.
        estimate_stokes_i_noise (bool, optional): See `rmsynth_3d`. Defaults to False.
        fit_order (int, optional): See `rmsynth_3d`. Defaults to 2.
        fit_function ("log", "linear", optional): See `rmsynth_3d`. Defaults to "log".
        stokes_i_snr_cut (float | None, optional): See `rmsynth_3d`. Defaults to 5.0.
        compute_model_error (bool, optional): See `rmsynth_3d`. Defaults to False.
        n_error_samples (int, optional): See `rmsynth_3d`. Defaults to 1000.
        per_pixel_rmsf (bool, optional): See `rmsynth_3d`. Defaults to False.
        nufft_nthreads (int, optional): See `rmsynth_3d`. Defaults to 1.
        target_chunk_mb (float, optional): Target per-chunk memory footprint
            in MB, see `read_fits_cube_dask`. Defaults to 256.
        log_level (int, optional): See `rmsynth_3d`. Defaults to `logging.WARNING`.

    Returns:
        RMSynth3DResults: Lazy dirty FDF cube, the shared RMSF, and associated
            parameters.
    """
    stokes_q, header_q = read_fits_cube_dask(
        stokes_q_file, target_chunk_mb=target_chunk_mb
    )
    stokes_u, _header_u = read_fits_cube_dask(
        stokes_u_file, target_chunk_mb=target_chunk_mb
    )

    freq_arr_hz = freq_arr_hz_from_header(header_q, n_freq=int(stokes_q.shape[0]))

    # Noise-based types use 1/sigma^2 as their base (uniform_lsq/briggs then apply
    # the geometric lambda^2 factor); per-channel `uniform` deliberately ignores noise.
    if weight_arr is None and weight_type in (
        "variance",
        "natural",
        "uniform_lsq",
        "briggs",
    ):
        weight_arr = get_weight_arr_from_fits(
            stokes_q_file,
            stokes_u_file,
            stokes_q_error_file,
            stokes_u_error_file,
            target_chunk_mb=target_chunk_mb,
            noise_files_are_weight=noise_files_are_weight,
        )

    stokes_i = None
    stokes_i_model = None
    stokes_i_error: NDArray[np.float64] | da.Array | None = None
    if stokes_i_model_file is not None:
        stokes_i_model, _ = read_fits_cube_dask(
            stokes_i_model_file, target_chunk_mb=target_chunk_mb
        )
    elif stokes_i_file is not None:
        stokes_i, _ = read_fits_cube_dask(
            stokes_i_file, target_chunk_mb=target_chunk_mb
        )
        if stokes_i_error_file is not None:
            stokes_i_error, _ = read_fits_cube_dask(
                stokes_i_error_file, target_chunk_mb=target_chunk_mb
            )
            if noise_files_are_weight:
                logger.warning(
                    "Interpreting Stokes I error file as weight! Will sqrt & invert"
                )
                stokes_i_error = error_from_weight_cube(stokes_i_error)
        elif estimate_stokes_i_noise:
            # Same reason as the Q/U noise above: a frequency-chunked read.
            i_planes, _ = read_fits_cube_channel_chunks(
                stokes_i_file, target_chunk_mb=target_chunk_mb
            )
            stokes_i_error = estimate_single_stokes_channel_noise(i_planes)

    return rmsynth_3d(
        stokes_q=stokes_q,
        stokes_u=stokes_u,
        freq_arr_hz=freq_arr_hz,
        weight_arr=weight_arr,
        lam_sq_0_m2=lam_sq_0_m2,
        phi_max_radm2=phi_max_radm2,
        d_phi_radm2=d_phi_radm2,
        n_samples=n_samples,
        weight_type=weight_type,
        robust=robust,
        stokes_i=stokes_i,
        stokes_i_error=stokes_i_error,
        stokes_i_model=stokes_i_model,
        estimate_stokes_i_noise=estimate_stokes_i_noise,
        fit_order=fit_order,
        fit_function=fit_function,
        stokes_i_snr_cut=stokes_i_snr_cut,
        compute_model_error=compute_model_error,
        n_error_samples=n_error_samples,
        per_pixel_rmsf=per_pixel_rmsf,
        nufft_nthreads=nufft_nthreads,
        log_level=log_level,
    )
