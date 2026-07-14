"""RM-synthesis on chunked 3D Stokes Q/U cubes via dask."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Literal, NamedTuple

import dask.array as da
import numpy as np
from numpy.typing import NDArray

from rm_lite.utils.dask_io import (
    DEFAULT_TARGET_CHUNK_MB,
    complex_pol_dask,
    estimate_channel_noise_mad,
    estimate_stokes_i_channel_noise,
    freq_arr_hz_from_header,
    read_fits_cube_dask,
)
from rm_lite.utils.fitting import (
    FitResult,
    StokesIFitOptions,
    draw_model_samples,
    fit_stokes_i_model,
)
from rm_lite.utils.logging import logger, quiet_logs
from rm_lite.utils.synthesis import (
    FDFOptions,
    RMSynthParams,
    TheoreticalNoise,
    WeightType,
    compute_rmsynth_params,
    compute_theoretical_noise,
    get_fwhm_rmsf,
    get_rmsf_nufft,
    lambda2_to_freq,
    make_double_phi_arr,
    rmsynth_nufft,
)


class RMSynth3DResults(NamedTuple):
    """Results of chunked 3D RM-synthesis."""

    fdf_dirty_cube: da.Array
    """Dirty FDF cube, lazy dask array of shape (n_phi, ny, nx)."""
    rmsf_cube: da.Array
    """RMSF cube, lazy dask array of shape (n_phi_double, ny, nx)."""
    phi_arr_radm2: NDArray[np.float64]
    """Faraday depth values in rad/m^2."""
    phi_double_arr_radm2: NDArray[np.float64]
    """Double-length Faraday depth values in rad/m^2, for the RMSF."""
    fwhm_rmsf_radm2: float
    """Analytic RMSF FWHM (per-pixel fitting is not performed in 3D)."""
    lam_sq_0_m2: float
    """Reference wavelength^2 value in m^2."""
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


def _compute_global_params(
    freq_arr_hz: NDArray[np.float64],
    weight_arr: NDArray[np.float64],
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
    with np.errstate(divide="ignore"):
        real_error = np.where(weight_arr > 0, 1.0 / np.sqrt(weight_arr), np.inf)
    complex_pol_error = (real_error + 1j * real_error).astype(np.complex128)
    complex_pol_arr = np.ones_like(freq_arr_hz, dtype=np.complex128)

    rmsynth_params = compute_rmsynth_params(
        freq_arr_hz=freq_arr_hz,
        complex_pol_arr=complex_pol_arr,
        complex_pol_error=complex_pol_error,
        fdf_options=fdf_options,
    )
    theoretical_noise = compute_theoretical_noise(
        complex_pol_error=complex_pol_error,
        weight_arr=weight_arr,
    )
    return rmsynth_params, theoretical_noise


def _pixel_stokes_i_error(
    err_block: NDArray[np.float64] | None,
    err_1d: NDArray[np.float64] | None,
    n_freq: int,
    y: int,
    x: int,
) -> NDArray[np.float64]:
    """Per-pixel Stokes I error spectrum from a 3D error cube, 1D per-channel
    array, or neither (zeros -> `fit_stokes_i_model` fits unweighted)."""
    if err_block is not None:
        return err_block[:, y, x]
    if err_1d is not None:
        return err_1d
    return np.zeros(n_freq, dtype=np.float64)


class PixelFit(NamedTuple):
    """One pixel's Stokes I fit within a chunk (see `_iter_pixel_fits`)."""

    y: int
    x: int
    i_spec: NDArray[np.float64]
    """The pixel's Stokes I spectrum (unmasked), for the flat-model fallback."""
    good: NDArray[np.bool_]
    """Finite-channel mask, for the flat-model fallback."""
    fit: FitResult | None
    """The fit, or None if the pixel was skipped (too few channels / low SNR)."""


def _alpha_at_ref(
    spec: NDArray[np.float64],
    freq_arr_hz: NDArray[np.float64],
    ref_freq_hz: float,
) -> float:
    """Spectral index (d ln I / d ln nu) of `spec` at the reference frequency.

    Sorts by frequency first: `np.interp`/`np.gradient` need an ascending axis,
    and a descending-frequency cube (negative CDELT3) otherwise gives garbage.
    """
    order = np.argsort(freq_arr_hz)
    ln_freq = np.log(freq_arr_hz[order])
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.gradient(np.log(np.abs(spec[order])), ln_freq)
    return float(np.interp(np.log(ref_freq_hz), ln_freq, slope))


def _iter_pixel_fits(
    i_block: NDArray[np.float64],
    err_block: NDArray[np.float64] | None,
    err_1d: NDArray[np.float64] | None,
    freq_arr_hz: NDArray[np.float64],
    ref_freq_hz: float,
    fit_options: StokesIFitOptions,
) -> Iterator[PixelFit]:
    """Yield a `PixelFit` for every pixel in a chunk.

    `fit` is None for a skipped pixel (`fit_stokes_i_model` returns None when
    there are too few finite channels or the SNR is below `fit_options.snr_cut`).
    Shared by the model/alpha pass and the opt-in error pass so the setup lives
    once.
    """
    n_freq, cy, cx = i_block.shape
    for y in range(cy):
        for x in range(cx):
            i_spec = i_block[:, y, x]
            e_spec = _pixel_stokes_i_error(err_block, err_1d, n_freq, y, x)
            good = np.isfinite(i_spec) & np.isfinite(e_spec)
            fit = fit_stokes_i_model(
                freq_arr_hz=freq_arr_hz,
                ref_freq_hz=ref_freq_hz,
                stokes_i_arr=i_spec,
                stokes_i_error_arr=e_spec,
                options=fit_options,
            )
            yield PixelFit(y, x, i_spec, good, fit)


def _fit_stokes_i_block(
    *arrays: NDArray[np.float64],
    freq_arr_hz: NDArray[np.float64],
    ref_freq_hz: float,
    err_1d: NDArray[np.float64] | None,
    fit_options: StokesIFitOptions,
    log_level: int,
) -> NDArray[np.float64]:
    """Fit a Stokes I model per pixel over one spatial chunk, in a single pass.

    Returns a stacked block of shape (n_out, cy, cx). Without error: planes
    0..n_freq-1 are the model cube, plane n_freq is the fitted spectral index
    alpha at the reference frequency, and plane n_freq+1 is the fitted polynomial
    order (`len(popt) - 1`; `n_out = n_freq + 2`). With
    `fit_options.compute_model_error`: planes n_freq+2..2*n_freq+1 are the 1-sigma
    model error and plane 2*n_freq+2 is the 1-sigma alpha error
    (`n_out = 2*n_freq + 3`). Both errors come from one Monte-Carlo over the same
    per-pixel fit covariance, so they cost no extra fit; when `compute_model_error`
    is False no error work is done at all.

    `arrays` is `(i_block,)` or `(i_block, err_block)`; the error cube is
    optional (see `_pixel_stokes_i_error`). A skipped pixel (too few finite
    channels or SNR below `fit_options.snr_cut`; see `fit_stokes_i_model`) falls
    back to a flat model at its mean Stokes I, so it gets no spectral correction
    (its FDF is the plain Q/U FDF), and its alpha, order and errors stay NaN. A
    pixel with no finite channels stays NaN throughout.
    """
    compute_error = fit_options.compute_model_error
    i_block = arrays[0]
    err_block = arrays[1] if len(arrays) > 1 else None
    n_freq, cy, cx = i_block.shape
    x_arr = freq_arr_hz / ref_freq_hz
    n_out = n_freq + 2 + (n_freq + 1 if compute_error else 0)
    out = np.full((n_out, cy, cx), np.nan, dtype=np.float64)
    # The 1D fitter logs per fit and per failure. At cube scale that floods, so
    # quiet it to at least ERROR whatever the caller's log_level.
    with quiet_logs(max(log_level, logging.ERROR)):
        for y, x, i_spec, good, fit in _iter_pixel_fits(
            i_block,
            err_block,
            err_1d,
            freq_arr_hz,
            ref_freq_hz,
            fit_options,
        ):
            if fit is not None:
                model = fit.stokes_i_model_func(x_arr, *np.asarray(fit.popt))
                out[:n_freq, y, x] = model
                out[n_freq, y, x] = _alpha_at_ref(model, freq_arr_hz, ref_freq_hz)
                out[n_freq + 1, y, x] = np.asarray(fit.popt).size - 1
                if compute_error:
                    # One MC draw feeds both the per-channel model error and the
                    # alpha error (alpha of each realisation at the ref freq).
                    samples = draw_model_samples(
                        fit, x_arr, fit_options.n_error_samples
                    )
                    low, high = np.nanpercentile(samples, [16, 84], axis=0)
                    model_error = np.abs(high - low)
                    model_error[model_error > 1e99] = np.nan
                    out[n_freq + 2 : 2 * n_freq + 2, y, x] = model_error
                    alpha_samples = np.array(
                        [_alpha_at_ref(m, freq_arr_hz, ref_freq_hz) for m in samples]
                    )
                    a_low, a_high = np.nanpercentile(alpha_samples, [16, 84])
                    out[2 * n_freq + 2, y, x] = abs(a_high - a_low)
            elif good.any():
                # Flat fallback: no correction, and alpha/order stay NaN (masked).
                out[:n_freq, y, x] = float(np.mean(i_spec[good]))
    return out


def _ref_flux_block(
    model_block: NDArray[np.float64],
    freq_arr_hz: NDArray[np.float64],
    ref_freq_hz: float,
) -> NDArray[np.float64]:
    """Interpolate a Stokes I model cube block at the reference frequency ->
    (cy, cx) reference-flux map block."""
    _, cy, cx = model_block.shape
    # np.interp needs an ascending axis; sort so a descending-frequency cube
    # (negative CDELT3) doesn't silently interpolate to garbage.
    order = np.argsort(freq_arr_hz)
    freq_sorted = freq_arr_hz[order]
    ref_flux = np.full((cy, cx), np.nan, dtype=np.float64)
    for y in range(cy):
        for x in range(cx):
            spec = model_block[:, y, x]
            if np.isfinite(spec).all():
                ref_flux[y, x] = np.interp(ref_freq_hz, freq_sorted, spec[order])
    return ref_flux


def _alpha_from_model_block(
    model_block: NDArray[np.float64],
    freq_arr_hz: NDArray[np.float64],
    ref_freq_hz: float,
) -> NDArray[np.float64]:
    """Spectral index alpha at the reference frequency from a supplied model cube.

    For a model given directly (not fitted), every pixel is modelled, so alpha is
    finite wherever the model is finite (0 for a flat model). NaN only where the
    model is not finite.
    """
    _, cy, cx = model_block.shape
    alpha = np.full((cy, cx), np.nan, dtype=np.float64)
    for y in range(cy):
        for x in range(cx):
            spec = model_block[:, y, x]
            if not np.isfinite(spec).all():
                continue
            value = _alpha_at_ref(spec, freq_arr_hz, ref_freq_hz)
            alpha[y, x] = value if np.isfinite(value) else 0.0
    return alpha


def _split_stokes_i_error(
    stokes_i_error: NDArray[np.float64] | da.Array | None,
    chunks: tuple[tuple[int, ...], ...],
) -> tuple[NDArray[np.float64] | None, da.Array | None]:
    """Split a Stokes I error into (per-channel 1D array, per-pixel dask cube).

    A 3D error (numpy or dask) becomes a dask cube rechunked to `chunks`; a 1D
    per-channel error becomes a numpy array; None stays None. `da.asarray` wraps
    a numpy cube so `.rechunk` works either way.
    """
    if stokes_i_error is None:
        return None, None
    if getattr(stokes_i_error, "ndim", 1) == 3:
        return None, da.asarray(stokes_i_error).rechunk(chunks)
    return np.asarray(stokes_i_error, dtype=np.float64), None


def _stokes_i_model_cube(
    stokes_i: da.Array,
    stokes_i_error: NDArray[np.float64] | da.Array | None,
    freq_arr_hz: NDArray[np.float64],
    ref_freq_hz: float,
    fit_options: StokesIFitOptions,
    log_level: int,
) -> tuple[da.Array, da.Array, da.Array, da.Array | None, da.Array | None]:
    """Lazy per-pixel Stokes I model cube, spectral-index map, fitted-order map,
    and (optional) model-error cube and alpha-error map.

    All come from one per-pixel fit pass (`_fit_stokes_i_block`): the model cube
    is chunked like `stokes_i`; the alpha and order maps are (ny, nx); and the
    error cube (n_freq, ny, nx) plus the alpha-error map (ny, nx) -- returned only
    when `fit_options.compute_model_error`, else None -- reuse the same fit, so
    they cost no extra fit. Alpha and order are NaN for pixels that were not
    fitted. `stokes_i_error` is a per-channel 1D array (n_freq,), a per-pixel
    error cube (n_freq, ny, nx), or None.
    """
    err_1d, err_cube = _split_stokes_i_error(stokes_i_error, stokes_i.chunks)

    compute_error = fit_options.compute_model_error
    n_freq = int(stokes_i.shape[0])
    n_out = n_freq + 2 + (n_freq + 1 if compute_error else 0)
    stacked = da.map_blocks(
        _fit_stokes_i_block,
        *((stokes_i,) if err_cube is None else (stokes_i, err_cube)),
        dtype=np.float64,
        chunks=((n_out,), stokes_i.chunks[1], stokes_i.chunks[2]),
        freq_arr_hz=freq_arr_hz,
        ref_freq_hz=ref_freq_hz,
        err_1d=err_1d,
        fit_options=fit_options,
        log_level=log_level,
    )
    model = stacked[:n_freq]
    alpha = stacked[n_freq]
    order = stacked[n_freq + 1]
    if not compute_error:
        return model, alpha, order, None, None
    error = stacked[n_freq + 2 : 2 * n_freq + 2]
    alpha_error = stacked[2 * n_freq + 2]
    return model, alpha, order, error, alpha_error


def rmsynth_3d(
    stokes_q: da.Array,
    stokes_u: da.Array,
    freq_arr_hz: NDArray[np.float64],
    weight_arr: NDArray[np.float64] | None = None,
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
    nufft_nthreads: int = 1,
    log_level: int = logging.WARNING,
) -> RMSynth3DResults:
    """Run RM-synthesis on chunked Stokes Q/U cubes.

    Given a Stokes I cube or model, Q/U are divided by a per-pixel Stokes I model
    (fitted or supplied) and the FDF is rescaled to flux at the reference
    frequency; otherwise the FDF stays in Q/U flux.

    Args:
        stokes_q (da.Array): Stokes Q cube (n_freq, ny, nx), chunked spatially only.
        stokes_u (da.Array): Stokes U cube, same shape/chunks as `stokes_q`.
        freq_arr_hz (NDArray[np.float64]): Frequency array in Hz.
        weight_arr (NDArray[np.float64] | None, optional): Per-channel (not
            per-pixel) weight array. Defaults to uniform.
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
            weight the fit. Defaults to None (unweighted, or estimated if
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
            Defaults to 5.0.
        compute_model_error (bool, optional): Also compute a per-pixel model error
            cube via Monte-Carlo over the fit covariance, in the same fit pass.
            Logs a warning about the compute coupling when enabled. Defaults to False.
        n_error_samples (int, optional): Monte-Carlo samples per pixel for
            `compute_model_error`. Defaults to 1000.
        nufft_nthreads (int, optional): finufft OpenMP threads per chunk. Defaults
            to 1 so dask parallelises across chunks without oversubscribing finufft's
            own threads (the fast config on many chunks). Set to 0 (finufft default,
            all cores) only when computing with few chunks on the synchronous scheduler.
        log_level (int, optional): `rm_lite` logger level while chunks run;
            defaults to WARNING to silence per-chunk noise.

    Returns:
        RMSynth3DResults: Lazy FDF cube, RMSF cube, and parameters. With a Stokes I
            model, also the model cube and the 2D reference-flux/spectral-index maps.
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
    )
    fit_options = StokesIFitOptions(
        fit_order=fit_order,
        fit_function=fit_function,
        snr_cut=stokes_i_snr_cut,
        compute_model_error=compute_model_error,
        n_error_samples=n_error_samples,
    )

    rmsynth_params, theoretical_noise = _compute_global_params(
        freq_arr_hz=freq_arr_hz,
        weight_arr=weight_arr,
        fdf_options=fdf_options,
    )
    n_phi = rmsynth_params.phi_arr_radm2.shape[0]
    phi_double_arr_radm2 = make_double_phi_arr(rmsynth_params.phi_arr_radm2)
    n_phi_double = phi_double_arr_radm2.shape[0]
    fwhm_rmsf_radm2 = get_fwhm_rmsf(rmsynth_params.lambda_sq_arr_m2).fwhm_rmsf_radm2

    pol_cube = complex_pol_dask(stokes_q, stokes_u)

    # Optional per-pixel Stokes I fractional-polarization correction. Either a
    # model cube is supplied directly, or one is fitted per pixel; Q/U are then
    # divided by it and the FDF is rescaled to absolute flux by the per-pixel
    # reference-frequency Stokes I flux (see module docstring).
    ref_freq_hz = float(lambda2_to_freq(rmsynth_params.lam_sq_0_m2))
    stokes_i_model_cube: da.Array | None = None
    stokes_i_model_error_cube: da.Array | None = None
    ref_flux_map: da.Array | None = None
    alpha_map: da.Array | None = None
    alpha_error_map: da.Array | None = None
    order_map: da.Array | None = None
    if stokes_i_model is not None:
        stokes_i_model_cube = stokes_i_model.rechunk(stokes_q.chunks)
        alpha_map = da.map_blocks(
            _alpha_from_model_block,
            stokes_i_model_cube,
            drop_axis=0,
            dtype=np.float64,
            freq_arr_hz=freq_arr_hz,
            ref_freq_hz=ref_freq_hz,
        )
    elif stokes_i is not None:
        stokes_i = stokes_i.rechunk(stokes_q.chunks)
        if stokes_i_error is None and estimate_stokes_i_noise:
            stokes_i_error = estimate_stokes_i_channel_noise(stokes_i)
        if compute_model_error:
            logger.warning(
                "compute_model_error=True: the model-error cube shares one dask "
                "task with the model cube, so computing the FDF also runs the "
                "Monte-Carlo error sampling. Compute the error cube together with "
                "the model/FDF in one pass to avoid recomputing the fit."
            )
        (
            stokes_i_model_cube,
            alpha_map,
            order_map,
            stokes_i_model_error_cube,
            alpha_error_map,
        ) = _stokes_i_model_cube(
            stokes_i=stokes_i,
            stokes_i_error=stokes_i_error,
            freq_arr_hz=freq_arr_hz,
            ref_freq_hz=ref_freq_hz,
            fit_options=fit_options,
            log_level=log_level,
        )

    if stokes_i_model_cube is not None:
        pol_cube = pol_cube / stokes_i_model_cube
        ref_flux_map = da.map_blocks(
            _ref_flux_block,
            stokes_i_model_cube,
            drop_axis=0,
            dtype=np.float64,
            freq_arr_hz=freq_arr_hz,
            ref_freq_hz=ref_freq_hz,
        )

    def _synth_block(block: NDArray[np.complex128]) -> NDArray[np.complex128]:
        _, cy, cx = block.shape
        with quiet_logs(log_level):
            fdf = rmsynth_nufft(
                complex_pol_arr=block,
                lambda_sq_arr_m2=rmsynth_params.lambda_sq_arr_m2,
                phi_arr_radm2=rmsynth_params.phi_arr_radm2,
                weight_arr=rmsynth_params.weight_arr,
                lam_sq_0_m2=rmsynth_params.lam_sq_0_m2,
                nthreads=nufft_nthreads,
            )
        # rmsynth_nufft squeezes size-1 spatial axes; restore the block shape.
        return fdf.reshape(n_phi, cy, cx)

    fdf_dirty_cube = da.map_blocks(
        _synth_block,
        pol_cube,
        chunks=((n_phi,), pol_cube.chunks[1], pol_cube.chunks[2]),
        dtype=np.complex128,
    )

    if ref_flux_map is not None:
        # Rescale fractional FDF to absolute polarised flux per pixel.
        fdf_dirty_cube = fdf_dirty_cube * ref_flux_map[np.newaxis, :, :]

    def _rmsf_block(block: NDArray[np.complex128]) -> NDArray[np.complex128]:
        _, cy, cx = block.shape
        with quiet_logs(log_level):
            rmsf_result = get_rmsf_nufft(
                lambda_sq_arr_m2=rmsynth_params.lambda_sq_arr_m2,
                phi_arr_radm2=rmsynth_params.phi_arr_radm2,
                weight_arr=rmsynth_params.weight_arr,
                lam_sq_0_m2=rmsynth_params.lam_sq_0_m2,
                mask_arr=~np.isfinite(block),
                do_fit_rmsf=False,
                nthreads=nufft_nthreads,
            )
        # RMSFResults.rmsf_cube is annotated NDArray[np.float64] but is
        # actually complex128 at runtime (built from a finufft complex output).
        return rmsf_result.rmsf_cube.reshape(n_phi_double, cy, cx)  # type: ignore[return-value]

    rmsf_cube = da.map_blocks(
        _rmsf_block,
        pol_cube,
        chunks=((n_phi_double,), pol_cube.chunks[1], pol_cube.chunks[2]),
        dtype=np.complex128,
    )

    return RMSynth3DResults(
        fdf_dirty_cube=fdf_dirty_cube,
        rmsf_cube=rmsf_cube,
        phi_arr_radm2=rmsynth_params.phi_arr_radm2,
        phi_double_arr_radm2=phi_double_arr_radm2,
        fwhm_rmsf_radm2=fwhm_rmsf_radm2,
        lam_sq_0_m2=rmsynth_params.lam_sq_0_m2,
        theoretical_noise=theoretical_noise,
        stokes_i_model_cube=stokes_i_model_cube,
        stokes_i_model_error_cube=stokes_i_model_error_cube,
        stokes_i_ref_flux_map=ref_flux_map,
        stokes_i_alpha_map=alpha_map,
        stokes_i_alpha_error_map=alpha_error_map,
        stokes_i_model_order_map=order_map,
    )


def rmsynth_3d_from_fits(
    stokes_q_file: str | Path,
    stokes_u_file: str | Path,
    weight_arr: NDArray[np.float64] | None = None,
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
        stokes_u_file (str | Path): Path to the Stokes U FITS cube, same
            shape as the Q cube.
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
        nufft_nthreads (int, optional): See `rmsynth_3d`. Defaults to 1.
        target_chunk_mb (float, optional): Target per-chunk memory footprint
            in MB, see `read_fits_cube_dask`. Defaults to 256.
        log_level (int, optional): See `rmsynth_3d`. Defaults to `logging.WARNING`.

    Returns:
        RMSynth3DResults: Lazy dirty FDF cube, RMSF cube, and associated parameters.
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
        noise_arr = estimate_channel_noise_mad(stokes_q, stokes_u)
        weight_arr = 1.0 / noise_arr**2

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

    return rmsynth_3d(
        stokes_q=stokes_q,
        stokes_u=stokes_u,
        freq_arr_hz=freq_arr_hz,
        weight_arr=weight_arr,
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
        nufft_nthreads=nufft_nthreads,
        log_level=log_level,
    )
