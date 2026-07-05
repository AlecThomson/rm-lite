"""RM-synthesis on chunked 3D Stokes Q/U cubes via dask.

This module is orchestration only: the actual per-pixel NUFFT math lives in
`rm_lite.utils.synthesis` (`rmsynth_nufft`, `get_rmsf_nufft`), which already
operates on a whole `(n_freq, ny, nx)` block in one vectorized call. Here we
apply that block function across a dask-chunked cube with `dask.array.map_blocks`,
chunked only on the two spatial axes (the frequency axis is always kept whole
per chunk, see `rm_lite.utils.dask_io`).

Optional per-pixel Stokes I fractional-polarization division is supported (see
`rmsynth_3d`'s `stokes_i`/`stokes_i_model` arguments): a Stokes I model is
either supplied as a cube or fitted per pixel (with `rm_lite.utils.fitting.
fit_stokes_i_model`, the same fitter the 1D tools use), Q/U are divided by it
to form fractional spectra `q = Q/I`, `u = U/I`, and the resulting FDF is scaled
back to absolute polarised flux by the per-pixel reference-frequency Stokes I
flux (mirroring `rm_lite.tools_1d.rmsynth`). This removes the spurious Faraday-
depth broadening a steep Stokes I spectral index would otherwise imprint on the
FDF. The per-pixel fit is a genuine per-pixel loop within each chunk, like the
Hogbom loop in `rm_lite.tools_3d.rmclean`; the fractional division and rescaling
are vectorized dask ops.

Per-pixel FDF parameter fitting (as done in `rm_lite.tools_1d.rmsynth`) is out
of scope here. RMSF per-pixel Gaussian fitting (`do_fit_rmsf`) is also not
performed in 3D -- the analytic FWHM is used everywhere, since fitting every
pixel would add a real per-pixel loop to what is otherwise a fully vectorized
stage.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, NamedTuple

import dask.array as da
import numpy as np
from numpy.typing import NDArray
from scipy import stats

from rm_lite.utils.dask_io import (
    DEFAULT_TARGET_CHUNK_MB,
    complex_pol_dask,
    estimate_channel_noise_mad,
    estimate_stokes_i_channel_noise,
    freq_arr_hz_from_header,
    read_fits_cube_dask,
)
from rm_lite.utils.fitting import fit_stokes_i_model
from rm_lite.utils.logging import quiet_logs
from rm_lite.utils.synthesis import (
    FDFOptions,
    RMSynthParams,
    TheoreticalNoise,
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
    """Theoretical FDF-domain noise from the per-channel weight array (uniform
    across the cube -- 3D RM-synthesis only carries a per-channel, not
    per-pixel, noise estimate). When a Stokes I model is used, the FDF is
    rescaled to absolute polarised flux per pixel; this noise stays in the
    absolute Q/U-error domain it was computed in, which the per-pixel rescaling
    keeps approximately consistent (exactly so for a flat Stokes I spectrum)."""
    stokes_i_model_cube: da.Array | None = None
    """Per-pixel Stokes I model cube, lazy dask array of shape (n_freq, ny, nx).
    None unless a Stokes I cube/model was supplied to `rmsynth_3d`."""
    stokes_i_model_error_cube: da.Array | None = None
    """Per-pixel Stokes I model 1-sigma error cube, shape (n_freq, ny, nx).
    None unless `compute_model_error=True` (opt-in; adds a second per-pixel
    Monte-Carlo fit pass)."""
    stokes_i_ref_flux_map: da.Array | None = None
    """Per-pixel reference-frequency Stokes I flux, shape (ny, nx). The factor
    the fractional FDF was multiplied by to reach absolute units. None unless a
    Stokes I cube/model was supplied."""


def _compute_global_params(
    freq_arr_hz: NDArray[np.float64],
    weight_arr: NDArray[np.float64],
    phi_max_radm2: float | None,
    d_phi_radm2: float | None,
    n_samples: float | None,
    weight_type: Literal["variance", "uniform"],
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

    fdf_options = FDFOptions(
        phi_max_radm2=phi_max_radm2,
        d_phi_radm2=d_phi_radm2,
        n_samples=n_samples,
        weight_type=weight_type,
    )
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


def _fit_stokes_i_block(
    *arrays: NDArray[np.float64],
    freq_arr_hz: NDArray[np.float64],
    ref_freq_hz: float,
    err_1d: NDArray[np.float64] | None,
    fit_order: int,
    fit_function: Literal["log", "linear"],
    log_level: int,
) -> NDArray[np.float64]:
    """Fit a Stokes I model per pixel over one spatial chunk -> model cube block.

    `arrays` is `(i_block,)` or `(i_block, err_block)`; the error cube is
    optional (see `_pixel_stokes_i_error`). Pixels with too few finite channels
    to constrain the fit are left NaN, which flags them downstream.
    """
    i_block = arrays[0]
    err_block = arrays[1] if len(arrays) > 1 else None
    n_freq, cy, cx = i_block.shape
    x_arr = freq_arr_hz / ref_freq_hz
    model = np.full((n_freq, cy, cx), np.nan, dtype=np.float64)
    # fit_stokes_i_model logs per fit (INFO), and per-pixel fit failures at
    # WARNING/ERROR -- at cube scale that's a flood, so quiet the fit to at
    # least ERROR regardless of the caller's log_level (a failed pixel just
    # yields a NaN model column, handled downstream).
    with quiet_logs(max(log_level, logging.ERROR)):
        for y in range(cy):
            for x in range(cx):
                i_spec = i_block[:, y, x]
                e_spec = _pixel_stokes_i_error(err_block, err_1d, n_freq, y, x)
                good = np.isfinite(i_spec) & np.isfinite(e_spec)
                if int(good.sum()) < fit_order + 2:
                    continue
                fit = fit_stokes_i_model(
                    freq_arr_hz=freq_arr_hz[good],
                    ref_freq_hz=ref_freq_hz,
                    stokes_i_arr=i_spec[good],
                    stokes_i_error_arr=e_spec[good],
                    fit_order=fit_order,
                    fit_type=fit_function,
                )
                model[:, y, x] = fit.stokes_i_model_func(x_arr, *np.asarray(fit.popt))
    return model


def _fit_stokes_i_error_block(
    *arrays: NDArray[np.float64],
    freq_arr_hz: NDArray[np.float64],
    ref_freq_hz: float,
    err_1d: NDArray[np.float64] | None,
    fit_order: int,
    fit_function: Literal["log", "linear"],
    n_error_samples: int,
    log_level: int,
) -> NDArray[np.float64]:
    """Per-pixel Stokes I model 1-sigma error via Monte-Carlo over the fit
    covariance -> error cube block. Mirrors `create_fractional_spectra`'s
    16th/84th-percentile spread. Opt-in: a second per-pixel fit pass."""
    i_block = arrays[0]
    err_block = arrays[1] if len(arrays) > 1 else None
    n_freq, cy, cx = i_block.shape
    x_arr = freq_arr_hz / ref_freq_hz
    model_error = np.full((n_freq, cy, cx), np.nan, dtype=np.float64)
    with quiet_logs(max(log_level, logging.ERROR)):
        for y in range(cy):
            for x in range(cx):
                i_spec = i_block[:, y, x]
                e_spec = _pixel_stokes_i_error(err_block, err_1d, n_freq, y, x)
                good = np.isfinite(i_spec) & np.isfinite(e_spec)
                if int(good.sum()) < fit_order + 2:
                    continue
                fit = fit_stokes_i_model(
                    freq_arr_hz=freq_arr_hz[good],
                    ref_freq_hz=ref_freq_hz,
                    stokes_i_arr=i_spec[good],
                    stokes_i_error_arr=e_spec[good],
                    fit_order=fit_order,
                    fit_type=fit_function,
                )
                dist = stats.multivariate_normal(
                    mean=np.asarray(fit.popt),
                    cov=np.asarray(fit.pcov),
                    allow_singular=True,
                )
                samples = np.atleast_2d(dist.rvs(n_error_samples))
                model_samples = np.array(
                    [fit.stokes_i_model_func(x_arr, *s) for s in samples]
                )
                low, high = np.nanpercentile(model_samples, [16, 84], axis=0)
                err = np.abs(high - low)
                err[err > 1e99] = np.nan
                model_error[:, y, x] = err
    return model_error


def _ref_flux_block(
    model_block: NDArray[np.float64],
    freq_arr_hz: NDArray[np.float64],
    ref_freq_hz: float,
) -> NDArray[np.float64]:
    """Interpolate a Stokes I model cube block at the reference frequency ->
    (cy, cx) reference-flux map block."""
    _, cy, cx = model_block.shape
    ref_flux = np.full((cy, cx), np.nan, dtype=np.float64)
    for y in range(cy):
        for x in range(cx):
            spec = model_block[:, y, x]
            if np.isfinite(spec).all():
                ref_flux[y, x] = np.interp(ref_freq_hz, freq_arr_hz, spec)
    return ref_flux


def _stokes_i_model_cube(
    stokes_i: da.Array,
    stokes_i_error: NDArray[np.float64] | da.Array | None,
    freq_arr_hz: NDArray[np.float64],
    ref_freq_hz: float,
    fit_order: int,
    fit_function: Literal["log", "linear"],
    log_level: int,
) -> da.Array:
    """Lazy per-pixel Stokes I model cube, chunked like `stokes_i`.

    `stokes_i_error` is a per-channel 1D array (n_freq,), a per-pixel error cube
    (n_freq, ny, nx), or None (unweighted fit).
    """
    err_1d: NDArray[np.float64] | None = None
    err_cube: da.Array | None = None
    if stokes_i_error is not None:
        if getattr(stokes_i_error, "ndim", 1) == 3:
            err_cube = stokes_i_error.rechunk(stokes_i.chunks)  # type: ignore[union-attr]
        else:
            err_1d = np.asarray(stokes_i_error, dtype=np.float64)

    kwargs = {
        "freq_arr_hz": freq_arr_hz,
        "ref_freq_hz": ref_freq_hz,
        "err_1d": err_1d,
        "fit_order": fit_order,
        "fit_function": fit_function,
        "log_level": log_level,
    }
    args = (stokes_i,) if err_cube is None else (stokes_i, err_cube)
    return da.map_blocks(
        _fit_stokes_i_block, *args, dtype=np.float64, chunks=stokes_i.chunks, **kwargs
    )


def rmsynth_3d(
    stokes_q: da.Array,
    stokes_u: da.Array,
    freq_arr_hz: NDArray[np.float64],
    weight_arr: NDArray[np.float64] | None = None,
    phi_max_radm2: float | None = None,
    d_phi_radm2: float | None = None,
    n_samples: float | None = 10.0,
    weight_type: Literal["variance", "uniform"] = "variance",
    stokes_i: da.Array | None = None,
    stokes_i_error: NDArray[np.float64] | da.Array | None = None,
    stokes_i_model: da.Array | None = None,
    estimate_stokes_i_noise: bool = False,
    fit_order: int = 2,
    fit_function: Literal["log", "linear"] = "log",
    compute_model_error: bool = False,
    n_error_samples: int = 1000,
    log_level: int = logging.WARNING,
) -> RMSynth3DResults:
    """Run RM-synthesis on chunked Stokes Q/U cubes.

    Args:
        stokes_q (da.Array): Stokes Q dask array, shape (n_freq, ny, nx),
            chunked spatially only (whole frequency axis per chunk).
        stokes_u (da.Array): Stokes U dask array, same shape/chunks as `stokes_q`.
        freq_arr_hz (NDArray[np.float64]): Frequency array in Hz.
        weight_arr (NDArray[np.float64] | None, optional): Per-channel weight
            array. Defaults to uniform weighting. Note this is a per-channel,
            not per-pixel, weight -- a per-pixel noise cube is not derived
            from the data here.
        phi_max_radm2 (float | None, optional): Maximum Faraday depth. Defaults to None.
        d_phi_radm2 (float | None, optional): Faraday depth resolution. Defaults to None.
        n_samples (float | None, optional): Number of samples across the RMSF. Defaults to 10.0.
        weight_type ("variance", "uniform", optional): Type of weighting. Defaults to "variance".
        stokes_i (da.Array | None, optional): Stokes I cube (measurements),
            shape (n_freq, ny, nx). If given, a Stokes I model is fitted per
            pixel with `rm_lite.utils.fitting.fit_stokes_i_model` and used to
            form fractional spectra. Ignored if `stokes_i_model` is given.
            Defaults to None (no fractional division -- FDF stays in Q/U flux).
        stokes_i_error (NDArray[np.float64] | da.Array | None, optional): Stokes
            I error, either per-channel 1D shape (n_freq,) or a per-pixel cube
            shape (n_freq, ny, nx), used to weight the per-pixel fit. Defaults
            to None (unweighted fit, or estimated if `estimate_stokes_i_noise`).
        stokes_i_model (da.Array | None, optional): Pre-computed Stokes I model
            cube, shape (n_freq, ny, nx). Used directly (no fitting) to form
            fractional spectra. Takes precedence over `stokes_i`. Defaults to None.
        estimate_stokes_i_noise (bool, optional): If True and `stokes_i` is given
            without `stokes_i_error`, derive a per-channel error from the Stokes
            I cube with `estimate_stokes_i_channel_noise`. Defaults to False.
        fit_order (int, optional): Stokes I fit order. Negative iterates orders
            and picks the best by AIC (see `fit_stokes_i_model`). Defaults to 2.
        fit_function ("log", "linear", optional): Stokes I fit family ("log" =
            power law, "linear" = polynomial). Defaults to "log".
        compute_model_error (bool, optional): Also compute a per-pixel Stokes I
            model error cube via Monte-Carlo over the fit covariance. Opt-in: a
            second per-pixel fit pass, so it roughly doubles fit cost. The FDF
            itself does not depend on this. Defaults to False.
        n_error_samples (int, optional): Monte-Carlo samples per pixel when
            `compute_model_error` is True. Defaults to 1000.
        log_level (int, optional): Log level applied to `rm_lite`'s logger while
            each chunk runs. `rmsynth_nufft`/`get_rmsf_nufft` log at INFO per
            call, which is only useful for a single spectrum -- repeated once
            per chunk it's just noise, so this defaults to WARNING. Pass
            `logging.INFO` to restore the per-chunk messages. Defaults to
            `logging.WARNING`.

    Returns:
        RMSynth3DResults: Lazy dirty FDF cube, RMSF cube, and associated
            parameters. When a Stokes I model is used, the FDF is fractional-
            corrected then rescaled to absolute polarised flux, and the Stokes I
            model cube / reference-flux map are populated.
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

    rmsynth_params, theoretical_noise = _compute_global_params(
        freq_arr_hz=freq_arr_hz,
        weight_arr=weight_arr,
        phi_max_radm2=phi_max_radm2,
        d_phi_radm2=d_phi_radm2,
        n_samples=n_samples,
        weight_type=weight_type,
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
    if stokes_i_model is not None:
        stokes_i_model_cube = stokes_i_model.rechunk(stokes_q.chunks)
    elif stokes_i is not None:
        stokes_i = stokes_i.rechunk(stokes_q.chunks)
        if stokes_i_error is None and estimate_stokes_i_noise:
            stokes_i_error = estimate_stokes_i_channel_noise(stokes_i)
        stokes_i_model_cube = _stokes_i_model_cube(
            stokes_i=stokes_i,
            stokes_i_error=stokes_i_error,
            freq_arr_hz=freq_arr_hz,
            ref_freq_hz=ref_freq_hz,
            fit_order=fit_order,
            fit_function=fit_function,
            log_level=log_level,
        )
        if compute_model_error:
            err_1d = (
                np.asarray(stokes_i_error, dtype=np.float64)
                if stokes_i_error is not None
                and getattr(stokes_i_error, "ndim", 1) != 3
                else None
            )
            err_args = (
                (stokes_i, stokes_i_error.rechunk(stokes_q.chunks))  # type: ignore[union-attr]
                if stokes_i_error is not None
                and getattr(stokes_i_error, "ndim", 1) == 3
                else (stokes_i,)
            )
            stokes_i_model_error_cube = da.map_blocks(
                _fit_stokes_i_error_block,
                *err_args,
                dtype=np.float64,
                chunks=stokes_i.chunks,
                freq_arr_hz=freq_arr_hz,
                ref_freq_hz=ref_freq_hz,
                err_1d=err_1d,
                fit_order=fit_order,
                fit_function=fit_function,
                n_error_samples=n_error_samples,
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
    )


def rmsynth_3d_from_fits(
    stokes_q_file: str | Path,
    stokes_u_file: str | Path,
    weight_arr: NDArray[np.float64] | None = None,
    phi_max_radm2: float | None = None,
    d_phi_radm2: float | None = None,
    n_samples: float | None = 10.0,
    weight_type: Literal["variance", "uniform"] = "variance",
    stokes_i_file: str | Path | None = None,
    stokes_i_error_file: str | Path | None = None,
    stokes_i_model_file: str | Path | None = None,
    estimate_stokes_i_noise: bool = False,
    fit_order: int = 2,
    fit_function: Literal["log", "linear"] = "log",
    compute_model_error: bool = False,
    n_error_samples: int = 1000,
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
        weight_type ("variance", "uniform", optional): Type of weighting. Defaults to "variance".
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
        compute_model_error (bool, optional): See `rmsynth_3d`. Defaults to False.
        n_error_samples (int, optional): See `rmsynth_3d`. Defaults to 1000.
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

    if weight_arr is None and weight_type == "variance":
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
        stokes_i=stokes_i,
        stokes_i_error=stokes_i_error,
        stokes_i_model=stokes_i_model,
        estimate_stokes_i_noise=estimate_stokes_i_noise,
        fit_order=fit_order,
        fit_function=fit_function,
        compute_model_error=compute_model_error,
        n_error_samples=n_error_samples,
        log_level=log_level,
    )
