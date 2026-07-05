"""RM-synthesis on chunked 3D Stokes Q/U cubes via dask."""

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
from rm_lite.utils.fitting import FitResult, fit_stokes_i_model
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
    unless `compute_model_error=True` (adds a second per-pixel fit pass)."""
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


def _pixel_stokes_i_snr(
    i_spec: NDArray[np.float64], e_spec: NDArray[np.float64]
) -> float:
    """Frequency-averaged per-pixel Stokes I SNR: `mean(I) * sqrt(n) / rms(error)`.

    Averaging `n` channels beats the noise down by `sqrt(n)`, hence the factor.
    Returns inf when there is no usable noise (all-zero or non-finite error), so
    the SNR cut becomes a no-op instead of skipping every pixel.
    """
    n = e_spec.size
    rms_err = float(np.sqrt(np.mean(e_spec**2))) if n else 0.0
    if not np.isfinite(rms_err) or rms_err <= 0:
        return np.inf
    return float(np.mean(i_spec)) * np.sqrt(n) / rms_err


def _fit_pixel_stokes_i(
    i_spec: NDArray[np.float64],
    e_spec: NDArray[np.float64],
    good: NDArray[np.bool_],
    freq_arr_hz: NDArray[np.float64],
    ref_freq_hz: float,
    fit_order: int,
    fit_function: Literal["log", "linear"],
    snr_cut: float | None,
) -> FitResult | None:
    """Fit one pixel's Stokes I spectrum, or return None if the fit is skipped.

    Returns None when there are too few finite channels or, when `snr_cut` is not
    None, the SNR is below it, so the caller can impose a flat (mean) model for
    that pixel (see `_fit_stokes_i_block`). A fit that cannot converge does not
    raise: `fit_stokes_i_model` itself falls back to a flat model.
    """
    if int(good.sum()) < fit_order + 2:
        return None
    if (
        snr_cut is not None
        and _pixel_stokes_i_snr(i_spec[good], e_spec[good]) < snr_cut
    ):
        return None
    return fit_stokes_i_model(
        freq_arr_hz=freq_arr_hz[good],
        ref_freq_hz=ref_freq_hz,
        stokes_i_arr=i_spec[good],
        stokes_i_error_arr=e_spec[good],
        fit_order=fit_order,
        fit_type=fit_function,
    )


def _fit_stokes_i_block(
    *arrays: NDArray[np.float64],
    freq_arr_hz: NDArray[np.float64],
    ref_freq_hz: float,
    err_1d: NDArray[np.float64] | None,
    fit_order: int,
    fit_function: Literal["log", "linear"],
    snr_cut: float | None,
    log_level: int,
) -> NDArray[np.float64]:
    """Fit a Stokes I model per pixel over one spatial chunk.

    Returns a stacked block of shape (n_freq + 1, cy, cx): planes 0..n_freq-1
    are the model cube, the last plane is the fitted spectral index alpha at the
    reference frequency.

    `arrays` is `(i_block,)` or `(i_block, err_block)`; the error cube is
    optional (see `_pixel_stokes_i_error`). A skipped pixel (too few finite
    channels or SNR below `snr_cut`; see `_fit_pixel_stokes_i`) falls back to a
    flat model at its mean Stokes I, so it gets no spectral correction (its FDF
    is the plain Q/U FDF) and its alpha is NaN. A pixel with no finite channels
    stays NaN in both.
    """
    i_block = arrays[0]
    err_block = arrays[1] if len(arrays) > 1 else None
    n_freq, cy, cx = i_block.shape
    x_arr = freq_arr_hz / ref_freq_hz
    ln_freq = np.log(freq_arr_hz)
    ln_ref = float(np.log(ref_freq_hz))
    out = np.full((n_freq + 1, cy, cx), np.nan, dtype=np.float64)
    # The 1D fitter logs per fit and per failure. At cube scale that floods, so
    # quiet it to at least ERROR whatever the caller's log_level.
    with quiet_logs(max(log_level, logging.ERROR)):
        for y in range(cy):
            for x in range(cx):
                i_spec = i_block[:, y, x]
                e_spec = _pixel_stokes_i_error(err_block, err_1d, n_freq, y, x)
                good = np.isfinite(i_spec) & np.isfinite(e_spec)
                fit = _fit_pixel_stokes_i(
                    i_spec,
                    e_spec,
                    good,
                    freq_arr_hz,
                    ref_freq_hz,
                    fit_order,
                    fit_function,
                    snr_cut,
                )
                if fit is not None:
                    model = fit.stokes_i_model_func(x_arr, *np.asarray(fit.popt))
                    out[:n_freq, y, x] = model
                    with np.errstate(divide="ignore", invalid="ignore"):
                        slope = np.gradient(np.log(np.abs(model)), ln_freq)
                    out[n_freq, y, x] = np.interp(ln_ref, ln_freq, slope)
                elif good.any():
                    # Flat fallback: no correction, and alpha stays NaN (masked).
                    out[:n_freq, y, x] = float(np.mean(i_spec[good]))
    return out


def _fit_stokes_i_error_block(
    *arrays: NDArray[np.float64],
    freq_arr_hz: NDArray[np.float64],
    ref_freq_hz: float,
    err_1d: NDArray[np.float64] | None,
    fit_order: int,
    fit_function: Literal["log", "linear"],
    snr_cut: float | None,
    n_error_samples: int,
    log_level: int,
) -> NDArray[np.float64]:
    """Per-pixel Stokes I model 1-sigma error cube via Monte-Carlo over the fit
    covariance, using the same 16th/84th-percentile spread as
    `create_fractional_spectra`. Opt-in, a second per-pixel fit pass. Pixels
    skipped by `_fit_pixel_stokes_i` are left NaN here: their model is an imposed
    flat model with no fitted uncertainty."""
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
                fit = _fit_pixel_stokes_i(
                    i_spec,
                    e_spec,
                    good,
                    freq_arr_hz,
                    ref_freq_hz,
                    fit_order,
                    fit_function,
                    snr_cut,
                )
                if fit is None:
                    continue
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
    ln_freq = np.log(freq_arr_hz)
    ln_ref = float(np.log(ref_freq_hz))
    alpha = np.full((cy, cx), np.nan, dtype=np.float64)
    for y in range(cy):
        for x in range(cx):
            spec = model_block[:, y, x]
            if not np.isfinite(spec).all():
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                slope = np.gradient(np.log(np.abs(spec)), ln_freq)
            value = np.interp(ln_ref, ln_freq, slope)
            alpha[y, x] = value if np.isfinite(value) else 0.0
    return alpha


def _stokes_i_model_cube(
    stokes_i: da.Array,
    stokes_i_error: NDArray[np.float64] | da.Array | None,
    freq_arr_hz: NDArray[np.float64],
    ref_freq_hz: float,
    fit_order: int,
    fit_function: Literal["log", "linear"],
    snr_cut: float | None,
    log_level: int,
) -> tuple[da.Array, da.Array]:
    """Lazy per-pixel Stokes I model cube and spectral-index map.

    Both come from one per-pixel fit pass. The model cube is chunked like
    `stokes_i`; the alpha map is (ny, nx). Alpha is NaN for pixels that were not
    fitted (see `_fit_stokes_i_block`). `stokes_i_error` is a per-channel 1D
    array (n_freq,), a per-pixel error cube (n_freq, ny, nx), or None.
    """
    err_1d: NDArray[np.float64] | None = None
    err_cube: da.Array | None = None
    if stokes_i_error is not None:
        if getattr(stokes_i_error, "ndim", 1) == 3:
            err_cube = stokes_i_error.rechunk(stokes_i.chunks)  # type: ignore[union-attr]
        else:
            err_1d = np.asarray(stokes_i_error, dtype=np.float64)

    n_freq = int(stokes_i.shape[0])
    # _fit_stokes_i_block stacks the model cube (n_freq planes) and the alpha
    # map (1 plane) so a single fit pass produces both.
    stacked = da.map_blocks(
        _fit_stokes_i_block,
        *((stokes_i,) if err_cube is None else (stokes_i, err_cube)),
        dtype=np.float64,
        chunks=((n_freq + 1,), stokes_i.chunks[1], stokes_i.chunks[2]),
        freq_arr_hz=freq_arr_hz,
        ref_freq_hz=ref_freq_hz,
        err_1d=err_1d,
        fit_order=fit_order,
        fit_function=fit_function,
        snr_cut=snr_cut,
        log_level=log_level,
    )
    return stacked[:n_freq], stacked[n_freq]


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
    stokes_i_snr_cut: float | None = 5.0,
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
            not per-pixel, weight. A per-pixel noise cube is not derived from
            the data here.
        phi_max_radm2 (float | None, optional): Maximum Faraday depth. Defaults to None.
        d_phi_radm2 (float | None, optional): Faraday depth resolution. Defaults to None.
        n_samples (float | None, optional): Number of samples across the RMSF. Defaults to 10.0.
        weight_type ("variance", "uniform", optional): Type of weighting. Defaults to "variance".
        stokes_i (da.Array | None, optional): Stokes I cube (measurements),
            shape (n_freq, ny, nx). If given, a Stokes I model is fitted per
            pixel with `rm_lite.utils.fitting.fit_stokes_i_model` and used to
            form fractional spectra. Ignored if `stokes_i_model` is given.
            Defaults to None, in which case the FDF stays in Q/U flux.
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
        stokes_i_snr_cut (float | None, optional): Frequency-averaged Stokes I
            SNR (`mean(I) * sqrt(n_chan) / rms(noise)`) below which a pixel is not
            fitted. Such a pixel, and any whose fit fails, falls back to a flat
            model at its mean Stokes I. It then gets no spectral correction, so
            its FDF is the plain Q/U FDF, and it is not blanked. Needs a Stokes I
            error (given or estimated). Pass None to disable the cut and fit
            every pixel. Only applies to the fit path, not a supplied
            `stokes_i_model`. Defaults to 5.0.
        compute_model_error (bool, optional): Also compute a per-pixel Stokes I
            model error cube via Monte-Carlo over the fit covariance. A second
            per-pixel fit pass, so it roughly doubles the fit cost. The FDF does
            not depend on it. Defaults to False.
        n_error_samples (int, optional): Monte-Carlo samples per pixel when
            `compute_model_error` is True. Defaults to 1000.
        log_level (int, optional): Log level applied to `rm_lite`'s logger while
            each chunk runs. `rmsynth_nufft`/`get_rmsf_nufft` log at INFO per
            call, which is only useful for a single spectrum. Repeated once per
            chunk it is just noise, so this defaults to WARNING. Pass
            `logging.INFO` to restore the per-chunk messages. Defaults to
            `logging.WARNING`.

    Returns:
        RMSynth3DResults: Lazy dirty FDF cube, RMSF cube, and associated
            parameters. When a Stokes I model is used the FDF is fractional
            corrected then rescaled to flux, and the Stokes I model cube plus the
            2D reference-flux and spectral-index maps are populated.
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
    alpha_map: da.Array | None = None
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
        stokes_i_model_cube, alpha_map = _stokes_i_model_cube(
            stokes_i=stokes_i,
            stokes_i_error=stokes_i_error,
            freq_arr_hz=freq_arr_hz,
            ref_freq_hz=ref_freq_hz,
            fit_order=fit_order,
            fit_function=fit_function,
            snr_cut=stokes_i_snr_cut,
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
                snr_cut=stokes_i_snr_cut,
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
        stokes_i_alpha_map=alpha_map,
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
    stokes_i_snr_cut: float | None = 5.0,
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
        stokes_i_snr_cut (float | None, optional): See `rmsynth_3d`. Defaults to 5.0.
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
        stokes_i_snr_cut=stokes_i_snr_cut,
        compute_model_error=compute_model_error,
        n_error_samples=n_error_samples,
        log_level=log_level,
    )
