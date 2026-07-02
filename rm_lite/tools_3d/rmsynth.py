"""RM-synthesis on chunked 3D Stokes Q/U cubes via dask.

This module is orchestration only: the actual per-pixel NUFFT math lives in
`rm_lite.utils.synthesis` (`rmsynth_nufft`, `get_rmsf_nufft`), which already
operates on a whole `(n_freq, ny, nx)` block in one vectorized call. Here we
apply that block function across a dask-chunked cube with `dask.array.map_blocks`,
chunked only on the two spatial axes (the frequency axis is always kept whole
per chunk, see `rm_lite.utils.dask_io`).

Per-pixel Stokes I fractional-polarization division and per-pixel FDF parameter
fitting (as done in `rm_lite.tools_1d.rmsynth`) are out of scope here; this
produces the dirty FDF cube and RMSF cube only. RMSF per-pixel Gaussian fitting
(`do_fit_rmsf`) is also not performed in 3D -- the analytic FWHM is used
everywhere, since fitting every pixel would add a real per-pixel loop to what
is otherwise a fully vectorized stage.
"""

from __future__ import annotations

from typing import NamedTuple

import dask.array as da
import numpy as np
from numpy.typing import NDArray

from rm_lite.utils.dask_io import complex_pol_dask
from rm_lite.utils.synthesis import (
    FDFOptions,
    RMSynthParams,
    compute_rmsynth_params,
    get_fwhm_rmsf,
    get_rmsf_nufft,
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


def _compute_global_params(
    freq_arr_hz: NDArray[np.float64],
    weight_arr: NDArray[np.float64],
    phi_max_radm2: float | None,
    d_phi_radm2: float | None,
    n_samples: float | None,
) -> RMSynthParams:
    """Compute phi_arr/lam_sq_0_m2/weight_arr once for the whole cube.

    `compute_rmsynth_params` is written for a single per-pixel spectrum, but
    its weight-array derivation round-trips exactly from a per-channel error
    spectrum (`weight = 1/error**2`), so a synthetic, fully-finite spectrum
    with `error = 1/sqrt(weight_arr)` reuses it unmodified for a per-channel
    (not per-pixel) weight array shared by every spatial chunk.
    """
    with np.errstate(divide="ignore"):
        real_error = np.where(weight_arr > 0, 1.0 / np.sqrt(weight_arr), np.inf)
    complex_pol_error = (real_error + 1j * real_error).astype(np.complex128)
    complex_pol_arr = np.ones_like(freq_arr_hz, dtype=np.complex128)

    fdf_options = FDFOptions(
        phi_max_radm2=phi_max_radm2,
        d_phi_radm2=d_phi_radm2,
        n_samples=n_samples,
        weight_type="variance",
    )
    return compute_rmsynth_params(
        freq_arr_hz=freq_arr_hz,
        complex_pol_arr=complex_pol_arr,
        complex_pol_error=complex_pol_error,
        fdf_options=fdf_options,
    )


def rmsynth_3d(
    stokes_q: da.Array,
    stokes_u: da.Array,
    freq_arr_hz: NDArray[np.float64],
    weight_arr: NDArray[np.float64] | None = None,
    phi_max_radm2: float | None = None,
    d_phi_radm2: float | None = None,
    n_samples: float | None = 10.0,
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

    Returns:
        RMSynth3DResults: Lazy dirty FDF cube, RMSF cube, and associated parameters.
    """
    if stokes_q.shape != stokes_u.shape:
        msg = f"Stokes Q and U must have the same shape. Got {stokes_q.shape} and {stokes_u.shape}."
        raise ValueError(msg)
    if stokes_q.chunks != stokes_u.chunks:
        msg = "Stokes Q and U must have identical chunking."
        raise ValueError(msg)

    n_freq = stokes_q.shape[0]
    if weight_arr is None:
        weight_arr = np.ones(n_freq, dtype=np.float64)

    rmsynth_params = _compute_global_params(
        freq_arr_hz=freq_arr_hz,
        weight_arr=weight_arr,
        phi_max_radm2=phi_max_radm2,
        d_phi_radm2=d_phi_radm2,
        n_samples=n_samples,
    )
    n_phi = rmsynth_params.phi_arr_radm2.shape[0]
    phi_double_arr_radm2 = make_double_phi_arr(rmsynth_params.phi_arr_radm2)
    n_phi_double = phi_double_arr_radm2.shape[0]
    fwhm_rmsf_radm2 = get_fwhm_rmsf(rmsynth_params.lambda_sq_arr_m2).fwhm_rmsf_radm2

    pol_cube = complex_pol_dask(stokes_q, stokes_u)

    def _synth_block(block: NDArray[np.complex128]) -> NDArray[np.complex128]:
        _, cy, cx = block.shape
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

    def _rmsf_block(block: NDArray[np.complex128]) -> NDArray[np.complex128]:
        _, cy, cx = block.shape
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
    )
