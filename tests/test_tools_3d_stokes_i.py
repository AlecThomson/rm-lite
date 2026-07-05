"""Tests for optional Stokes I fractional-polarization correction in 3D RM-synthesis.

Mirrors the per-pixel-vs-1D convention of `test_tools_3d_dask.py`: uniform
weighting and no flagged channels, so the global (per-cube) lam_sq_0/weight_arr
used by the 3D orchestration matches what the 1D tools derive per pixel.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import dask.array as da
import numpy as np
import pytest
from numpy.typing import NDArray
from rm_lite.tools_1d.rmsynth import run_rmsynth
from rm_lite.tools_3d.rmsynth import rmsynth_3d
from rm_lite.utils.dask_io import estimate_stokes_i_channel_noise
from rm_lite.utils.synthesis import freq_to_lambda2

RNG = np.random.default_rng(2026)
D_PHI_RADM2 = 1.0


class StokesICube(NamedTuple):
    freq_arr_hz: NDArray[np.float64]
    stokes_q: NDArray[np.float64]
    stokes_u: NDArray[np.float64]
    stokes_i: NDArray[np.float64]
    rm_map: NDArray[np.float64]


def _make_cube(
    ny: int = 3,
    nx: int = 4,
    alpha: float = -0.8,
    frac_pol: float = 0.6,
    noise: float = 0.0,
) -> StokesICube:
    """Polarised Q/U/I cube with a per-pixel RM/PA and a power-law Stokes I."""
    freq_arr_hz = (np.arange(744, 1032, 3) * 1e6).astype(np.float64)
    lambda_sq_arr_m2 = freq_to_lambda2(freq_arr_hz)
    ref_freq_hz = float(np.median(freq_arr_hz))

    rm_map = RNG.uniform(-120, 120, size=(ny, nx))
    pa_map = RNG.uniform(0, 180, size=(ny, nx))
    amp_map = RNG.uniform(1.0, 3.0, size=(ny, nx))

    x = (freq_arr_hz / ref_freq_hz)[:, None, None]
    stokes_i = amp_map[None] * x**alpha

    stokes_q = np.zeros((freq_arr_hz.size, ny, nx))
    stokes_u = np.zeros_like(stokes_q)
    for j in range(ny):
        for i in range(nx):
            angle = 2 * rm_map[j, i] * lambda_sq_arr_m2 + 2 * np.deg2rad(pa_map[j, i])
            stokes_q[:, j, i] = frac_pol * np.cos(angle) * stokes_i[:, j, i]
            stokes_u[:, j, i] = frac_pol * np.sin(angle) * stokes_i[:, j, i]

    if noise:
        stokes_q += RNG.normal(0, noise, stokes_q.shape)
        stokes_u += RNG.normal(0, noise, stokes_u.shape)
        stokes_i += RNG.normal(0, noise, stokes_i.shape)

    return StokesICube(freq_arr_hz, stokes_q, stokes_u, stokes_i, rm_map)


def _chunked(array: NDArray[np.float64], cy: int = 2, cx: int = 2) -> da.Array:
    return da.from_array(array, chunks=(-1, cy, cx))


def _half_max_fwhm(
    amp: NDArray[np.float64], phi_arr_radm2: NDArray[np.float64]
) -> float:
    a = amp / amp.max()
    peak = int(a.argmax())
    left = peak - int(np.argmax(a[peak::-1] < 0.5))
    right = peak + int(np.argmax(a[peak:] < 0.5))
    return float(phi_arr_radm2[right] - phi_arr_radm2[left])


def test_stokes_i_model_path_matches_1d_per_pixel():
    """3D fractional-then-rescaled FDF (model supplied) must match 1D per pixel."""
    cube = _make_cube(alpha=-1.2)
    ny, nx = cube.rm_map.shape
    q_dask, u_dask = _chunked(cube.stokes_q), _chunked(cube.stokes_u)
    i_dask = _chunked(cube.stokes_i)

    result = rmsynth_3d(
        q_dask,
        u_dask,
        cube.freq_arr_hz,
        stokes_i_model=i_dask,
        d_phi_radm2=D_PHI_RADM2,
        weight_type="uniform",
    )
    fdf_cube = result.fdf_dirty_cube.compute()

    for j in range(ny):
        for i in range(nx):
            complex_pol = cube.stokes_q[:, j, i] + 1j * cube.stokes_u[:, j, i]
            ref = run_rmsynth(
                freq_arr_hz=cube.freq_arr_hz,
                complex_pol_arr=complex_pol,
                complex_pol_error=np.ones_like(complex_pol),
                stokes_i_model_arr=cube.stokes_i[:, j, i],
                stokes_i_model_error=np.zeros(cube.freq_arr_hz.size),
                d_phi_radm2=D_PHI_RADM2,
                weight_type="uniform",
            )
            ref_fdf = ref.fdf_arrs["fdf_dirty_complex_arr"].to_numpy().astype(complex)
            np.testing.assert_allclose(fdf_cube[:, j, i], ref_fdf, atol=1e-8)


@pytest.mark.parametrize("fit_function", ["log", "linear"])
def test_stokes_i_fit_recovers_model_and_rm(
    fit_function: Literal["log", "linear"],
):
    """Per-pixel fit recovers the input Stokes I cube and the correct peak RM."""
    cube = _make_cube(alpha=-1.0, noise=0.0)
    i_err = np.full_like(cube.stokes_i, 1e-3)

    result = rmsynth_3d(
        _chunked(cube.stokes_q),
        _chunked(cube.stokes_u),
        cube.freq_arr_hz,
        stokes_i=_chunked(cube.stokes_i),
        stokes_i_error=_chunked(i_err),
        fit_function=fit_function,
        d_phi_radm2=D_PHI_RADM2,
        weight_type="uniform",
    )
    model_cube = result.stokes_i_model_cube.compute()
    # A degree-2 power law / polynomial fits a clean power law to <0.5%.
    np.testing.assert_allclose(model_cube, cube.stokes_i, rtol=5e-3)

    fdf_cube = result.fdf_dirty_cube.compute()
    peak_rm = result.phi_arr_radm2[np.abs(fdf_cube).argmax(0)]
    # Coarse d_phi=1 grid: recovered peak within one resolution element.
    np.testing.assert_allclose(peak_rm, cube.rm_map, atol=D_PHI_RADM2)


def _single_pixel_fdf_width(alpha: float, corrected: bool) -> float:
    freq_arr_hz = (np.linspace(700, 1800, 300) * 1e6).astype(np.float64)
    lambda_sq_arr_m2 = freq_to_lambda2(freq_arr_hz)
    ref_freq_hz = float(np.median(freq_arr_hz))
    stokes_i = (freq_arr_hz / ref_freq_hz) ** alpha
    angle = 2 * 30.0 * lambda_sq_arr_m2
    q = (0.5 * np.cos(angle) * stokes_i)[:, None, None]
    u = (0.5 * np.sin(angle) * stokes_i)[:, None, None]
    i_cube = stokes_i[:, None, None]

    kwargs = {"d_phi_radm2": 0.1, "weight_type": "uniform", "phi_max_radm2": 200.0}
    result = rmsynth_3d(
        da.from_array(q, chunks=(-1, 1, 1)),
        da.from_array(u, chunks=(-1, 1, 1)),
        freq_arr_hz,
        stokes_i_model=(
            da.from_array(i_cube, chunks=(-1, 1, 1)) if corrected else None
        ),
        **kwargs,
    )
    amp = np.abs(result.fdf_dirty_cube.compute()[:, 0, 0])
    return _half_max_fwhm(amp, result.phi_arr_radm2)


def test_stokes_i_correction_removes_spectral_index_distortion():
    """The Stokes I spectral index imprints a spurious FDF width change unrelated
    to Faraday structure; dividing by the model removes it.

    The corrected FDF width is set only by the RMSF, so it is invariant to the
    spectral index. The uncorrected width is distorted by it.
    """
    corr_flat = _single_pixel_fdf_width(alpha=0.0, corrected=True)
    corr_steep = _single_pixel_fdf_width(alpha=-3.0, corrected=True)
    raw_flat = _single_pixel_fdf_width(alpha=0.0, corrected=False)
    raw_steep = _single_pixel_fdf_width(alpha=-3.0, corrected=False)

    # Flat spectrum: correction is a no-op, so raw == corrected.
    np.testing.assert_allclose(raw_flat, corr_flat, rtol=1e-6)
    # Corrected width is invariant to spectral index.
    np.testing.assert_allclose(corr_steep, corr_flat, rtol=0.02)
    # Uncorrected width is significantly distorted by the spectral index.
    assert abs(raw_steep - raw_flat) > 0.1 * raw_flat


def test_no_stokes_i_leaves_outputs_none():
    """Without any Stokes I input the extra outputs stay None (raw Q/U FDF)."""
    cube = _make_cube()
    result = rmsynth_3d(
        _chunked(cube.stokes_q),
        _chunked(cube.stokes_u),
        cube.freq_arr_hz,
        d_phi_radm2=D_PHI_RADM2,
        weight_type="uniform",
    )
    assert result.stokes_i_model_cube is None
    assert result.stokes_i_model_error_cube is None
    assert result.stokes_i_ref_flux_map is None


def test_stokes_i_output_shapes():
    """Model cube / ref-flux map have the expected shapes when Stokes I is used."""
    cube = _make_cube()
    ny, nx = cube.rm_map.shape
    result = rmsynth_3d(
        _chunked(cube.stokes_q),
        _chunked(cube.stokes_u),
        cube.freq_arr_hz,
        stokes_i=_chunked(cube.stokes_i),
        d_phi_radm2=D_PHI_RADM2,
        weight_type="uniform",
    )
    assert result.stokes_i_model_cube.shape == cube.stokes_i.shape
    assert result.stokes_i_ref_flux_map.shape == (ny, nx)
    ref_flux = result.stokes_i_ref_flux_map.compute()
    assert np.isfinite(ref_flux).all()
    assert (ref_flux > 0).all()


def test_stokes_i_error_1d_matches_3d_error():
    """A per-channel 1D error and the equivalent broadcast 3D error cube agree."""
    cube = _make_cube(alpha=-1.0)
    err_1d = np.full(cube.freq_arr_hz.size, 1e-3)
    err_cube = np.broadcast_to(err_1d[:, None, None], cube.stokes_i.shape).copy()

    common = {
        "freq_arr_hz": cube.freq_arr_hz,
        "d_phi_radm2": D_PHI_RADM2,
        "weight_type": "uniform",
    }
    res_1d = rmsynth_3d(
        _chunked(cube.stokes_q),
        _chunked(cube.stokes_u),
        stokes_i=_chunked(cube.stokes_i),
        stokes_i_error=err_1d,
        **common,
    )
    res_3d = rmsynth_3d(
        _chunked(cube.stokes_q),
        _chunked(cube.stokes_u),
        stokes_i=_chunked(cube.stokes_i),
        stokes_i_error=_chunked(err_cube),
        **common,
    )
    np.testing.assert_allclose(
        res_1d.stokes_i_model_cube.compute(),
        res_3d.stokes_i_model_cube.compute(),
        rtol=1e-10,
    )


def test_estimate_stokes_i_noise_runs():
    """The estimate flag derives a per-channel error and fits a finite model."""
    cube = _make_cube(alpha=-1.0, noise=0.01)
    result = rmsynth_3d(
        _chunked(cube.stokes_q),
        _chunked(cube.stokes_u),
        cube.freq_arr_hz,
        stokes_i=_chunked(cube.stokes_i),
        estimate_stokes_i_noise=True,
        d_phi_radm2=D_PHI_RADM2,
        weight_type="uniform",
    )
    assert np.isfinite(result.stokes_i_model_cube.compute()).all()

    noise = estimate_stokes_i_channel_noise(_chunked(cube.stokes_i))
    assert noise.shape == (cube.freq_arr_hz.size,)
    assert (noise > 0).all()


def test_stokes_i_model_error_cube_opt_in():
    """compute_model_error yields a finite, non-negative error cube; off by default."""
    cube = _make_cube(alpha=-1.0)
    i_err = np.full_like(cube.stokes_i, 1e-3)

    off = rmsynth_3d(
        _chunked(cube.stokes_q),
        _chunked(cube.stokes_u),
        cube.freq_arr_hz,
        stokes_i=_chunked(cube.stokes_i),
        stokes_i_error=_chunked(i_err),
        d_phi_radm2=D_PHI_RADM2,
        weight_type="uniform",
    )
    assert off.stokes_i_model_error_cube is None

    on = rmsynth_3d(
        _chunked(cube.stokes_q),
        _chunked(cube.stokes_u),
        cube.freq_arr_hz,
        stokes_i=_chunked(cube.stokes_i),
        stokes_i_error=_chunked(i_err),
        compute_model_error=True,
        n_error_samples=200,
        d_phi_radm2=D_PHI_RADM2,
        weight_type="uniform",
    )
    err_cube = on.stokes_i_model_error_cube.compute()
    assert err_cube.shape == cube.stokes_i.shape
    assert np.isfinite(err_cube).all()
    assert (err_cube >= 0).all()
