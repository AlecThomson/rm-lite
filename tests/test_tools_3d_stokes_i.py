"""Tests for optional Stokes I fractional-polarization correction in 3D RM-synthesis.

Mirrors the per-pixel-vs-1D convention of `test_tools_3d_dask.py`: uniform
weighting and no flagged channels, so the global (per-cube) lam_sq_0/weight_arr
used by the 3D orchestration matches what the 1D tools derive per pixel.
"""

from __future__ import annotations

from typing import Any, Literal, NamedTuple

import dask.array as da
import numpy as np
import pytest
from numpy.typing import NDArray
from rm_lite.tools_1d.rmsynth import run_rmsynth
from rm_lite.tools_3d.rmsynth import rmsynth_3d
from rm_lite.utils.dask_io import estimate_stokes_i_channel_noise
from rm_lite.utils.fitting import fit_stokes_i_model
from rm_lite.utils.synthesis import freq_to_lambda2
from scipy import optimize


def _raise_curve_fit(*_args: Any, **_kwargs: Any) -> None:
    msg = "curve_fit forced to fail"
    raise RuntimeError(msg)


def _require(arr: da.Array | None) -> da.Array:
    """Assert an optional result field is populated, for the type checker."""
    assert arr is not None
    return arr


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
    model_cube = _require(result.stokes_i_model_cube).compute()
    # A degree-2 power law / polynomial fits a clean power law to <0.5%.
    np.testing.assert_allclose(model_cube, cube.stokes_i, rtol=5e-3)

    fdf_cube = result.fdf_dirty_cube.compute()
    peak_rm = result.phi_arr_radm2[np.abs(fdf_cube).argmax(0)]
    # Coarse d_phi=1 grid: recovered peak within one resolution element.
    np.testing.assert_allclose(peak_rm, cube.rm_map, atol=D_PHI_RADM2)


def _single_pixel_fdf_and_rmsf_fwhm(
    alpha: float, corrected: bool
) -> tuple[float, float]:
    """Return (FDF main-lobe FWHM, reported RMSF FWHM) for a single-RM pixel
    with a power-law Stokes I of index `alpha`."""
    freq_arr_hz = (np.linspace(700, 1800, 300) * 1e6).astype(np.float64)
    lambda_sq_arr_m2 = freq_to_lambda2(freq_arr_hz)
    ref_freq_hz = float(np.median(freq_arr_hz))
    stokes_i = (freq_arr_hz / ref_freq_hz) ** alpha
    angle = 2 * 30.0 * lambda_sq_arr_m2
    q = (0.5 * np.cos(angle) * stokes_i)[:, None, None]
    u = (0.5 * np.sin(angle) * stokes_i)[:, None, None]
    i_cube = stokes_i[:, None, None]

    result = rmsynth_3d(
        da.from_array(q, chunks=(-1, 1, 1)),
        da.from_array(u, chunks=(-1, 1, 1)),
        freq_arr_hz,
        stokes_i_model=(
            da.from_array(i_cube, chunks=(-1, 1, 1)) if corrected else None
        ),
        d_phi_radm2=0.1,
        weight_type="uniform",
        phi_max_radm2=200.0,
    )
    fdf_fwhm = _half_max_fwhm(
        np.abs(result.fdf_dirty_cube.compute()[:, 0, 0]), result.phi_arr_radm2
    )
    rmsf_fwhm = _half_max_fwhm(
        np.abs(result.rmsf_cube.compute()[:, 0, 0]), result.phi_double_arr_radm2
    )
    return fdf_fwhm, rmsf_fwhm


def test_stokes_i_correction_keeps_fdf_consistent_with_rmsf():
    """A Stokes I spectral index is equivalent to reweighting the data by
    ``I(lambda^2)``, so the uncorrected FDF's *effective* RMSF is not the
    reported ``rmsf_cube`` (which is built from the per-channel weights alone).
    RM-CLEAN and the moments deconvolve against ``rmsf_cube``, so that mismatch
    biases them.

    Dividing by the Stokes I model removes the reweighting, so the corrected FDF
    matches the reported RMSF for any spectral index, exactly as a flat-spectrum
    source does. (The FDF is never narrower than its own RMSF; the effective RMSF
    of the uncorrected, reweighted data simply differs from the reported one.)
    """
    fdf_flat, rmsf_flat = _single_pixel_fdf_and_rmsf_fwhm(alpha=0.0, corrected=True)
    fdf_corr, rmsf_corr = _single_pixel_fdf_and_rmsf_fwhm(alpha=-3.0, corrected=True)
    fdf_raw, rmsf_raw = _single_pixel_fdf_and_rmsf_fwhm(alpha=-3.0, corrected=False)

    # The reported RMSF is set by the weights alone, identical in every case.
    np.testing.assert_allclose(rmsf_corr, rmsf_flat, rtol=1e-6)
    np.testing.assert_allclose(rmsf_raw, rmsf_flat, rtol=1e-6)

    # Corrected: the FDF matches the reported RMSF, independent of alpha (just as
    # a flat-spectrum source's FDF equals the RMSF).
    np.testing.assert_allclose(fdf_flat, rmsf_flat, atol=2 * 0.1)
    np.testing.assert_allclose(fdf_corr, rmsf_corr, atol=2 * 0.1)

    # Uncorrected: the spectral index reweights the data, so the FDF's effective
    # RMSF differs from the reported one: the mismatch the correction removes.
    assert abs(fdf_raw - rmsf_raw) > 0.1 * rmsf_raw


def test_stokes_i_alpha_map_recovers_spectral_index():
    """The spectral-index map recovers the input power-law index per pixel."""
    cube = _make_cube(alpha=-1.3)
    result = rmsynth_3d(
        _chunked(cube.stokes_q),
        _chunked(cube.stokes_u),
        cube.freq_arr_hz,
        stokes_i_model=_chunked(cube.stokes_i),
        d_phi_radm2=D_PHI_RADM2,
        weight_type="uniform",
    )
    alpha_map = _require(result.stokes_i_alpha_map).compute()
    assert alpha_map.shape == cube.rm_map.shape
    # The cube is a pure power law of index -1.3, recovered at every pixel.
    np.testing.assert_allclose(alpha_map, -1.3, atol=1e-6)


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
    assert result.stokes_i_alpha_map is None


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
    assert _require(result.stokes_i_model_cube).shape == cube.stokes_i.shape
    assert _require(result.stokes_i_ref_flux_map).shape == (ny, nx)
    assert _require(result.stokes_i_alpha_map).shape == (ny, nx)
    ref_flux = _require(result.stokes_i_ref_flux_map).compute()
    assert np.isfinite(ref_flux).all()
    assert (ref_flux > 0).all()
    assert np.isfinite(_require(result.stokes_i_alpha_map).compute()).all()


def test_stokes_i_error_1d_matches_3d_error():
    """A per-channel 1D error and the equivalent broadcast 3D error cube agree."""
    cube = _make_cube(alpha=-1.0)
    err_1d = np.full(cube.freq_arr_hz.size, 1e-3)
    err_cube = np.broadcast_to(err_1d[:, None, None], cube.stokes_i.shape).copy()

    common: dict[str, Any] = {
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
        _require(res_1d.stokes_i_model_cube).compute(),
        _require(res_3d.stokes_i_model_cube).compute(),
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
    assert np.isfinite(_require(result.stokes_i_model_cube).compute()).all()

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
    err_cube = _require(on.stokes_i_model_error_cube).compute()
    assert err_cube.shape == cube.stokes_i.shape
    assert np.isfinite(err_cube).all()
    assert (err_cube >= 0).all()


def _cube_with_faint_pixels(faint: list[tuple[int, int]], noise: float = 1e-3):
    """A uniformly bright cube with a few deliberately faint (low-SNR) pixels.

    Self-contained with a fixed local RNG (not the module RNG) and low noise, so
    every bright pixel fits reliably and the result is deterministic.
    """
    rng = np.random.default_rng(11)
    freq_arr_hz = (np.arange(744, 1032, 3) * 1e6).astype(np.float64)
    lambda_sq_arr_m2 = freq_to_lambda2(freq_arr_hz)
    ref_freq_hz = float(np.median(freq_arr_hz))
    ny, nx = 3, 4

    stokes_i = 2.0 * (freq_arr_hz / ref_freq_hz)[:, None, None] ** -1.0
    stokes_i = np.broadcast_to(stokes_i, (freq_arr_hz.size, ny, nx)).copy()
    rm_map = rng.uniform(-100, 100, size=(ny, nx))
    pa_map = rng.uniform(0, 180, size=(ny, nx))
    stokes_q = np.zeros((freq_arr_hz.size, ny, nx))
    stokes_u = np.zeros_like(stokes_q)
    for j in range(ny):
        for i in range(nx):
            angle = 2 * rm_map[j, i] * lambda_sq_arr_m2 + 2 * np.deg2rad(pa_map[j, i])
            stokes_q[:, j, i] = 0.6 * np.cos(angle) * stokes_i[:, j, i]
            stokes_u[:, j, i] = 0.6 * np.sin(angle) * stokes_i[:, j, i]

    for j, i in faint:
        stokes_q[:, j, i] *= 1e-4
        stokes_u[:, j, i] *= 1e-4
        stokes_i[:, j, i] *= 1e-4

    stokes_q += rng.normal(0, noise, stokes_q.shape)
    stokes_u += rng.normal(0, noise, stokes_u.shape)
    stokes_i_obs = stokes_i + rng.normal(0, noise, stokes_i.shape)
    err = np.full_like(stokes_i, noise)
    return stokes_q, stokes_u, stokes_i_obs, err, freq_arr_hz


def test_stokes_i_snr_cut_falls_back_to_flat_model():
    """Pixels below the SNR cut fall back to a flat model (no spectral
    correction): their model is a finite constant, their alpha is NaN (masked),
    and their FDF equals the uncorrected Q/U FDF. Bright pixels are fitted."""
    faint = [(0, 0), (2, 3)]
    q, u, i_obs, err, freq = _cube_with_faint_pixels(faint)
    common: dict[str, Any] = {
        "d_phi_radm2": D_PHI_RADM2,
        "weight_type": "uniform",
        "phi_max_radm2": 200.0,
    }
    result = rmsynth_3d(
        _chunked(q),
        _chunked(u),
        freq,
        stokes_i=_chunked(i_obs),
        stokes_i_error=_chunked(err),
        stokes_i_snr_cut=5.0,
        **common,
    )
    # The uncorrected reference FDF (no Stokes I at all).
    raw = rmsynth_3d(_chunked(q), _chunked(u), freq, **common)

    model = _require(result.stokes_i_model_cube).compute()
    alpha = _require(result.stokes_i_alpha_map).compute()
    fdf = result.fdf_dirty_cube.compute()
    fdf_raw = raw.fdf_dirty_cube.compute()

    faint_mask = np.zeros(model.shape[1:], dtype=bool)
    for j, i in faint:
        faint_mask[j, i] = True
        # Flat model: finite and constant across frequency.
        assert np.isfinite(model[:, j, i]).all()
        np.testing.assert_allclose(model[:, j, i], model[0, j, i], rtol=1e-10)
        # No spectral correction -> FDF matches the uncorrected Q/U FDF.
        np.testing.assert_allclose(fdf[:, j, i], fdf_raw[:, j, i], atol=1e-8)

    # Masked (unfitted) pixels have NaN alpha; fitted pixels have finite alpha.
    assert np.isnan(alpha[faint_mask]).all()
    assert np.isfinite(alpha[~faint_mask]).all()
    # The model cube is never blanked, even for masked pixels.
    assert np.isfinite(model).all()


def test_stokes_i_snr_cut_zero_fits_all_pixels():
    """A cut of 0 disables the SNR gate: even faint pixels are fitted (their
    model is not forced flat)."""
    faint = [(0, 0)]
    q, u, i_obs, err, freq = _cube_with_faint_pixels(faint)
    result = rmsynth_3d(
        _chunked(q),
        _chunked(u),
        freq,
        stokes_i=_chunked(i_obs),
        stokes_i_error=_chunked(err),
        stokes_i_snr_cut=0.0,
        d_phi_radm2=D_PHI_RADM2,
        weight_type="uniform",
    )
    assert np.isfinite(_require(result.stokes_i_model_cube).compute()).all()


def test_fit_stokes_i_model_flat_fallback_on_failure(monkeypatch):
    """When curve_fit cannot converge, the fitter returns a flat (mean) model
    instead of raising; the graceful handling lives in the fitting code."""
    monkeypatch.setattr(optimize, "curve_fit", _raise_curve_fit)

    freq_arr_hz = (np.arange(744, 1032, 3) * 1e6).astype(np.float64)
    ref_freq_hz = float(np.median(freq_arr_hz))
    stokes_i = 2.0 * (freq_arr_hz / ref_freq_hz) ** -1.0
    err = np.full_like(stokes_i, 0.01)

    fit = fit_stokes_i_model(
        freq_arr_hz, ref_freq_hz, stokes_i, err, fit_order=2, fit_type="log"
    )
    assert fit is not None
    model = fit.stokes_i_model_func(freq_arr_hz / ref_freq_hz, *np.asarray(fit.popt))
    np.testing.assert_allclose(model, model[0], rtol=1e-10)  # flat
    np.testing.assert_allclose(model[0], np.mean(stokes_i), rtol=1e-6)
    assert np.allclose(np.asarray(fit.pcov), 0.0)


def test_stokes_i_fit_failure_is_graceful(monkeypatch):
    """If every per-pixel fit fails to converge, the cube still completes and
    each pixel falls back to a finite flat model (via the fitter), no crash."""
    monkeypatch.setattr(optimize, "curve_fit", _raise_curve_fit)

    cube = _make_cube(alpha=-1.0)
    i_err = np.full_like(cube.stokes_i, 1e-3)
    result = rmsynth_3d(
        _chunked(cube.stokes_q),
        _chunked(cube.stokes_u),
        cube.freq_arr_hz,
        stokes_i=_chunked(cube.stokes_i),
        stokes_i_error=_chunked(i_err),
        stokes_i_snr_cut=0.0,  # ensure the fit is attempted (then fails over)
        d_phi_radm2=D_PHI_RADM2,
        weight_type="uniform",
    )
    model = _require(result.stokes_i_model_cube).compute()
    assert np.isfinite(model).all()
    # Flat fallback everywhere: constant across frequency per pixel.
    assert np.allclose(model, model[:1], rtol=1e-10)


def test_stokes_i_descending_frequency_matches_ascending():
    """A descending frequency axis (negative CDELT3) must give the same ref-flux
    and alpha as the ascending storage of the same physical data. Guards the
    np.interp/np.gradient ascending-axis assumption in ref_flux / alpha."""
    # Curved (order-2) model so a bad interp can't be hidden by a constant slope.
    cube = _make_cube(alpha=-1.0)
    ref_freq_hz = float(np.median(cube.freq_arr_hz))
    curve = 1.0 + 0.3 * np.log10(cube.freq_arr_hz / ref_freq_hz) ** 2
    model = cube.stokes_i * curve[:, None, None]

    def run(
        freq: NDArray[np.float64],
        q: NDArray[np.float64],
        u: NDArray[np.float64],
        m: NDArray[np.float64],
    ) -> Any:
        return rmsynth_3d(
            _chunked(q),
            _chunked(u),
            freq,
            stokes_i_model=_chunked(m),
            d_phi_radm2=D_PHI_RADM2,
            weight_type="uniform",
        )

    asc = run(cube.freq_arr_hz, cube.stokes_q, cube.stokes_u, model)
    # Same physical cube, stored with the frequency axis reversed.
    desc = run(
        cube.freq_arr_hz[::-1],
        cube.stokes_q[::-1],
        cube.stokes_u[::-1],
        model[::-1],
    )
    np.testing.assert_allclose(
        _require(desc.stokes_i_ref_flux_map).compute(),
        _require(asc.stokes_i_ref_flux_map).compute(),
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        _require(desc.stokes_i_alpha_map).compute(),
        _require(asc.stokes_i_alpha_map).compute(),
        rtol=1e-10,
    )


def test_stokes_i_error_numpy_3d_cube():
    """A per-pixel error passed as a plain NumPy 3D array (not dask) is accepted
    and fits a finite model. Guards the ndim==3 -> .rechunk() crash path."""
    cube = _make_cube(alpha=-1.0)
    err = np.full_like(cube.stokes_i, 1e-3)  # numpy, not dask
    result = rmsynth_3d(
        _chunked(cube.stokes_q),
        _chunked(cube.stokes_u),
        cube.freq_arr_hz,
        stokes_i=_chunked(cube.stokes_i),
        stokes_i_error=err,
        d_phi_radm2=D_PHI_RADM2,
        weight_type="uniform",
    )
    assert np.isfinite(_require(result.stokes_i_model_cube).compute()).all()


def test_stokes_i_model_error_fit_order_zero():
    """compute_model_error with fit_order=0 (length-1 popt) yields a finite,
    non-negative error cube. Guards the multivariate_normal.rvs orientation."""
    cube = _make_cube(alpha=-1.0)
    i_err = np.full_like(cube.stokes_i, 1e-3)
    result = rmsynth_3d(
        _chunked(cube.stokes_q),
        _chunked(cube.stokes_u),
        cube.freq_arr_hz,
        stokes_i=_chunked(cube.stokes_i),
        stokes_i_error=_chunked(i_err),
        fit_order=0,
        stokes_i_snr_cut=0.0,
        compute_model_error=True,
        n_error_samples=100,
        d_phi_radm2=D_PHI_RADM2,
        weight_type="uniform",
    )
    err_cube = _require(result.stokes_i_model_error_cube).compute()
    assert err_cube.shape == cube.stokes_i.shape
    assert np.isfinite(err_cube).all()
    assert (err_cube >= 0).all()
