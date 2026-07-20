"""Tests for multiscale RM-CLEAN."""

from __future__ import annotations

import logging
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray
from rm_lite.tools_1d.rmclean import run_rmclean_from_synth
from rm_lite.tools_1d.rmsynth import run_rmsynth
from rm_lite.utils.clean import (
    MultiscaleOptions,
    RMCleanOptions,
    RMCleanResults,
    RMSynthArrays,
    _reconvolve_model,
    compute_scale_kernels,
    convolve_fdf_scale,
    default_scales,
    rmclean,
)
from rm_lite.utils.logging import quiet_logs
from rm_lite.utils.simulate import (
    build_geometry,
    build_model_fdf,
    delta,
    gauss,
    simulate_fdf,
)
from rm_lite.utils.synthesis import (
    calc_faraday_moments,
    freq_to_lambda2,
    get_fwhm_rmsf,
    get_rmsf_nufft,
    make_phi_arr,
)

RNG = np.random.default_rng(1234)


def _coverage_grid(
    freq_lo_hz: float,
    freq_hi_hz: float,
    phi_max_radm2: float = 250.0,
    quiet: bool = True,
) -> NDArray[np.float64]:
    """Auto multiscale grid for a contiguous band, from the band's own RMSF.

    Mirrors how rmclean derives the grid: phi axis at fwhm/10 sampling, phi
    window >= 2*phi_max_scale so the window never caps the wideband bands, and
    phi_max_scale_radm2 = pi / lambda_sq_min. `quiet=False` lets the degeneration
    warning through for caplog.
    """
    freq = np.linspace(freq_lo_hz, freq_hi_hz, 200)
    lsq = freq_to_lambda2(freq)
    fwhm = float(get_fwhm_rmsf(lsq).fwhm_rmsf_radm2)
    phi = make_phi_arr(phi_max_radm2=phi_max_radm2, d_phi_radm2=fwhm / 10)
    phi_max_scale = float(np.pi / lsq.min())

    def _call() -> NDArray[np.float64]:
        return default_scales(
            phi, fwhm, MultiscaleOptions(), phi_max_scale_radm2=phi_max_scale
        )

    if quiet:
        with quiet_logs(logging.ERROR):
            return _call()
    return _call()


def burn_slab(
    lsq: NDArray[np.float64],
    frac_pol: float,
    psi0_deg: float,
    rm_radm2: float,
    delta_rm_radm2: float,
) -> NDArray[np.complex128]:
    """Burn slab P(lambda^2): a Faraday-thick component (top-hat in phi)."""
    return cast(
        "NDArray[np.complex128]",
        (
            frac_pol
            * np.exp(2j * (np.deg2rad(psi0_deg) + rm_radm2 * lsq))
            * np.sinc(delta_rm_radm2 * lsq / np.pi)
        ).astype(np.complex128),
    )


def _run_synth(complex_pol: NDArray[np.complex128], freq_hz: NDArray[np.float64]):
    rms = 0.02
    err = np.ones_like(complex_pol) * (rms + 1j * rms)
    with quiet_logs(logging.ERROR):
        return run_rmsynth(freq_hz, complex_pol, err, n_samples=10, phi_max_radm2=250.0)


def test_default_scales_capped_to_phi_window() -> None:
    """A huge phi_max_scale must not inflate scales past the FDF phi window:
    a scale kernel wider than the window is meaningless (Bug A)."""
    fwhm = 6.0
    phi = make_phi_arr(120.0, fwhm / 10)  # window = 240 rad/m^2
    window = float(phi.max() - phi.min())
    scales = default_scales(phi, fwhm, MultiscaleOptions(), phi_max_scale_radm2=1.0e5)
    # Every scale kernel (scale * fwhm) fits inside the window.
    assert scales.max() * fwhm <= window
    # And the huge phi_max did not win over the window bound.
    assert scales.max() * fwhm <= 1.0e5 / 2


def test_multiscale_oversized_scales_do_not_diverge() -> None:
    """Explicit scales far larger than the phi window must not produce a runaway
    clean FDF: the whole-array divergence guard reverts to the best state so the
    result stays finite and bounded by the dirty peak (Bug B)."""
    freq_hz = np.linspace(0.8e9, 2.2e9, 400)
    lsq = freq_to_lambda2(freq_hz)
    model = burn_slab(lsq, 0.5, 20, 30, 12.0)
    noisy = (
        model
        + RNG.normal(0, 0.02, freq_hz.size)
        + 1j * RNG.normal(0, 0.02, freq_hz.size)
    ).astype(np.complex128)
    synth = _run_synth(noisy, freq_hz)
    phi = synth.fdf_arrs["phi_arr_radm2"].to_numpy().astype(float)
    fwhm = float(synth.fdf_parameters["fwhm_rmsf_radm2"][0])
    phi2 = synth.rmsf_arrs["phi2_arr_radm2"].to_numpy().astype(float)
    dirty = synth.fdf_arrs["fdf_dirty_complex_arr"].to_numpy().astype(complex)
    rmsf = synth.rmsf_arrs["rmsf_complex_arr"].to_numpy().astype(complex)
    noise = float(synth.fdf_parameters["fdf_error_noise"][0])

    window = float(phi.max() - phi.min())
    # Scales up to ~10x the phi window: without the guard these over-subtract and
    # grow spurious features outside the source region.
    oversized = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]) * window / fwhm / 4
    oversized[0] = 0.0
    with quiet_logs(logging.ERROR):
        result = rmclean(
            RMSynthArrays(
                dirty_fdf_arr=dirty,
                phi_arr_radm2=phi,
                rmsf_arr=rmsf,
                phi_double_arr_radm2=phi2,
                fwhm_rmsf_arr=np.array(fwhm),
            ),
            RMCleanOptions(mask=8 * noise, threshold=3 * noise, fdf_noise=noise),
            multiscale_options=MultiscaleOptions(
                scales=oversized,
                max_iter_sub_minor=2000,
            ),
        )
    clean_peak = float(np.nanmax(np.abs(result.clean_fdf_arr)))
    dirty_peak = float(np.nanmax(np.abs(dirty)))
    assert np.all(np.isfinite(result.clean_fdf_arr))
    # Guard keeps the clean FDF from exceeding the dirty peak by a large factor.
    assert clean_peak < 2 * dirty_peak


def test_multiscale_stall_terminates() -> None:
    """An unreachable threshold must not run the major loop to max_iter: the
    stall guard stops once the residual peak stops improving (issue C)."""
    freq_hz = np.linspace(0.8e9, 2.2e9, 400)
    lsq = freq_to_lambda2(freq_hz)
    model = burn_slab(lsq, 0.5, 30, 20, 8.0)
    noisy = (
        model
        + RNG.normal(0, 0.02, freq_hz.size)
        + 1j * RNG.normal(0, 0.02, freq_hz.size)
    ).astype(np.complex128)
    synth = _run_synth(noisy, freq_hz)
    phi = synth.fdf_arrs["phi_arr_radm2"].to_numpy().astype(float)
    fwhm = float(synth.fdf_parameters["fwhm_rmsf_radm2"][0])
    phi2 = synth.rmsf_arrs["phi2_arr_radm2"].to_numpy().astype(float)
    dirty = synth.fdf_arrs["fdf_dirty_complex_arr"].to_numpy().astype(complex)
    rmsf = synth.rmsf_arrs["rmsf_complex_arr"].to_numpy().astype(complex)
    noise = float(synth.fdf_parameters["fdf_error_noise"][0])

    max_iter = 300
    with quiet_logs(logging.ERROR):
        result = rmclean(
            RMSynthArrays(
                dirty_fdf_arr=dirty,
                phi_arr_radm2=phi,
                rmsf_arr=rmsf,
                phi_double_arr_radm2=phi2,
                fwhm_rmsf_arr=np.array(fwhm),
            ),
            RMCleanOptions(
                mask=8 * noise,
                # unreachable threshold: forces a stall, not convergence
                threshold=1e-4 * noise,
                max_iter=max_iter,
                fdf_noise=noise,
            ),
            multiscale_options=MultiscaleOptions(max_iter_sub_minor=2000),
        )
    n_iter = int(np.ravel(result.clean_iter_arr)[0])
    assert 0 < n_iter < max_iter  # stalled early, did not grind to the cap
    assert np.all(np.isfinite(result.clean_fdf_arr))


def test_iteration_counts_reported_fairly() -> None:
    """The 1D tool reports both the minor-cycle count (`n_iter`) and the total
    component-placement count (`n_sub_minor_iter`). Single-scale: the two are
    equal (one component per minor iteration). Multiscale: the sub-minor total is
    at least the minor-cycle count, and is the number comparable to single-scale."""
    freq_hz = np.linspace(0.8e9, 2.2e9, 400)
    lsq = freq_to_lambda2(freq_hz)
    model = burn_slab(lsq, 0.5, 20, 30, 12.0)
    noisy = (
        model
        + RNG.normal(0, 0.02, freq_hz.size)
        + 1j * RNG.normal(0, 0.02, freq_hz.size)
    ).astype(np.complex128)
    synth = _run_synth(noisy, freq_hz)
    with quiet_logs(logging.ERROR):
        single = run_rmclean_from_synth(synth, auto_mask=8, auto_threshold=1)
        multi = run_rmclean_from_synth(
            synth, auto_mask=8, auto_threshold=1, multiscale=True
        )

    single_n = int(single.clean_parameters["n_iter"][0])
    single_sub = int(single.clean_parameters["n_sub_minor_iter"][0])
    multi_n = int(multi.clean_parameters["n_iter"][0])
    multi_sub = int(multi.clean_parameters["n_sub_minor_iter"][0])
    assert single_sub == single_n
    assert multi_sub >= multi_n


def test_multiscale_options_validation() -> None:
    with pytest.raises(ValueError, match="sub_minor_fraction"):
        MultiscaleOptions(sub_minor_fraction=1.5)
    with pytest.raises(ValueError, match="max_iter_sub_minor"):
        MultiscaleOptions(max_iter_sub_minor=0)


def test_coupling_identity() -> None:
    """A pure scale-s component is recovered exactly (amplitude and footprint)."""
    freq_hz = np.linspace(0.8e9, 2.2e9, 300)
    lsq = freq_to_lambda2(freq_hz)
    weight = np.ones_like(freq_hz)
    lam0 = float(np.sum(weight * lsq) / np.sum(weight))
    fwhm = float(get_fwhm_rmsf(lsq).fwhm_rmsf_radm2)
    phi = make_phi_arr(250.0, fwhm / 10)
    rmsf_res = get_rmsf_nufft(lsq, phi, weight, lam0)
    rmsf = rmsf_res.rmsf_cube.astype(complex)
    phi2 = rmsf_res.phi_double_arr_radm2

    scales = np.array([0.0, 2.0, 4.0])
    kernels = compute_scale_kernels(scales, rmsf, fwhm, phi2, "tapered_quad")
    si = 2
    peak_response = kernels.peak_response[si]

    # Unit scale-si component -> its dirty footprint.
    deltas = np.zeros(phi.size, dtype=complex)
    deltas[phi.size // 2] = 1.0
    dirty = _reconvolve_model(deltas, kernels.rmsf_conv_scale[si], phi, phi2)
    r_s = np.asarray(
        convolve_fdf_scale(scales[si], fwhm, dirty, phi2, "tapered_quad"), complex
    )
    peak = int(np.argmax(np.abs(r_s)))
    # Recovered amplitude R_s_peak / peak_response == 1 for a unit component.
    assert np.isclose(np.abs(r_s[peak]) / peak_response, 1.0, atol=1e-3)
    # Subtracting the recovered footprint zeroes the residual.
    d2 = np.zeros(phi.size, dtype=complex)
    d2[peak] = r_s[peak] / peak_response
    resid = dirty - _reconvolve_model(d2, kernels.rmsf_conv_scale[si], phi, phi2)
    assert np.nanmax(np.abs(resid)) < 1e-6


def test_multiscale_recovers_thick_flux() -> None:
    """Multiscale recovers a thin+thick source; converges without diverging."""
    freq_hz = np.linspace(0.8e9, 2.2e9, 400)
    lsq = freq_to_lambda2(freq_hz)
    model = burn_slab(lsq, 0.4, 10, -40, 0.0) + burn_slab(lsq, 0.5, 50, 30, 15.0)
    noisy = (
        model
        + RNG.normal(0, 0.02, freq_hz.size)
        + 1j * RNG.normal(0, 0.02, freq_hz.size)
    ).astype(np.complex128)
    synth = _run_synth(noisy, freq_hz)

    params = synth.fdf_parameters
    noise = float(params["fdf_error_noise"][0])
    fwhm = float(params["fwhm_rmsf_radm2"][0])
    phi = synth.fdf_arrs["phi_arr_radm2"].to_numpy().astype(float)
    phi2 = synth.rmsf_arrs["phi2_arr_radm2"].to_numpy().astype(float)
    dirty = synth.fdf_arrs["fdf_dirty_complex_arr"].to_numpy().astype(complex)
    rmsf = synth.rmsf_arrs["rmsf_complex_arr"].to_numpy().astype(complex)

    with quiet_logs(logging.ERROR):
        result = rmclean(
            RMSynthArrays(
                dirty_fdf_arr=dirty,
                phi_arr_radm2=phi,
                rmsf_arr=rmsf,
                phi_double_arr_radm2=phi2,
                fwhm_rmsf_arr=np.array(fwhm),
            ),
            RMCleanOptions(mask=8 * noise, threshold=1 * noise, fdf_noise=noise),
            multiscale_options=MultiscaleOptions(max_iter_sub_minor=2000),
        )
    mom0 = calc_faraday_moments(np.abs(result.clean_fdf_arr), phi, fwhm).mom0
    true_flux = 0.9
    # Recovers most of the true integrated flux, no runaway divergence.
    assert 0.75 * true_flux < mom0 < 1.3 * true_flux
    assert np.all(np.isfinite(result.clean_fdf_arr))
    assert int(np.ravel(result.clean_iter_arr)[0]) < 1000


def test_multiscale_thin_matches_single_scale() -> None:
    """On a Faraday-thin source multiscale agrees with single-scale RM-CLEAN."""
    freq_hz = np.linspace(0.8e9, 2.2e9, 400)
    lsq = freq_to_lambda2(freq_hz)
    model = burn_slab(lsq, 0.6, 30, 45, 0.0)
    noisy = (
        model
        + RNG.normal(0, 0.02, freq_hz.size)
        + 1j * RNG.normal(0, 0.02, freq_hz.size)
    ).astype(np.complex128)
    synth = _run_synth(noisy, freq_hz)
    phi = synth.fdf_arrs["phi_arr_radm2"].to_numpy().astype(float)
    fwhm = float(synth.fdf_parameters["fwhm_rmsf_radm2"][0])

    with quiet_logs(logging.ERROR):
        single = run_rmclean_from_synth(synth, auto_mask=8, auto_threshold=1)
        multi = run_rmclean_from_synth(
            synth, auto_mask=8, auto_threshold=1, multiscale=True
        )

    single_fdf = single.fdf_arrs["fdf_clean_complex_arr"].to_numpy().astype(complex)
    multi_fdf = multi.fdf_arrs["fdf_clean_complex_arr"].to_numpy().astype(complex)
    # Peak Faraday depth agrees within one channel.
    assert (
        abs(phi[np.argmax(np.abs(single_fdf))] - phi[np.argmax(np.abs(multi_fdf))])
        <= abs(phi[1] - phi[0]) + 1e-6
    )
    m0_single = calc_faraday_moments(np.abs(single_fdf), phi, fwhm).mom0
    m0_multi = calc_faraday_moments(np.abs(multi_fdf), phi, fwhm).mom0
    assert np.isclose(m0_single, m0_multi, rtol=0.3)


# Contiguous survey bands (MHz -> Hz) and their auto-grid outcome. The four
# narrowband bands have RMSF FWHM so wide that no extended scale (>= 4*FWHM)
# fits the recoverable Faraday range, so the grid degenerates to single-scale
# [0.0]. Only genuinely wide fractional bandwidth (racs_all, gmims) yields an
# extended grid. Values verified against the package synthesis utils.
_DEGENERATE = "degenerate"
_RACS_ALL = "racs_all"
_GMIMS = "gmims"
COVERAGES: list[tuple[str, float, float, str]] = [
    ("racs_low", 744e6, 1032e6, _DEGENERATE),
    ("possum_b1", 800e6, 1088e6, _DEGENERATE),
    ("meerkat_l", 886e6, 1682e6, _DEGENERATE),
    ("lofar", 120e6, 168e6, _DEGENERATE),
    ("racs_all", 744e6, 1800e6, _RACS_ALL),
    ("gmims_wide", 300e6, 1800e6, _GMIMS),
]


@pytest.mark.parametrize(
    ("lo", "hi", "kind"),
    [(lo, hi, kind) for _, lo, hi, kind in COVERAGES],
    ids=[name for name, *_ in COVERAGES],
)
def test_default_scales_coverage_matrix(lo: float, hi: float, kind: str) -> None:
    """Auto grid per survey band: narrowband degenerates, wideband extends.

    Locks the scale grid (root cause 3), not the scale selection: a single
    narrow band must be seen to collapse to single-scale so it is never again
    mistaken for a working multiscale run.
    """
    scales = _coverage_grid(lo, hi)
    if kind == _DEGENERATE:
        assert scales.tolist() == [0.0]
    elif kind == _RACS_ALL:
        # Just enough fractional bandwidth for extended scales; fine grid
        # anchors at 3 (make_fine_scales).
        assert len(scales) > 1
        assert scales[0] == 0.0
        assert scales[1] == 3.0
    elif kind == _GMIMS:
        # Wide band: delta plus >= 3 extended scales; fine grid anchors at 3
        # then doubles geometrically from 6.
        assert scales[0] == 0.0
        extended = scales[1:]
        assert len(extended) >= 3
        assert extended[0] == 3.0
        assert np.allclose(extended[1:], 6.0 * 2.0 ** np.arange(len(extended) - 1))


def test_default_scales_degeneration_warning(caplog: pytest.LogCaptureFixture) -> None:
    """The degenerate grid warns loudly; a wideband grid does not.

    The silent collapse to single-scale was the original wheel-spin; the warning
    is the shippable signal, so guard that it fires exactly when it should.
    """
    with caplog.at_level(logging.WARNING, logger="rmtools-lite"):
        degenerate = _coverage_grid(744e6, 1032e6, quiet=False)  # racs_low
    assert degenerate.tolist() == [0.0]
    assert "degenerated to [0.0]" in caplog.text
    assert "multiscale_scales" in caplog.text  # names the escape hatch

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="rmtools-lite"):
        wideband = _coverage_grid(300e6, 1800e6, quiet=False)  # gmims
    assert len(wideband) > 1
    assert "degenerated" not in caplog.text


def test_multiscale_wideband_preserves_point_flux() -> None:
    """On a wideband band, multiscale on a thin source keeps single-scale flux.

    racs_all's grid starts [0, 3, ...]: a true delta must stay on scale 0, so mom0 must
    match single-scale CLEAN. This is the shippable guarantee (point flux is not
    destroyed), distinct from the known thin-stealing on deeper grids.
    """
    rng = np.random.default_rng(20240717)
    freq_hz = np.linspace(744e6, 1800e6, 400)  # racs_all
    lsq = freq_to_lambda2(freq_hz)
    model = burn_slab(lsq, 0.6, 30, 45, 0.0)  # Faraday-thin point source
    noisy = (
        model
        + rng.normal(0, 0.02, freq_hz.size)
        + 1j * rng.normal(0, 0.02, freq_hz.size)
    ).astype(np.complex128)
    synth = _run_synth(noisy, freq_hz)
    phi = synth.fdf_arrs["phi_arr_radm2"].to_numpy().astype(float)
    fwhm = float(synth.fdf_parameters["fwhm_rmsf_radm2"][0])
    phi2 = synth.rmsf_arrs["phi2_arr_radm2"].to_numpy().astype(float)
    dirty = synth.fdf_arrs["fdf_dirty_complex_arr"].to_numpy().astype(complex)
    rmsf = synth.rmsf_arrs["rmsf_complex_arr"].to_numpy().astype(complex)
    noise = float(synth.fdf_parameters["fdf_error_noise"][0])
    phi_max_scale = float(np.pi / lsq.min())

    arrays = RMSynthArrays(
        dirty_fdf_arr=dirty,
        phi_arr_radm2=phi,
        rmsf_arr=rmsf,
        phi_double_arr_radm2=phi2,
        fwhm_rmsf_arr=np.array(fwhm),
    )
    clean_options = RMCleanOptions(mask=8 * noise, threshold=1 * noise, fdf_noise=noise)
    with quiet_logs(logging.ERROR):
        single = rmclean(arrays, clean_options)
        multi = rmclean(
            arrays,
            clean_options,
            multiscale_options=MultiscaleOptions(max_iter_sub_minor=2000),
            phi_max_scale_radm2=phi_max_scale,
        )
    m0_single = calc_faraday_moments(np.abs(single.clean_fdf_arr), phi, fwhm).mom0
    m0_multi = calc_faraday_moments(np.abs(multi.clean_fdf_arr), phi, fwhm).mom0
    # Point flux preserved: the thin source stays on scale 0. rtol 0.2 covers
    # noise realisation while still catching flux being destroyed or doubled.
    assert np.isclose(m0_single, m0_multi, rtol=0.2)
    assert np.all(np.isfinite(multi.clean_fdf_arr))


def test_multiscale_wideband_does_not_diverge() -> None:
    """The deep gmims grid [0, 3, 6, 12, 24] stays finite and bounded by the dirty peak.

    Exercises the full extended grid: even when scale selection is imperfect
    (known thin-stealing), the clean must not run away.
    """
    rng = np.random.default_rng(20240718)
    freq_hz = np.linspace(300e6, 1800e6, 400)  # gmims
    lsq = freq_to_lambda2(freq_hz)
    model = burn_slab(lsq, 0.5, 20, 30, 12.0)  # thick source engages extended scales
    noisy = (
        model
        + rng.normal(0, 0.02, freq_hz.size)
        + 1j * rng.normal(0, 0.02, freq_hz.size)
    ).astype(np.complex128)
    synth = _run_synth(noisy, freq_hz)
    phi = synth.fdf_arrs["phi_arr_radm2"].to_numpy().astype(float)
    fwhm = float(synth.fdf_parameters["fwhm_rmsf_radm2"][0])
    phi2 = synth.rmsf_arrs["phi2_arr_radm2"].to_numpy().astype(float)
    dirty = synth.fdf_arrs["fdf_dirty_complex_arr"].to_numpy().astype(complex)
    rmsf = synth.rmsf_arrs["rmsf_complex_arr"].to_numpy().astype(complex)
    noise = float(synth.fdf_parameters["fdf_error_noise"][0])
    phi_max_scale = float(np.pi / lsq.min())

    with quiet_logs(logging.ERROR):
        multi = rmclean(
            RMSynthArrays(
                dirty_fdf_arr=dirty,
                phi_arr_radm2=phi,
                rmsf_arr=rmsf,
                phi_double_arr_radm2=phi2,
                fwhm_rmsf_arr=np.array(fwhm),
            ),
            RMCleanOptions(mask=8 * noise, threshold=1 * noise, fdf_noise=noise),
            multiscale_options=MultiscaleOptions(max_iter_sub_minor=2000),
            phi_max_scale_radm2=phi_max_scale,
        )
    clean_peak = float(np.nanmax(np.abs(multi.clean_fdf_arr)))
    dirty_peak = float(np.nanmax(np.abs(dirty)))
    assert np.all(np.isfinite(multi.clean_fdf_arr))
    # Bounded by the dirty peak (factor 2 leaves headroom for reconvolution).
    assert clean_peak < 2 * dirty_peak


def _clean_single_and_hybrid(
    sim_dirty: NDArray[np.complex128],
    rmsf: NDArray[np.complex128],
    phi: NDArray[np.float64],
    phi_double: NDArray[np.float64],
    fwhm: float,
    noise: float,
    phi_max_scale: float,
) -> tuple[RMCleanResults, RMCleanResults]:
    """Single-scale and hybrid-multiscale cleans at matched depth."""
    arrays = RMSynthArrays(
        dirty_fdf_arr=sim_dirty,
        phi_arr_radm2=phi,
        rmsf_arr=rmsf,
        phi_double_arr_radm2=phi_double,
        fwhm_rmsf_arr=np.array([fwhm]),
    )
    clean_options = RMCleanOptions(
        mask=5.0 * noise, threshold=3.0 * noise, fdf_noise=noise
    )
    with quiet_logs(logging.ERROR):
        single = rmclean(arrays, clean_options)
        multi = rmclean(
            arrays,
            clean_options,
            multiscale_options=MultiscaleOptions(),
            phi_max_scale_radm2=phi_max_scale,
        )
    return single, multi


def _model_shape_err(
    model: NDArray[np.complex128], truth: NDArray[np.complex128]
) -> float:
    """Scale-free rms of |model| against best-fit-amplitude |truth|."""
    a = np.abs(model)
    t = np.abs(truth)
    amp = float((a * t).sum() / (t * t).sum())
    return float(np.sqrt(np.mean((a - amp * t) ** 2)) / (amp * t.max()))


def test_hybrid_model_quality_wideband() -> None:
    """On a wide band (300-1800 MHz) a thick Gaussian's raw component model is
    much closer to truth under hybrid selection than single-scale's spike comb.

    The full benchmark bounds this ratio at 0.6 over 24 realisations; loosened
    to 0.7 here for 4 realisations with distinct seeds.
    """
    freqs = np.linspace(300e6, 1800e6, 300)
    geom = build_geometry(freqs)
    spec = gauss(1.5, amp=1.0)
    truth = build_model_fdf(spec, geom.phi_arr_radm2, geom.fwhm)
    pms = float(np.pi / np.nanmin(geom.lambda_sq_arr_m2))
    ratios = []
    for i in range(4):
        rng = np.random.default_rng(800000 + i)
        sim = simulate_fdf(spec, freqs, rng=rng, sn=24.0, geometry=geom)
        single, multi = _clean_single_and_hybrid(
            sim.dirty_fdf,
            sim.rmsf_arr,
            geom.phi_arr_radm2,
            geom.phi_double_arr_radm2,
            geom.fwhm,
            sim.fdf_noise,
            pms,
        )
        err_single = _model_shape_err(np.asarray(single.model_fdf_arr).ravel(), truth)
        err_multi = _model_shape_err(np.asarray(multi.model_fdf_arr).ravel(), truth)
        ratios.append(err_multi / err_single)
    assert float(np.median(ratios)) <= 0.7


def test_hybrid_delta_steps_parity() -> None:
    """On a bright offset delta, hybrid multiscale does single-scale work: the
    same flux and effectively the same sub-minor step count (within the one-step
    slack of the adaptive two-phase clean), all on the delta scale."""
    freqs = np.linspace(300e6, 1800e6, 300)
    geom = build_geometry(freqs)
    spec = delta(center_fwhm=3.3, amp=1.0)
    pms = float(np.pi / np.nanmin(geom.lambda_sq_arr_m2))
    for i in range(4):
        rng = np.random.default_rng(810000 + i)
        sim = simulate_fdf(spec, freqs, rng=rng, sn=24.0, geometry=geom)
        single, multi = _clean_single_and_hybrid(
            sim.dirty_fdf,
            sim.rmsf_arr,
            geom.phi_arr_radm2,
            geom.phi_double_arr_radm2,
            geom.fwhm,
            sim.fdf_noise,
            pms,
        )
        steps_single = int(np.ravel(single.sub_minor_iter_arr)[0])
        steps_multi = int(np.ravel(multi.sub_minor_iter_arr)[0])
        assert abs(steps_multi - steps_single) <= 1
        m0_single = calc_faraday_moments(
            np.abs(single.clean_fdf_arr), geom.phi_arr_radm2, geom.fwhm
        ).mom0
        m0_multi = calc_faraday_moments(
            np.abs(multi.clean_fdf_arr), geom.phi_arr_radm2, geom.fwhm
        ).mom0
        # Sub-percent, not bit-identical: the adaptive two-phase clean restores
        # the delta a hair differently from single-scale. Still catches flux
        # being destroyed or doubled.
        assert np.isclose(m0_single, m0_multi, rtol=0.02)


def test_explicit_scales_sorted_and_require_delta() -> None:
    # Unsorted input is normalised to ascending order (selectors index scales[0]
    # as the delta scale and assume ascending).
    opts = MultiscaleOptions(scales=np.array([8.0, 0.0, 3.0]))
    assert opts.scales is not None
    np.testing.assert_array_equal(opts.scales, [0.0, 3.0, 8.0])

    # Missing the delta (0) scale is rejected.
    with pytest.raises(ValueError, match="must include 0"):
        MultiscaleOptions(scales=np.array([3.0, 8.0]))

    # Empty is still rejected.
    with pytest.raises(ValueError, match="non-empty"):
        MultiscaleOptions(scales=np.array([]))
