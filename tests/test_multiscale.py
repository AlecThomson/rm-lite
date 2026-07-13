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
    _reconvolve_model,
    compute_scale_kernels,
    convolve_fdf_scale,
    default_scales,
    make_scales,
    rmclean,
    scale_bias_function,
)
from rm_lite.utils.logging import quiet_logs
from rm_lite.utils.synthesis import (
    calc_faraday_moments,
    freq_to_lambda2,
    get_fwhm_rmsf,
    get_rmsf_nufft,
    make_phi_arr,
)

RNG = np.random.default_rng(1234)


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


def test_scale_bias_function() -> None:
    scales = np.array([0.0, 1.0, 2.0, 4.0])
    bias = scale_bias_function(scales, 0.6)
    assert bias[0] == 1.0
    # Lower scale_bias favours larger scales: weights increase with scale.
    assert np.all(np.diff(bias[1:]) > 0)
    # A single scale gets unit weight.
    assert scale_bias_function(np.array([0.0]), 0.6)[0] == 1.0


def test_make_scales() -> None:
    scales = make_scales(10.0)
    assert scales[0] == 0.0
    assert np.allclose(scales, [0, 1, 2, 4, 8])
    assert len(make_scales(10.0, n_scales=3)) == 3


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
            dirty_fdf_arr=dirty,
            phi_arr_radm2=phi,
            rmsf_arr=rmsf,
            phi_double_arr_radm2=phi2,
            fwhm_rmsf_arr=np.array(fwhm),
            mask=8 * noise,
            threshold=3 * noise,
            multiscale=True,
            multiscale_scales=oversized,
            multiscale_max_iter_sub_minor=2000,
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
            dirty_fdf_arr=dirty,
            phi_arr_radm2=phi,
            rmsf_arr=rmsf,
            phi_double_arr_radm2=phi2,
            fwhm_rmsf_arr=np.array(fwhm),
            mask=8 * noise,
            threshold=1e-4 * noise,  # unreachable: forces a stall, not convergence
            max_iter=max_iter,
            multiscale=True,
            multiscale_max_iter_sub_minor=2000,
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
            dirty_fdf_arr=dirty,
            phi_arr_radm2=phi,
            rmsf_arr=rmsf,
            phi_double_arr_radm2=phi2,
            fwhm_rmsf_arr=np.array(fwhm),
            mask=8 * noise,
            threshold=1 * noise,
            multiscale=True,
            multiscale_max_iter_sub_minor=2000,
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
