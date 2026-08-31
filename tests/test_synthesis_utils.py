"""Tests for the synthesis utilities"""

from __future__ import annotations

import dask.array as da
import numpy as np
import pytest
from numpy.random import Generator
from numpy.typing import NDArray
from rm_lite.utils.fitting import StokesIFitOptions, gaussian, gaussian_integrand
from rm_lite.utils.synthesis import (
    StokesData,
    calc_faraday_moments,
    calc_faraday_peaks,
    create_fractional_spectra,
    debias_fdf,
    freq_to_lambda2,
    lambda2_to_freq,
    make_double_phi_arr,
    make_phi_arr,
)

RNG: Generator = np.random.default_rng()


def test_freq_lsq_freq():
    freqs = RNG.uniform(1e6, 10e9, 1000)

    assert np.allclose(lambda2_to_freq(freq_to_lambda2(freqs)), freqs)


def test_lsq_freq_lsq():
    lambda2s = RNG.uniform(1e-6, 1, 1000)

    assert np.allclose(freq_to_lambda2(lambda2_to_freq(lambda2s)), lambda2s)


def test_phi_arr():
    for max_val in [100, 1000, 10000]:
        for step_val in [1, 10, 100]:
            phi_one = make_phi_arr(max_val, step_val)
            assert len(phi_one) == max_val // step_val * 2 + 1
            assert np.isclose(np.max(phi_one), max_val)
            assert np.isclose(np.min(phi_one), -max_val)
            # Check that middle pixel is 0
            assert phi_one[len(phi_one) // 2] == 0


def test_doubel_phi_arr():
    for max_val in [100, 1000, 10000]:
        for step_val in [1, 10, 100]:
            phi_arr = make_phi_arr(max_val, step_val)
            phi_double_arr = make_double_phi_arr(phi_arr)
            assert len(phi_double_arr) == len(phi_arr) * 2 + 1
            assert np.isclose(np.max(phi_double_arr), (max_val * 2) + step_val)
            assert np.isclose(np.min(phi_double_arr), -(max_val * 2) - step_val)
            # Check that middle pixel is 0
            assert phi_double_arr[len(phi_double_arr) // 2] == 0


def test_moments_unresolved_gaussian():
    phi_arr = make_phi_arr(1000, 1)
    fwhm = 60.0
    amplitude = 3.0
    center = 123.0
    fdf = gaussian(phi_arr, amplitude, center, fwhm=fwhm).astype(np.complex128)

    moments = calc_faraday_moments(fdf, phi_arr, fwhm)

    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    assert np.isclose(float(moments.mom0), amplitude, rtol=1e-3)
    assert np.isclose(float(moments.mom1), center, rtol=1e-3)
    assert np.isclose(float(moments.mom2), sigma, rtol=1e-3)


def test_moments_ignore_phase():
    phi_arr = make_phi_arr(1000, 1)
    fwhm = 60.0
    fdf_real = gaussian(phi_arr, 2.0, -50.0, fwhm=fwhm).astype(np.complex128)
    fdf_rotated = fdf_real * np.exp(2j * 0.5 * phi_arr)

    moments_real = calc_faraday_moments(fdf_real, phi_arr, fwhm)
    moments_rotated = calc_faraday_moments(fdf_rotated, phi_arr, fwhm)

    assert np.allclose(moments_real, moments_rotated)


def test_moments_delta_function():
    phi_arr = make_phi_arr(100, 2)
    fwhm = 20.0
    amplitude = 5.0
    fdf = np.zeros_like(phi_arr, dtype=np.complex128)
    index = 60
    fdf[index] = amplitude

    moments = calc_faraday_moments(fdf, phi_arr, fwhm)

    delta_phi = phi_arr[1] - phi_arr[0]
    assert np.isclose(
        float(moments.mom0),
        amplitude * delta_phi / gaussian_integrand(1.0, fwhm=fwhm),
    )
    assert np.isclose(float(moments.mom1), phi_arr[index])
    assert np.isclose(float(moments.mom2), 0.0)


def test_moments_nd_and_axis():
    phi_arr = make_phi_arr(1000, 1)
    fwhm = 60.0
    fdf_1d = gaussian(phi_arr, 3.0, 123.0, fwhm=fwhm).astype(np.complex128)
    moments_1d = calc_faraday_moments(fdf_1d, phi_arr, fwhm)

    fdf_3d = np.tile(fdf_1d[:, np.newaxis, np.newaxis], (1, 2, 3))
    moments_3d = calc_faraday_moments(fdf_3d, phi_arr, fwhm)
    for moment_3d, moment_1d in zip(moments_3d, moments_1d, strict=True):
        assert moment_3d.shape == (2, 3)
        assert np.allclose(moment_3d, moment_1d)

    fdf_last = np.moveaxis(fdf_3d, 0, -1)
    moments_last = calc_faraday_moments(fdf_last, phi_arr, fwhm, axis=-1)
    for moment_last, moment_3d in zip(moments_last, moments_3d, strict=True):
        assert np.allclose(moment_last, moment_3d)


def test_moments_threshold():
    phi_arr = make_phi_arr(1000, 1)
    fwhm = 60.0
    strong = gaussian(phi_arr, 3.0, 100.0, fwhm=fwhm)
    weak = gaussian(phi_arr, 0.05, -300.0, fwhm=fwhm)
    fdf = (strong + weak).astype(np.complex128)

    moments = calc_faraday_moments(fdf, phi_arr, fwhm, threshold=0.5)
    moments_clean = calc_faraday_moments(
        strong.astype(np.complex128), phi_arr, fwhm, threshold=0.5
    )

    assert np.allclose(moments, moments_clean)
    assert np.isclose(float(moments.mom1), 100.0, atol=1.0)


def test_moments_auto_threshold():
    rng = np.random.default_rng(42)
    phi_arr = make_phi_arr(1000, 1)
    fwhm = 60.0
    center = 100.0
    noise_sigma = 0.1
    signal = gaussian(phi_arr, 5.0, center, fwhm=fwhm)
    noise = rng.normal(0, noise_sigma, phi_arr.shape) + 1j * rng.normal(
        0, noise_sigma, phi_arr.shape
    )
    fdf = signal + noise

    moments = calc_faraday_moments(fdf, phi_arr, fwhm, auto_threshold_sigma=5.0)

    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    assert np.isclose(float(moments.mom1), center, atol=1.0)
    assert np.isclose(float(moments.mom2), sigma, rtol=0.2)


def test_moments_mutually_exclusive_thresholds():
    phi_arr = make_phi_arr(100, 1)
    fdf = np.ones_like(phi_arr, dtype=np.complex128)
    with pytest.raises(ValueError, match="mutually exclusive"):
        calc_faraday_moments(
            fdf, phi_arr, 20.0, threshold=1.0, auto_threshold_sigma=5.0
        )


def test_moments_shape_mismatch():
    phi_arr = make_phi_arr(100, 1)
    fdf = np.ones(len(phi_arr) + 1, dtype=np.complex128)
    with pytest.raises(ValueError, match="length"):
        calc_faraday_moments(fdf, phi_arr, 20.0)


def test_moments_empty_spectrum():
    phi_arr = make_phi_arr(100, 1)
    fdf = np.zeros((len(phi_arr), 2), dtype=np.complex128)
    fdf[:, 1] = gaussian(phi_arr, 1.0, 0.0, fwhm=20.0)

    moments = calc_faraday_moments(fdf, phi_arr, 20.0)

    assert moments.mom0[0] == 0.0
    assert np.isnan(moments.mom1[0])
    assert np.isnan(moments.mom2[0])
    assert np.isfinite(moments.mom1[1])


def test_moments_dask():
    phi_arr = make_phi_arr(1000, 1)
    fwhm = 60.0
    fdf_1d = gaussian(phi_arr, 3.0, 123.0, fwhm=fwhm).astype(np.complex128)
    fdf_3d = np.tile(fdf_1d[:, np.newaxis, np.newaxis], (1, 4, 4))
    fdf_dask = da.from_array(fdf_3d, chunks=(len(phi_arr), 2, 2))

    moments_numpy = calc_faraday_moments(fdf_3d, phi_arr, fwhm)
    moments_dask = calc_faraday_moments(fdf_dask, phi_arr, fwhm)

    for moment_dask, moment_numpy in zip(moments_dask, moments_numpy, strict=True):
        assert isinstance(moment_dask, da.Array)
        assert np.allclose(moment_dask.compute(), moment_numpy)

    moments_dask_auto = calc_faraday_moments(
        fdf_dask, phi_arr, fwhm, auto_threshold_sigma=5.0
    )
    moments_numpy_auto = calc_faraday_moments(
        fdf_3d, phi_arr, fwhm, auto_threshold_sigma=5.0
    )
    for moment_dask, moment_numpy in zip(
        moments_dask_auto, moments_numpy_auto, strict=True
    ):
        assert isinstance(moment_dask, da.Array)
        assert np.allclose(moment_dask.compute(), moment_numpy)


def thin_fdf(
    phi_arr_radm2: NDArray[np.float64],
    fwhm: float,
    amplitude: float,
    rm_radm2: float,
    pa0_deg: float,
    lam_sq_0_m2: float,
) -> NDArray[np.complex128]:
    """A Faraday-thin FDF: RMSF-shaped lobe, angle 2*(pa0 + RM*lam_sq_0)."""
    lobe = gaussian(phi_arr_radm2, amplitude, rm_radm2, fwhm=fwhm)
    phase = np.exp(2j * (np.deg2rad(pa0_deg) + rm_radm2 * lam_sq_0_m2))
    return np.asarray(lobe * phase, dtype=np.complex128)


def test_peaks_recover_a_thin_source():
    phi_arr = make_phi_arr(300.0, 5.0)
    fwhm, amplitude, rm, pa0, lam_sq_0 = 40.0, 2.0, 37.3, 30.0, 0.05
    fdf = thin_fdf(phi_arr, fwhm, amplitude, rm, pa0, lam_sq_0)

    peaks = calc_faraday_peaks(
        fdf,
        phi_arr,
        fwhm,
        fdf_error=0.01,
        lam_sq_0_m2=lam_sq_0,
        lambda_sq_arr_m2=np.linspace(0.03, 0.07, 36),
    )

    # The parabola is sub-sample: the RM lands well inside the 5 rad/m^2 grid.
    assert np.isclose(float(peaks.peak_pi), amplitude, rtol=1e-3)
    assert np.isclose(float(peaks.peak_rm_radm2), rm, atol=0.5)
    assert np.isclose(float(peaks.peak_pa0_deg), pa0, atol=0.5)
    # Angle at the reference lambda^2 is the intrinsic angle plus the rotation.
    assert np.isclose(
        float(peaks.peak_pa_deg), (pa0 + np.degrees(rm * lam_sq_0)) % 180, atol=0.5
    )
    # Errors all scale as 1/SNR, so a 200-sigma peak is pinned down tightly.
    assert np.isclose(float(peaks.peak_pi_error), 0.01)
    assert 0 < float(peaks.peak_rm_error_radm2) < 1.0
    assert 0 < float(peaks.peak_pa_error_deg) < 1.0
    assert 0 < float(peaks.peak_pa0_error_deg) < 5.0
    # Debiasing a high-SNR peak barely moves it, and never up.
    assert 0 < float(peaks.peak_pi_debias) <= float(peaks.peak_pi)


def test_peaks_nd_and_axis():
    phi_arr = make_phi_arr(300.0, 5.0)
    fwhm = 40.0
    fdf_1d = thin_fdf(phi_arr, fwhm, 2.0, 37.3, 30.0, 0.05)
    peaks_1d = calc_faraday_peaks(fdf_1d, phi_arr, fwhm, fdf_error=0.01)

    fdf_3d = np.tile(fdf_1d[:, np.newaxis, np.newaxis], (1, 2, 3))
    peaks_3d = calc_faraday_peaks(fdf_3d, phi_arr, fwhm, fdf_error=0.01)
    for peak_3d, peak_1d in zip(peaks_3d, peaks_1d, strict=True):
        assert peak_3d.shape == (2, 3)
        assert np.allclose(peak_3d, peak_1d, equal_nan=True)

    fdf_last = np.moveaxis(fdf_3d, 0, -1)
    peaks_last = calc_faraday_peaks(fdf_last, phi_arr, fwhm, fdf_error=0.01, axis=-1)
    for peak_last, peak_3d in zip(peaks_last, peaks_3d, strict=True):
        assert np.allclose(peak_last, peak_3d, equal_nan=True)


def test_peaks_dask_matches_numpy():
    phi_arr = make_phi_arr(300.0, 5.0)
    fwhm = 40.0
    fdf_1d = thin_fdf(phi_arr, fwhm, 2.0, 37.3, 30.0, 0.05)
    fdf_3d = np.tile(fdf_1d[:, np.newaxis, np.newaxis], (1, 4, 4))
    # Chunked along Faraday depth too: the peak is a reduction, not a per-chunk
    # search, so a split axis must give the same answer.
    fdf_dask = da.from_array(fdf_3d, chunks=(len(phi_arr) // 2, 2, 2))

    peaks_numpy = calc_faraday_peaks(fdf_3d, phi_arr, fwhm, fdf_error=0.01)
    peaks_dask = calc_faraday_peaks(fdf_dask, phi_arr, fwhm, fdf_error=0.01)

    for peak_dask, peak_numpy in zip(peaks_dask, peaks_numpy, strict=True):
        assert isinstance(peak_dask, da.Array)
        assert np.allclose(peak_dask.compute(), peak_numpy, equal_nan=True)


def test_peaks_without_a_peak():
    phi_arr = make_phi_arr(100.0, 1.0)
    flat = np.zeros((len(phi_arr), 3), dtype=np.complex128)
    all_nan = np.full((len(phi_arr), 3), np.nan, dtype=np.complex128)
    edge = np.zeros((len(phi_arr), 3), dtype=np.complex128)
    edge[0] = 5.0

    for fdf in (flat, all_nan, edge):
        peaks = calc_faraday_peaks(fdf, phi_arr, 20.0, fdf_error=0.01)
        for name, peak in peaks._asdict().items():
            # The noise is a property of the observation, not of a detection,
            # so it is reported whether or not there is a peak to go with it.
            if name == "peak_pi_error":
                assert (peak == 0.01).all()
            else:
                assert np.isnan(peak).all()


def test_peaks_threshold_blanks_faint_peaks():
    phi_arr = make_phi_arr(300.0, 5.0)
    fwhm = 40.0
    bright = thin_fdf(phi_arr, fwhm, 2.0, 37.3, 30.0, 0.05)
    faint = thin_fdf(phi_arr, fwhm, 0.05, -100.0, 30.0, 0.05)
    fdf = np.stack([bright, faint], axis=-1)

    peaks = calc_faraday_peaks(fdf, phi_arr, fwhm, threshold=0.5)

    assert np.isfinite(peaks.peak_pi[0])
    assert np.isnan(peaks.peak_pi[1])
    assert np.isnan(peaks.peak_rm_radm2[1])
    assert np.isnan(peaks.peak_pa_deg[1])


def test_peaks_blank_what_they_were_not_given():
    phi_arr = make_phi_arr(300.0, 5.0)
    fwhm = 40.0
    fdf = thin_fdf(phi_arr, fwhm, 2.0, 37.3, 30.0, 0.05)

    bare = calc_faraday_peaks(fdf, phi_arr, fwhm)
    assert np.isfinite(bare.peak_pi)
    assert np.isfinite(bare.peak_rm_radm2)
    assert np.isnan(bare.peak_pi_error)
    assert np.isnan(bare.peak_pi_debias)
    assert np.isnan(bare.peak_rm_error_radm2)
    assert np.isnan(bare.peak_pa_error_deg)
    # An angle needs no noise estimate, an intrinsic angle needs a reference.
    assert np.isfinite(bare.peak_pa_deg)
    assert np.isnan(bare.peak_pa0_deg)

    no_channels = calc_faraday_peaks(
        fdf, phi_arr, fwhm, fdf_error=0.01, lam_sq_0_m2=0.05
    )
    assert np.isfinite(no_channels.peak_pa0_deg)
    assert np.isnan(no_channels.peak_pa0_error_deg)


def test_peaks_debias_leaves_low_snr_peaks_alone():
    phi_arr = make_phi_arr(300.0, 5.0)
    fwhm = 40.0
    fdf = thin_fdf(phi_arr, fwhm, 2.0, 37.3, 30.0, 0.05)

    # Noise a third of the peak: below the 5-sigma cut, so no correction.
    peaks = calc_faraday_peaks(fdf, phi_arr, fwhm, fdf_error=0.67)
    assert float(peaks.peak_pi_debias) == float(peaks.peak_pi)

    peaks = calc_faraday_peaks(fdf, phi_arr, fwhm, fdf_error=0.1)
    assert float(peaks.peak_pi_debias) < float(peaks.peak_pi)


def test_peaks_reject_real_input():
    phi_arr = make_phi_arr(100.0, 1.0)
    fdf = np.abs(thin_fdf(phi_arr, 20.0, 1.0, 0.0, 30.0, 0.05))
    with pytest.raises(ValueError, match="must be complex"):
        calc_faraday_peaks(fdf, phi_arr, 20.0)


def test_peaks_reject_a_bad_phi_arr():
    fdf = np.ones(4, dtype=np.complex128)
    for phi_arr in (np.array([1.0]), np.ones((4, 2))):
        with pytest.raises(ValueError, match="1D with at least two samples"):
            calc_faraday_peaks(fdf, phi_arr, 20.0)


def test_peaks_shape_mismatch():
    phi_arr = make_phi_arr(100.0, 1.0)
    fdf = np.ones(len(phi_arr) + 1, dtype=np.complex128)
    with pytest.raises(ValueError, match="length"):
        calc_faraday_peaks(fdf, phi_arr, 20.0)


def test_debias_fdf_noise_only():
    rng = np.random.default_rng(1234)
    sigma = 0.1
    noise = rng.normal(0, sigma, (2, 100, 100)) + 1j * rng.normal(
        0, sigma, (2, 100, 100)
    )
    phi_arr = np.array([-10.0, 10.0])

    debiased = debias_fdf(noise, phi_arr, lam_sq_0_m2=0.0)

    # abs() has a Rayleigh mean of sigma*sqrt(pi/2); the projection should be
    # near-zero-mean noise (a small residual bias remains, per the paper)
    rayleigh_mean = float(np.mean(np.abs(noise)))
    assert np.isclose(rayleigh_mean, sigma * np.sqrt(np.pi / 2), rtol=0.05)
    assert np.abs(np.mean(debiased)) < 0.1 * rayleigh_mean
    assert np.isclose(np.std(debiased), sigma, rtol=0.1)


def test_debias_fdf_recovers_signal():
    rng = np.random.default_rng(1234)
    sigma = 0.05
    signal = 2.0
    phi_arr = np.array([-10.0, 10.0])
    # Angles either side of the -pi/pi wrap to exercise the component filtering.
    # theta = 0 also discriminates the paper projection (U sin + Q cos) from a
    # Q/U-swapped one, which would return ~0 rather than the signal.
    for theta in [0.0, np.pi / 3, np.pi - 0.05, -np.pi + 0.05]:
        fdf = signal * np.exp(1j * theta) + (
            rng.normal(0, sigma, (2, 64, 64)) + 1j * rng.normal(0, sigma, (2, 64, 64))
        )
        debiased = debias_fdf(fdf, phi_arr, lam_sq_0_m2=0.0)
        assert np.isclose(np.mean(debiased), signal, rtol=0.02)


def gradient_rm_cube(
    rng: Generator,
) -> tuple[NDArray[np.complex128], NDArray[np.float64], float, NDArray[np.float64]]:
    """Thin-source FDF cube with a steep RM gradient across the map."""
    lam_sq_0_m2 = 0.1
    fwhm = 60.0
    phi_arr = make_phi_arr(150, 10)
    ny = nx = 32
    # RM gradient of ~5 rad/m^2 per pixel: the FDF angle spins ~1 rad per
    # pixel, breaking the smooth-angle assumption of the underotated filter
    rm_map = np.broadcast_to(np.linspace(-80, 80, nx), (ny, nx))
    psi0 = 0.5
    profile = gaussian(phi_arr[:, None, None], 1.0, rm_map[None], fwhm=fwhm)
    fdf = profile * np.exp(2j * (psi0 + rm_map[None] * lam_sq_0_m2))
    fdf += rng.normal(0, 0.02, fdf.shape) + 1j * rng.normal(0, 0.02, fdf.shape)
    return fdf, phi_arr, lam_sq_0_m2, np.sum(profile, axis=0)


def test_debias_fdf_rm_gradient():
    rng = np.random.default_rng(1234)
    fdf, phi_arr, lam_sq_0_m2, signal_total = gradient_rm_cube(rng)

    debiased = debias_fdf(fdf, phi_arr, lam_sq_0_m2=lam_sq_0_m2)
    plain = debias_fdf(fdf, phi_arr, lam_sq_0_m2=0.0)

    # The peak-RM derotation makes the filtered angle spatially smooth, so
    # the signed sum recovers the true signal despite the RM gradient...
    assert np.allclose(np.sum(debiased, axis=0), signal_total, rtol=0.07)
    # ...while the original (underotated) method depolarises here
    assert np.median(np.sum(plain, axis=0) / signal_total) < 0.9


def test_debias_fdf_dask_matches_numpy():
    rng = np.random.default_rng(1234)
    fdf = rng.normal(0, 1, (4, 32, 32)) + 1j * rng.normal(0, 1, (4, 32, 32))
    fdf[:2] += 3.0 * np.exp(1j * 1.0)
    phi_arr = np.arange(4, dtype=np.float64) * 10.0

    debiased_numpy = debias_fdf(fdf, phi_arr, lam_sq_0_m2=0.1)
    debiased_dask = debias_fdf(
        da.from_array(fdf, chunks=(4, 16, 16)), phi_arr, lam_sq_0_m2=0.1
    )

    assert isinstance(debiased_dask, da.Array)
    assert np.allclose(debiased_dask.compute(), debiased_numpy)

    with pytest.raises(ValueError, match="single chunk"):
        debias_fdf(da.from_array(fdf, chunks=(2, 16, 16)), phi_arr, lam_sq_0_m2=0.1)


def test_debias_fdf_validation():
    fdf = np.ones((4, 8, 8), dtype=np.complex128)
    phi_arr = np.arange(4, dtype=np.float64) * 10.0
    with pytest.raises(ValueError, match="odd"):
        debias_fdf(fdf, phi_arr, lam_sq_0_m2=0.0, filter_size=4)
    with pytest.raises(ValueError, match="spatial"):
        debias_fdf(np.ones(4, dtype=np.complex128), phi_arr, lam_sq_0_m2=0.0)
    with pytest.raises(ValueError, match="length"):
        debias_fdf(fdf, np.arange(5, dtype=np.float64) * 10.0, lam_sq_0_m2=0.0)


def test_moments_signed_real_input():
    phi_arr = make_phi_arr(100, 1)
    fwhm = 20.0
    signed = gaussian(phi_arr, 2.0, 0.0, fwhm=fwhm) - 0.5

    moments = calc_faraday_moments(signed, phi_arr, fwhm)

    delta_phi = phi_arr[1] - phi_arr[0]
    expected = np.sum(signed) * delta_phi / gaussian_integrand(1.0, fwhm=fwhm)
    assert np.isclose(float(moments.mom0), expected)
    # abs() of the same values gives a strictly larger mom0
    folded = calc_faraday_moments(np.abs(signed).astype(np.complex128), phi_arr, fwhm)
    assert float(folded.mom0) > float(moments.mom0)


def test_moments_debias_option():
    rng = np.random.default_rng(1234)
    fdf, phi_arr, lam_sq_0_m2, signal_total = gradient_rm_cube(rng)
    fwhm = 60.0

    moments = calc_faraday_moments(
        fdf, phi_arr, fwhm, debias=True, lam_sq_0_m2=lam_sq_0_m2
    )
    manual = calc_faraday_moments(
        debias_fdf(fdf, phi_arr, lam_sq_0_m2=lam_sq_0_m2), phi_arr, fwhm
    )

    assert np.allclose(moments.mom0, manual.mom0, equal_nan=True)
    delta_phi = phi_arr[1] - phi_arr[0]
    expected_mom0 = signal_total * delta_phi / gaussian_integrand(1.0, fwhm=fwhm)
    assert np.allclose(moments.mom0, expected_mom0, rtol=0.1)

    with pytest.raises(ValueError, match="required"):
        calc_faraday_moments(fdf, phi_arr, fwhm, debias=True)
    with pytest.raises(ValueError, match="not supported"):
        calc_faraday_moments(
            fdf,
            phi_arr,
            fwhm,
            debias=True,
            lam_sq_0_m2=lam_sq_0_m2,
            auto_threshold_sigma=5.0,
        )


def test_moments_auto_threshold_dask_multichunk_guard():
    # calc_faraday_moments' auto-threshold noise estimate reduces over the
    # Faraday depth axis, which dask cannot do across chunks, so it must raise a
    # clear error rather than an opaque dask failure (mirrors debias_fdf).
    phi_arr = make_phi_arr(200, 1)
    fdf = np.zeros((len(phi_arr), 4, 4), dtype=np.complex128)
    fdf_dask = da.from_array(fdf, chunks=(50, 4, 4))
    with pytest.raises(ValueError, match="single chunk"):
        calc_faraday_moments(fdf_dask, phi_arr, 20.0, auto_threshold_sigma=5.0)


def test_moments_auto_threshold_broad_source():
    # A broad component fills most of the band, so median(|FDF|) is dominated
    # by signal and would inflate the noise estimate past the peak, masking the
    # source entirely. The mad_std estimate on the real/imag parts stays robust.
    rng = np.random.default_rng(7)
    phi_arr = make_phi_arr(150, 1)
    fwhm = 200.0
    noise_sigma = 0.05
    signal = gaussian(phi_arr, 5.0, 0.0, fwhm=fwhm)
    noise = rng.normal(0, noise_sigma, phi_arr.shape) + 1j * rng.normal(
        0, noise_sigma, phi_arr.shape
    )
    fdf = signal + noise

    moments = calc_faraday_moments(fdf, phi_arr, fwhm, auto_threshold_sigma=5.0)

    assert np.isfinite(moments.mom1)
    assert float(moments.mom0) > 0
    assert abs(float(moments.mom1)) < 30.0


def test_moments_debias_threshold_guard():
    # A positive threshold on the signed debiased amplitudes would clip the
    # negative noise samples the debias relies on, so it must be rejected.
    phi_arr = make_phi_arr(50, 1)
    fdf = np.ones((len(phi_arr), 4, 4), dtype=np.complex128)
    with pytest.raises(ValueError, match="not supported"):
        calc_faraday_moments(
            fdf, phi_arr, 20.0, debias=True, lam_sq_0_m2=0.1, threshold=0.5
        )


def test_moments_min_weight_fraction_signed():
    phi_arr = make_phi_arr(100, 1)
    fwhm = 20.0
    # Signed spectrum that nearly cancels: net weight is a tiny positive value
    signed = np.zeros_like(phi_arr)
    signed[10] = 1.0
    signed[20] = -0.999

    # Default guard (weight_sum > 0) admits the tiny positive sum -> the mean
    # Faraday depth is a finite but meaningless value.
    default = calc_faraday_moments(signed, phi_arr, fwhm)
    assert np.isfinite(default.mom1)

    # Opt-in floor masks the near-cancelling spectrum symmetrically.
    guarded = calc_faraday_moments(signed, phi_arr, fwhm, min_weight_fraction=0.1)
    assert np.isnan(guarded.mom1)
    assert np.isnan(guarded.mom2)
    # mom0 (integrated flux) is unaffected by the guard.
    assert np.isclose(float(guarded.mom0), float(default.mom0))

    # For non-negative |FDF| input the guard is a no-op (ratio == 1).
    pos = gaussian(phi_arr, 2.0, 0.0, fwhm=fwhm).astype(np.complex128)
    plain = calc_faraday_moments(pos, phi_arr, fwhm)
    floored = calc_faraday_moments(pos, phi_arr, fwhm, min_weight_fraction=0.5)
    assert np.allclose(
        [plain.mom0, plain.mom1, plain.mom2],
        [floored.mom0, floored.mom1, floored.mom2],
    )


def _stokes_data(
    stokes_i: NDArray[np.float64],
    stokes_i_error: NDArray[np.float64],
    freq_arr_hz: NDArray[np.float64],
) -> StokesData:
    """A StokesData with flat, noiseless Q/U, so only the Stokes I fit matters."""
    pol = np.full(freq_arr_hz.size, 0.1 + 0.0j, dtype=np.complex128)
    return StokesData(
        complex_pol_arr=pol,
        complex_pol_error=np.full_like(pol, 1e-3),
        freq_arr_hz=freq_arr_hz,
        stokes_i_arr=stokes_i,
        stokes_i_error_arr=stokes_i_error,
    )


def test_fractional_spectra_snr_cut_needs_an_error() -> None:
    """The 1D path rejects the same silent no-op the cube path does."""
    freq = np.linspace(800e6, 1800e6, 64)
    stokes_i = np.full(freq.size, 1.0)
    data = _stokes_data(stokes_i, np.zeros(freq.size), freq)

    with pytest.raises(ValueError, match="needs a Stokes I error"):
        create_fractional_spectra(data, float(freq.mean()), StokesIFitOptions())

    # No cut, no requirement: an unweighted fit is fine when nothing depends on SNR.
    result = create_fractional_spectra(
        data, float(freq.mean()), StokesIFitOptions(snr_cut=None)
    )
    assert result is not None


def test_fractional_spectra_falls_back_rather_than_raising() -> None:
    """An unusable model is a data condition, so 1D warns and carries on.

    A polynomial through noise dips below zero at a band edge; dividing Q/U by
    it would flip their sign there, so the flat model stands in.
    """
    rng = np.random.default_rng(20260826)
    freq = np.linspace(800e6, 1800e6, 125)
    stokes_i = rng.normal(0, 0.05, freq.size) + 0.01
    data = _stokes_data(stokes_i, np.full(freq.size, 0.05), freq)

    options = StokesIFitOptions(snr_cut=None, fit_function="linear")
    result = create_fractional_spectra(data, float(freq.mean()), options)
    assert result is not None
    model = result.stokes_data.stokes_i_model_arr
    assert model is not None
    assert np.all(np.isfinite(model))
    assert np.allclose(model, np.mean(stokes_i)), "not the flat fallback"
    assert np.all(np.isfinite(result.stokes_data.complex_pol_arr))

    # And it is the gate doing it, not a fit that happened to come out flat:
    # the same fit on data shifted clear of zero keeps its spectral shape.
    lifted = _stokes_data(stokes_i + 1.0, np.full(freq.size, 0.05), freq)
    ungated = create_fractional_spectra(lifted, float(freq.mean()), options)
    assert ungated is not None
    ungated_model = ungated.stokes_data.stokes_i_model_arr
    assert ungated_model is not None
    assert not np.allclose(ungated_model, np.mean(stokes_i + 1.0))
