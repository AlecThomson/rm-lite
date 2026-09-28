"""Tests for the inverse-variance weights that follow the Stokes I division."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import dask.array as da
import numpy as np
import pytest
import rm_lite.tools_3d.rmclean as rmclean3d_mod
import rm_lite.utils.fitting as fitting_mod
from dask.base import compute
from numpy.typing import NDArray
from rm_lite.tools_1d.rmsynth import run_rmsynth
from rm_lite.tools_3d.rmsynth import RMSynth3DResults, rmsynth_3d
from rm_lite.utils.fitting import StokesIFitOptions, field_spectral_index
from rm_lite.utils.synthesis import (
    FDFOptions,
    StokesIWeighting,
    freq_to_lambda2,
    lambda2_to_freq,
    stokes_i_template,
)

if TYPE_CHECKING:
    from collections.abc import Callable

FREQ_ARR_HZ = (np.arange(744, 1032, 3) * 1e6).astype(np.float64)
WIDE_FREQ_ARR_HZ = (np.arange(800, 1800, 8) * 1e6).astype(np.float64)
MODES: list[StokesIWeighting | None] = [None, "global", "per_pixel"]


def fading_model() -> NDArray[np.float64]:
    """Stokes I that falls to 2% of its peak at the top of the band."""
    t = (FREQ_ARR_HZ - FREQ_ARR_HZ[0]) / (FREQ_ARR_HZ[-1] - FREQ_ARR_HZ[0])
    return np.asarray(1 - 0.98 * t**6, dtype=np.float64)


def thin_source(
    model: NDArray[np.float64],
    frac_pol: float,
    sigma: float,
    rm_radm2: float = 30.0,
    seed: int = 1,
    freq_arr_hz: NDArray[np.float64] = FREQ_ARR_HZ,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Faraday-thin Q/U following `model`, with Gaussian noise of `sigma`."""
    rng = np.random.default_rng(seed)
    angle = (2 * rm_radm2 * freq_to_lambda2(freq_arr_hz))[:, np.newaxis, np.newaxis]
    stokes_q = frac_pol * model * np.cos(angle) + rng.normal(0, sigma, model.shape)
    stokes_u = frac_pol * model * np.sin(angle) + rng.normal(0, sigma, model.shape)
    return stokes_q, stokes_u


def synth_with_model(
    model: NDArray[np.float64],
    stokes_q: NDArray[np.float64],
    stokes_u: NDArray[np.float64],
    sigma: float,
    d_phi_radm2: float = 2.0,
    freq_arr_hz: NDArray[np.float64] = FREQ_ARR_HZ,
    **kwargs: Any,
) -> RMSynth3DResults:
    """3D synthesis with a supplied Stokes I model and 1/sigma^2 noise weights."""
    chunks = (-1, 20, 20)
    return rmsynth_3d(
        da.from_array(stokes_q, chunks=chunks),
        da.from_array(stokes_u, chunks=chunks),
        freq_arr_hz,
        weight_arr=np.full(freq_arr_hz.size, 1 / sigma**2),
        stokes_i_model=da.from_array(model, chunks=chunks),
        d_phi_radm2=d_phi_radm2,
        phi_max_radm2=400.0,
        **kwargs,
    )


def broadcast(spectrum: NDArray[np.float64], ny: int, nx: int) -> NDArray[np.float64]:
    """One spectrum in every pixel of an (n_freq, ny, nx) cube."""
    return np.broadcast_to(spectrum[:, None, None], (spectrum.size, ny, nx)).copy()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"stokes_i_weighting": "bogus"}, "stokes_i_weighting must be one of"),
        (
            {"stokes_i_weighting": "per_pixel", "lam_sq_0_m2": "per_pixel"},
            "cannot be combined",
        ),
        ({"stokes_i_weight_alpha": np.nan}, "stokes_i_weight_alpha must be"),
        ({"stokes_i_weight_alpha": "bogus"}, "stokes_i_weight_alpha must be"),
    ],
)
def test_options_reject_bad_values(kwargs: dict[str, Any], match: str) -> None:
    """Unknown modes, a non-finite alpha and per-pixel weights with a per-pixel reference raise."""
    with pytest.raises(ValueError, match=match):
        FDFOptions(n_samples=10.0, **kwargs)


@pytest.mark.parametrize("mode", MODES)
def test_theoretical_noise_matches_monte_carlo(mode: StokesIWeighting | None) -> None:
    """The reported FDF noise is the scatter of the FDF over noise-only pixels."""
    model = broadcast(fading_model(), 40, 40)
    sigma = 0.02
    stokes_q, stokes_u = thin_source(model, frac_pol=0.0, sigma=sigma)
    synth = synth_with_model(model, stokes_q, stokes_u, sigma, stokes_i_weighting=mode)

    fdf = synth.fdf_dirty_cube.compute()
    measured = np.std(fdf.real)
    theory = np.median(np.asarray(synth.theoretical_noise.fdf_q_noise))
    assert measured == pytest.approx(theory, rel=0.05)


def test_weights_raise_snr_where_the_model_falls() -> None:
    """Following the model gives more SNR, and PI stays unbiased."""
    model = broadcast(fading_model(), 40, 40)
    sigma, frac_pol = 0.02, 0.1
    stokes_q, stokes_u = thin_source(model, frac_pol=frac_pol, sigma=sigma)

    snr = {}
    for mode in MODES:
        synth = synth_with_model(
            model, stokes_q, stokes_u, sigma, stokes_i_weighting=mode
        )
        truth = frac_pol * np.asarray(synth.stokes_i_ref_flux_map)
        peak_pi = np.abs(synth.fdf_dirty_cube.compute()).max(axis=0)
        snr[mode] = np.median(truth / np.asarray(synth.theoretical_noise.fdf_q_noise))
        if mode is not None:
            assert np.median(peak_pi / truth) == pytest.approx(1.0, abs=0.02)
    assert snr[None] < snr["global"] < snr["per_pixel"]
    assert snr["per_pixel"] > 3 * snr[None]


def test_global_weights_set_lam_sq_0() -> None:
    """The reference lambda^2 is the weighted mean of lambda^2 under template^2/sigma^2."""
    model = broadcast(FREQ_ARR_HZ**-1.5, 2, 2)
    sigma = np.linspace(0.01, 0.03, FREQ_ARR_HZ.size)
    stokes_q, stokes_u = thin_source(model, frac_pol=0.1, sigma=0.0)
    chunks = (-1, 2, 2)
    synth = rmsynth_3d(
        da.from_array(stokes_q, chunks=chunks),
        da.from_array(stokes_u, chunks=chunks),
        FREQ_ARR_HZ,
        weight_arr=1 / sigma**2,
        stokes_i_model=da.from_array(model, chunks=chunks),
        d_phi_radm2=2.0,
        stokes_i_weight_alpha=-1.5,
    )
    weight = stokes_i_template(FREQ_ARR_HZ, -1.5) ** 2 / sigma**2
    lambda_sq = freq_to_lambda2(FREQ_ARR_HZ)
    assert synth.lam_sq_0_m2 == pytest.approx(np.sum(weight * lambda_sq) / weight.sum())
    assert synth.stokes_i_weight_alpha == -1.5
    assert synth.stokes_i_weighting == "global"


def two_spectral_indices(
    freq_arr_hz: NDArray[np.float64] = FREQ_ARR_HZ,
) -> NDArray[np.float64]:
    """Two pixels, one flat-ish and one steep, as a (n_freq, 1, 2) model cube."""
    x = freq_arr_hz / 1e9
    return np.stack([x**-0.5, x**-2.5], axis=-1)[:, np.newaxis, :]


def test_global_mode_gives_every_pixel_one_rmsf_and_fdf_shape() -> None:
    """Sources of different alpha get the same RMSF and the same FDF shape."""
    model = two_spectral_indices(WIDE_FREQ_ARR_HZ)
    stokes_q, stokes_u = thin_source(
        model, frac_pol=0.1, sigma=0.0, freq_arr_hz=WIDE_FREQ_ARR_HZ
    )
    synth = synth_with_model(
        model,
        stokes_q,
        stokes_u,
        1.0,
        freq_arr_hz=WIDE_FREQ_ARR_HZ,
        stokes_i_weight_alpha=-1.5,
        per_pixel_rmsf=True,
    )
    rmsf = (
        np.asarray(synth.rmsf_cube.compute()) if synth.rmsf_cube is not None else None
    )
    assert rmsf is not None
    np.testing.assert_allclose(rmsf[:, 0, 0], rmsf[:, 0, 1], atol=1e-10)

    amp = np.abs(synth.fdf_dirty_cube.compute())
    shape = amp / amp.max(axis=0)
    np.testing.assert_allclose(shape[:, 0, 0], shape[:, 0, 1], atol=1e-10)


def test_per_pixel_mode_gives_each_pixel_its_own_rmsf() -> None:
    """Per-pixel weights make the RMSF follow alpha, so the per-pixel RMSF is forced."""
    model = two_spectral_indices(WIDE_FREQ_ARR_HZ)
    stokes_q, stokes_u = thin_source(
        model, frac_pol=0.1, sigma=0.0, freq_arr_hz=WIDE_FREQ_ARR_HZ
    )
    synth = synth_with_model(
        model,
        stokes_q,
        stokes_u,
        1.0,
        d_phi_radm2=0.2,
        freq_arr_hz=WIDE_FREQ_ARR_HZ,
        stokes_i_weighting="per_pixel",
    )
    assert synth.rmsf_cube is not None
    assert synth.per_pixel_rmsf is not None
    rmsf = np.abs(np.asarray(synth.rmsf_cube.compute()))
    phi = synth.phi_double_arr_radm2
    widths = [np.ptp(phi[rmsf[:, 0, i] >= 0.5]) for i in range(2)]
    assert widths[1] > widths[0] * 1.04


@pytest.mark.parametrize("mode", MODES)
def test_pixels_below_the_snr_cut_are_blank_in_every_map(
    mode: StokesIWeighting | None,
) -> None:
    """Only corrected pixels reach the maps, and they report the right fraction."""
    freq_arr_hz = WIDE_FREQ_ARR_HZ
    ny, nx = 1, 12
    frac_pol, sigma_i, alpha = 0.1, 1e-3, -2.0
    amplitude = np.geomspace(0.05, 1.0, nx)[np.newaxis, :]
    stokes_i = (freq_arr_hz / 1.2e9)[:, None, None] ** alpha * amplitude[None]
    stokes_i = stokes_i / np.median(stokes_i[:, 0, -1]) * 0.02
    stokes_q, stokes_u = thin_source(
        stokes_i, frac_pol=frac_pol, sigma=0.0, freq_arr_hz=freq_arr_hz
    )
    chunks = (-1, ny, nx)
    synth = rmsynth_3d(
        da.from_array(stokes_q, chunks=chunks),
        da.from_array(stokes_u, chunks=chunks),
        freq_arr_hz,
        weight_arr=np.full(freq_arr_hz.size, 1 / 1e-3**2),
        stokes_i=da.from_array(stokes_i, chunks=chunks),
        stokes_i_error=np.full(freq_arr_hz.size, sigma_i),
        fit_order=1,
        stokes_i_snr_cut=50.0,
        d_phi_radm2=0.5,
        phi_max_radm2=100.0,
        stokes_i_weighting=mode,
        per_pixel_rmsf=True,
    )
    fitted = np.isfinite(np.asarray(synth.stokes_i_model_order_map))[0]
    assert fitted.any(), "no pixel is above the cut"
    assert (~fitted).any(), "no pixel is below the cut"

    ref_hz = float(lambda2_to_freq(synth.lam_sq_0_m2))
    i_at_ref = np.array(
        [np.interp(ref_hz, freq_arr_hz, stokes_i[:, 0, i]) for i in range(nx)]
    )
    peak_pi = np.abs(synth.fdf_dirty_cube.compute()).max(axis=0)[0]
    np.testing.assert_allclose(peak_pi[fitted] / i_at_ref[fitted], frac_pol, rtol=1e-3)

    clean = rmclean3d_mod.run_rmclean_from_synth(synth, max_iter=10)
    maps = {
        f"{label}.{name}": value
        for label, cube in (
            ("dirty", synth.fdf_dirty_cube),
            ("clean", clean.clean_fdf_cube),
            ("model", clean.model_fdf_cube),
        )
        for name, value in rmclean3d_mod.faraday_maps(
            cube,
            phi_arr_radm2=synth.phi_arr_radm2,
            fwhm_rmsf_radm2=synth.fwhm_rmsf_radm2,
            fdf_units="integrated" if label == "model" else "per_rmsf",
            lam_sq_0_m2=synth.lam_sq_0_m2,
            lambda_sq_arr_m2=synth.lambda_sq_arr_m2,
            fdf_noise=synth.theoretical_noise.fdf_error_noise,
            moment_threshold=None,
        ).items()
    }
    (computed,) = compute(maps)
    finite = [
        k for k, v in computed.items() if np.isfinite(np.asarray(v)[0, ~fitted]).any()
    ]
    assert not finite


def test_field_spectral_index_recovers_alpha() -> None:
    """The mean spectrum of a field of one power law gives back its index."""
    rng = np.random.default_rng(3)
    amplitude = rng.uniform(0.5, 2.0, size=(1, 10, 10))
    stokes_i = FREQ_ARR_HZ[:, None, None] ** -1.2 * amplitude
    stokes_i = stokes_i / stokes_i.mean() + rng.normal(0, 0.01, stokes_i.shape)
    stokes_i[5:20, 0, 0] = np.nan
    alpha = field_spectral_index(
        da.from_array(stokes_i, chunks=(-1, 5, 5)),
        np.full(FREQ_ARR_HZ.size, 0.01),
        FREQ_ARR_HZ,
        StokesIFitOptions(),
    )
    assert alpha == pytest.approx(-1.2, abs=0.02)


@pytest.mark.parametrize("mode", ["global", "per_pixel"])
def test_1d_matches_3d_under_variance_weighting(mode: StokesIWeighting) -> None:
    """The 1D tool weights a spectrum as the 3D tool weights its pixel."""
    model = two_spectral_indices()
    sigma = 0.01
    stokes_q, stokes_u = thin_source(model, frac_pol=0.2, sigma=sigma, seed=4)
    synth = synth_with_model(
        model,
        stokes_q,
        stokes_u,
        sigma,
        stokes_i_weighting=mode,
        stokes_i_weight_alpha=-1.5,
    )
    fdf_cube = synth.fdf_dirty_cube.compute()
    for i in range(2):
        ref = run_rmsynth(
            freq_arr_hz=FREQ_ARR_HZ,
            complex_pol_arr=stokes_q[:, 0, i] + 1j * stokes_u[:, 0, i],
            complex_pol_error=np.full(FREQ_ARR_HZ.size, sigma * (1 + 1j)),
            stokes_i_model_arr=model[:, 0, i],
            stokes_i_model_error=np.zeros(FREQ_ARR_HZ.size),
            d_phi_radm2=2.0,
            phi_max_radm2=400.0,
            stokes_i_weighting=mode,
            stokes_i_weight_alpha=-1.5,
        )
        ref_fdf = ref.fdf_arrs["fdf_dirty_complex_arr"].to_numpy().astype(complex)
        np.testing.assert_allclose(fdf_cube[:, 0, i], ref_fdf, rtol=1e-6, atol=1e-10)


@pytest.mark.filterwarnings("ignore: All channels masked")
def test_per_pixel_mode_fits_once_per_chunk(
    monkeypatch: pytest.MonkeyPatch, chunked: Callable[..., da.Array]
) -> None:
    """The model feeds the weights, the FDF and CLEAN's RMSFs without a refit."""
    calls = {"n": 0}
    original = fitting_mod._fit_stokes_i_block

    def counting(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(fitting_mod, "_fit_stokes_i_block", counting)
    rng = np.random.default_rng(5)
    stokes_i = rng.uniform(1.0, 3.0, size=(1, 6, 8)) * (
        (FREQ_ARR_HZ / np.median(FREQ_ARR_HZ))[:, None, None] ** -0.8
    )
    stokes_q, stokes_u = thin_source(stokes_i, frac_pol=0.6, sigma=0.0)
    synth = rmsynth_3d(
        chunked(stokes_q, 3, 4),
        chunked(stokes_u, 3, 4),
        FREQ_ARR_HZ,
        stokes_i=chunked(stokes_i, 3, 4),
        stokes_i_error=np.full(FREQ_ARR_HZ.size, 1e-3),
        phi_max_radm2=20.0,
        d_phi_radm2=2.0,
        stokes_i_weighting="per_pixel",
    )
    assert synth.per_pixel_rmsf is not None
    clean = rmclean3d_mod.run_rmclean(
        synth.fdf_dirty_cube,
        synth.rmsf_arr,
        synth.phi_arr_radm2,
        synth.phi_double_arr_radm2,
        synth.fwhm_rmsf_radm2,
        mask=1e-3,
        threshold=1e-3,
        per_pixel_rmsf=synth.per_pixel_rmsf,
    )
    n_chunks = synth.fdf_dirty_cube.numblocks[1] * synth.fdf_dirty_cube.numblocks[2]
    # `map_blocks` probes the block function once while the graph is built.
    calls["n"] = 0
    compute(
        clean.clean_fdf_cube,
        clean.mom0_map,
        synth.fdf_dirty_cube,
        synth.theoretical_noise.fdf_error_noise,
        scheduler="synchronous",
    )
    assert calls["n"] == n_chunks
