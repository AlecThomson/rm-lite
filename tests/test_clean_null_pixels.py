"""Tests for the RM-CLEAN null-pixel screen (`_null_clean_pixels`)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pytest
from numpy.typing import NDArray
from rm_lite.utils import clean as clean_mod
from rm_lite.utils.clean import (
    CleanProgress,
    CleanState,
    MinorLoopArrays,
    MinorLoopOptions,
    MultiscaleOptions,
    RMCleanOptions,
    RMCleanResults,
    RMSynthArrays,
    _blank_pixels,
    _null_clean_pixels,
    minor_loop,
    rmclean,
)
from rm_lite.utils.logging import quiet_logs
from rm_lite.utils.simulate import Component, FDFSpec, simulate_fdf
from rm_lite.utils.synthesis import (
    freq_to_lambda2,
    get_fwhm_rmsf,
    get_rmsf_nufft,
    make_phi_arr,
    rmsynth_nufft,
)

if TYPE_CHECKING:
    from collections.abc import Callable

NOISE = 0.02
MASK = 6 * NOISE
# Pixel makeup of the test cube, in flat (C order) index order.
N_BLANK, N_NOISE, N_FAINT, N_BRIGHT = 6, 12, 6, 6


class Cube(NamedTuple):
    """A small dirty-FDF cube plus the axes and RMSF `rmclean` needs."""

    dirty: NDArray[np.complex128]
    rmsf: NDArray[np.complex128]
    phi: NDArray[np.float64]
    phi2: NDArray[np.float64]
    fwhm: float


def make_cube(*, with_blanks: bool) -> Cube:
    """A mix of blank, noise-only, faint and bright pixels on one phi axis."""
    rng = np.random.default_rng(20260823)
    freq_hz = np.linspace(0.8e9, 1.8e9, 125)
    lsq = freq_to_lambda2(freq_hz)
    fwhm = float(get_fwhm_rmsf(lsq).fwhm_rmsf_radm2)
    phi = make_phi_arr(phi_max_radm2=250.0, d_phi_radm2=fwhm / 10)

    amps = np.concatenate(
        [
            np.zeros(N_BLANK),  # blank (set to NaN below)
            np.zeros(N_NOISE),  # noise only
            np.full(N_FAINT, 0.3),
            np.full(N_BRIGHT, 3.0),
        ]
    )
    source = amps[:, np.newaxis] * np.exp(2j * (0.3 + 25.0 * lsq))[np.newaxis, :]
    spectra = (
        source
        + rng.normal(0, NOISE, source.shape)
        + 1j * rng.normal(0, NOISE, source.shape)
    ).astype(np.complex128)
    if with_blanks:
        spectra[:N_BLANK] = np.nan + 1j * np.nan
    else:
        spectra = spectra[N_BLANK:]

    # Pixels as a 2D image (rows of 6), so nothing is squeezed away and the flat
    # pixel index keeps the grouping above.
    n_pix = spectra.shape[0]
    pol_arr = np.ascontiguousarray(spectra.T).reshape(lsq.size, n_pix // 6, 6)
    weight = np.ones_like(lsq)
    lsq_0 = float(np.nanmean(lsq))

    with quiet_logs(logging.ERROR):
        dirty = rmsynth_nufft(pol_arr, lsq, phi, weight, lsq_0, nthreads=1)
        rmsf_res = get_rmsf_nufft(
            lambda_sq_arr_m2=lsq,
            phi_arr_radm2=phi,
            weight_arr=weight,
            lam_sq_0_m2=lsq_0,
            mask_arr=~np.isfinite(pol_arr),
            nthreads=1,
        )
    return Cube(
        dirty=np.asarray(dirty, dtype=np.complex128),
        rmsf=np.asarray(rmsf_res.rmsf_cube, dtype=np.complex128),
        phi=phi,
        phi2=np.asarray(rmsf_res.phi_double_arr_radm2, dtype=np.float64),
        fwhm=float(np.nanmedian(np.real(rmsf_res.fwhm_rmsf_arr))),
    )


@pytest.fixture
def cube() -> Cube:
    """The mixed cube without blank columns."""
    return make_cube(with_blanks=False)


@pytest.fixture
def blanked_cube() -> Cube:
    """The same cube with fully blanked columns mixed in, as a mosaic edge gives."""
    return make_cube(with_blanks=True)


def run_clean(cube: Cube, *, adaptive: bool, multiscale: bool) -> RMCleanResults:
    with quiet_logs(logging.ERROR):
        return rmclean(
            RMSynthArrays(
                dirty_fdf_arr=cube.dirty,
                phi_arr_radm2=cube.phi,
                phi_double_arr_radm2=cube.phi2,
                rmsf_arr=cube.rmsf,
                fwhm_rmsf_arr=np.array(cube.fwhm),
            ),
            RMCleanOptions(
                mask=MASK,
                threshold=3 * NOISE,
                max_iter=2000,
                fdf_noise=NOISE if adaptive else None,
            ),
            multiscale_options=MultiscaleOptions(max_iter_sub_minor=2000)
            if multiscale
            else None,
        )


def assert_identical(new: RMCleanResults, old: RMCleanResults) -> None:
    for name, a, b in zip(new._fields, new, old, strict=True):
        assert np.array_equal(a, b, equal_nan=True), name


def blank_columns_only(cube: Cube) -> Cube:
    """Just the blank columns of a cube, as a one-row image."""
    flat = cube.dirty.reshape(cube.dirty.shape[0], -1)[:, :N_BLANK]
    return cube._replace(
        dirty=flat[:, np.newaxis, :],
        rmsf=cube.rmsf.reshape(cube.rmsf.shape[0], -1)[:, :N_BLANK][:, np.newaxis, :],
    )


@pytest.fixture
def reference_clean(monkeypatch: pytest.MonkeyPatch) -> Callable[..., RMCleanResults]:
    """`rmclean` with the screen forced empty, i.e. the pre-screen function."""
    # Skipping is the only behaviour the screen changes, so this is an exact
    # stand-in for the unpatched loop.

    def screen_nothing(
        dirty_fdf_arr_2d: NDArray[np.complex128], *_args: float
    ) -> NDArray[np.bool_]:
        return np.zeros(dirty_fdf_arr_2d.shape[1], dtype=bool)

    def run(cube: Cube, *, adaptive: bool, multiscale: bool) -> RMCleanResults:
        monkeypatch.setattr(clean_mod, "_null_clean_pixels", screen_nothing)
        monkeypatch.setattr(clean_mod, "_blank_pixels", screen_nothing)
        return run_clean(cube, adaptive=adaptive, multiscale=multiscale)

    return run


@pytest.mark.parametrize(
    ("multiscale", "adaptive"),
    [
        pytest.param(False, True, id="single-scale-adaptive"),
        pytest.param(False, False, id="single-scale-fixed-mask"),
        pytest.param(True, True, id="multiscale-adaptive"),
        pytest.param(True, False, id="multiscale-fixed-mask"),
    ],
)
def test_null_pixel_screen_is_bit_identical(
    cube: Cube,
    reference_clean: Callable[..., RMCleanResults],
    multiscale: bool,
    adaptive: bool,
) -> None:
    """Screening null pixels must reproduce the loop bit-for-bit, in every mode."""
    screened = run_clean(cube, adaptive=adaptive, multiscale=multiscale)
    every_pixel = reference_clean(cube, adaptive=adaptive, multiscale=multiscale)
    assert_identical(screened, every_pixel)


@pytest.mark.parametrize(
    ("multiscale", "adaptive"),
    [
        pytest.param(False, True, id="single-scale-adaptive"),
        pytest.param(False, False, id="single-scale-fixed-mask"),
    ],
)
def test_null_pixel_screen_is_bit_identical_with_blanks(
    blanked_cube: Cube,
    reference_clean: Callable[..., RMCleanResults],
    multiscale: bool,
    adaptive: bool,
) -> None:
    """Same, with fully blanked columns mixed in."""
    screened = run_clean(blanked_cube, adaptive=adaptive, multiscale=multiscale)
    every_pixel = reference_clean(
        blanked_cube, adaptive=adaptive, multiscale=multiscale
    )
    assert_identical(screened, every_pixel)
    assert np.isnan(
        screened.clean_fdf_arr.reshape(screened.clean_fdf_arr.shape[0], -1)[:, :N_BLANK]
    ).all()


def test_null_pixel_screen_actually_skips(blanked_cube: Cube) -> None:
    """The screen has to fire, or the equality tests above prove nothing."""
    dirty_2d = blanked_cube.dirty.reshape(blanked_cube.dirty.shape[0], -1)

    skip = _null_clean_pixels(dirty_2d, MASK)
    # Blank and noise-only pixels skipped; faint and bright ones cleaned.
    assert skip[: N_BLANK + N_NOISE].all()
    assert not skip[N_BLANK + N_NOISE :].any()

    # Non-adaptive multiscale gets the narrower screen: blanks only.
    blanks = _blank_pixels(dirty_2d)
    assert blanks[:N_BLANK].all()
    assert not blanks[N_BLANK:].any()


@pytest.mark.parametrize(
    ("multiscale", "adaptive"),
    [
        pytest.param(False, True, id="single-scale-adaptive"),
        pytest.param(False, False, id="single-scale-fixed-mask"),
        pytest.param(True, True, id="multiscale-adaptive"),
        pytest.param(True, False, id="multiscale-fixed-mask"),
    ],
)
def test_blank_spectrum_does_not_crash(
    blanked_cube: Cube, multiscale: bool, adaptive: bool
) -> None:
    """A fully blanked spectrum used to crash multiscale RM-CLEAN."""
    blank_only = blank_columns_only(blanked_cube)
    result = run_clean(blank_only, adaptive=adaptive, multiscale=multiscale)
    assert np.isnan(result.clean_fdf_arr).all()
    assert not result.clean_iter_arr.any()
    assert not np.asarray(result.model_fdf_arr).any()


def test_multiscale_blank_spectrum_crashes_without_the_screen(
    blanked_cube: Cube, reference_clean: Callable[..., RMCleanResults]
) -> None:
    """Pins the bug the screen fixes: without it, a blank pixel raises."""
    blank_only = blank_columns_only(blanked_cube)
    with (
        pytest.raises(ValueError, match="must not contain infs or NaNs"),
        np.errstate(invalid="ignore"),
    ):
        reference_clean(blank_only, adaptive=True, multiscale=True)


def test_null_pixel_screen_quiet_on_blank_columns(
    recwarn: pytest.WarningsRecorder,
) -> None:
    """An all-NaN column must not emit an "All-NaN slice" warning per call."""
    dirty = np.full((64, 8), np.nan + 1j * np.nan, dtype=np.complex128)
    dirty[:, 0] = 1.0 + 0j
    skip = _null_clean_pixels(dirty, 0.5)
    assert not skip[0]
    assert skip[1:].all()
    assert not [w for w in recwarn if "All-NaN" in str(w.message)]


def test_null_pixel_screen_strips_match_whole_array() -> None:
    """Strip-wise reduction must give the same answer as one pass."""
    rng = np.random.default_rng(7)
    dirty = (rng.normal(size=(2003, 97)) + 1j * rng.normal(size=(2003, 97))).astype(
        np.complex128
    )
    dirty[:, ::7] = np.nan
    expected = ~(np.fmax.reduce(np.abs(dirty), axis=0) > 1.0)
    assert np.array_equal(_null_clean_pixels(dirty, 1.0), expected)


def test_divergence_guard_never_fires_on_a_converging_clean(caplog) -> None:
    """The backstop must not change a clean that was already working."""
    n_phi, fwhm = 401, 40.0
    phi_arr_radm2 = np.linspace(-2000, 2000, n_phi)
    phi_double_arr_radm2 = np.linspace(-4000, 4000, 2 * n_phi - 1)
    rmsf_spectrum = np.exp(-0.5 * (phi_double_arr_radm2 / (fwhm / 2.355)) ** 2).astype(
        np.complex128
    )

    with caplog.at_level(logging.WARNING, logger="rm-lite"):
        for case in range(40):
            rng = np.random.default_rng(case)
            noise = 10 ** rng.uniform(-4, -2)
            spectrum = rng.normal(0, noise, n_phi) + 1j * rng.normal(0, noise, n_phi)
            kind = case % 4
            if kind == 1:
                spectrum += 10 ** rng.uniform(-3, -1) * np.exp(
                    -0.5
                    * ((phi_arr_radm2 - rng.uniform(-500, 500)) / (fwhm / 2.355)) ** 2
                )
            elif kind == 2:
                for depth in np.linspace(-300, 300, 7):
                    spectrum += 10 ** rng.uniform(-3, -2) * np.exp(
                        -0.5 * ((phi_arr_radm2 - depth) / (fwhm / 2.355)) ** 2
                    )
            elif kind == 3:
                spectrum[:] = np.nan
            for update_mask in (False, True):
                minor_loop(
                    MinorLoopArrays(
                        resid_fdf_spectrum_mask=np.ma.array(
                            spectrum.copy(), mask=np.zeros(n_phi, bool)
                        ),
                        phi_arr_radm2=phi_arr_radm2,
                        phi_double_arr_radm2=phi_double_arr_radm2,
                        rmsf_spectrum=rmsf_spectrum,
                        rmsf_fwhm=fwhm,
                    ),
                    MinorLoopOptions(
                        max_iter=2000,
                        gain=0.1,
                        mask_threshold=3 * noise,
                        stopping_threshold=noise,
                        update_mask=update_mask,
                        noise=noise if update_mask else None,
                    ),
                )

    assert "diverging" not in caplog.text


def test_stall_count_resets_while_the_peak_keeps_falling() -> None:
    """A loop halving its peak every iteration is converging, not stalled."""
    zeros = np.zeros(4, dtype=complex)
    progress = CleanProgress(zeros, zeros, stall_patience=5)
    peaks = [1.0 * 0.5**i for i in range(8)]
    assert all(progress.check(p, zeros, zeros) is CleanState.CONVERGING for p in peaks)


def test_stall_still_fires_when_the_peak_barely_moves() -> None:
    """A peak falling 0.1% an iteration stalls once patience runs out."""
    zeros = np.zeros(4, dtype=complex)
    progress = CleanProgress(zeros, zeros, stall_patience=5)
    states = [progress.check(1.0 * 0.999**i, zeros, zeros) for i in range(8)]
    assert states[:5] == [CleanState.CONVERGING] * 5
    assert states[5:] == [CleanState.STALLED] * 3


def test_divergence_guard_stops_and_keeps_the_best_state(caplog) -> None:
    """A runaway clean stops at its best peak rather than running to max_iter."""
    n_phi, fwhm, noise = 201, 20.0, 1e-3
    phi_arr_radm2 = np.linspace(-1000, 1000, n_phi)
    phi_double_arr_radm2 = np.linspace(-2000, 2000, 2 * n_phi - 1)
    sigma = fwhm / 2.355
    # A sidelobe towering over the main lobe: subtracting a component injects
    # more flux than it removes, so the residual peak climbs every iteration.
    rmsf_spectrum = (
        np.exp(-0.5 * (phi_double_arr_radm2 / sigma) ** 2)
        + 6.0 * np.exp(-0.5 * ((phi_double_arr_radm2 - 400) / sigma) ** 2)
    ).astype(np.complex128)
    rng = np.random.default_rng(0)
    spectrum = rng.normal(0, noise, n_phi) + 1j * rng.normal(0, noise, n_phi)
    spectrum += 0.05 * np.exp(-0.5 * (phi_arr_radm2 / sigma) ** 2)

    with caplog.at_level(logging.WARNING, logger="rm-lite"):
        results = minor_loop(
            MinorLoopArrays(
                resid_fdf_spectrum_mask=np.ma.array(
                    spectrum, mask=np.zeros(n_phi, bool)
                ),
                phi_arr_radm2=phi_arr_radm2,
                phi_double_arr_radm2=phi_double_arr_radm2,
                rmsf_spectrum=rmsf_spectrum,
                rmsf_fwhm=fwhm,
            ),
            MinorLoopOptions(
                max_iter=200,
                gain=0.5,
                mask_threshold=3 * noise,
                stopping_threshold=noise,
                update_mask=False,
            ),
        )

    assert "diverging" in caplog.text
    assert results.iter_count < 200
    # Reverted to the best state, not left at the runaway one.
    assert np.isclose(
        float(np.nanmax(np.abs(results.resid_fdf_spectrum))),
        float(np.nanmax(np.abs(spectrum))),
    )


def test_adaptive_mask_reaches_a_second_source_behind_a_null() -> None:
    """A bright pair must not leave the fainter source uncleaned at max_iter."""
    freq_hz = np.linspace(800e6, 1088e6, 288)
    spec = FDFSpec(
        (Component("delta", 0.0, 0.0, 1.0), Component("delta", 0.0, 2.5, 0.6))
    )
    sim = simulate_fdf(spec, freq_hz, rng=np.random.default_rng(1), sigma=0.003)
    noise = sim.fdf_noise
    max_iter = 100_000
    with quiet_logs():
        results = rmclean(
            RMSynthArrays(
                dirty_fdf_arr=sim.dirty_fdf,
                phi_arr_radm2=sim.phi_arr_radm2,
                phi_double_arr_radm2=sim.phi_double_arr_radm2,
                rmsf_arr=sim.rmsf_arr,
                fwhm_rmsf_arr=np.array(sim.fwhm),
            ),
            RMCleanOptions(
                mask=7 * noise, threshold=noise, fdf_noise=noise, max_iter=max_iter
            ),
        )
    second = np.abs(sim.phi_arr_radm2 - 2.5 * sim.fwhm) <= 0.5 * sim.fwhm
    assert int(results.clean_iter_arr.squeeze()) < max_iter
    assert np.abs(results.resid_fdf_arr).max() < 10 * noise
    assert abs(np.abs(results.model_fdf_arr[second].sum()) - 0.6) < 0.03
