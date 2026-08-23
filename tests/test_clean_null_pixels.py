"""Tests for the RM-CLEAN null-pixel screen (`_null_clean_pixels`)."""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
import pytest
from numpy.typing import NDArray
from rm_lite.utils import clean as clean_mod
from rm_lite.utils.clean import (
    MultiscaleOptions,
    RMCleanOptions,
    RMCleanResults,
    RMSynthArrays,
    _blank_pixels,
    _null_clean_pixels,
    rmclean,
)
from rm_lite.utils.logging import quiet_logs
from rm_lite.utils.synthesis import (
    freq_to_lambda2,
    get_fwhm_rmsf,
    get_rmsf_nufft,
    make_phi_arr,
    rmsynth_nufft,
)

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


def _make_cube(*, with_blanks: bool) -> Cube:
    """A mix of blank, noise-only, faint and bright pixels on one phi axis.

    Blank pixels are all-NaN columns, as a mosaic edge gives; the noise pixels
    are what the screen is for; faint sits just above the CLEAN mask and bright
    well above it, so both take the loop.
    """
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
        source + rng.normal(0, NOISE, source.shape) + 1j * rng.normal(0, NOISE, source.shape)
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


def _run(cube: Cube, *, adaptive: bool, multiscale: bool) -> RMCleanResults:
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


def _assert_identical(new: RMCleanResults, old: RMCleanResults, label: str) -> None:
    for name, a, b in zip(new._fields, new, old, strict=True):
        assert np.array_equal(a, b, equal_nan=True), f"{label}: {name}"


ALL_MODES = [
    ("single-scale, adaptive", False, True),
    ("single-scale, fixed mask", False, False),
    ("multiscale, adaptive", True, True),
    ("multiscale, fixed mask", True, False),
]
SINGLE_SCALE_MODES = ALL_MODES[:2]


def _reference(
    monkeypatch: pytest.MonkeyPatch, cube: Cube, *, adaptive: bool, multiscale: bool
) -> RMCleanResults:
    """`rmclean` with the screen forced empty, i.e. the pre-screen function.

    Skipping is the only behaviour the screen changes, so this is an exact
    stand-in for the unpatched loop.
    """
    empty = lambda arr, *args: np.zeros(arr.shape[1], dtype=bool)  # noqa: E731
    monkeypatch.setattr(clean_mod, "_null_clean_pixels", empty)
    monkeypatch.setattr(clean_mod, "_blank_pixels", empty)
    return _run(cube, adaptive=adaptive, multiscale=multiscale)


@pytest.mark.parametrize(("label", "multiscale", "adaptive"), ALL_MODES)
def test_null_pixel_screen_is_bit_identical(
    monkeypatch: pytest.MonkeyPatch, label: str, multiscale: bool, adaptive: bool
) -> None:
    """Screening null pixels must reproduce the loop bit-for-bit, in every mode."""
    cube = _make_cube(with_blanks=False)
    screened = _run(cube, adaptive=adaptive, multiscale=multiscale)
    every_pixel = _reference(
        monkeypatch, cube, adaptive=adaptive, multiscale=multiscale
    )
    _assert_identical(screened, every_pixel, label)


@pytest.mark.parametrize(("label", "multiscale", "adaptive"), SINGLE_SCALE_MODES)
def test_null_pixel_screen_is_bit_identical_with_blanks(
    monkeypatch: pytest.MonkeyPatch, label: str, multiscale: bool, adaptive: bool
) -> None:
    """Same, with fully blanked columns mixed in.

    Single-scale only: the unpatched multiscale loop cannot run this cube at all
    (see `test_multiscale_blank_spectrum_crashes_without_the_screen`), so there
    is nothing to compare it against.
    """
    cube = _make_cube(with_blanks=True)
    screened = _run(cube, adaptive=adaptive, multiscale=multiscale)
    every_pixel = _reference(
        monkeypatch, cube, adaptive=adaptive, multiscale=multiscale
    )
    _assert_identical(screened, every_pixel, label)
    assert np.isnan(screened.clean_fdf_arr.reshape(screened.clean_fdf_arr.shape[0], -1)[:, :N_BLANK]).all()


def test_null_pixel_screen_actually_skips() -> None:
    """The screen has to fire, or the equality tests above prove nothing."""
    cube = _make_cube(with_blanks=True)
    dirty_2d = cube.dirty.reshape(cube.dirty.shape[0], -1)

    skip = _null_clean_pixels(dirty_2d, MASK)
    # Blank and noise-only pixels skipped; faint and bright ones cleaned.
    assert skip[: N_BLANK + N_NOISE].all()
    assert not skip[N_BLANK + N_NOISE :].any()

    # Non-adaptive multiscale gets the narrower screen: blanks only.
    blanks = _blank_pixels(dirty_2d)
    assert blanks[:N_BLANK].all()
    assert not blanks[N_BLANK:].any()


@pytest.mark.parametrize(("label", "multiscale", "adaptive"), ALL_MODES)
def test_blank_spectrum_does_not_crash(
    label: str, multiscale: bool, adaptive: bool
) -> None:
    """A fully blanked spectrum used to crash multiscale RM-CLEAN.

    `compute_scale_kernels` -> `fit_rmsf` -> `curve_fit` raised
    `ValueError: array must not contain infs or NaNs` on an all-NaN pixel, which
    a mosaic edge has many of. The screen removes those pixels first.
    """
    cube = _make_cube(with_blanks=True)
    flat = cube.dirty.reshape(cube.dirty.shape[0], -1)[:, :N_BLANK]
    blank_only = cube._replace(
        dirty=flat[:, np.newaxis, :],
        rmsf=cube.rmsf.reshape(cube.rmsf.shape[0], -1)[:, :N_BLANK][:, np.newaxis, :],
    )
    result = _run(blank_only, adaptive=adaptive, multiscale=multiscale)
    assert np.isnan(result.clean_fdf_arr).all()
    assert not result.clean_iter_arr.any()
    assert not np.asarray(result.model_fdf_arr).any()


def test_multiscale_blank_spectrum_crashes_without_the_screen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pins the bug the screen fixes: without it, a blank pixel raises."""
    cube = _make_cube(with_blanks=True)
    flat = cube.dirty.reshape(cube.dirty.shape[0], -1)[:, :N_BLANK]
    blank_only = cube._replace(
        dirty=flat[:, np.newaxis, :],
        rmsf=cube.rmsf.reshape(cube.rmsf.shape[0], -1)[:, :N_BLANK][:, np.newaxis, :],
    )
    with (
        pytest.raises(ValueError, match="must not contain infs or NaNs"),
        np.errstate(invalid="ignore"),
    ):
        _reference(monkeypatch, blank_only, adaptive=True, multiscale=True)


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
