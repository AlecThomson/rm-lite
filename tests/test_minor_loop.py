"""Tests for the Hogbom minor loop and its RMSF shift."""

from __future__ import annotations

import dataclasses
from typing import Any, Literal, TypeAlias

import numpy as np
import pytest
from numpy.typing import NDArray
from rm_lite.utils.clean import (
    CleanProgress,
    CleanState,
    MinorLoopArrays,
    MinorLoopOptions,
    MinorLoopResults,
    minor_loop,
    shift_rmsf,
)
from rm_lite.utils.logging import quiet_logs
from rm_lite.utils.simulate import Component, FDFSpec, simulate_fdf
from scipy.ndimage import binary_dilation

Source: TypeAlias = Literal["noise", "point", "pair", "extended"]
MaskMode: TypeAlias = Literal["adaptive", "threshold", "fixed"]

NOISE = 0.003


@pytest.mark.parametrize(("fdf_len", "n_phi_pad"), [(101, 50), (64, 64), (7, 3)])
def test_shift_rmsf_matches_roll_and_clip(fdf_len: int, n_phi_pad: int) -> None:
    """Every shift, wrapping or not, is the RMSF rolled to the channel and clipped."""
    rng = np.random.default_rng(fdf_len)
    double_len = fdf_len + 2 * n_phi_pad
    rmsf = rng.normal(size=double_len) + 1j * rng.normal(size=double_len)
    max_rmsf_index = int(np.argmax(np.abs(rmsf)))
    for fdf_index in range(fdf_len):
        rolled = np.roll(rmsf, fdf_index + n_phi_pad - max_rmsf_index)
        shifted = shift_rmsf(rmsf, fdf_index, n_phi_pad, max_rmsf_index)
        np.testing.assert_array_equal(shifted, rolled[n_phi_pad : n_phi_pad + fdf_len])
        assert not np.shares_memory(shifted, rmsf)


def python_seed_mask(
    resid: NDArray[np.complexfloating], cap: NDArray[np.bool_], floor: float
) -> NDArray[np.bool_]:
    mask = np.zeros(resid.shape, dtype=bool)
    allowed = np.where(cap, np.abs(resid), 0.0)
    seed = int(np.argmax(allowed))
    if allowed[seed] > floor:
        mask[seed] = True
    return mask


def python_offsource_rms(
    resid: NDArray[np.complexfloating], mask: NDArray[np.bool_]
) -> float:
    off = resid[~mask]
    if off.size < 8:
        return 0.0
    pooled = np.concatenate([off.real, off.imag])
    return float(1.4826 * np.median(np.abs(pooled - np.median(pooled))))


def python_minor_loop(
    arrays: MinorLoopArrays, options: MinorLoopOptions
) -> MinorLoopResults:
    """`minor_loop` as it was in NumPy (`rm_lite/utils/clean.py` at e97dbbc), to
    check the compiled one by"""
    resid = arrays.resid_fdf_spectrum_mask.data.copy()
    model = np.zeros_like(resid)
    mask = ~np.ma.getmaskarray(arrays.resid_fdf_spectrum_mask).copy()
    mask_original = mask.copy()
    iter_count = options.start_iter
    last_recompute_peak, last_recompute_iter = np.inf, options.start_iter
    if options.update_mask and options.noise is not None:
        mask = python_seed_mask(resid, mask_original, options.mask_threshold)
    max_rmsf_index = int(np.nanargmax(np.abs(arrays.rmsf_spectrum)))
    n_phi_pad = (len(arrays.phi_double_arr_radm2) - len(arrays.phi_arr_radm2)) // 2
    guard_divergence = bool(np.isfinite(resid).any())
    progress = CleanProgress(model.copy(), resid.copy())
    for iter_count in range(options.start_iter, options.max_iter + 1):
        if not mask.any() or iter_count == options.max_iter:
            break
        masked_abs = np.where(mask, np.abs(resid), -np.inf)
        peak_index = int(np.argmax(masked_abs))
        if masked_abs[peak_index] < options.stopping_threshold:
            if options.noise is not None:
                seed = python_seed_mask(resid, mask_original, options.mask_threshold)
                if seed.any():
                    mask |= seed
                    last_recompute_peak, last_recompute_iter = np.inf, iter_count
                    continue
            break
        if guard_divergence and (
            progress.check(float(np.nanmax(np.abs(resid))), model, resid)
            is not CleanState.CONVERGING
        ):
            model, resid = progress.model_fdf_spectrum, progress.resid_fdf_spectrum
            break
        peak = float(np.abs(resid[peak_index]))
        component = options.gain * resid[peak_index]
        model[peak_index] += component
        resid -= component * shift_rmsf(
            arrays.rmsf_spectrum, peak_index, n_phi_pad, max_rmsf_index
        )
        if not options.update_mask:
            continue
        if options.noise is None:
            mask = (np.abs(resid) > options.mask_threshold) & mask_original
            continue
        stalled = iter_count - last_recompute_iter >= options.stall_iters
        if peak < 0.8 * last_recompute_peak or stalled:
            ratio = python_offsource_rms(resid, mask) / options.noise
            threshold = options.mask_threshold * max(1.0, ratio)
            last_recompute_peak, last_recompute_iter = peak, iter_count
            above = (np.abs(resid) > threshold) & mask_original
            mask = mask | (binary_dilation(mask) & above)
            if stalled:
                mask |= python_seed_mask(resid, mask_original, options.mask_threshold)
    return MinorLoopResults(resid, np.ma.array(resid, mask=~mask), model, iter_count)


def assert_same_clean(compiled: MinorLoopResults, python: MinorLoopResults) -> None:
    """Same iterations and mask; arrays equal to rounding (NumPy's own complex
    abs and multiply round differently from machine to machine)."""
    assert compiled.iter_count == python.iter_count
    np.testing.assert_array_equal(
        np.ma.getmaskarray(compiled.resid_fdf_spectrum_mask),
        np.ma.getmaskarray(python.resid_fdf_spectrum_mask),
    )
    finite = np.isfinite(python.resid_fdf_spectrum)
    scale = max(
        float(np.abs(python.resid_fdf_spectrum[finite]).max(initial=0.0)),
        float(np.abs(python.model_fdf_spectrum).max(initial=0.0)),
        1e-30,
    )
    atol = 100 * np.finfo(python.resid_fdf_spectrum.dtype).eps * scale
    for got, expected in (
        (compiled.resid_fdf_spectrum, python.resid_fdf_spectrum),
        (compiled.model_fdf_spectrum, python.model_fdf_spectrum),
    ):
        assert got.dtype == expected.dtype
        np.testing.assert_allclose(got, expected, rtol=0, atol=atol)


def simulated_arrays(
    source: Source, dtype: type[np.complexfloating] = np.complex128
) -> MinorLoopArrays:
    """A dirty FDF with rm-lite's own RMSF and Faraday depth axes"""
    components = {
        "noise": (),
        "point": (Component("delta", 0.0, 0.0, 1.0),),
        # The fainter one sits behind an RMSF null: the adaptive mask stalls
        "pair": (
            Component("delta", 0.0, 0.0, 1.0),
            Component("delta", 0.0, 2.5, 0.6),
        ),
        # Wider than the RMSF in Faraday depth
        "extended": tuple(
            Component("delta", 0.0, depth, 0.2) for depth in np.linspace(-1, 1, 9)
        ),
    }[source]
    sim = simulate_fdf(
        FDFSpec(components),
        np.linspace(800e6, 1088e6, 288),
        rng=np.random.default_rng(1),
        sigma=NOISE,
    )
    return MinorLoopArrays(
        resid_fdf_spectrum_mask=np.ma.array(
            sim.dirty_fdf.astype(dtype), mask=np.zeros(sim.dirty_fdf.shape, bool)
        ),
        phi_arr_radm2=sim.phi_arr_radm2,
        phi_double_arr_radm2=sim.phi_double_arr_radm2,
        rmsf_spectrum=sim.rmsf_arr.astype(dtype),
        rmsf_fwhm=float(sim.fwhm),
    )


def options_for(
    arrays: MinorLoopArrays, mask_mode: MaskMode, noise: float, **kwargs: Any
) -> tuple[MinorLoopArrays, MinorLoopOptions]:
    """Options for a mask mode, and the arrays with a fixed mask applied"""
    if mask_mode == "fixed":
        dirty = arrays.resid_fdf_spectrum_mask.data
        arrays = arrays._replace(
            resid_fdf_spectrum_mask=np.ma.array(
                dirty, mask=~(np.abs(dirty) > 5 * noise)
            )
        )
    options = MinorLoopOptions(
        max_iter=5000,
        gain=0.1,
        mask_threshold=5 * noise,
        stopping_threshold=noise,
        update_mask=mask_mode != "fixed",
        noise=noise if mask_mode == "adaptive" else None,
    )
    return arrays, dataclasses.replace(options, **kwargs)


@pytest.mark.parametrize("source", ["noise", "point", "pair", "extended"])
@pytest.mark.parametrize("mask_mode", ["adaptive", "threshold", "fixed"])
@pytest.mark.parametrize("dtype", [np.complex128, np.complex64])
def test_compiled_loop_matches_python(
    source: Source, mask_mode: MaskMode, dtype: type[np.complexfloating]
) -> None:
    arrays, options = options_for(simulated_arrays(source, dtype), mask_mode, NOISE)
    with quiet_logs():
        compiled = minor_loop(arrays, options)
    assert_same_clean(compiled, python_minor_loop(arrays, options))
    if source != "noise":
        assert compiled.iter_count > 0


@pytest.mark.parametrize("mask_mode", ["adaptive", "threshold", "fixed"])
@pytest.mark.parametrize(
    "overrides",
    [{"max_iter": 7}, {"max_iter": 40}, {"stall_iters": 5}, {"start_iter": 123}],
    ids=["max_iter=7", "max_iter=40", "stall_iters=5", "start_iter=123"],
)
def test_compiled_loop_matches_python_when_cut_short(
    mask_mode: MaskMode, overrides: dict[str, int]
) -> None:
    """Running out of iterations, stalling, and starting part way through"""
    arrays, options = options_for(
        simulated_arrays("pair"), mask_mode, NOISE, **overrides
    )
    with quiet_logs():
        compiled = minor_loop(arrays, options)
    assert_same_clean(compiled, python_minor_loop(arrays, options))
    if "max_iter" in overrides:
        assert compiled.iter_count == overrides["max_iter"]


@pytest.mark.parametrize("mask_mode", ["adaptive", "threshold", "fixed"])
def test_compiled_loop_matches_python_when_diverging(mask_mode: MaskMode) -> None:
    """A sidelobe taller than the main lobe makes CLEAN run away"""
    n_phi = 201
    phi_arr_radm2 = np.linspace(-1000, 1000, n_phi)
    phi_double_arr_radm2 = np.linspace(-2000, 2000, 2 * n_phi - 1)
    sigma = 20.0 / 2.355
    rmsf = np.exp(-0.5 * (phi_double_arr_radm2 / sigma) ** 2) + 6.0 * np.exp(
        -0.5 * ((phi_double_arr_radm2 - 400) / sigma) ** 2
    )
    rng = np.random.default_rng(0)
    spectrum = 1e-3 * (rng.normal(size=n_phi) + 1j * rng.normal(size=n_phi))
    spectrum += 0.05 * np.exp(-0.5 * (phi_arr_radm2 / sigma) ** 2)
    arrays, options = options_for(
        MinorLoopArrays(
            np.ma.array(spectrum, mask=np.zeros(n_phi, bool)),
            phi_arr_radm2,
            phi_double_arr_radm2,
            rmsf.astype(complex),
            20.0,
        ),
        mask_mode,
        1e-3,
        max_iter=200,
    )
    options = dataclasses.replace(options, gain=0.5)
    with quiet_logs():
        compiled = minor_loop(arrays, options)
    assert_same_clean(compiled, python_minor_loop(arrays, options))
    assert compiled.iter_count < 200


@pytest.mark.parametrize("mask_mode", ["adaptive", "threshold", "fixed"])
def test_compiled_loop_matches_python_on_a_blank_spectrum(mask_mode: MaskMode) -> None:
    arrays = simulated_arrays("noise")
    blank = np.full(arrays.resid_fdf_spectrum_mask.shape, np.nan, dtype=complex)
    arrays = arrays._replace(
        resid_fdf_spectrum_mask=np.ma.array(blank, mask=np.zeros(blank.shape, bool))
    )
    _, options = options_for(arrays, mask_mode, NOISE, max_iter=30)
    with quiet_logs():
        compiled = minor_loop(arrays, options)
    assert_same_clean(compiled, python_minor_loop(arrays, options))


def test_compiled_loop_matches_python_when_reseeding_changes_nothing() -> None:
    """A mask floor below the stopping threshold reseeds the same channel
    forever, so the loop runs to max_iter without adding a component."""
    arrays, options = options_for(
        simulated_arrays("point"), "adaptive", NOISE, max_iter=3000
    )
    options = dataclasses.replace(
        options, mask_threshold=NOISE, stopping_threshold=0.05
    )
    with quiet_logs():
        compiled = minor_loop(arrays, options)
    assert_same_clean(compiled, python_minor_loop(arrays, options))
    assert compiled.iter_count == 3000


@pytest.mark.parametrize("n_chan", [64, 288])
def test_compiled_loop_matches_python_on_a_cyclic_axis(n_chan: int) -> None:
    """CLEAN in delay: sample-index axes, and the response tiled three times so
    the shift wraps round"""
    rng = np.random.default_rng(n_chan)
    n_samples = 4 * n_chan
    weights = (rng.random(n_chan) > 0.1).astype(float)
    channels = np.arange(n_chan)
    spectrum = rng.normal(size=n_chan) + 1j * rng.normal(size=n_chan)
    spectrum += 300 * np.exp(2j * np.pi * 0.137 * channels)
    dirty = np.fft.fftshift(np.fft.fft(weights * spectrum, n=n_samples)) / weights.sum()
    response = np.fft.fftshift(np.fft.fft(weights, n=n_samples)) / weights.sum()
    noise = float(np.sqrt(np.sum(weights**2)) / weights.sum())
    arrays = MinorLoopArrays(
        resid_fdf_spectrum_mask=np.ma.masked_array(
            dirty, mask=np.zeros(n_samples, bool)
        ),
        phi_arr_radm2=np.arange(n_samples, dtype=float),
        phi_double_arr_radm2=np.arange(3 * n_samples, dtype=float) - n_samples,
        rmsf_spectrum=np.tile(response, 3),
        rmsf_fwhm=4.0,
    )
    for mask_mode in ("adaptive", "threshold"):
        _, options = options_for(arrays, mask_mode, noise, max_iter=100_000)
        options = dataclasses.replace(options, stopping_threshold=3 * noise)
        with quiet_logs():
            compiled = minor_loop(arrays, options)
        assert_same_clean(compiled, python_minor_loop(arrays, options))
        assert compiled.iter_count > 0


def test_minor_loop_takes_non_native_and_strided_arrays() -> None:
    arrays, options = options_for(simulated_arrays("point"), "adaptive", NOISE)
    expected = python_minor_loop(arrays, options)
    dirty = arrays.resid_fdf_spectrum_mask.data
    swapped = MinorLoopArrays(
        np.ma.array(
            dirty.astype(dirty.dtype.newbyteorder(">")),
            mask=np.zeros(dirty.shape, bool),
        ),
        arrays.phi_arr_radm2,
        arrays.phi_double_arr_radm2,
        np.stack([arrays.rmsf_spectrum, arrays.rmsf_spectrum], axis=1)[:, 0],
        arrays.rmsf_fwhm,
    )
    with quiet_logs():
        compiled = minor_loop(swapped, options)
    assert_same_clean(compiled, expected)
    np.testing.assert_array_equal(swapped.resid_fdf_spectrum_mask.data, dirty)


def test_minor_loop_rejects_an_all_nan_rmsf() -> None:
    arrays, options = options_for(simulated_arrays("point"), "adaptive", NOISE)
    with pytest.raises(ValueError, match="all NaN"):
        minor_loop(
            arrays._replace(rmsf_spectrum=np.full_like(arrays.rmsf_spectrum, np.nan)),
            options,
        )


def test_minor_loop_rejects_mismatched_axes() -> None:
    arrays, options = options_for(simulated_arrays("point"), "adaptive", NOISE)
    with pytest.raises(ValueError, match="clipped to the FDF axis"):
        minor_loop(
            arrays._replace(phi_double_arr_radm2=arrays.phi_double_arr_radm2[1:]),
            options,
        )
