"""Tests for the Hogbom minor loop and its RMSF shift."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import replace
from importlib import resources
from typing import Any, Literal, TypeAlias

import numpy as np
import pytest
from numpy.typing import NDArray
from rm_lite.utils import clean
from rm_lite.utils.clean import (
    CleanProgress,
    CleanState,
    HogbomStop,
    MinorLoopArrays,
    MinorLoopBatchArrays,
    MinorLoopBatchOptions,
    MinorLoopBatchResults,
    MinorLoopOptions,
    MinorLoopResults,
    RMCleanOptions,
    RMSynthArrays,
    _HogbomBatchResults,
    minor_cycle,
    minor_cycle_batch,
    minor_loop,
    minor_loop_batch,
    offsource_rms,
    rmclean,
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
            noise = options.noise
            ratio = python_offsource_rms(resid, mask) / noise if noise > 0 else 1.0
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


def diverging(mask_mode: MaskMode) -> tuple[MinorLoopArrays, MinorLoopOptions]:
    """A sidelobe taller than the main lobe, which makes CLEAN run away"""
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
    return arrays, dataclasses.replace(options, gain=0.5)


@pytest.mark.parametrize("mask_mode", ["adaptive", "threshold", "fixed"])
def test_compiled_loop_matches_python_when_diverging(mask_mode: MaskMode) -> None:
    """A sidelobe taller than the main lobe makes CLEAN run away"""
    arrays, options = diverging(mask_mode)
    with quiet_logs():
        compiled = minor_loop(arrays, options)
    assert_same_clean(compiled, python_minor_loop(arrays, options))
    assert compiled.iter_count < 200


def blank_arrays() -> MinorLoopArrays:
    """An all-NaN spectrum"""
    arrays = simulated_arrays("noise")
    blank = np.full(arrays.resid_fdf_spectrum_mask.shape, np.nan, dtype=complex)
    return arrays._replace(
        resid_fdf_spectrum_mask=np.ma.array(blank, mask=np.zeros(blank.shape, bool))
    )


@pytest.mark.parametrize("mask_mode", ["adaptive", "threshold", "fixed"])
def test_compiled_loop_matches_python_on_a_blank_spectrum(mask_mode: MaskMode) -> None:
    arrays = blank_arrays()
    _, options = options_for(arrays, mask_mode, NOISE, max_iter=30)
    with quiet_logs():
        compiled = minor_loop(arrays, options)
    assert_same_clean(compiled, python_minor_loop(arrays, options))


@pytest.mark.parametrize("mask_mode", ["adaptive", "threshold", "fixed"])
@pytest.mark.parametrize(
    "nan_options",
    [
        ("mask_threshold",),
        ("stopping_threshold",),
        ("noise",),
        ("mask_threshold", "stopping_threshold", "noise"),
    ],
    ids=["mask_threshold", "stopping_threshold", "noise", "all"],
)
def test_compiled_loop_matches_python_with_nan_options(
    mask_mode: MaskMode, nan_options: tuple[str, ...]
) -> None:
    """NaN thresholds and noise, as from a blanked pixel of a map"""
    arrays, options = options_for(
        simulated_arrays("pair"), mask_mode, NOISE, max_iter=300
    )
    nans: dict[str, Any] = dict.fromkeys(nan_options, np.nan)
    options = dataclasses.replace(options, **nans)
    with quiet_logs():
        compiled = minor_loop(arrays, options)
    assert_same_clean(compiled, python_minor_loop(arrays, options))


def test_offsource_rms_needs_eight_channels() -> None:
    resid = np.random.default_rng(0).normal(size=10).astype(complex)
    mask = np.arange(10) < 3
    assert offsource_rms(resid, mask) == 0.0
    assert offsource_rms(resid, ~mask) == 0.0
    assert offsource_rms(resid, np.arange(10) < 2) > 0.0


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


@pytest.fixture
def stop_reasons(monkeypatch: pytest.MonkeyPatch) -> list[HogbomStop]:
    """Why each minor loop stopped, as `_log_stops` is told"""
    reasons: list[HogbomStop] = []

    def capture(results: _HogbomBatchResults) -> None:
        reasons.extend(HogbomStop(stop) for stop in results.stop)

    monkeypatch.setattr(clean, "_log_stops", capture)
    return reasons


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("threshold", HogbomStop.THRESHOLD),
        ("empty_mask", HogbomStop.EMPTY_MASK),
        ("max_iter", HogbomStop.MAX_ITER),
        ("diverged", HogbomStop.DIVERGED),
        ("reseed", HogbomStop.MAX_ITER),
        ("blank", HogbomStop.EMPTY_MASK),
        ("not_started", HogbomStop.NOT_STARTED),
    ],
)
def test_minor_loop_stops_for_its_reason(
    case: str, expected: HogbomStop, stop_reasons: list[HogbomStop]
) -> None:
    """Each way the loop can end reports its own reason"""
    point = simulated_arrays("point")
    cases: dict[str, Callable[[], tuple[MinorLoopArrays, MinorLoopOptions]]] = {
        "threshold": lambda: options_for(point, "fixed", NOISE),
        "empty_mask": lambda: options_for(point, "threshold", NOISE),
        "max_iter": lambda: options_for(
            simulated_arrays("pair"), "adaptive", NOISE, max_iter=50
        ),
        "diverged": lambda: diverging("fixed"),
        # A mask floor below the stopping threshold reseeds the same channel
        "reseed": lambda: options_for(
            point, "adaptive", NOISE, mask_threshold=NOISE, stopping_threshold=0.05
        ),
        "blank": lambda: options_for(blank_arrays(), "adaptive", NOISE, max_iter=30),
        "not_started": lambda: options_for(
            point, "fixed", NOISE, start_iter=10, max_iter=5
        ),
    }
    arrays, options = cases[case]()
    with quiet_logs():
        minor_loop(arrays, options)
    assert stop_reasons == [expected]


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


def delay_spectra(
    n_spectra: int, n_chan: int, seed: int = 0
) -> tuple[MinorLoopBatchArrays, NDArray[np.float64]]:
    """Delay spectra with their own flagging (so their own response), a mix of
    noise-only, point and bright sources, and the noise of each"""
    rng = np.random.default_rng(seed)
    n_samples = 4 * n_chan
    channels = np.arange(n_chan)
    dirty, response, noise = [], [], []
    for spec in range(n_spectra):
        weights = (rng.random(n_chan) > 0.1).astype(float)
        spectrum = rng.normal(size=n_chan) + 1j * rng.normal(size=n_chan)
        amp = (0.0, 30.0, 300.0, 3000.0)[spec % 4]
        spectrum += amp * np.exp(2j * np.pi * rng.uniform(-0.4, 0.4) * channels)
        dirty.append(
            np.fft.fftshift(np.fft.fft(weights * spectrum, n=n_samples)) / weights.sum()
        )
        response.append(
            np.fft.fftshift(np.fft.fft(weights, n=n_samples)) / weights.sum()
        )
        noise.append(np.sqrt(np.sum(weights**2)) / weights.sum())
    arrays = MinorLoopBatchArrays(
        resid_fdf_arr_mask=np.ma.array(
            dirty, mask=np.zeros((n_spectra, n_samples), bool)
        ),
        phi_arr_radm2=np.arange(n_samples, dtype=float),
        phi_double_arr_radm2=np.arange(3 * n_samples, dtype=float) - n_samples,
        rmsf_arr=np.tile(np.array(response), 3),
    )
    return arrays, np.array(noise)


def looped(
    arrays: MinorLoopBatchArrays, options: MinorLoopBatchOptions
) -> list[MinorLoopResults]:
    """`minor_loop` on each spectrum of a batch"""
    n_spectra = arrays.resid_fdf_arr_mask.shape[0]
    rmsf_arr = np.atleast_2d(arrays.rmsf_arr)
    mask_threshold, stopping_threshold, start_iter, noise = (
        np.broadcast_to(np.asarray(value), n_spectra)
        for value in (
            options.mask_threshold,
            options.stopping_threshold,
            options.start_iter,
            np.nan if options.noise is None else options.noise,
        )
    )
    with quiet_logs():
        return [
            minor_loop(
                MinorLoopArrays(
                    arrays.resid_fdf_arr_mask[spec],
                    arrays.phi_arr_radm2,
                    arrays.phi_double_arr_radm2,
                    rmsf_arr[spec if len(rmsf_arr) > 1 else 0],
                    1.0,
                ),
                MinorLoopOptions(
                    max_iter=options.max_iter,
                    gain=options.gain,
                    mask_threshold=float(mask_threshold[spec]),
                    stopping_threshold=float(stopping_threshold[spec]),
                    start_iter=int(start_iter[spec]),
                    update_mask=options.update_mask,
                    noise=None if options.noise is None else float(noise[spec]),
                    stall_iters=options.stall_iters,
                ),
            )
            for spec in range(n_spectra)
        ]


def assert_batch_matches(
    batch: MinorLoopBatchResults, singles: list[MinorLoopResults]
) -> None:
    """The batch is bit for bit `minor_loop` on each spectrum"""
    for spec, single in enumerate(singles):
        assert batch.iter_count_arr[spec] == single.iter_count
        np.testing.assert_array_equal(
            batch.resid_fdf_arr[spec], single.resid_fdf_spectrum
        )
        np.testing.assert_array_equal(
            batch.model_fdf_arr[spec], single.model_fdf_spectrum
        )
        np.testing.assert_array_equal(
            np.ma.getmaskarray(batch.resid_fdf_arr_mask)[spec],
            np.ma.getmaskarray(single.resid_fdf_spectrum_mask),
        )


@pytest.mark.slow_uncompiled
@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("mask_mode", ["adaptive", "threshold", "fixed"])
@pytest.mark.parametrize("response", ["per-spectrum", "shared"])
def test_batch_matches_looped_minor_loop(
    parallel: bool, mask_mode: MaskMode, response: Literal["per-spectrum", "shared"]
) -> None:
    """Each spectrum of a batch cleans as it would alone, serially or in parallel"""
    arrays, noise = delay_spectra(12, 64)
    if response == "shared":
        arrays = arrays._replace(rmsf_arr=arrays.rmsf_arr[0])
    if mask_mode == "fixed":
        dirty = arrays.resid_fdf_arr_mask.data
        arrays = arrays._replace(
            resid_fdf_arr_mask=np.ma.array(
                dirty, mask=~(np.abs(dirty) > 5 * noise[:, None])
            )
        )
    options = MinorLoopBatchOptions(
        max_iter=20_000,
        gain=0.1,
        mask_threshold=5 * noise,
        stopping_threshold=3 * noise,
        update_mask=mask_mode != "fixed",
        noise=noise if mask_mode == "adaptive" else None,
        parallel=parallel,
    )
    with quiet_logs():
        batch = minor_loop_batch(arrays, options)
    assert_batch_matches(batch, looped(arrays, options))
    # delay_spectra cycles through amplitudes 0, 30, 300 and 3000
    assert (batch.iter_count_arr[np.arange(12) % 4 != 0] > 0).all()


def test_batch_takes_scalars_and_mixed_per_spectrum_options() -> None:
    """Scalar thresholds, a per-spectrum start, NaN noise (a blanked pixel of a
    noise map) for some spectra, a short max_iter, and complex64"""
    arrays, noise = delay_spectra(8, 32, seed=1)
    arrays = arrays._replace(
        resid_fdf_arr_mask=np.ma.array(arrays.resid_fdf_arr_mask, dtype=np.complex64),
        rmsf_arr=arrays.rmsf_arr.astype(np.complex64),
    )
    options = MinorLoopBatchOptions(
        max_iter=300,
        gain=0.2,
        mask_threshold=float(5 * noise.mean()),
        stopping_threshold=float(3 * noise.mean()),
        start_iter=np.arange(8) * 10,
        noise=np.where(np.arange(8) % 2, noise, np.nan),
        stall_iters=20,
        # The uncompiled coverage run reaches the parallel loop only here, as
        # the parametrised test is too slow for it
        parallel=True,
    )
    with quiet_logs():
        batch = minor_loop_batch(arrays, options)
    assert batch.resid_fdf_arr.dtype == np.complex64
    assert_batch_matches(batch, looped(arrays, options))


def test_batch_rejects_misshapen_inputs() -> None:
    arrays, noise = delay_spectra(3, 16)
    options = MinorLoopBatchOptions(
        max_iter=10, gain=0.1, mask_threshold=1.0, stopping_threshold=1.0
    )
    with pytest.raises(ValueError, match="2D"):
        minor_loop_batch(
            arrays._replace(resid_fdf_arr_mask=arrays.resid_fdf_arr_mask[0]), options
        )
    with pytest.raises(ValueError, match="one for each"):
        minor_loop_batch(arrays._replace(rmsf_arr=arrays.rmsf_arr[:2]), options)
    with pytest.raises(ValueError, match="clipped to the FDF axis"):
        minor_loop_batch(
            arrays._replace(phi_double_arr_radm2=arrays.phi_double_arr_radm2[1:]),
            options,
        )
    with pytest.raises(ValueError, match="`mask_threshold` must be one value"):
        minor_loop_batch(arrays, replace(options, mask_threshold=noise[:2]))
    with pytest.raises(ValueError, match="all NaN"):
        minor_loop_batch(
            arrays._replace(rmsf_arr=np.full_like(arrays.rmsf_arr, np.nan)), options
        )


def test_batch_of_no_spectra() -> None:
    arrays, _ = delay_spectra(1, 16)
    results = minor_loop_batch(
        arrays._replace(resid_fdf_arr_mask=arrays.resid_fdf_arr_mask[:0]),
        MinorLoopBatchOptions(
            max_iter=10, gain=0.1, mask_threshold=1.0, stopping_threshold=1.0
        ),
    )
    assert results.resid_fdf_arr.shape == (0, 64)
    assert results.iter_count_arr.shape == (0,)


@pytest.mark.parametrize("fdf_noise", [None, 1.0])
def test_minor_cycle_batch_rejects_misshapen_inputs(fdf_noise: float | None) -> None:
    """The same message for a fixed or an adaptive mask"""
    arrays, noise = delay_spectra(4, 16)
    rm_synth_arrays = RMSynthArrays(
        dirty_fdf_arr=arrays.resid_fdf_arr_mask.data.T,
        phi_arr_radm2=arrays.phi_arr_radm2,
        phi_double_arr_radm2=arrays.phi_double_arr_radm2,
        rmsf_arr=arrays.rmsf_arr.T,
        fwhm_rmsf_arr=np.array(1.0),
    )
    options = RMCleanOptions(mask=1.0, threshold=1.0, fdf_noise=fdf_noise)
    with pytest.raises(ValueError, match="`mask` must be one value"):
        minor_cycle_batch(rm_synth_arrays, replace(options, mask=noise[:3]))
    with pytest.raises(ValueError, match=r"2D \(phi, spectrum\)"):
        minor_cycle_batch(
            rm_synth_arrays._replace(dirty_fdf_arr=rm_synth_arrays.dirty_fdf_arr[:, 0]),
            options,
        )


@pytest.mark.slow_uncompiled
@pytest.mark.parametrize("adaptive", [True, False])
@pytest.mark.parametrize("per_pixel", [True, False])
def test_rmclean_matches_minor_cycle_on_each_pixel(
    adaptive: bool, per_pixel: bool
) -> None:
    """rmclean cleans its pixels in batches, and each cleans as minor_cycle on
    it alone, with per-pixel maps, an FDF mask, and a shared or per-pixel RMSF"""
    sources = (
        (),
        (Component("delta", 0.0, 0.0, 1.0),),
        (Component("delta", 0.0, 1.0, 0.05),),
        (Component("delta", 0.0, 0.0, 1.0), Component("delta", 0.0, 2.5, 0.6)),
    )
    sims = [
        simulate_fdf(
            FDFSpec(components),
            np.linspace(800e6, 1088e6, 288),
            rng=np.random.default_rng(pix),
            sigma=NOISE,
        )
        for pix, components in enumerate(sources * 2)
    ]
    # Enough pixels to clean that rmclean takes more than one batch
    shape = (16, 24)
    dirty = np.tile(np.stack([sim.dirty_fdf for sim in sims], axis=1), 48)
    dirty = dirty.reshape(-1, *shape)
    sim = sims[0]
    n_double = len(sim.rmsf_arr)
    if per_pixel:
        scale = 1 + 0.001 * np.arange(dirty[0].size).reshape(shape)
        rmsf_arr = sim.rmsf_arr[:, None, None] * scale
    else:
        rmsf_arr = np.broadcast_to(sim.rmsf_arr[:, None, None], (n_double, *shape))
    noise = np.asarray(
        sim.fdf_noise * (1 + 0.2 * (np.arange(dirty[0].size) % 8).reshape(shape)),
        dtype=np.float64,
    )
    fdf_mask_arr = np.ones(dirty.shape, dtype=bool)
    fdf_mask_arr[: len(sim.phi_arr_radm2) // 4] = False
    clean_options = RMCleanOptions(
        mask=5 * noise, threshold=noise, fdf_noise=noise if adaptive else None
    )
    with quiet_logs():
        results = rmclean(
            RMSynthArrays(
                dirty_fdf_arr=dirty,
                phi_arr_radm2=sim.phi_arr_radm2,
                phi_double_arr_radm2=sim.phi_double_arr_radm2,
                rmsf_arr=rmsf_arr,
                fwhm_rmsf_arr=np.array(sim.fwhm),
                fdf_mask_arr=fdf_mask_arr,
            ),
            clean_options,
        )
        for y, x in np.ndindex(shape):
            pixel = minor_cycle(
                RMSynthArrays(
                    dirty_fdf_arr=dirty[:, y, x],
                    phi_arr_radm2=sim.phi_arr_radm2,
                    phi_double_arr_radm2=sim.phi_double_arr_radm2,
                    rmsf_arr=np.ascontiguousarray(rmsf_arr[:, y, x]),
                    fwhm_rmsf_arr=np.array(sim.fwhm),
                    fdf_mask_arr=fdf_mask_arr[:, y, x],
                ),
                replace(
                    clean_options,
                    mask=noise[y, x] * 5,
                    threshold=noise[y, x],
                    fdf_noise=noise[y, x] if adaptive else None,
                ),
            )
            assert results.clean_iter_arr[y, x] == pixel.iter_count
            for got, expected in (
                # rmclean adds the residual back in
                (
                    results.clean_fdf_arr[:, y, x],
                    pixel.clean_fdf_spectrum + pixel.resid_fdf_spectrum,
                ),
                (results.model_fdf_arr[:, y, x], pixel.model_fdf_spectrum),
                (results.resid_fdf_arr[:, y, x], pixel.resid_fdf_spectrum),
            ):
                np.testing.assert_array_equal(got, expected)
    assert (results.clean_iter_arr > 0).sum() > 256


@pytest.mark.slow_uncompiled
@pytest.mark.parametrize("mode", ["adaptive", "fixed"])
def test_rmclean_matches_main(mode: Literal["adaptive", "fixed"]) -> None:
    """rmclean as `main` (aa208dc) ran it on a saved cube: delay spectra with
    per-pixel RMSFs, noise and thresholds, and an FDF mask"""
    saved = np.load(str(resources.files("rm_lite.data.tests") / "rmclean_main.npz"))
    noise = saved["noise"]
    with quiet_logs():
        results = rmclean(
            RMSynthArrays(
                dirty_fdf_arr=saved["dirty"],
                phi_arr_radm2=saved["phi"],
                phi_double_arr_radm2=saved["phi_double"],
                rmsf_arr=saved["rmsf"],
                fwhm_rmsf_arr=saved["fwhm"],
                fdf_mask_arr=saved["fdf_mask"],
            ),
            RMCleanOptions(
                mask=5 * noise,
                threshold=3 * noise,
                fdf_noise=noise if mode == "adaptive" else None,
            ),
        )
    np.testing.assert_array_equal(
        results.clean_iter_arr, saved[f"{mode}_clean_iter_arr"]
    )
    # Equal to rounding, as `assert_same_clean`
    for name in ("clean_fdf_arr", "model_fdf_arr", "resid_fdf_arr"):
        expected = saved[f"{mode}_{name}"]
        atol = 100 * np.finfo(expected.dtype).eps * np.abs(expected).max()
        np.testing.assert_allclose(getattr(results, name), expected, rtol=0, atol=atol)
