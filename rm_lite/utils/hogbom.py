"""Compiled Hogbom minor loop for RM-CLEAN"""

from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple

import numba
import numpy as np
from numpy.typing import NDArray

from rm_lite.utils.arrays import nb_median


class HogbomStop(IntEnum):
    """Why the minor loop stopped"""

    EMPTY_MASK = 0
    MAX_ITER = 1
    THRESHOLD = 2
    DIVERGED = 3
    NOT_STARTED = 4
    """`start_iter` was already past `max_iter`"""


class Peaks(NamedTuple):
    """Peaks of the residual. Amplitudes are squared: they order channels as
    |resid| does, without a sqrt each."""

    mask_sq: float
    """Largest |resid|**2 in the mask (NaN if the mask holds a NaN)"""
    mask_index: int
    """Its first channel, or the mask's first NaN, as np.argmax picks"""
    global_sq: float
    """Largest |resid|**2 anywhere, ignoring NaN (-1 if all NaN)"""
    mask_any: bool
    """Whether the mask has any channel"""


@numba.njit(cache=True, nogil=True, inline="always")  # type: ignore[misc]
def abs_sq(value: complex) -> float:
    """|value|**2 in float64, which cannot overflow for complex64.

    Args:
        value (complex): Value.

    Returns:
        float: Squared amplitude.
    """
    re, im = np.float64(value.real), np.float64(value.imag)
    return float(re * re + im * im)


@numba.njit(cache=True, nogil=True)  # type: ignore[misc]
def find_peaks(
    resid: NDArray[np.complexfloating],
    mask: NDArray[np.bool_],
    mask_original: NDArray[np.bool_],
    rmsf: NDArray[np.complexfloating],
    component: complex,
    rmsf_start: int,
    threshold_sq: float,
    subtract: bool,
    threshold_mask_update: bool,
) -> Peaks:
    """Find the residual's peaks, first subtracting a component's RMSF if
    `subtract`, in one pass.

    Args:
        resid (NDArray[np.complexfloating]): Residual, updated in place.
        mask (NDArray[np.bool_]): Channels to clean, updated in place.
        mask_original (NDArray[np.bool_]): Channels the mask may hold.
        rmsf (NDArray[np.complexfloating]): Double-length RMSF.
        component (complex): Clean component to subtract.
        rmsf_start (int): Channel `chan` subtracts `rmsf[(chan + rmsf_start) % len(rmsf)]`.
        threshold_sq (float): Squared mask threshold.
        subtract (bool): Subtract the component.
        threshold_mask_update (bool): Remake the mask as `|resid|**2 > threshold_sq`
            within `mask_original` while subtracting.

    Returns:
        Peaks: Peaks of the (new) residual.
    """
    n_chan, n_rmsf = resid.size, rmsf.size
    mask_sq, mask_index, nan_index, global_sq = -1.0, -1, -1, -1.0
    mask_any = False
    # Two runs either side of where the RMSF index wraps: faster than testing
    # for the wrap every channel
    wrap = min(n_chan, n_rmsf - rmsf_start)
    for start, stop, offset in ((0, wrap, rmsf_start), (wrap, n_chan, -wrap)):
        for chan in range(start, stop):
            value = resid[chan]
            if subtract:
                value = value - component * rmsf[chan + offset]
                resid[chan] = value
            amp_sq = abs_sq(value)
            global_sq = amp_sq if amp_sq > global_sq else global_sq
            if subtract and threshold_mask_update:
                mask[chan] = mask_original[chan] and amp_sq > threshold_sq
            if mask[chan]:
                mask_any = True
                if amp_sq > mask_sq:
                    mask_sq, mask_index = amp_sq, chan
                elif nan_index < 0 and np.isnan(amp_sq):
                    nan_index = chan
    if nan_index >= 0:
        mask_sq, mask_index = np.nan, nan_index
    return Peaks(mask_sq, mask_index, global_sq, mask_any)


@numba.njit(cache=True, nogil=True)  # type: ignore[misc]
def offsource_rms(resid: NDArray[np.complexfloating], mask: NDArray[np.bool_]) -> float:
    """Robust (MAD) RMS of the residual outside the mask.

    Sidelobe-inflated while a bright source remains, falling to the noise once
    it subtracts; this is what makes the auto-mask contract off the sidelobes.

    Args:
        resid (NDArray[np.complexfloating]): Residual.
        mask (NDArray[np.bool_]): Current clean region.

    Returns:
        float: RMS of the real and imaginary parts pooled, or 0.0 with fewer
            than 8 channels outside the mask.
    """
    off = resid[~mask]
    if off.size < 8:
        return 0.0
    pooled = np.concatenate((off.real, off.imag))
    return float(1.4826 * nb_median(np.abs(pooled - nb_median(pooled))))


@numba.njit(cache=True, nogil=True)  # type: ignore[misc]
def seed_index(
    resid: NDArray[np.complexfloating], cap: NDArray[np.bool_], floor: float
) -> int:
    """Channel to seed the adaptive mask with: the brightest one allowed.

    The global peak is always real emission, so seeding it is safe. Below the
    floor (noise only) there is no seed, so nothing cleans.

    Args:
        resid (NDArray[np.complexfloating]): Residual.
        cap (NDArray[np.bool_]): Channels allowed.
        floor (float): Amplitude the seed must clear.

    Returns:
        int: Seed channel, or -1 for none.
    """
    peak_sq, peak_index = -1.0, -1
    for chan in range(resid.size):
        if cap[chan]:
            amp_sq = abs_sq(resid[chan])
            if np.isnan(amp_sq):
                # np.argmax would pick it, and NaN never clears the floor
                return -1
            if amp_sq > peak_sq:
                peak_sq, peak_index = amp_sq, chan
    if peak_index >= 0 and np.sqrt(peak_sq) > floor:
        return peak_index
    return -1


@numba.njit(cache=True, nogil=True)  # type: ignore[misc]
def hogbom_minor_loop(
    resid: NDArray[np.complexfloating],
    mask: NDArray[np.bool_],
    mask_original: NDArray[np.bool_],
    rmsf: NDArray[np.complexfloating],
    max_rmsf_index: int,
    gain: float,
    stopping_threshold: float,
    mask_threshold: float,
    update_mask: bool,
    noise: float,
    stall_iters: int,
    start_iter: int,
    max_iter: int,
    guard_divergence: bool,
    divergence_fraction: float,
) -> tuple[NDArray[np.complexfloating], int, int, float, int]:
    """Hogbom iterations from a starting mask, in place, until the loop stops.

    Args:
        resid (NDArray[np.complexfloating]): Residual, cleaned in place.
        mask (NDArray[np.bool_]): Starting clean region, updated in place.
        mask_original (NDArray[np.bool_]): Channels that may be cleaned.
        rmsf (NDArray[np.complexfloating]): Double-length RMSF.
        max_rmsf_index (int): Index of the RMSF peak.
        gain (float): Loop gain, in the residual's real dtype.
        stopping_threshold (float): Peak amplitude at which the loop stops.
        mask_threshold (float): Amplitude below which a channel is not cleaned.
        update_mask (bool): Update the mask as the loop goes.
        noise (float): FDF noise for the adaptive mask; NaN for a fixed one.
        stall_iters (int): Iterations without progress before the adaptive mask
            seeds the global peak.
        start_iter (int): Starting iteration.
        max_iter (int): Maximum number of iterations.
        guard_divergence (bool): Stop if the residual peak runs away.
        divergence_fraction (float): Rise of the residual peak over its lowest
            that counts as running away.

    Returns:
        tuple[NDArray[np.complexfloating], int, int, float, int]: Model,
            iteration count, `HogbomStop`, lowest residual peak and the
            iteration it was seen at.
    """
    adaptive = not np.isnan(noise)
    adaptive_mask_update = update_mask and adaptive
    threshold_mask_update = update_mask and not adaptive
    threshold_sq = mask_threshold * mask_threshold if mask_threshold >= 0 else -1.0
    model = np.zeros_like(resid)
    iter_count = start_iter
    stop = HogbomStop.NOT_STARTED
    best_peak, best_iter = np.inf, -1
    last_recompute_peak, last_recompute_iter = np.inf, iter_count
    peaks = find_peaks(
        resid, mask, mask_original, rmsf, 0j, 0, threshold_sq, False, False
    )
    while iter_count <= max_iter:
        if not peaks.mask_any:
            stop = HogbomStop.EMPTY_MASK
            break
        if iter_count == max_iter:
            stop = HogbomStop.MAX_ITER
            break
        peak = np.sqrt(peaks.mask_sq)
        if peak < stopping_threshold:
            stop = HogbomStop.THRESHOLD
            if not adaptive:
                break
            # Region exhausted, so the main source and its sidelobes are gone:
            # anything still above the mask floor is real emission. Start a new
            # region there instead of stopping.
            seed = seed_index(resid, mask_original, mask_threshold)
            if seed < 0:
                break
            if mask[seed]:
                # Already in the region: every iteration from here on reseeds
                # it and changes nothing
                iter_count = max_iter
            else:
                mask[seed] = True
                last_recompute_peak, last_recompute_iter = np.inf, iter_count
                iter_count += 1
            peaks = find_peaks(
                resid, mask, mask_original, rmsf, 0j, 0, threshold_sq, False, False
            )
            continue
        if guard_divergence:
            # As `CleanProgress.check`, on the peak of the whole residual
            global_peak = np.sqrt(peaks.global_sq) if peaks.global_sq >= 0 else np.nan
            if global_peak - best_peak > divergence_fraction * best_peak:
                stop = HogbomStop.DIVERGED
                break
            if global_peak < best_peak:
                best_peak, best_iter = global_peak, iter_count

        # Add the component, and subtract its RMSF: at channel `chan` that is
        # rmsf[chan + max_rmsf_index - peak channel], wrapping round
        component = gain * resid[peaks.mask_index]
        model[peaks.mask_index] += component
        peaks = find_peaks(
            resid,
            mask,
            mask_original,
            rmsf,
            component,
            (max_rmsf_index - peaks.mask_index) % rmsf.size,
            threshold_sq,
            True,
            threshold_mask_update,
        )

        stalled = iter_count - last_recompute_iter >= stall_iters
        if adaptive_mask_update and (peak < 0.8 * last_recompute_peak or stalled):
            # Each time the peak falls by 0.8, raise the mask threshold by how
            # far sidelobes lift the off-source RMS over the noise, and grow the
            # region (it only grows) into touching channels above it. Growing
            # only by touch keeps it from jumping RMSF nulls onto sidelobe
            # islands.
            ratio = offsource_rms(resid, mask) / noise if noise > 0 else 1.0
            threshold = mask_threshold * max(1.0, ratio)
            last_recompute_peak, last_recompute_iter = peak, iter_count
            grown = mask.copy()
            for chan in range(resid.size):
                touches = (
                    mask[chan]
                    or (chan > 0 and mask[chan - 1])
                    or (chan < resid.size - 1 and mask[chan + 1])
                )
                if (
                    touches
                    and mask_original[chan]
                    and np.sqrt(abs_sq(resid[chan])) > threshold
                ):
                    grown[chan] = True
            mask[:] = grown
            if stalled:
                # The peak stopped falling: it is held up by the sidelobes of a
                # brighter source the growth cannot reach across a null, so seed
                # that one too. A noise-like residual clears no floor and seeds
                # nothing, so a noise-only spectrum still cleans nothing.
                seed = seed_index(resid, mask_original, mask_threshold)
                if seed >= 0:
                    mask[seed] = True
            peaks = find_peaks(
                resid, mask, mask_original, rmsf, 0j, 0, threshold_sq, False, False
            )
        iter_count += 1

    return model, iter_count, stop, best_peak, best_iter


@numba.njit(cache=True, nogil=True)  # type: ignore[misc]
def minor_loop_spectrum(
    resid: NDArray[np.complexfloating],
    resid_start: NDArray[np.complexfloating],
    mask_original: NDArray[np.bool_],
    rmsf: NDArray[np.complexfloating],
    max_rmsf_index: int,
    gain: float,
    stopping_threshold: float,
    mask_threshold: float,
    update_mask: bool,
    noise: float,
    stall_iters: int,
    start_iter: int,
    max_iter: int,
    divergence_fraction: float,
) -> tuple[NDArray[np.complexfloating], NDArray[np.bool_], int, int, float, int]:
    """Hogbom minor loop over one spectrum: the compiled body of
    `rm_lite.utils.clean.minor_loop`.

    Args:
        resid (NDArray[np.complexfloating]): Residual, a copy of `resid_start`,
            cleaned in place.
        resid_start (NDArray[np.complexfloating]): Residual to start from.
        mask_original (NDArray[np.bool_]): Channels that may be cleaned.
        rmsf (NDArray[np.complexfloating]): Double-length RMSF.
        max_rmsf_index (int): Index of the RMSF peak.
        gain (float): Loop gain, in the residual's real dtype.
        stopping_threshold (float): Peak amplitude at which the loop stops.
        mask_threshold (float): Amplitude below which a channel is not cleaned.
        update_mask (bool): Update the mask as the loop goes.
        noise (float): FDF noise for the adaptive mask; NaN for a fixed one.
        stall_iters (int): Iterations without progress before the adaptive mask
            seeds the global peak.
        start_iter (int): Starting iteration.
        max_iter (int): Maximum number of iterations.
        divergence_fraction (float): Rise of the residual peak over its lowest
            that counts as diverging.

    Returns:
        tuple[NDArray[np.complexfloating], NDArray[np.bool_], int, int, float, int]:
            Model, final mask, iteration count, `HogbomStop`, lowest residual
            peak, and channels in the starting mask. On divergence the residual
            and model are those at the lowest peak.
    """
    if update_mask and not np.isnan(noise):
        # Start tight (seed only) so the RMSF sidelobe ringing sits off-source
        # and inflates the MAD; the incoming mask is the cap the mask grows in.
        mask = np.zeros(resid.size, dtype=np.bool_)
        seed = seed_index(resid, mask_original, mask_threshold)
        if seed >= 0:
            mask[seed] = True
    else:
        mask = mask_original.copy()
    mask_start = mask.copy()
    # A blank spectrum has no peak to watch, and cannot run away.
    guard_divergence = False
    for value in resid:
        if np.isfinite(value):
            guard_divergence = True
            break
    args = (
        mask_original,
        rmsf,
        max_rmsf_index,
        gain,
        stopping_threshold,
        mask_threshold,
        update_mask,
        noise,
        stall_iters,
        start_iter,
    )
    model, iter_count, stop, best_peak, best_iter = hogbom_minor_loop(
        resid, mask, *args, max_iter, guard_divergence, divergence_fraction
    )
    if stop == HogbomStop.DIVERGED:
        # The loop is deterministic, so the state at the lowest peak is a rerun
        # that stops at the iteration it was seen at.
        resid[:] = resid_start
        model = hogbom_minor_loop(
            resid,
            mask_start.copy(),
            *args,
            best_iter,
            guard_divergence,
            divergence_fraction,
        )[0]
    return model, mask, iter_count, stop, best_peak, int(np.count_nonzero(mask_start))
