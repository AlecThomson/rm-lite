"""RM-CLEAN utils"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import IntEnum, StrEnum
from typing import Any, Literal, NamedTuple, TypeAlias, TypeVar, get_args

import numba
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import (
    binary_dilation,
    convolve,
    maximum_filter1d,
    minimum_filter1d,
)
from tqdm.auto import tqdm

# numba's cache tracks this file only: clear __pycache__ after editing the nb_
# kernels in arrays.py
from rm_lite.utils.arrays import (
    nb_median,
    nb_nanargmax_abs,
    nd_to_two_d,
    two_d_to_nd,
)
from rm_lite.utils.fitting import (
    fit_rmsf,
    fwhm_to_sigma,
    unit_centred_gaussian,
)
from rm_lite.utils.logging import TqdmToLogger, logger

TQDM_OUT = TqdmToLogger(logger, level=logging.INFO)

DType = TypeVar("DType", bound=np.generic)

KernelType: TypeAlias = Literal["tapered_quad", "gaussian"]

SelectionType: TypeAlias = Literal["snr", "hybrid"]


class CleanState(StrEnum):
    """What a CLEAN loop is doing"""

    CONVERGING = "converging"
    DIVERGING = "diverging"
    STALLED = "stalled"


@dataclass
class CleanProgress:
    """Whether a CLEAN loop is still converging, diverging, or cleaning noise.

    Divergence follows CASA tclean: the peak is taken over the whole spectrum,
    never the masked part, which fluctuates as the mask grows.

    `stall_patience` of 0 leaves the stall test off, for a loop whose peak
    falls too slowly per iteration for it to mean anything.
    """

    model_fdf_spectrum: NDArray[np.complexfloating]
    resid_fdf_spectrum: NDArray[np.complexfloating]
    divergence_fraction: float = 3.0
    stall_patience: int = 0
    stall_rel_improvement: float = 1e-2
    best_peak: float = np.inf
    stall_count: int = 0

    def check(
        self,
        peak: float,
        model_fdf_spectrum: NDArray[np.complexfloating],
        resid_fdf_spectrum: NDArray[np.complexfloating],
    ) -> CleanState:
        """Record this iteration's peak, and say whether the loop should stop."""
        if peak - self.best_peak > self.divergence_fraction * self.best_peak:
            return CleanState.DIVERGING
        if peak < self.best_peak:
            improved = peak < self.best_peak * (1 - self.stall_rel_improvement)
            self.best_peak = peak
            self.model_fdf_spectrum = model_fdf_spectrum.copy()
            self.resid_fdf_spectrum = resid_fdf_spectrum.copy()
            self.stall_count = 0 if improved else self.stall_count + 1
        else:
            self.stall_count += 1
        if self.stall_patience and self.stall_count >= self.stall_patience:
            return CleanState.STALLED
        return CleanState.CONVERGING


class RMCleanResults(NamedTuple):
    """Results of the RM-CLEAN calculation"""

    clean_fdf_arr: NDArray[np.complexfloating]
    """The cleaned Faraday dispersion function cube"""
    model_fdf_arr: NDArray[np.complexfloating]
    """The clean components cube"""
    clean_iter_arr: NDArray[np.int64]
    """CLEAN iterations per pixel. Single-scale: minor iterations (one component
    each). Multiscale: minor cycles (scale re-selections); see
    `sub_minor_iter_arr` for the comparable component count."""
    resid_fdf_arr: NDArray[np.complexfloating]
    """The residual Faraday dispersion function cube"""
    sub_minor_iter_arr: NDArray[np.int64]
    """Total component-placement steps per pixel: single-scale equals
    `clean_iter_arr`; multiscale sums the sub-minor (per-scale Hogbom) iterations
    over all minor cycles. This is the count comparable to single-scale."""


class CleanLoopResults(NamedTuple):
    """Results of the RM-CLEAN loop"""

    clean_fdf_spectrum: NDArray[np.complexfloating]
    """The cleaned Faraday dispersion function cube"""
    resid_fdf_spectrum: NDArray[np.complexfloating]
    """The residual Faraday dispersion function cube"""
    model_fdf_spectrum: NDArray[np.complexfloating]
    """The clean components cube"""
    iter_count: int
    """The number of iterations"""


class MinorLoopResults(NamedTuple):
    """Results of the RM-CLEAN minor loop"""

    resid_fdf_spectrum: NDArray[np.complexfloating]
    """The residual Faraday dispersion function cube"""
    resid_fdf_spectrum_mask: np.ma.MaskedArray
    """The masked residual Faraday dispersion function cube"""
    model_fdf_spectrum: NDArray[np.complexfloating]
    """The clean components cube"""
    iter_count: int
    """The number of iterations"""


PER_PIXEL_CLEAN_FIELDS = ("mask", "threshold", "fdf_noise")
"""`RMCleanOptions` fields that take a per-pixel map as well as a scalar."""


@dataclass(frozen=True, kw_only=True, slots=True)
class RMCleanOptions:
    """Options for RM-CLEAN, shared by the 1D and 3D tools"""

    mask: float | NDArray[np.float64]
    """Masking threshold - pixels below this value are not cleaned. A scalar, or
    a per-pixel map shaped like the FDF's spatial axes."""
    threshold: float | NDArray[np.float64]
    """Cleaning threshold - stop when all pixels are below this value. A scalar,
    or a per-pixel map shaped like the FDF's spatial axes."""
    max_iter: int = 100_000
    """Maximum clean iterations"""
    gain: float = 0.1
    """Clean loop gain"""
    fdf_noise: float | NDArray[np.float64] | None = None
    """Theoretical FDF noise; enables the adaptive off-source auto-mask when set.
    A scalar, or a per-pixel map shaped like the FDF's spatial axes."""

    def __post_init__(self) -> None:
        if self.max_iter < 1:
            msg = f"max_iter must be >= 1, got {self.max_iter}."
            raise ValueError(msg)
        if not 0 < self.gain <= 1:
            msg = f"gain must be in (0, 1], got {self.gain}."
            raise ValueError(msg)


class MinorLoopArrays(NamedTuple):
    """Arrays for the RM-CLEAN minor loop"""

    resid_fdf_spectrum_mask: np.ma.MaskedArray
    """Residual Faraday dispersion function spectrum"""
    phi_arr_radm2: NDArray[np.float64]
    """Faraday depth array in rad/m^2"""
    phi_double_arr_radm2: NDArray[np.float64]
    """Double-length Faraday depth array in rad/m^2"""
    rmsf_spectrum: NDArray[np.complexfloating]
    """RMSF spectrum"""
    rmsf_fwhm: float
    """FWHM of the RMSF"""


@dataclass(frozen=True, kw_only=True, slots=True)
class MinorLoopOptions:
    """Options for the RM-CLEAN minor loop"""

    max_iter: int
    """Maximum number of iterations"""
    gain: float
    """Loop gain"""
    mask_threshold: float
    """Amplitude below which a channel is not cleaned"""
    stopping_threshold: float
    """Peak amplitude at which the loop stops"""
    start_iter: int = 0
    """Starting iteration"""
    update_mask: bool = True
    """Update the mask after each iteration"""
    noise: float | None = None
    """FDF noise floor for the adaptive off-source auto-mask; None disables it
    (fixed mask, legacy behaviour)."""
    stall_iters: int = 300
    """Iterations without progress before the adaptive mask adds the global peak"""


class MinorLoopBatchArrays(NamedTuple):
    """Arrays for the RM-CLEAN minor loop over a stack of spectra"""

    resid_fdf_arr_mask: np.ma.MaskedArray
    """Residual spectra, one per row; masked channels are not cleaned"""
    phi_arr_radm2: NDArray[np.float64]
    """Faraday depth array in rad/m^2"""
    phi_double_arr_radm2: NDArray[np.float64]
    """Double-length Faraday depth array in rad/m^2"""
    rmsf_arr: NDArray[np.complexfloating]
    """RMSF shared by every spectrum (1D), or one per row (2D)"""


@dataclass(frozen=True, kw_only=True, slots=True)
class MinorLoopBatchOptions:
    """`MinorLoopOptions` for a stack of spectra, one value or one per spectrum"""

    max_iter: int
    """Maximum number of iterations"""
    gain: float
    """Loop gain"""
    mask_threshold: float | NDArray[np.float64]
    """Amplitude below which a channel is not cleaned"""
    stopping_threshold: float | NDArray[np.float64]
    """Peak amplitude at which the loop stops"""
    start_iter: int | NDArray[np.int64] = 0
    """Starting iteration"""
    update_mask: bool = True
    """Update the mask after each iteration"""
    noise: float | NDArray[np.float64] | None = None
    """FDF noise for the adaptive off-source auto-mask; None keeps the mask fixed"""
    stall_iters: int = 300
    """Iterations without progress before the adaptive mask adds the global peak"""
    parallel: bool = False
    """Clean the spectra on numba's threads, for a caller that is not already
    parallel. Inside dask tasks leave it off: dask's threads over blocks are
    faster, and numba's workqueue threading layer aborts on launches from
    several threads."""


class MinorLoopBatchResults(NamedTuple):
    """Results of the RM-CLEAN minor loop over a stack of spectra"""

    resid_fdf_arr: NDArray[np.complexfloating]
    """Residual spectra, one per row"""
    resid_fdf_arr_mask: np.ma.MaskedArray
    """Residual spectra, masked outside each one's final clean region"""
    model_fdf_arr: NDArray[np.complexfloating]
    """Clean components, one row per spectrum"""
    iter_count_arr: NDArray[np.int64]
    """Iterations per spectrum"""


def shift_rmsf(
    rmsf_spectrum: NDArray[np.complexfloating],
    fdf_index: int,
    n_phi_pad: int,
    max_rmsf_index: int,
) -> NDArray[np.complexfloating]:
    """Roll the double-length RMSF so its peak sits at FDF channel `fdf_index`,
    then clip to FDF length.

    Shared shift-and-subtract primitive (Hogbom and multiscale); assumes integer
    shifts and a symmetric RMSF.

    Args:
        rmsf_spectrum (NDArray[np.complexfloating]): Double-length RMSF.
        fdf_index (int): Target FDF channel for the RMSF peak.
        n_phi_pad (int): Half the length difference between the double and FDF axes.
        max_rmsf_index (int): Index of the RMSF peak.

    Returns:
        NDArray[np.complexfloating]: RMSF shifted to `fdf_index`, clipped to FDF length.
    """
    fdf_len = len(rmsf_spectrum) - 2 * n_phi_pad
    return np.array(_rmsf_window(rmsf_spectrum, fdf_index, fdf_len, max_rmsf_index))


def _rmsf_window(
    rmsf_spectrum: NDArray[np.complexfloating],
    fdf_index: int,
    fdf_len: int,
    max_rmsf_index: int,
) -> NDArray[np.complexfloating]:
    """`shift_rmsf` as a view of `rmsf_spectrum` where it does not wrap.

    Rolling the RMSF by `fdf_index + n_phi_pad - max_rmsf_index` and clipping
    `n_phi_pad` off each end leaves `rmsf[(j + max_rmsf_index - fdf_index) mod len]`,
    a contiguous slice unless it runs off the end.
    """
    start = (max_rmsf_index - fdf_index) % len(rmsf_spectrum)
    stop = start + fdf_len
    if stop <= len(rmsf_spectrum):
        return rmsf_spectrum[start:stop]
    return np.concatenate(
        (rmsf_spectrum[start:], rmsf_spectrum[: stop - len(rmsf_spectrum)])
    )


class HogbomStop(IntEnum):
    """Why the minor loop stopped"""

    EMPTY_MASK = 0
    """No channel left to clean"""
    MAX_ITER = 1
    """Reached `max_iter`"""
    THRESHOLD = 2
    """Peak in the mask fell below the stopping threshold"""
    DIVERGED = 3
    """Peak of the whole residual ran away; results are from its lowest point"""
    NOT_STARTED = 4
    """`start_iter` was already past `max_iter`"""


class HogbomOptions(NamedTuple):
    """`MinorLoopOptions` as numba can take them"""

    gain: float
    """Loop gain, in the residual's real dtype"""
    stopping_threshold: float
    """Peak amplitude at which the loop stops"""
    mask_threshold: float
    """Amplitude below which a channel is not cleaned"""
    update_mask: bool
    """Update the mask as the loop goes"""
    adaptive: bool
    """Grow the mask from a seed against the off-source noise"""
    noise: float
    """FDF noise for the adaptive mask"""
    stall_iters: int
    """Iterations without progress before the adaptive mask seeds the global peak"""
    start_iter: int
    """Starting iteration"""
    max_iter: int
    """Maximum number of iterations"""
    divergence_fraction: float
    """Rise of the residual peak over its lowest that counts as diverging"""


class _HogbomSettings(NamedTuple):
    """The `HogbomOptions` shared by every spectrum of a batch"""

    gain: float
    """Loop gain, in the residual's real dtype"""
    update_mask: bool
    """Update the mask as the loop goes"""
    adaptive: bool
    """Grow the mask from a seed against the off-source noise"""
    stall_iters: int
    """Iterations without progress before the adaptive mask seeds the global peak"""
    max_iter: int
    """Maximum number of iterations"""
    divergence_fraction: float
    """Rise of the residual peak over its lowest that counts as diverging"""


class _HogbomBatch(NamedTuple):
    """Spectra for the Hogbom loop, one per row, with their own thresholds"""

    resid: NDArray[np.complexfloating]
    """Residuals, copies of `resid_start`, cleaned in place"""
    resid_start: NDArray[np.complexfloating]
    """Residuals to start from"""
    mask_original: NDArray[np.bool_]
    """Channels that may be cleaned"""
    rmsf: NDArray[np.complexfloating]
    """Double-length RMSFs: one row for every spectrum, or one each"""
    max_rmsf_index: NDArray[np.int64]
    """Index of each RMSF's peak"""
    stopping_threshold: NDArray[np.float64]
    """Peak amplitude at which each loop stops"""
    mask_threshold: NDArray[np.float64]
    """Amplitude below which a channel is not cleaned"""
    noise: NDArray[np.float64]
    """FDF noise for the adaptive mask"""
    start_iter: NDArray[np.int64]
    """Starting iteration"""


class _HogbomSpectrumResults(NamedTuple):
    """`minor_loop_spectrum` on one spectrum"""

    model: NDArray[np.complexfloating]
    """Clean components"""
    mask: NDArray[np.bool_]
    """Final mask"""
    iter_count: int
    """Iterations"""
    stop: int
    """`HogbomStop`"""
    best_peak: float
    """Lowest peak of the whole residual"""
    n_start: int
    """Channels in the starting mask"""


class _HogbomBatchResults(NamedTuple):
    """`_HogbomSpectrumResults` for a stack of spectra, one per row"""

    model: NDArray[np.complexfloating]
    """Clean components"""
    mask: NDArray[np.bool_]
    """Final masks"""
    iter_count: NDArray[np.int64]
    """Iterations"""
    stop: NDArray[np.int64]
    """`HogbomStop`"""
    best_peak: NDArray[np.float64]
    """Lowest peak of the whole residual"""
    n_start: NDArray[np.int64]
    """Channels in the starting mask"""


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
    # Below 8 channels (16 values) the estimate is too noisy to raise the mask
    # threshold with: at 8 it scatters by 29%, and is 1.5x too high 3% of the
    # time. 0.0 leaves the threshold at the FDF noise.
    if off.size < 8:
        return 0.0
    pooled = np.concatenate((off.real, off.imag))
    # 1.4826 = 1 / Phi^-1(3/4), the MAD to standard deviation of a Gaussian
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
    options: HogbomOptions,
    max_iter: int,
    guard_divergence: bool,
) -> tuple[NDArray[np.complexfloating], int, int, float, int]:
    """Hogbom iterations from a starting mask, in place, until the loop stops.

    Args:
        resid (NDArray[np.complexfloating]): Residual, cleaned in place.
        mask (NDArray[np.bool_]): Starting clean region, updated in place.
        mask_original (NDArray[np.bool_]): Channels that may be cleaned.
        rmsf (NDArray[np.complexfloating]): Double-length RMSF.
        max_rmsf_index (int): Index of the RMSF peak.
        options (HogbomOptions): Loop options.
        max_iter (int): Iteration to stop at (`options.max_iter`, or earlier for
            the rerun after divergence).
        guard_divergence (bool): Stop if the residual peak runs away.

    Returns:
        tuple[NDArray[np.complexfloating], int, int, float, int]: Model,
            iteration count, `HogbomStop`, lowest residual peak and the
            iteration it was seen at.
    """
    gain, stopping_threshold, mask_threshold = (
        options.gain,
        options.stopping_threshold,
        options.mask_threshold,
    )
    noise, stall_iters = options.noise, options.stall_iters
    adaptive = options.adaptive
    adaptive_mask_update = options.update_mask and adaptive
    threshold_mask_update = options.update_mask and not adaptive
    # A NaN threshold stays NaN, so no channel clears it, as in NumPy
    threshold_sq = -1.0 if mask_threshold < 0 else mask_threshold * mask_threshold
    model = np.zeros_like(resid)
    iter_count = options.start_iter
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
            if global_peak - best_peak > options.divergence_fraction * best_peak:
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
    options: HogbomOptions,
) -> _HogbomSpectrumResults:
    """Hogbom minor loop over one spectrum: the compiled body of `minor_loop`.

    Args:
        resid (NDArray[np.complexfloating]): Residual, a copy of `resid_start`,
            cleaned in place.
        resid_start (NDArray[np.complexfloating]): Residual to start from.
        mask_original (NDArray[np.bool_]): Channels that may be cleaned.
        rmsf (NDArray[np.complexfloating]): Double-length RMSF.
        max_rmsf_index (int): Index of the RMSF peak.
        options (HogbomOptions): Loop options.

    Returns:
        _HogbomSpectrumResults: Model, final mask, iterations, why it stopped, lowest
            residual peak and starting mask size. On divergence the residual and
            model are those at the lowest peak.
    """
    if options.update_mask and options.adaptive:
        # Start tight (seed only) so the RMSF sidelobe ringing sits off-source
        # and inflates the MAD; the incoming mask is the cap the mask grows in.
        mask = np.zeros(resid.size, dtype=np.bool_)
        seed = seed_index(resid, mask_original, options.mask_threshold)
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
    model, iter_count, stop, best_peak, best_iter = hogbom_minor_loop(
        resid,
        mask,
        mask_original,
        rmsf,
        max_rmsf_index,
        options,
        options.max_iter,
        guard_divergence,
    )
    if stop == HogbomStop.DIVERGED:
        # The loop is deterministic, so the state at the lowest peak is a rerun
        # that stops at the iteration it was seen at.
        resid[:] = resid_start
        model = hogbom_minor_loop(
            resid,
            mask_start.copy(),
            mask_original,
            rmsf,
            max_rmsf_index,
            options,
            best_iter,
            guard_divergence,
        )[0]
    return _HogbomSpectrumResults(
        model, mask, iter_count, stop, best_peak, int(np.count_nonzero(mask_start))
    )


@numba.njit(cache=True, nogil=True)  # type: ignore[misc]
def _minor_loop_spectra(
    batch: _HogbomBatch,
    settings: _HogbomSettings,
    results: _HogbomBatchResults,
    first: int,
    last: int,
) -> None:
    """`minor_loop_spectrum` on spectra (rows) `first` to `last - 1`, in turn.

    Args:
        batch (_HogbomBatch): Spectra, RMSFs and per-spectrum thresholds.
        settings (_HogbomSettings): Settings for every spectrum.
        results (_HogbomBatchResults): Filled in place.
        first (int): First spectrum to clean.
        last (int): One past the last spectrum to clean.
    """
    for spec in range(first, last):
        row = spec if batch.rmsf.shape[0] > 1 else 0
        result = minor_loop_spectrum(
            batch.resid[spec],
            batch.resid_start[spec],
            batch.mask_original[spec],
            batch.rmsf[row],
            batch.max_rmsf_index[row],
            HogbomOptions(
                gain=settings.gain,
                stopping_threshold=batch.stopping_threshold[spec],
                mask_threshold=batch.mask_threshold[spec],
                update_mask=settings.update_mask,
                adaptive=settings.adaptive,
                noise=batch.noise[spec],
                stall_iters=settings.stall_iters,
                start_iter=batch.start_iter[spec],
                max_iter=settings.max_iter,
                divergence_fraction=settings.divergence_fraction,
            ),
        )
        results.model[spec] = result.model
        results.mask[spec] = result.mask
        results.iter_count[spec] = result.iter_count
        results.stop[spec] = result.stop
        results.best_peak[spec] = result.best_peak
        results.n_start[spec] = result.n_start


@numba.njit(cache=True, nogil=True, parallel=True)  # type: ignore[misc]
def _minor_loop_spectra_parallel(
    batch: _HogbomBatch, settings: _HogbomSettings, results: _HogbomBatchResults
) -> None:
    """`_minor_loop_spectra` over every spectrum, spread over numba's threads.

    Args:
        batch (_HogbomBatch): As `_minor_loop_spectra`.
        settings (_HogbomSettings): As `_minor_loop_spectra`.
        results (_HogbomBatchResults): As `_minor_loop_spectra`.
    """
    for spec in numba.prange(batch.resid.shape[0]):
        _minor_loop_spectra(batch, settings, results, spec, spec + 1)


def minor_loop(
    minor_loop_arrays: MinorLoopArrays,
    minor_loop_options: MinorLoopOptions,
) -> MinorLoopResults:
    """Hogbom minor loop: shift-subtract the RMSF at each residual peak within the
    mask until the peak drops below `threshold` (or max_iter).

    Args:
        minor_loop_arrays (MinorLoopArrays): Masked residual, phi axes, RMSF, FWHM.
        minor_loop_options (MinorLoopOptions): Gain, mask, threshold, max_iter, mask update.

    Raises:
        ValueError: If the RMSF, clipped to the FDF axis, is not as long as the residual.

    Returns:
        MinorLoopResults: Clean, residual, model, masked residual, and iteration count.
    """
    results = minor_loop_batch(
        MinorLoopBatchArrays(
            resid_fdf_arr_mask=minor_loop_arrays.resid_fdf_spectrum_mask[np.newaxis],
            phi_arr_radm2=minor_loop_arrays.phi_arr_radm2,
            phi_double_arr_radm2=minor_loop_arrays.phi_double_arr_radm2,
            rmsf_arr=minor_loop_arrays.rmsf_spectrum,
        ),
        MinorLoopBatchOptions(**dataclasses.asdict(minor_loop_options)),
    )
    return MinorLoopResults(
        resid_fdf_spectrum=results.resid_fdf_arr[0],
        resid_fdf_spectrum_mask=results.resid_fdf_arr_mask[0],
        model_fdf_spectrum=results.model_fdf_arr[0],
        iter_count=int(results.iter_count_arr[0]),
    )


def _check_per_spectrum(options: dict[str, Any], n_spectra: int) -> None:
    """Check each option is one value, or one for each spectrum.

    Args:
        options (dict[str, Any]): Options by name.
        n_spectra (int): Number of spectra.

    Raises:
        ValueError: If an option has any other shape.
    """
    for name, value in options.items():
        if np.ndim(value) != 0 and np.shape(value) != (n_spectra,):
            msg = (
                f"`{name}` must be one value, or one for each of the {n_spectra} "
                f"spectra; got shape {np.shape(value)}."
            )
            raise ValueError(msg)


def minor_loop_batch(
    minor_loop_arrays: MinorLoopBatchArrays,
    minor_loop_options: MinorLoopBatchOptions,
) -> MinorLoopBatchResults:
    """`minor_loop` over a stack of spectra in one call, with the same results.

    Args:
        minor_loop_arrays (MinorLoopBatchArrays): Masked residuals, phi axes, RMSF(s).
        minor_loop_options (MinorLoopBatchOptions): Gain, thresholds, noise, max_iter.

    Raises:
        ValueError: If the arrays or per-spectrum options do not line up.

    Returns:
        MinorLoopBatchResults: Residuals, models, final masks and iteration counts.
    """
    resid_fdf_arr_mask = minor_loop_arrays.resid_fdf_arr_mask
    if resid_fdf_arr_mask.ndim != 2:
        msg = (
            f"Residuals must be 2D (spectrum, channel), got {resid_fdf_arr_mask.ndim}D."
        )
        raise ValueError(msg)
    n_spectra, n_chan = resid_fdf_arr_mask.shape
    # The compiled loop wants contiguous, native-byte-order arrays
    resid_start = np.ascontiguousarray(
        resid_fdf_arr_mask.data,
        dtype=resid_fdf_arr_mask.dtype.newbyteorder("="),
    )
    rmsf_arr = np.ascontiguousarray(
        np.atleast_2d(minor_loop_arrays.rmsf_arr),
        dtype=minor_loop_arrays.rmsf_arr.dtype.newbyteorder("="),
    )
    if rmsf_arr.shape[0] not in (1, n_spectra):
        msg = f"Need one RMSF, or one for each of the {n_spectra} spectra; got {rmsf_arr.shape[0]}."
        raise ValueError(msg)
    n_phi_pad = (
        len(minor_loop_arrays.phi_double_arr_radm2)
        - len(minor_loop_arrays.phi_arr_radm2)
    ) // 2
    fdf_len = rmsf_arr.shape[1] - 2 * n_phi_pad
    if fdf_len != n_chan:
        msg = (
            f"The RMSF clipped to the FDF axis has {fdf_len} channels, but the "
            f"residuals have {n_chan}."
        )
        raise ValueError(msg)
    max_rmsf_index = nb_nanargmax_abs(rmsf_arr)
    if (max_rmsf_index < 0).any():
        msg = "An RMSF is all NaN."
        raise ValueError(msg)
    noise = minor_loop_options.noise
    per_spectrum = {
        "stopping_threshold": minor_loop_options.stopping_threshold,
        "mask_threshold": minor_loop_options.mask_threshold,
        "noise": 0.0 if noise is None else noise,
        "start_iter": minor_loop_options.start_iter,
    }
    _check_per_spectrum(per_spectrum, n_spectra)

    batch = _HogbomBatch(
        resid=resid_start.copy(),
        resid_start=resid_start,
        mask_original=np.ascontiguousarray(~np.ma.getmaskarray(resid_fdf_arr_mask)),
        rmsf=rmsf_arr,
        max_rmsf_index=max_rmsf_index,
        stopping_threshold=np.full(
            n_spectra, per_spectrum["stopping_threshold"], dtype=np.float64
        ),
        mask_threshold=np.full(
            n_spectra, per_spectrum["mask_threshold"], dtype=np.float64
        ),
        noise=np.full(n_spectra, per_spectrum["noise"], dtype=np.float64),
        start_iter=np.full(n_spectra, per_spectrum["start_iter"], dtype=np.int64),
    )
    settings = _HogbomSettings(
        # gain * peak in the residuals' own precision, as NumPy does it
        gain=resid_start.real.dtype.type(minor_loop_options.gain),
        update_mask=minor_loop_options.update_mask,
        adaptive=noise is not None,
        stall_iters=minor_loop_options.stall_iters,
        max_iter=minor_loop_options.max_iter,
        divergence_fraction=CleanProgress.divergence_fraction,
    )
    results = _HogbomBatchResults(
        model=np.zeros_like(resid_start),
        mask=np.zeros((n_spectra, n_chan), dtype=bool),
        iter_count=np.zeros(n_spectra, dtype=np.int64),
        stop=np.zeros(n_spectra, dtype=np.int64),
        best_peak=np.zeros(n_spectra),
        n_start=np.zeros(n_spectra, dtype=np.int64),
    )
    if minor_loop_options.parallel:
        # Spectra cost anything from microseconds to seconds, so hand them to
        # the threads one at a time rather than in equal blocks
        with numba.parallel_chunksize(1):
            _minor_loop_spectra_parallel(batch, settings, results)
    else:
        _minor_loop_spectra(batch, settings, results, 0, n_spectra)
    _log_stops(results)
    return MinorLoopBatchResults(
        resid_fdf_arr=batch.resid,
        resid_fdf_arr_mask=np.ma.array(batch.resid, mask=~results.mask),
        model_fdf_arr=results.model,
        iter_count_arr=results.iter_count,
    )


def _log_stops(results: _HogbomBatchResults) -> None:
    """Log why each minor loop stopped: in full for one spectrum, as a count of
    each reason for a stack."""
    if results.stop.size == 0:
        return
    if results.stop.size > 1:
        stops = dict(zip(*np.unique(results.stop, return_counts=True), strict=True))
        logger.info(
            f"Minor loop over {results.stop.size} spectra: "
            + ", ".join(
                f"{count} {HogbomStop(stop).name.lower()}"
                for stop, count in stops.items()
            )
        )
        for stop in (HogbomStop.EMPTY_MASK, HogbomStop.MAX_ITER, HogbomStop.DIVERGED):
            if stop in stops:
                logger.warning(f"{stops[stop]} spectra stopped: {stop.name.lower()}")
        return
    iter_count, reason = int(results.iter_count[0]), int(results.stop[0])
    logger.info(f"Starting minor loop... {results.n_start[0]} pixels in the mask")
    if reason == HogbomStop.EMPTY_MASK:
        logger.warning(
            f"All channels masked. Exiting loop...performed {iter_count} iterations"
        )
    elif reason == HogbomStop.MAX_ITER:
        logger.warning(
            f"Max iterations reached. Exiting loop...performed {iter_count} iterations"
        )
    elif reason == HogbomStop.THRESHOLD:
        logger.info(
            f"Threshold reached. Exiting loop...performed {iter_count} iterations"
        )
    elif reason == HogbomStop.DIVERGED:
        logger.warning(
            f"CLEAN {CleanState.DIVERGING} at iter {iter_count} (best peak "
            f"{results.best_peak[0]:0.3g}); "
            "stopping at best."
        )


def restore_model(
    model_fdf_spectrum: NDArray[np.complexfloating],
    phi_arr_radm2: NDArray[np.float64],
    rmsf_fwhm: float,
) -> NDArray[np.complexfloating]:
    """Restore a delta model with a unit-peak clean beam (one pass).

    Sum of per-component Gaussians == the delta model convolved with the clean
    beam; doing it once from the final model is identical to the old
    per-iteration restore but avoids a full-array Gaussian every minor iteration.
    Distinct component channels are bounded by the FDF length, so the outer sum
    is at most len(phi)^2.
    """
    nonzero = np.nonzero(model_fdf_spectrum)[0]
    if nonzero.size == 0:
        return np.zeros_like(model_fdf_spectrum)
    sigma = fwhm_to_sigma(rmsf_fwhm)
    beams = np.exp(
        -0.5 * ((phi_arr_radm2[None, :] - phi_arr_radm2[nonzero][:, None]) / sigma) ** 2
    )
    return np.asarray(
        (model_fdf_spectrum[nonzero][:, None] * beams).sum(axis=0),
        dtype=model_fdf_spectrum.dtype,
    )


class RMSynthArrays(NamedTuple):
    """Arrays for RM-synthesis"""

    dirty_fdf_arr: NDArray[np.complexfloating]
    """Dirty Faraday dispersion function array"""
    phi_arr_radm2: NDArray[np.float64]
    """Faraday depth array in rad/m^2"""
    phi_double_arr_radm2: NDArray[np.float64]
    """Double-length Faraday depth array in rad/m^2"""
    rmsf_arr: NDArray[np.complexfloating]
    """RMSF array"""
    fwhm_rmsf_arr: NDArray[np.float64]
    """FWHM of the RMSF array"""
    fdf_mask_arr: NDArray[np.bool_] | None = None
    """Mask of pixels to clean"""


def _fdf_peak_abs(
    dirty_fdf_arr_2d: NDArray[np.complexfloating],
) -> NDArray[np.float64]:
    """Per-pixel max |dirty FDF|; NaN for a fully blanked (all-NaN) pixel."""
    n_phi, n_pix = dirty_fdf_arr_2d.shape
    peak_abs_arr = np.empty(n_pix, dtype=np.float64)
    # Strip-wise, so the temporary |FDF| is a slice rather than a second array
    # the size of the whole block.
    strip = max(1, 2**20 // max(1, n_phi))
    for start in range(0, n_pix, strip):
        # `fmax.reduce`, not `nanmax`: same NaN-skipping maximum, but an
        # all-NaN column (a blanked pixel, routine at a mosaic edge) yields
        # NaN quietly instead of an "All-NaN slice" warning per strip.
        peak_abs_arr[start : start + strip] = np.fmax.reduce(
            np.abs(dirty_fdf_arr_2d[:, start : start + strip]), axis=0
        )
    return peak_abs_arr


def _null_clean_pixels(
    dirty_fdf_arr_2d: NDArray[np.complexfloating],
    mask: float | NDArray[np.float64],
) -> NDArray[np.bool_]:
    """Pixels whose dirty FDF peak cannot clear `mask`, so CLEAN is a no-op.

    Such a pixel is not an approximation to skip, it is already solved. The
    initial minor loop's mask is empty for it (the adaptive seed gate and the
    fixed mask are both `|FDF| > mask`), so that loop breaks at iteration 0
    with a zero model; the deep loop's mask (`|model| > 0`) is empty too; and
    `restore_model` of a zero model is zero. `rmclean`'s pre-loop state --
    zero clean, zero model, residual = dirty, zero iterations -- is exactly
    that answer, and its post-loop `clean += resid` then gives `clean = dirty`.

    Worth screening for because off-source pixels outnumber detections by
    orders of magnitude in any real cube, and each one otherwise spends
    ~200 us of masked-array construction and array copies to rediscover it.

    A fully blanked (all-NaN) spectrum is included: `~(peak > mask)` is True
    for NaN, and the skipped result (residual = NaN, zero model) is what the
    loop produces for it as well.
    """
    return ~(_fdf_peak_abs(dirty_fdf_arr_2d) > mask)


def _blank_pixels(
    dirty_fdf_arr_2d: NDArray[np.complexfloating],
) -> NDArray[np.bool_]:
    """Fully blanked (all-NaN) pixels, which no CLEAN mode can say anything about.

    Narrower than `_null_clean_pixels`, for non-adaptive multiscale where the
    general no-op argument does not hold. A blank pixel is still hopeless there,
    and worse than useless: its RMSF is all zeros, so `compute_scale_kernels`
    normalises by zero and `fit_rmsf` raises `array must not contain infs or
    NaNs`. Skipping leaves the NaN residual the pixel came in with.
    """
    return np.isnan(_fdf_peak_abs(dirty_fdf_arr_2d))


class PixelCleanOptions:
    """Hands each pixel the CLEAN options that apply to it.

    `mask`, `threshold` and `fdf_noise` are scalars, or per-pixel maps when the
    noise varies across the image. Maps are flattened once, like the FDF, and
    indexed as the pixel loop reaches them; all-scalar options pass straight
    through and allocate nothing.
    """

    def __init__(self, clean_options: RMCleanOptions, spatial_shape: tuple[int, ...]):
        self._options = clean_options
        self._per_pixel = {
            field: nd_to_two_d(np.asarray(value)).reshape(-1)
            for field in PER_PIXEL_CLEAN_FIELDS
            if np.ndim(value := getattr(clean_options, field)) != 0
        }
        for field, values in self._per_pixel.items():
            if values.shape != spatial_shape:
                msg = (
                    f"A per-pixel CLEAN {field} must match the FDF's spatial "
                    f"shape {spatial_shape}, got {values.shape}."
                )
                raise ValueError(msg)

    @property
    def mask_1d(self) -> float | NDArray[np.float64]:
        """The mask as the pixel loop sees it: a scalar, or one value per pixel."""
        return self._per_pixel.get("mask", self._options.mask)

    def describe_mask(self) -> str:
        mask = self.mask_1d
        if np.ndim(mask) == 0:
            return f"{float(mask):0.3g}"
        return f"{np.min(mask):0.3g} to {np.max(mask):0.3g} per-pixel"

    def at(self, pix_idx: int | NDArray[np.intp]) -> RMCleanOptions:
        """The options for one pixel, or for an array of pixels, where each
        per-pixel map becomes one value per pixel."""
        if not self._per_pixel:
            return self._options
        per_pixel: dict[str, Any] = {
            field: values[pix_idx] for field, values in self._per_pixel.items()
        }
        return replace(self._options, **per_pixel)


def rmclean(
    rm_synth_arrays: RMSynthArrays,
    clean_options: RMCleanOptions,
    multiscale_options: MultiscaleOptions | None = None,
    phi_max_scale_radm2: float | None = None,
) -> RMCleanResults:
    """Validate array shapes and CLEAN each pixel (single- or multi-scale).

    Args:
        rm_synth_arrays (RMSynthArrays): Dirty FDF cube, phi axes, RMSF, masks.
        clean_options (RMCleanOptions): Mask, threshold, gain, max_iter, FDF noise.
        multiscale_options (MultiscaleOptions | None): Multiscale RM-CLEAN settings
            (recovers Faraday-thick structure). None runs single-scale RM-CLEAN.
        phi_max_scale_radm2 (float | None): Largest recoverable Faraday scale
            (pi / lambda_sq_min); sets the auto scale range. None falls back to the
            phi window.

    Raises:
        ValueError: If the input array shapes are inconsistent.

    Returns:
        RMCleanResults: Per-pixel clean/model/resid cubes and iteration counts.
    """
    multiscale = multiscale_options is not None
    # Sanity checks on array sizes
    checks: list[tuple[bool, str]] = [
        (
            rm_synth_arrays.phi_arr_radm2.shape[0]
            == rm_synth_arrays.dirty_fdf_arr.shape[0],
            f"'phi_arr_radm2' (size {rm_synth_arrays.phi_arr_radm2.shape[0]}) and 'dirty_fdf_arr' (size {rm_synth_arrays.dirty_fdf_arr.shape[0]}) are not the same length.",
        ),
        (
            rm_synth_arrays.phi_double_arr_radm2.shape[0]
            == rm_synth_arrays.rmsf_arr.shape[0],
            f"Mismatch in 'phi_double_arr_radm2' (size {rm_synth_arrays.phi_double_arr_radm2.shape[0]}) and 'rmsf_arr' (size {rm_synth_arrays.rmsf_arr.shape[0]}) length.",
        ),
        (
            len(rm_synth_arrays.phi_double_arr_radm2)
            >= 2 * len(rm_synth_arrays.phi_arr_radm2),
            f"The Faraday depth of the RMSF (size {len(rm_synth_arrays.phi_double_arr_radm2)}) must be at least twice the FDF (size {len(rm_synth_arrays.phi_arr_radm2)}).",
        ),
        (
            rm_synth_arrays.dirty_fdf_arr.ndim <= 3,
            f"FDF array dimensions ({rm_synth_arrays.dirty_fdf_arr.ndim}) must be <= 3.",
        ),
        (
            rm_synth_arrays.dirty_fdf_arr.ndim == rm_synth_arrays.rmsf_arr.ndim,
            f"The input RMSF (ndim {rm_synth_arrays.rmsf_arr.ndim}) and FDF (ndim {rm_synth_arrays.dirty_fdf_arr.ndim}) must have the same number of axes.",
        ),
        (
            rm_synth_arrays.rmsf_arr.shape[1:]
            == rm_synth_arrays.dirty_fdf_arr.shape[1:],
            f"The xy dimensions of the RMSF {rm_synth_arrays.rmsf_arr.shape[1:]} and FDF {rm_synth_arrays.dirty_fdf_arr.shape[1:]} must match.",
        ),
    ]
    if rm_synth_arrays.fdf_mask_arr is not None:
        checks.append(
            (
                rm_synth_arrays.fdf_mask_arr.shape
                == rm_synth_arrays.dirty_fdf_arr.shape,
                f"Mask array dimensions {rm_synth_arrays.fdf_mask_arr.shape} must match the xy dimensions of the FDF cube {rm_synth_arrays.dirty_fdf_arr}.",
            )
        )

    for condition, error_msg in checks:
        if not condition:
            raise ValueError(error_msg)

    # Reshape the arrays to 2D i.e. [phi, x, y] -> [phi, x*y]
    dirty_fdf_arr_2d = nd_to_two_d(rm_synth_arrays.dirty_fdf_arr)
    rmsf_arr_2d = nd_to_two_d(rm_synth_arrays.rmsf_arr)
    # A shared RMSF is broadcast over the pixels (zero stride along them), so it
    # goes in once
    shared_rmsf = rmsf_arr_2d.strides[1] == 0
    iter_count_arr_2d = np.zeros(dirty_fdf_arr_2d.shape[1:], dtype=int)
    sub_minor_iter_arr_2d = np.zeros(dirty_fdf_arr_2d.shape[1:], dtype=int)
    fwhm_rmsf_arr_2d = nd_to_two_d(rm_synth_arrays.fwhm_rmsf_arr)

    # Initialise arrays to hold the residual FDF, clean components, clean FDF
    # Residual is initially copies of dirty FDF, so that pixels that are not
    #  processed get correct values (but will be overridden when processed)
    clean_fdf_spectrum_2d = np.zeros_like(dirty_fdf_arr_2d)
    model_fdf_spectrum_2d = np.zeros(
        dirty_fdf_arr_2d.shape, dtype=dirty_fdf_arr_2d.dtype
    )
    resid_fdf_arr_2d = dirty_fdf_arr_2d.copy()

    if multiscale:
        logger.warning("Multiscale RM-CLEAN enabled - note this is still experimental!")
        ms_options = multiscale_options or MultiscaleOptions()
        rmsf_fwhm = float(np.nanmedian(np.real(fwhm_rmsf_arr_2d)))
        scales = default_scales(
            rm_synth_arrays.phi_arr_radm2,
            rmsf_fwhm,
            ms_options,
            phi_max_scale_radm2=phi_max_scale_radm2,
        )
        logger.info(f"Multiscale scales (RMSF FWHM units): {scales}")
        # The scale kernels depend only on the RMSF, so a shared one needs them once
        ms_kernels = (
            compute_scale_kernels(
                scales,
                rmsf_arr_2d[:, 0],
                rmsf_fwhm,
                rm_synth_arrays.phi_double_arr_radm2,
                ms_options.kernel,
            )
            if shared_rmsf
            else None
        )

    # Only pixels that can actually clean need the per-pixel loop; the rest are
    # already in their final state (see `_null_clean_pixels`). Non-adaptive
    # multiscale gets the narrower blank-pixel screen instead: there a
    # scale-convolved dirty FDF can clear `mask * peak_response` even when the
    # raw peak does not, so the general no-op argument does not hold.
    pixel_options = PixelCleanOptions(clean_options, dirty_fdf_arr_2d.shape[1:])
    if not multiscale or clean_options.fdf_noise is not None:
        skip_arr = _null_clean_pixels(dirty_fdf_arr_2d, pixel_options.mask_1d)
        logger.info(
            f"Skipping {int(skip_arr.sum())} of {skip_arr.size} pixels with no "
            f"emission above the {pixel_options.describe_mask()} CLEAN mask."
        )
    else:
        skip_arr = _blank_pixels(dirty_fdf_arr_2d)
        logger.info(
            f"Skipping {int(skip_arr.sum())} of {skip_arr.size} fully blanked pixels."
        )

    todo = np.flatnonzero(~skip_arr)
    # Multiscale cleans one pixel at a time. Single-scale cleans 256 per call:
    # a call's working copies of its FDFs (and per-pixel RMSFs) stay a few MB,
    # while each call still covers enough pixels to make the call cheap.
    pixel_groups: Sequence[int | NDArray[np.intp]]
    if multiscale:
        pixel_groups = todo.tolist()
    else:
        pixel_groups = [todo[i : i + 256] for i in range(0, todo.size, 256)]
    fdf_mask_arr_2d = (
        None
        if rm_synth_arrays.fdf_mask_arr is None
        else nd_to_two_d(rm_synth_arrays.fdf_mask_arr)
    )
    for pix_idx in tqdm(
        pixel_groups,
        file=TQDM_OUT,
        leave=False,
        # tqdm.auto uses the notebook widget in Jupyter, which ignores `file`
        # and so escapes quiet_logs; gate on the logger level to actually mute it.
        disable=not logger.isEnabledFor(logging.INFO),
    ):
        if multiscale:
            (clean_spec, resid_spec, model_spec, iters, sub_minor_iters) = (
                multiscale_clean_spectrum(
                    dirty_fdf_spectrum=resid_fdf_arr_2d[:, pix_idx],
                    phi_arr_radm2=rm_synth_arrays.phi_arr_radm2,
                    phi_double_arr_radm2=rm_synth_arrays.phi_double_arr_radm2,
                    rmsf_spectrum=rmsf_arr_2d[:, pix_idx],
                    rmsf_fwhm=rmsf_fwhm,
                    scales=scales,
                    clean_options=pixel_options.at(pix_idx),
                    multiscale_options=ms_options,
                    kernels=ms_kernels,
                )
            )
            clean_fdf_spectrum_2d[:, pix_idx] = clean_spec
            resid_fdf_arr_2d[:, pix_idx] = resid_spec
            model_fdf_spectrum_2d[:, pix_idx] = model_spec
            iter_count_arr_2d[pix_idx] = iters
            sub_minor_iter_arr_2d[pix_idx] = sub_minor_iters
            continue
        results = minor_cycle_batch(
            RMSynthArrays(
                dirty_fdf_arr=resid_fdf_arr_2d[:, pix_idx],
                phi_arr_radm2=rm_synth_arrays.phi_arr_radm2,
                rmsf_arr=rmsf_arr_2d[:, 0] if shared_rmsf else rmsf_arr_2d[:, pix_idx],
                phi_double_arr_radm2=rm_synth_arrays.phi_double_arr_radm2,
                fwhm_rmsf_arr=fwhm_rmsf_arr_2d,
                fdf_mask_arr=None
                if fdf_mask_arr_2d is None
                else fdf_mask_arr_2d[:, pix_idx],
            ),
            pixel_options.at(pix_idx),
        )
        clean_fdf_spectrum_2d[:, pix_idx] = results.clean_fdf_arr
        resid_fdf_arr_2d[:, pix_idx] = results.resid_fdf_arr
        model_fdf_spectrum_2d[:, pix_idx] = results.model_fdf_arr
        iter_count_arr_2d[pix_idx] = results.clean_iter_arr
        sub_minor_iter_arr_2d[pix_idx] = results.sub_minor_iter_arr

    # Restore the residual to the cleaned FDF (moved outside of loop:
    # will now work for pixels/spectra without clean components)
    clean_fdf_spectrum_2d += resid_fdf_arr_2d

    # Reshape the arrays back to their original shape
    clean_fdf_spectrum = two_d_to_nd(
        clean_fdf_spectrum_2d, rm_synth_arrays.dirty_fdf_arr.shape
    )
    model_fdf_spectrum = two_d_to_nd(
        model_fdf_spectrum_2d, rm_synth_arrays.dirty_fdf_arr.shape
    )
    if rm_synth_arrays.dirty_fdf_arr.shape[1:] == ():
        iter_count_arr = two_d_to_nd(iter_count_arr_2d, (1,))
        sub_minor_iter_arr = two_d_to_nd(sub_minor_iter_arr_2d, (1,))
    else:
        iter_count_arr = two_d_to_nd(
            iter_count_arr_2d, rm_synth_arrays.dirty_fdf_arr.shape[1:]
        )
        sub_minor_iter_arr = two_d_to_nd(
            sub_minor_iter_arr_2d, rm_synth_arrays.dirty_fdf_arr.shape[1:]
        )
    resid_fdf_arr = two_d_to_nd(resid_fdf_arr_2d, rm_synth_arrays.dirty_fdf_arr.shape)

    return RMCleanResults(
        clean_fdf_spectrum,
        model_fdf_spectrum,
        iter_count_arr,
        resid_fdf_arr,
        sub_minor_iter_arr,
    )


def minor_cycle(
    rm_synth_1d_arrays: RMSynthArrays,
    clean_options: RMCleanOptions,
) -> CleanLoopResults:
    """Single-scale CLEAN one spectrum: an initial masked loop to `threshold`, then
    a deep loop frozen to the cleaned footprint.

    Args:
        rm_synth_1d_arrays (RMSynthArrays): 1D dirty FDF, phi axes, RMSF, optional mask.
        clean_options (RMCleanOptions): Mask, threshold, gain, max_iter.

    Raises:
        ValueError: If any input array is not 1D.

    Returns:
        CleanLoopResults: Clean, residual, and model spectra and the iteration count.
    """
    fdf_mask_arr = rm_synth_1d_arrays.fdf_mask_arr
    for array in (
        rm_synth_1d_arrays.dirty_fdf_arr,
        rm_synth_1d_arrays.phi_arr_radm2,
        rm_synth_1d_arrays.phi_double_arr_radm2,
        *(() if fdf_mask_arr is None else (fdf_mask_arr,)),
    ):
        if array.ndim != 1:
            msg = "Arrays in minor cycle must be 1D."
            raise ValueError(msg)
    results = minor_cycle_batch(
        rm_synth_1d_arrays._replace(
            dirty_fdf_arr=rm_synth_1d_arrays.dirty_fdf_arr[:, np.newaxis],
            fdf_mask_arr=None if fdf_mask_arr is None else fdf_mask_arr[:, np.newaxis],
        ),
        clean_options,
    )
    return CleanLoopResults(
        clean_fdf_spectrum=results.clean_fdf_arr[:, 0],
        resid_fdf_spectrum=results.resid_fdf_arr[:, 0],
        model_fdf_spectrum=results.model_fdf_arr[:, 0],
        iter_count=int(results.clean_iter_arr[0]),
    )


def minor_cycle_batch(
    rm_synth_2d_arrays: RMSynthArrays,
    clean_options: RMCleanOptions,
) -> RMCleanResults:
    """`minor_cycle` on a stack of spectra, each minor loop cleaning them all in
    one call.

    Args:
        rm_synth_2d_arrays (RMSynthArrays): Dirty FDFs [phi, spectrum], phi axes,
            RMSF (shared 1D, or [phi_double, spectrum]), one FWHM for all
            spectra, optional mask [phi, spectrum].
        clean_options (RMCleanOptions): Mask, threshold, gain, max_iter, FDF
            noise; per-spectrum values as 1D arrays.

    Raises:
        ValueError: If the FDFs are not 2D, or an option or RMSF does not have
            one value for every spectrum.

    Returns:
        RMCleanResults: Clean, model and residual FDFs [phi, spectrum], and the
            iterations per spectrum.
    """
    if rm_synth_2d_arrays.dirty_fdf_arr.ndim != 2:
        msg = (
            "Dirty FDFs must be 2D (phi, spectrum), got "
            f"{rm_synth_2d_arrays.dirty_fdf_arr.ndim}D."
        )
        raise ValueError(msg)
    dirty_fdf_arr = rm_synth_2d_arrays.dirty_fdf_arr.T
    _check_per_spectrum(
        {
            "mask": clean_options.mask,
            "threshold": clean_options.threshold,
            "fdf_noise": 0.0
            if clean_options.fdf_noise is None
            else clean_options.fdf_noise,
        },
        len(dirty_fdf_arr),
    )
    # A scalar mask stays a scalar, so it compares at the FDF's own precision
    mask_level = clean_options.mask
    if np.ndim(mask_level) != 0:
        mask_level = np.asarray(mask_level, dtype=np.float64)[:, np.newaxis]

    if clean_options.fdf_noise is not None:
        # Adaptive mask: cap is the whole allowed region (minor_loop starts tight
        # from a seed and grows the mask against the falling off-source noise).
        mask_arr = np.ones(dirty_fdf_arr.shape, dtype=bool)
    else:
        mask_arr = np.abs(dirty_fdf_arr) > mask_level
    if rm_synth_2d_arrays.fdf_mask_arr is not None:
        mask_arr &= rm_synth_2d_arrays.fdf_mask_arr.T.astype(bool)

    minor_loop_arrays = MinorLoopBatchArrays(
        resid_fdf_arr_mask=np.ma.array(dirty_fdf_arr, mask=~mask_arr),
        phi_arr_radm2=rm_synth_2d_arrays.phi_arr_radm2,
        phi_double_arr_radm2=rm_synth_2d_arrays.phi_double_arr_radm2,
        rmsf_arr=rm_synth_2d_arrays.rmsf_arr.T,
    )
    minor_loop_options = MinorLoopBatchOptions(
        max_iter=clean_options.max_iter,
        gain=clean_options.gain,
        mask_threshold=clean_options.mask,
        stopping_threshold=clean_options.threshold,
        noise=clean_options.fdf_noise,
    )
    logger.info("Starting initial minor loop...")
    initial = minor_loop_batch(minor_loop_arrays, minor_loop_options)

    # Deep clean, masked to where clean components have been added
    logger.info("Initial loop complete. Starting deep clean...")
    deep = minor_loop_batch(
        minor_loop_arrays._replace(
            resid_fdf_arr_mask=np.ma.array(
                initial.resid_fdf_arr, mask=initial.model_fdf_arr == 0
            )
        ),
        replace(
            minor_loop_options,
            start_iter=initial.iter_count_arr,
            update_mask=False,
        ),
    )

    model_fdf_arr = deep.model_fdf_arr + initial.model_fdf_arr
    rmsf_fwhm = float(np.squeeze(rm_synth_2d_arrays.fwhm_rmsf_arr))
    clean_fdf_arr = np.stack(
        [
            restore_model(model, rm_synth_2d_arrays.phi_arr_radm2, rmsf_fwhm)
            for model in model_fdf_arr
        ]
    )
    return RMCleanResults(
        clean_fdf_arr=clean_fdf_arr.T,
        model_fdf_arr=model_fdf_arr.T,
        clean_iter_arr=deep.iter_count_arr,
        resid_fdf_arr=deep.resid_fdf_arr.T,
        # Each minor iteration places one component
        sub_minor_iter_arr=deep.iter_count_arr,
    )


@dataclass(frozen=True, kw_only=True, slots=True)
class MultiscaleOptions:
    """Options for multiscale RM-CLEAN."""

    scales: NDArray[np.float64] | None = None
    """Explicit scales (RMSF FWHM units); None auto-selects WSClean-style"""
    n_scales: int | None = None
    """Cap on the auto-selected scale count (ignored if `scales` given)"""
    kernel: KernelType = "tapered_quad"
    """Scale kernel shape"""
    max_iter_sub_minor: int = 10_000
    """Maximum sub-minor (per-scale Hogbom) iterations"""
    sub_minor_fraction: float = 0.5
    """Re-select a scale once the activated scale's peak drops by this fraction"""
    selection: SelectionType = "hybrid"
    """Scale selector. "snr": best matched-filter score, preferring smaller scales
    within `selection_margin`. "hybrid": a wide scale only when the residual peak is
    clearly wide (the `hybrid_*` fields), else as "snr"."""
    selection_margin: float = 0.08
    """SNR selector (and the hybrid fallback): relative margin in [0, 1) favouring
    smaller scales. Among scales within this fraction of the best score, the
    smallest wins, which keeps points on the delta scale. 0 = raw argmax."""
    hybrid_width_factor: float = 1.2
    """Hybrid: min peak width, in measured dirty-beam widths"""
    hybrid_score_factor: float = 0.85
    """Hybrid: min extended score, relative to the delta scale"""
    hybrid_engage_factor: float = 2.0
    """Hybrid: min residual peak, in CLEAN masks"""

    def __post_init__(self) -> None:
        if self.kernel not in get_args(KernelType):
            msg = f"kernel must be one of {get_args(KernelType)}, got {self.kernel!r}."
            raise ValueError(msg)
        if self.selection not in get_args(SelectionType):
            msg = (
                f"selection must be one of {get_args(SelectionType)}, "
                f"got {self.selection!r}."
            )
            raise ValueError(msg)
        if not 0 <= self.selection_margin < 1:
            msg = f"selection_margin must be in [0, 1), got {self.selection_margin}."
            raise ValueError(msg)
        if not 0 < self.sub_minor_fraction < 1:
            msg = (
                f"sub_minor_fraction must be in (0, 1), got {self.sub_minor_fraction}."
            )
            raise ValueError(msg)
        if self.max_iter_sub_minor < 1:
            msg = "max_iter_sub_minor must be >= 1."
            raise ValueError(msg)
        for name in ("hybrid_width_factor", "hybrid_score_factor"):
            if getattr(self, name) <= 0:
                msg = f"{name} must be > 0, got {getattr(self, name)}."
                raise ValueError(msg)
        if self.hybrid_engage_factor < 0:
            msg = f"hybrid_engage_factor must be >= 0, got {self.hybrid_engage_factor}."
            raise ValueError(msg)
        if self.scales is not None:
            if len(self.scales) == 0:
                msg = "scales must be non-empty."
                raise ValueError(msg)
            scales = np.sort(np.asarray(self.scales, dtype=np.float64))
            if scales[0] != 0.0:
                msg = f"scales must include 0 (the delta scale), got {scales}."
                raise ValueError(msg)
            # Selectors index scales[0] as scale 0 and assume ascending order.
            object.__setattr__(self, "scales", scales)


def make_fine_scales(
    max_scale: float,
    n_scales: int | None = None,
) -> NDArray[np.float64]:
    """Scale set: 0, 3, then geometric doubling from 6 to `max_scale`.

    The first anchor is 3, not 1.5: a point's matched-filter score at s=1.5 is only
    ~1% below scale 0, so noise flipped bright points onto it.

    Args:
        max_scale (float): Upper bound in RMSF FWHM units (exclusive).
        n_scales (int | None): Cap on the scale count; None keeps all.

    Returns:
        NDArray[np.float64]: Scales in RMSF FWHM units, starting at 0.
    """
    scales = [0.0]
    if max_scale > 3.0:
        scales.append(3.0)
    scale = 6.0
    while scale < max_scale:
        scales.append(scale)
        scale *= 2
    scale_arr = np.array(scales, dtype=np.float64)
    if n_scales is not None:
        scale_arr = scale_arr[:n_scales]
    return scale_arr


def hanning(x_arr: NDArray[np.float64], length: float) -> NDArray[np.float64]:
    """Hanning window function."""
    han = (1 / length) * np.cos(np.pi * x_arr / length) ** 2
    return np.where(np.abs(x_arr) < length / 2, han, 0)


def tapered_quad_kernel_function(
    phi_double_arr_radm2: NDArray[np.float64],
    scale: float,
    rmsf_fwhm: float,
    sum_normalised: bool = True,
) -> NDArray[np.float64]:
    """Tapered quadratic scale kernel (Offringa & Smirnov 2017, eq. 2).

    Args:
        phi_double_arr_radm2 (NDArray[np.float64]): Grid to sample the kernel on.
        scale (float): Scale in RMSF FWHM units.
        rmsf_fwhm (float): RMSF FWHM in rad/m^2.
        sum_normalised (bool): Normalise to unit sum (else unit peak).

    Returns:
        NDArray[np.float64]: The sampled kernel.
    """
    scale_radm2 = scale * rmsf_fwhm
    kernel = hanning(phi_double_arr_radm2, scale_radm2) * (
        1 - (np.abs(phi_double_arr_radm2) / scale_radm2) ** 2
    )
    if sum_normalised:
        kernel /= kernel.sum()
    else:
        kernel /= kernel.max()
    return kernel


def gaussian_scale_kernel_function(
    phi_double_arr_radm2: NDArray[np.float64],
    scale: float,
    rmsf_fwhm: float,
    sum_normalised: bool = True,
) -> NDArray[np.float64]:
    """Gaussian scale kernel; sigma = (3/16) * scale (Offringa & Smirnov 2017).

    O&S define sigma = (3/16) * alpha with alpha the scale in physical units; here
    that is `scale * rmsf_fwhm`, matching the tapered_quad width.

    Args:
        phi_double_arr_radm2 (NDArray[np.float64]): Grid to sample the kernel on.
        scale (float): Scale in RMSF FWHM units.
        rmsf_fwhm (float): RMSF FWHM in rad/m^2.
        sum_normalised (bool): Normalise to unit sum (else unit peak).

    Returns:
        NDArray[np.float64]: The sampled kernel.
    """
    sigma = (3 / 16) * scale * rmsf_fwhm
    kernel = unit_centred_gaussian(x=phi_double_arr_radm2, stddev=sigma)
    if sum_normalised:
        kernel /= kernel.sum()
    else:
        kernel /= kernel.max()
    return kernel


KERNEL_FUNCS: dict[str, Callable[..., NDArray[np.float64]]] = {
    "tapered_quad": tapered_quad_kernel_function,
    "gaussian": gaussian_scale_kernel_function,
}

# Kernel half-width as a fraction of scale_radm2 (= scale * rmsf_fwhm), used to trim
# the convolution kernel to its real support instead of sampling it across the whole
# signal. tapered_quad is exactly zero beyond scale_radm2/2 (the Hanning gate), so
# 0.5 loses nothing. gaussian has no hard cutoff; 6 sigma (sigma = (3/16)*scale_radm2)
# leaves a truncation error of ~1e-8 relative, well below any numerical tolerance here.
KERNEL_SUPPORT_FACTOR: dict[str, float] = {
    "tapered_quad": 0.5,
    "gaussian": 6 * 3 / 16,
}


def convolve_fdf_scale(
    scale: float,
    fwhm: float,
    fdf_arr: NDArray[np.complexfloating] | NDArray[np.float64],
    phi_double_arr_radm2: NDArray[np.float64],
    kernel: KernelType = "gaussian",
    sum_normalised: bool = True,
) -> NDArray[np.complexfloating] | NDArray[np.float64]:
    """Convolve an FDF (complex or real) with a real scale kernel.

    The real and imaginary parts are convolved separately, since
    scipy.ndimage.convolve drops the imaginary part of complex input.

    Args:
        scale (float): Scale in RMSF FWHM units; 0 returns the input unchanged.
        fwhm (float): RMSF FWHM in rad/m^2.
        fdf_arr (NDArray): FDF (or RMSF) to convolve.
        phi_double_arr_radm2 (NDArray[np.float64]): Sets the sample spacing d_phi.
        kernel (KernelType): Scale-kernel shape.
        sum_normalised (bool): Normalise the kernel to unit sum (else unit peak).

    Returns:
        NDArray: The scale-convolved FDF, same length as `fdf_arr`.
    """
    if scale == 0:
        return fdf_arr
    kernel_func = KERNEL_FUNCS[kernel]
    # Sample the kernel on a zero-centred grid trimmed to its real support (see
    # KERNEL_SUPPORT_FACTOR), not the full signal length: scipy.ndimage.convolve is
    # O(n * kernel_len), and the kernel is negligible (tapered_quad: exactly zero)
    # beyond that support. Capped at n so a kernel wider than the signal (large
    # scale, narrow phi window) still can't exceed it and produce boundary garbage
    # under scipy's reflect mode.
    d_phi = float(phi_double_arr_radm2[1] - phi_double_arr_radm2[0])
    n = len(fdf_arr)
    support_factor = KERNEL_SUPPORT_FACTOR[kernel]
    half_width = min(int(np.ceil(support_factor * scale * fwhm / d_phi)), (n - 1) // 2)
    kernel_grid = np.arange(-half_width, half_width + 1) * d_phi
    kernel_arr = kernel_func(kernel_grid, scale, fwhm, sum_normalised=sum_normalised)

    if np.iscomplexobj(fdf_arr):
        conv_spec = convolve(fdf_arr.real, kernel_arr, mode="reflect") + 1j * convolve(
            fdf_arr.imag, kernel_arr, mode="reflect"
        )
    else:
        conv_spec = convolve(fdf_arr, kernel_arr, mode="reflect")

    assert len(conv_spec) == len(fdf_arr), "Convolved FDF has wrong length."
    return np.asarray(conv_spec)


def _restore_multiscale(
    model_fdf_spectrum: NDArray[np.complexfloating],
    phi_double_arr_radm2: NDArray[np.float64],
    rmsf_fwhm: float,
) -> NDArray[np.complexfloating]:
    """Convolve the model with a unit-peak clean beam.

    Matches single-scale `minor_loop`, so `calc_faraday_moments` mom0
    normalisation recovers the component flux.
    """
    clean_beam = unit_centred_gaussian(x=phi_double_arr_radm2, fwhm=rmsf_fwhm)
    return np.asarray(
        convolve(model_fdf_spectrum.real, clean_beam, mode="reflect")
        + 1j * convolve(model_fdf_spectrum.imag, clean_beam, mode="reflect"),
        dtype=np.complex128,
    )


def _reconvolve_model(
    model_fdf_spectrum: NDArray[np.complexfloating],
    rmsf_spectrum: NDArray[np.complexfloating],
    phi_arr_radm2: NDArray[np.float64],
    phi_double_arr_radm2: NDArray[np.float64],
) -> NDArray[np.complexfloating]:
    """Footprint of a sparse delta model in the dirty FDF: sum_i m_i * RMSF@i.

    Same shift-and-add primitive as `minor_loop`, so the multiscale residual
    update matches how single-scale CLEAN subtracts components.
    """
    out = np.zeros_like(model_fdf_spectrum)
    max_rmsf_index = int(np.nanargmax(np.abs(rmsf_spectrum)))
    n_phi_pad = int((len(phi_double_arr_radm2) - len(phi_arr_radm2)) / 2)
    for idx in np.nonzero(model_fdf_spectrum)[0]:
        shifted = shift_rmsf(rmsf_spectrum, int(idx), n_phi_pad, max_rmsf_index)
        out += model_fdf_spectrum[idx] * shifted
    return out


class ScaleKernels(NamedTuple):
    """Per-scale RMSF responses for one spectrum (Offringa & Smirnov 2017).

    The scale kernel `K_s` (eq. 2) is the extended shape a scale-s component is
    built from. A scale-s structure appears in the residual as `RMSF conv K_s`,
    and in the scale-convolved residual (which the sub-minor loop works in) as
    `RMSF conv K_s conv K_s`: the RMSF picks up the scale twice, once from the
    structure and once from convolving the residual (Section 2.2).
    """

    scales: NDArray[np.float64]
    """Scales (RMSF FWHM units)"""
    rmsf_conv_scale: list[NDArray[np.complexfloating]]
    """`RMSF conv K_s` per scale: the scale-s response in the full-res residual,
    used for the residual subtraction. On the double phi axis."""
    rmsf_conv_scale_twice: list[NDArray[np.complexfloating]]
    """`RMSF conv K_s conv K_s` per scale: the scale-s response in the
    scale-convolved residual, i.e. the effective RMSF the sub-minor loop cleans
    against (Section 2.2)."""
    peak_response: NDArray[np.float64]
    """`max|RMSF conv K_s conv K_s|` per scale: peak of the scale-convolved
    response, dividing it out recovers the true scale-s amplitude (cf. the
    per-scale gain, eq. 4)."""
    fwhm_conv_scale_twice: NDArray[np.float64]
    """Fitted FWHM of `|RMSF conv K_s conv K_s|` per scale (sub-minor restore
    width)."""
    sigma_s: NDArray[np.float64]
    """Per-scale FDF noise std relative to scale 0 under correlated FDF noise,
    for the matched-filter (snr) score inside hybrid selection:
    sqrt((K_s conv K_s conv RMSF)(0) / RMSF(0)). sigma_s[0] == 1."""
    point_width: float
    """Measured half-max width of |RMSF| in rad/m^2: the dirty-beam width a
    delta presents in the dirty |FDF|, near-in sidelobe shoulders included.
    Wider than the theoretical fwhm under non-uniform lambda^2 sampling; the
    hybrid selector's width test calibrates against this, never the fwhm."""


def compute_scale_kernels(
    scales: NDArray[np.float64],
    rmsf_spectrum: NDArray[np.complexfloating],
    rmsf_fwhm: float,
    phi_double_arr_radm2: NDArray[np.float64],
    kernel: KernelType,
) -> ScaleKernels:
    """Precompute the per-scale RMSF responses for one spectrum.

    Follows Offringa & Smirnov (2017), Section 2.2; see `ScaleKernels`.

    Args:
        scales (NDArray[np.float64]): Scales in RMSF FWHM units.
        rmsf_spectrum (NDArray[np.complexfloating]): RMSF on the double-phi axis.
        rmsf_fwhm (float): RMSF FWHM in rad/m^2.
        phi_double_arr_radm2 (NDArray[np.float64]): Double-phi axis.
        kernel (KernelType): Scale-kernel shape.

    Returns:
        ScaleKernels: Per-scale responses (see that type).
    """
    rmsf_conv_scale: list[NDArray[np.complexfloating]] = []
    rmsf_conv_scale_twice: list[NDArray[np.complexfloating]] = []
    peak_response = np.ones_like(scales)
    fwhm_conv_scale_twice = np.full_like(scales, rmsf_fwhm)
    # sigma_s = sqrt((K_s conv K_s conv C)(0) / C(0)): the zero-lag value of the
    # noise covariance C twice-convolved with the scale kernel. C is the ordinary
    # RMSF (exact for uniform weights). "Zero lag" is the RMSF peak channel; the
    # twice-conv response peaks there too. Since C == RMSF, the twice-conv of C is
    # exactly conv_twice, so qform reuses it.
    cov_peak_index = int(np.nanargmax(np.abs(rmsf_spectrum)))
    qform = np.ones_like(scales)
    for i, scale in enumerate(scales):
        conv_once = convolve_fdf_scale(
            scale, rmsf_fwhm, rmsf_spectrum, phi_double_arr_radm2, kernel
        )
        conv_twice = convolve_fdf_scale(
            scale, rmsf_fwhm, conv_once, phi_double_arr_radm2, kernel
        )
        rmsf_conv_scale.append(np.asarray(conv_once, dtype=np.complex128))
        rmsf_conv_scale_twice.append(np.asarray(conv_twice, dtype=np.complex128))
        peak_response[i] = float(np.nanmax(np.abs(conv_twice)))
        if scale != 0:
            fwhm_conv_scale_twice[i] = fit_rmsf(
                np.abs(conv_twice),
                phi_double_arr_radm2=phi_double_arr_radm2,
                fwhm_rmsf_radm2=rmsf_fwhm * scale,
            )
        qform[i] = float(np.real(conv_twice[cov_peak_index]))
    sigma_s = np.sqrt(qform / qform[0])
    d_phi = float(phi_double_arr_radm2[1] - phi_double_arr_radm2[0])
    return ScaleKernels(
        scales,
        rmsf_conv_scale,
        rmsf_conv_scale_twice,
        peak_response,
        fwhm_conv_scale_twice,
        sigma_s,
        _halfmax_width_radm2(np.abs(rmsf_spectrum), d_phi),
    )


def _halfmax_width_radm2(profile: NDArray[np.float64], d_phi: float) -> float:
    """Half-max width of a peaked profile; on |RMSF| this is the measured
    dirty-beam width (>> theoretical fwhm under non-uniform lambda^2 sampling)."""
    peak_index = int(np.argmax(profile))
    half = profile[peak_index] / 2.0
    hi = peak_index
    while hi + 1 < len(profile) and profile[hi + 1] > half:
        hi += 1
    lo = peak_index
    while lo - 1 >= 0 and profile[lo - 1] > half:
        lo -= 1
    return (hi - lo + 1) * d_phi


def find_significant_scale(
    resid_fdf_spectrum: NDArray[np.complexfloating],
    scale_kernels: ScaleKernels,
    rmsf_fwhm: float,
    phi_double_arr_radm2: NDArray[np.float64],
    multiscale_options: MultiscaleOptions,
    active: NDArray[np.bool_] | None = None,
    engage_floor: float = 0.0,
) -> int:
    """Index of the most-significant scale (Offringa & Smirnov 2017).

    "snr" scores the matched filter `max|resid conv K_s| / sigma_s`, with
    `selection_margin` preferring the smallest scale within that margin of the best.
    "hybrid" is width-gated snr: engage an extended scale only when the residual
    peak fits wider than the measured dirty beam AND its score competes with
    scale 0, else fall back to "snr" (see `_hybrid_scale_selection`).

    Args:
        resid_fdf_spectrum (NDArray[np.complexfloating]): Current residual FDF.
        scale_kernels (ScaleKernels): Precomputed per-scale responses.
        rmsf_fwhm (float): RMSF FWHM in rad/m^2.
        phi_double_arr_radm2 (NDArray[np.float64]): Double-phi axis.
        multiscale_options (MultiscaleOptions): Kernel, selector, margin and
            hybrid gates.
        active (NDArray[np.bool_] | None): Scales still allowed to be selected.
        engage_floor (float): Hybrid only: below this residual peak (absolute FDF
            units) always fall back to snr. 0 disables the gate.

    Returns:
        int: Index into `scale_kernels.scales` of the selected scale.
    """
    if multiscale_options.selection == "hybrid":
        return _hybrid_scale_selection(
            resid_fdf_spectrum,
            scale_kernels,
            rmsf_fwhm,
            phi_double_arr_radm2,
            multiscale_options,
            active=active,
            engage_floor=engage_floor,
        )
    return _snr_scale_selection(
        resid_fdf_spectrum,
        scale_kernels,
        rmsf_fwhm,
        phi_double_arr_radm2,
        multiscale_options,
        active=active,
    )


def _snr_scale_selection(
    resid_fdf_spectrum: NDArray[np.complexfloating],
    scale_kernels: ScaleKernels,
    rmsf_fwhm: float,
    phi_double_arr_radm2: NDArray[np.float64],
    multiscale_options: MultiscaleOptions,
    active: NDArray[np.bool_] | None,
) -> int:
    """Matched-filter snr selection, preferring the smallest scale in the margin"""
    selection_margin = multiscale_options.selection_margin
    scores = np.full_like(scale_kernels.scales, -np.inf)
    for i, scale in enumerate(scale_kernels.scales):
        if active is not None and not active[i]:
            continue
        resid_conv = convolve_fdf_scale(
            scale,
            rmsf_fwhm,
            resid_fdf_spectrum,
            phi_double_arr_radm2,
            multiscale_options.kernel,
        )
        peak = float(np.nanmax(np.abs(resid_conv)))
        scores[i] = peak / scale_kernels.sigma_s[i]
    if selection_margin > 0:
        best = float(np.max(scores))
        if np.isfinite(best) and best > 0:
            # Scales are sorted ascending, so the first within-margin index is the
            # smallest such scale.
            within = scores >= best * (1 - selection_margin)
            return int(np.argmax(within))
    return int(np.argmax(scores))


def _hybrid_scale_selection(
    resid_fdf_spectrum: NDArray[np.complexfloating],
    scale_kernels: ScaleKernels,
    rmsf_fwhm: float,
    phi_double_arr_radm2: NDArray[np.float64],
    multiscale_options: MultiscaleOptions,
    active: NDArray[np.bool_] | None,
    engage_floor: float,
) -> int:
    """Width-gated snr selection.

    Engage an extended scale only when two independent tests agree, else fall
    back to plain snr selection with `selection_margin`:
    1. Width: bounded half-max fit of the residual peak wider than
       `hybrid_width_factor` x the measured dirty-beam width (a delta fits 1.0x,
       the narrowest genuinely extended source ~2x).
    2. Score: best active extended matched-filter score at least
       `hybrid_score_factor` x the scale-0 score (kills residual junk that fits
       wide but scores poorly at wide scales).
    Engagement takes the argmax score among active extended scales. Below
    `engage_floor` (endgame) always falls back.
    """

    def snr_fallback() -> int:
        return _snr_scale_selection(
            resid_fdf_spectrum,
            scale_kernels,
            rmsf_fwhm,
            phi_double_arr_radm2,
            multiscale_options,
            active=active,
        )

    abs_resid = np.abs(resid_fdf_spectrum)
    peak_index = int(np.argmax(abs_resid))
    if engage_floor > 0 and abs_resid[peak_index] < engage_floor:
        return snr_fallback()

    point_width = scale_kernels.point_width
    d_phi = float(phi_double_arr_radm2[1] - phi_double_arr_radm2[0])
    # Walk just past the decision point; hitting the bound means "wide".
    max_steps = int(2.0 * point_width / d_phi) + 1
    half_val = abs_resid[peak_index] / 2.0
    hi = peak_index
    while (
        hi + 1 < len(abs_resid)
        and abs_resid[hi + 1] > half_val
        and hi - peak_index < max_steps
    ):
        hi += 1
    lo = peak_index
    while lo - 1 >= 0 and abs_resid[lo - 1] > half_val and peak_index - lo < max_steps:
        lo -= 1
    if (hi - lo + 1) * d_phi < multiscale_options.hybrid_width_factor * point_width:
        return snr_fallback()

    scales = scale_kernels.scales
    ext_active = np.array(
        [
            (active is None or bool(active[i])) and float(s) > 0
            for i, s in enumerate(scales)
        ]
    )
    zero_active = bool(float(scales[0]) == 0 and (active is None or bool(active[0])))
    if not ext_active.any() or not zero_active:
        return snr_fallback()
    scores = np.full(len(scales), -np.inf)
    for i, scale in enumerate(scales):
        if not (ext_active[i] or float(scale) == 0):
            continue
        resid_conv = convolve_fdf_scale(
            float(scale),
            rmsf_fwhm,
            resid_fdf_spectrum,
            phi_double_arr_radm2,
            multiscale_options.kernel,
        )
        scores[i] = float(np.nanmax(np.abs(resid_conv))) / float(
            scale_kernels.sigma_s[i]
        )
    best_ext = int(np.argmax(np.where(ext_active, scores, -np.inf)))
    if scores[best_ext] < multiscale_options.hybrid_score_factor * scores[0]:
        return snr_fallback()
    return best_ext


def adaptive_scale_supports(
    dirty_fdf_spectrum: NDArray[np.complexfloating],
    source_region: NDArray[np.bool_],
    kernels: ScaleKernels,
    scales: NDArray[np.float64],
    phi_double_arr_radm2: NDArray[np.float64],
    kernel: KernelType,
    rmsf_fwhm: float,
    mask: float,
) -> list[NDArray[np.bool_]]:
    """Opening-filter scale supports (eye-patch flint-crew, 1D RM analogue).

    Per scale: scale-convolve the dirty FDF, morphologically OPEN it (min then max
    filter of the scale width) to isolate structure at least that wide, gate at the
    flux-correct amplitude `mask * peak_response`, and confine to `source_region`
    dilated by the scale width. `source_region` is the tight adaptive mask from the
    single-scale off-source pre-pass; dilating per scale is the eye-patch
    beam-shape-erosion analogue and keeps extended scales from spreading onto the
    RMSF sidelobe islands (the pond). A scale with an empty support is inactive.
    """
    d_phi = float(phi_double_arr_radm2[1] - phi_double_arr_radm2[0])
    supports: list[NDArray[np.bool_]] = []
    for i, scale in enumerate(scales):
        w = max(1, round(float(scale) * rmsf_fwhm / d_phi))
        region = (
            source_region
            if float(scale) == 0
            else binary_dilation(source_region, iterations=w)
        )
        abs_conv = np.abs(
            convolve_fdf_scale(
                float(scale),
                rmsf_fwhm,
                dirty_fdf_spectrum,
                phi_double_arr_radm2,
                kernel,
            )
        )
        opened = (
            maximum_filter1d(minimum_filter1d(abs_conv, w), w) if w > 1 else abs_conv
        )
        supports.append((opened > mask * float(kernels.peak_response[i])) & region)
    return supports


def _multiscale_minor_cycles(
    resid_fdf_spectrum: NDArray[np.complexfloating],
    model_fdf_spectrum: NDArray[np.complexfloating],
    *,
    kernels: ScaleKernels,
    scales: NDArray[np.float64],
    allowed_supports: list[NDArray[np.bool_]],
    phi_arr_radm2: NDArray[np.float64],
    phi_double_arr_radm2: NDArray[np.float64],
    rmsf_fwhm: float,
    clean_options: RMCleanOptions,
    multiscale_options: MultiscaleOptions,
    stop_threshold: float,
    record: list[list[int]] | None,
) -> tuple[NDArray[np.complexfloating], NDArray[np.complexfloating], int, int]:
    """One phase of multiscale minor cycles, restricted to `allowed_supports`.

    Cleans down to `stop_threshold` with the O&S divergence and stall guards. A
    scale that exhausts its allowed region is retired (not reselected). If
    `record` is given, appends the cleaned channel indices per scale, to build
    the phase-2 auto-mask. Returns (resid, model, n_iter, sub_minor_total).
    """
    kernel = multiscale_options.kernel
    gain = clean_options.gain
    max_iter = clean_options.max_iter
    max_iter_sub_minor = multiscale_options.max_iter_sub_minor
    sub_minor_fraction = multiscale_options.sub_minor_fraction
    active = np.array([bool(s.any()) for s in allowed_supports])
    support = np.logical_or.reduce(allowed_supports)
    n_iter = 0
    sub_minor_total = 0
    if not support.any():
        return resid_fdf_spectrum, model_fdf_spectrum, n_iter, sub_minor_total

    progress = CleanProgress(
        model_fdf_spectrum.copy(),
        resid_fdf_spectrum.copy(),
        divergence_fraction=1.0,
        stall_patience=5,
    )

    for n_iter in range(1, max_iter + 1):
        # Over the whole array, not just `support`: an over-large scale can grow
        # a spurious feature outside the clean region.
        peak = float(np.nanmax(np.abs(resid_fdf_spectrum)))
        support_peak = float(np.nanmax(np.abs(resid_fdf_spectrum[support])))
        state = progress.check(peak, model_fdf_spectrum, resid_fdf_spectrum)
        if state is CleanState.DIVERGING:
            logger.warning(
                f"Multiscale CLEAN diverging at iter {n_iter} (peak {peak:0.3g}, "
                f"best {progress.best_peak:0.3g}); stopping at best."
            )
            model_fdf_spectrum = progress.model_fdf_spectrum
            resid_fdf_spectrum = progress.resid_fdf_spectrum
            break
        if support_peak < stop_threshold:
            break
        if state is CleanState.STALLED:
            logger.warning(
                f"Multiscale CLEAN stalled at iter {n_iter} (peak {peak:0.3g}, "
                f"threshold {stop_threshold:0.3g}); stopping at best."
            )
            model_fdf_spectrum = progress.model_fdf_spectrum
            resid_fdf_spectrum = progress.resid_fdf_spectrum
            break

        scale_index = find_significant_scale(
            resid_fdf_spectrum,
            kernels,
            rmsf_fwhm,
            phi_double_arr_radm2,
            multiscale_options,
            active=active,
            engage_floor=multiscale_options.hybrid_engage_factor
            * float(clean_options.mask),
        )
        scale = float(scales[scale_index])
        peak_response = float(kernels.peak_response[scale_index])
        sigma_s_i = float(kernels.sigma_s[scale_index])

        # thresh_scale sets the detection threshold in the once-convolved residual,
        # where the noise std scales by sigma_s (= sqrt(peak_response) for uniform
        # weights) relative to scale 0. Amplitude recovery (dividing by
        # peak_response) is separate, see below.
        thresh_scale = sigma_s_i

        # Scale-convolved residual; sub-minor cleans in this space, restricted to
        # the scale's allowed region and down to the stop threshold.
        resid_conv = np.asarray(
            convolve_fdf_scale(
                scale, rmsf_fwhm, resid_fdf_spectrum, phi_double_arr_radm2, kernel
            ),
            dtype=np.complex128,
        )
        mask_conv = (np.abs(resid_conv) > stop_threshold * thresh_scale) & (
            allowed_supports[scale_index]
        )
        if not mask_conv.any():
            # This scale has cleaned out its allowed region; retire it so it is
            # not reselected, and stop once every scale is exhausted.
            active[scale_index] = False
            if not active.any():
                break
            support = np.logical_or.reduce(
                [s for s, a in zip(allowed_supports, active, strict=True) if a]
            )
            if not support.any():
                break
            continue
        resid_conv_masked = np.ma.array(resid_conv, mask=~mask_conv)

        # Sub-minor cleans this scale only until its peak drops by
        # `sub_minor_fraction`, then re-select a scale.
        peak_conv = float(np.ma.max(np.ma.abs(resid_conv_masked)))
        # Threshold uses thresh_scale (noise); the sub-minor rmsf below divides by
        # peak_response (amplitude).
        threshold_sub = max(
            stop_threshold * thresh_scale,
            sub_minor_fraction * peak_conv,
        )

        sub_minor = minor_loop(
            MinorLoopArrays(
                resid_fdf_spectrum_mask=resid_conv_masked,
                phi_arr_radm2=phi_arr_radm2,
                phi_double_arr_radm2=phi_double_arr_radm2,
                rmsf_spectrum=kernels.rmsf_conv_scale_twice[scale_index]
                / peak_response,
                rmsf_fwhm=float(kernels.fwhm_conv_scale_twice[scale_index]),
            ),
            MinorLoopOptions(
                max_iter=max_iter_sub_minor,
                gain=gain,
                mask_threshold=stop_threshold * thresh_scale,
                stopping_threshold=threshold_sub,
                update_mask=True,
            ),
        )

        sub_minor_total += int(sub_minor.iter_count)

        # Undo the peak-response normalisation to recover true k_s amplitudes.
        true_deltas = sub_minor.model_fdf_spectrum / peak_response
        if not np.any(true_deltas):
            # Sub-minor placed nothing for this scale; retire it (as the empty-mask
            # branch above) so a single stuck scale does not kill the whole phase.
            active[scale_index] = False
            if not active.any():
                break
            support = np.logical_or.reduce(
                [s for s, a in zip(allowed_supports, active, strict=True) if a]
            )
            if not support.any():
                break
            continue
        if record is not None:
            record[scale_index].extend(np.nonzero(true_deltas)[0].tolist())

        model_fdf_spectrum = model_fdf_spectrum + np.asarray(
            convolve_fdf_scale(
                scale, rmsf_fwhm, true_deltas, phi_double_arr_radm2, kernel
            ),
            dtype=np.complex128,
        )
        # Full-res residual update: dirty footprint = model (conv) RMSF.
        resid_fdf_spectrum = resid_fdf_spectrum - _reconvolve_model(
            true_deltas,
            kernels.rmsf_conv_scale[scale_index],
            phi_arr_radm2,
            phi_double_arr_radm2,
        )

    return resid_fdf_spectrum, model_fdf_spectrum, n_iter, sub_minor_total


def _build_scale_masks(
    record: list[list[int]],
    scales: NDArray[np.float64],
    phi_len: int,
    rmsf_fwhm: float,
    d_phi: float,
) -> list[NDArray[np.bool_]]:
    """Phase-2 per-scale masks: each scale's phase-1 centres dilated by
    `scale*fwhm/2` each side (WSClean 2.2.3 scale-dependent masking); empty for a
    scale that cleaned nothing.

    The dilation is the scale's own extent only, with no added RMSF core: the core
    over-dilated the delta scale into the RMSF sidelobes on wide-RMSF narrowband
    cells. At scale 0 this confines phase 2 to the exact phase-1 footprint, like
    single-scale's frozen deep clean.
    """
    masks: list[NDArray[np.bool_]] = []
    for scale, rec in zip(scales, record, strict=True):
        mask_arr = np.zeros(phi_len, dtype=bool)
        if rec:
            half = round(float(scale) * rmsf_fwhm / d_phi) // 2
            for i in set(rec):
                mask_arr[max(0, i - half) : i + half + 1] = True
        masks.append(mask_arr)
    return masks


def _classify_hybrid_source(
    dirty_fdf_spectrum: NDArray[np.complexfloating],
    phi_double_arr_radm2: NDArray[np.float64],
    kernels: ScaleKernels,
    multiscale_options: MultiscaleOptions,
) -> MultiscaleOptions:
    """Classify the source once on the DIRTY spectrum for "hybrid" selection.

    A peak that fits at beam width is a point source: lock the whole clean to
    "snr" so mid-clean sidelobe pedestals (which can fit wide while the peak is
    still above the engage floor) never re-open extended selection. A wide fit
    keeps "hybrid" with its per-cycle gates. No-op for other selectors.
    """
    if multiscale_options.selection != "hybrid":
        return multiscale_options
    d_phi = float(phi_double_arr_radm2[1] - phi_double_arr_radm2[0])
    width = _halfmax_width_radm2(np.abs(dirty_fdf_spectrum), d_phi)
    if width < multiscale_options.hybrid_width_factor * kernels.point_width:
        return dataclasses.replace(multiscale_options, selection="snr")
    return multiscale_options


def multiscale_clean_spectrum(
    dirty_fdf_spectrum: NDArray[np.complexfloating],
    phi_arr_radm2: NDArray[np.float64],
    phi_double_arr_radm2: NDArray[np.float64],
    rmsf_spectrum: NDArray[np.complexfloating],
    rmsf_fwhm: float,
    scales: NDArray[np.float64],
    clean_options: RMCleanOptions,
    multiscale_options: MultiscaleOptions,
    kernels: ScaleKernels | None = None,
) -> tuple[
    NDArray[np.complexfloating],
    NDArray[np.complexfloating],
    NDArray[np.complexfloating],
    int,
    int,
]:
    """Multiscale CLEAN one FDF spectrum (Offringa & Smirnov 2017).

    Two-phase auto-mask (Section 2.2.3): phase 1 cleans to `mask` and records where
    each scale places flux; phase 2 deep-cleans to `threshold`, restricting each
    scale to its dilated phase-1 footprint (stops a scale smearing flux across the
    window). Give a generous phi window, as scale kernels wider than it pick up
    reflect-mode boundary artefacts.

    Args:
        dirty_fdf_spectrum (NDArray[np.complexfloating]): Dirty FDF on the phi axis.
        phi_arr_radm2 (NDArray[np.float64]): Faraday depth axis in rad/m^2.
        phi_double_arr_radm2 (NDArray[np.float64]): Double-phi axis (RMSF length).
        rmsf_spectrum (NDArray[np.complexfloating]): RMSF on the double-phi axis.
        rmsf_fwhm (float): RMSF FWHM in rad/m^2.
        scales (NDArray[np.float64]): Scales in RMSF FWHM units.
        clean_options (RMCleanOptions): Mask, threshold, gain, max_iter.
        multiscale_options (MultiscaleOptions): Kernel, selector, sub-minor settings.
        kernels (ScaleKernels | None): `compute_scale_kernels` for this RMSF,
            scales and kernel, to share between spectra; computed here if None.

    Returns:
        (clean, resid, model, minor_iters, sub_minor_iters): minor_iters counts
        minor cycles (scale re-selections); sub_minor_iters the per-scale Hogbom
        steps (comparable to single-scale's iteration count).
    """
    # A delta-only grid IS single-scale (narrow bands degenerate here): delegate so
    # the result is bit-identical to single-scale CLEAN, not just numerically close.
    if len(scales) == 1 and float(scales[0]) == 0:
        r = minor_cycle(
            RMSynthArrays(
                dirty_fdf_arr=dirty_fdf_spectrum,
                phi_arr_radm2=phi_arr_radm2,
                phi_double_arr_radm2=phi_double_arr_radm2,
                rmsf_arr=rmsf_spectrum,
                fwhm_rmsf_arr=np.array([rmsf_fwhm]),
            ),
            clean_options,
        )
        return (
            r.clean_fdf_spectrum,
            r.resid_fdf_spectrum,
            r.model_fdf_spectrum,
            r.iter_count,
            r.iter_count,
        )

    mask = float(clean_options.mask)
    threshold = float(clean_options.threshold)
    kernel = multiscale_options.kernel
    resid_fdf_spectrum = dirty_fdf_spectrum.copy()
    model_fdf_spectrum = np.zeros_like(resid_fdf_spectrum)
    if kernels is None:
        kernels = compute_scale_kernels(
            scales,
            rmsf_spectrum,
            rmsf_fwhm,
            phi_double_arr_radm2,
            kernel,
        )
    # Adaptive opening-filter masking replaces the frozen dirty-based supports and
    # the dirty hybrid classification (activation is the opening filter).
    adaptive = clean_options.fdf_noise is not None
    if not adaptive:
        multiscale_options = _classify_hybrid_source(
            dirty_fdf_spectrum, phi_double_arr_radm2, kernels, multiscale_options
        )

    if adaptive:
        # Single-scale off-source pre-pass gives the tight source region (its
        # component footprint), which bounds every scale's opening-filter support.
        # This keeps the flux-correct multiscale minor cycles unchanged while
        # confining them off the RMSF sidelobe islands. A noise-only spectrum yields
        # an empty region -> empty supports -> the guard below returns (no clean).
        pre_pass = minor_cycle(
            RMSynthArrays(
                dirty_fdf_arr=dirty_fdf_spectrum,
                phi_arr_radm2=phi_arr_radm2,
                phi_double_arr_radm2=phi_double_arr_radm2,
                rmsf_arr=rmsf_spectrum,
                fwhm_rmsf_arr=np.array([rmsf_fwhm]),
            ),
            clean_options,
        )
        source_region = np.abs(pre_pass.model_fdf_spectrum) > 0
        scale_supports = adaptive_scale_supports(
            dirty_fdf_spectrum,
            source_region,
            kernels,
            scales,
            phi_double_arr_radm2,
            kernel,
            rmsf_fwhm,
            mask,
        )
    else:
        # Phase-1 detection support: each scale may clean where its scale-convolved
        # dirty exceeds the mask. A source at the mask amplitude has a scale-convolved
        # peak of `mask * peak_response`, so the scale kernel sets the mask width.
        scale_supports = [
            np.abs(
                convolve_fdf_scale(
                    float(scale),
                    rmsf_fwhm,
                    dirty_fdf_spectrum,
                    phi_double_arr_radm2,
                    kernel,
                )
            )
            > mask * float(peak_response_s)
            for scale, peak_response_s in zip(
                scales, kernels.peak_response, strict=True
            )
        ]
    if not np.logical_or.reduce(scale_supports).any():
        return (
            _restore_multiscale(model_fdf_spectrum, phi_double_arr_radm2, rmsf_fwhm),
            resid_fdf_spectrum,
            model_fdf_spectrum,
            0,
            0,
        )

    # Phase 1: clean to the mask level, recording where each scale places flux.
    record: list[list[int]] = [[] for _ in scales]
    resid_fdf_spectrum, model_fdf_spectrum, iter1, sub1 = _multiscale_minor_cycles(
        resid_fdf_spectrum,
        model_fdf_spectrum,
        kernels=kernels,
        scales=scales,
        allowed_supports=scale_supports,
        phi_arr_radm2=phi_arr_radm2,
        phi_double_arr_radm2=phi_double_arr_radm2,
        rmsf_fwhm=rmsf_fwhm,
        clean_options=clean_options,
        multiscale_options=multiscale_options,
        stop_threshold=mask,
        record=record,
    )

    # Phase 2: deep-clean to the threshold, each scale confined to its phase-1
    # footprint.
    d_phi = float(phi_double_arr_radm2[1] - phi_double_arr_radm2[0])
    allowed = _build_scale_masks(record, scales, len(phi_arr_radm2), rmsf_fwhm, d_phi)
    resid_fdf_spectrum, model_fdf_spectrum, iter2, sub2 = _multiscale_minor_cycles(
        resid_fdf_spectrum,
        model_fdf_spectrum,
        kernels=kernels,
        scales=scales,
        allowed_supports=allowed,
        phi_arr_radm2=phi_arr_radm2,
        phi_double_arr_radm2=phi_double_arr_radm2,
        rmsf_fwhm=rmsf_fwhm,
        clean_options=clean_options,
        multiscale_options=multiscale_options,
        stop_threshold=threshold,
        record=None,
    )

    clean_fdf_spectrum = _restore_multiscale(
        model_fdf_spectrum, phi_double_arr_radm2, rmsf_fwhm
    )
    return (
        clean_fdf_spectrum,
        resid_fdf_spectrum,
        model_fdf_spectrum,
        iter1 + iter2,
        sub1 + sub2,
    )


def default_scales(
    phi_arr_radm2: NDArray[np.float64],
    rmsf_fwhm: float,
    multiscale_options: MultiscaleOptions,
    phi_max_scale_radm2: float | None = None,
) -> NDArray[np.float64]:
    """Scales (RMSF FWHM units): explicit if given, else WSClean-style auto.

    The auto max scale is `phi_max_scale_radm2` (pi / lambda_sq_min) when given,
    capped at the FDF phi window: a kernel wider than the window is meaningless (its
    response flattens, its normalisation blows up). Uses the `make_fine_scales` grid.

    Args:
        phi_arr_radm2 (NDArray[np.float64]): Faraday depth axis (sets the window cap).
        rmsf_fwhm (float): RMSF FWHM in rad/m^2.
        multiscale_options (MultiscaleOptions): Selector, explicit scales, count cap.
        phi_max_scale_radm2 (float | None): Largest recoverable scale; None uses
            the window.

    Returns:
        NDArray[np.float64]: Scales in RMSF FWHM units.
    """
    if multiscale_options.scales is not None:
        return multiscale_options.scales
    window_max_scale = float(phi_arr_radm2.max() - phi_arr_radm2.min()) / (
        2 * rmsf_fwhm
    )
    if phi_max_scale_radm2 is not None:
        max_scale = min(phi_max_scale_radm2 / rmsf_fwhm, window_max_scale)
    else:
        max_scale = window_max_scale
    scales = make_fine_scales(max_scale, n_scales=multiscale_options.n_scales)
    if len(scales) == 1:
        logger.warning(
            "Multiscale auto grid degenerated to [0.0]: the phi window is narrower "
            f"than 2*3*RMSF FWHM ({2 * 3.0 * rmsf_fwhm:0.3g} rad/m^2), "
            "so no extended scale fits and multiscale CLEAN reduces to single-scale. "
            "Pass explicit `multiscale_scales` to force an extended grid."
        )
    return scales
