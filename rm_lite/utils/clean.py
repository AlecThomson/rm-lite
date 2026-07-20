"""RM-CLEAN utils"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, NamedTuple, TypeAlias, TypeVar

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import convolve
from tqdm.auto import tqdm

from rm_lite.utils.arrays import nd_to_two_d, two_d_to_nd
from rm_lite.utils.fitting import (
    fit_rmsf,
    gaussian,
    unit_centred_gaussian,
)
from rm_lite.utils.logging import TqdmToLogger, logger

TQDM_OUT = TqdmToLogger(logger, level=logging.INFO)

DType = TypeVar("DType", bound=np.generic)

KernelType: TypeAlias = Literal["tapered_quad", "gaussian"]

SelectionType: TypeAlias = Literal["bias", "snr", "hybrid"]

# Hybrid (width-gated snr) selection.
# Engage an extended scale only when the residual peak fits wider than
# HYBRID_WIDTH_FACTOR x the measured dirty-beam width AND the best extended
# matched-filter score reaches HYBRID_SCORE_FACTOR x the scale-0 score, while
# the peak is above HYBRID_ENGAGE_MASK_FACTOR x the component-finding mask;
# otherwise fall back to plain snr selection. The width test must use the
# MEASURED half-max width of |RMSF| (the dirty beam, near-in sidelobe
# shoulders included), never the theoretical fwhm: under non-uniform lambda^2
# sampling a pure delta fits far wider than 1.2 x the theoretical fwhm.
HYBRID_WIDTH_FACTOR = 1.2
HYBRID_SCORE_FACTOR = 0.85
HYBRID_ENGAGE_MASK_FACTOR = 2.0

# Bail out of a spectrum's minor-cycle (scale-selection) loop if the residual
# peak exceeds this factor times the best (lowest) peak seen: a runaway backstop.
DIVERGENCE_FACTOR = 2.0

# Stall backstop: if the residual peak fails to improve by this relative fraction
# for STALL_PATIENCE consecutive minor cycles, the loop has plateaued above the
# threshold (e.g. grinding scale-0 components into the noise floor) and stops at
# the best state rather than running to max_iter. A converging minor cycle drops
# the peak by roughly the sub-minor fraction (~50%), far above this 1%, so real
# cleaning never trips it; slow noise-grinding (sub-1% ratcheting) does.
STALL_REL_IMPROVEMENT = 1e-2
STALL_PATIENCE = 5


class RMCleanResults(NamedTuple):
    """Results of the RM-CLEAN calculation"""

    clean_fdf_arr: NDArray[np.complex128]
    """The cleaned Faraday dispersion function cube"""
    model_fdf_arr: NDArray[np.complex128]
    """The clean components cube"""
    clean_iter_arr: NDArray[np.int16]
    """CLEAN iterations per pixel. Single-scale: minor iterations (one component
    each). Multiscale: minor cycles (scale re-selections); see
    `sub_minor_iter_arr` for the comparable component count."""
    resid_fdf_arr: NDArray[np.complex128]
    """The residual Faraday dispersion function cube"""
    sub_minor_iter_arr: NDArray[np.int16]
    """Total component-placement steps per pixel: single-scale equals
    `clean_iter_arr`; multiscale sums the sub-minor (per-scale Hogbom) iterations
    over all minor cycles. This is the count comparable to single-scale."""


class CleanLoopResults(NamedTuple):
    """Results of the RM-CLEAN loop"""

    clean_fdf_spectrum: NDArray[np.complex128]
    """The cleaned Faraday dispersion function cube"""
    resid_fdf_spectrum: NDArray[np.complex128]
    """The residual Faraday dispersion function cube"""
    model_fdf_spectrum: NDArray[np.complex128]
    """The clean components cube"""
    iter_count: int
    """The number of iterations"""


class MinorLoopResults(NamedTuple):
    """Results of the RM-CLEAN minor loop"""

    clean_fdf_spectrum: NDArray[np.complex128]
    """The cleaned Faraday dispersion function cube"""
    resid_fdf_spectrum: NDArray[np.complex128]
    """The residual Faraday dispersion function cube"""
    resid_fdf_spectrum_mask: np.ma.MaskedArray
    """The masked residual Faraday dispersion function cube"""
    model_fdf_spectrum: NDArray[np.complex128]
    """The clean components cube"""
    model_rmsf_spectrum: NDArray[np.complex128]
    """ Model * RMSF """
    iter_count: int
    """The number of iterations"""


@dataclass(frozen=True, kw_only=True, slots=True)
class RMCleanOptions:
    """Options for RM-CLEAN, shared by the 1D and 3D tools"""

    mask: float
    """Masking threshold - pixels below this value are not cleaned"""
    threshold: float
    """Cleaning threshold - stop when all pixels are below this value"""
    max_iter: int = 1000
    """Maximum clean iterations"""
    gain: float = 0.1
    """Clean loop gain"""

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
    rmsf_spectrum: NDArray[np.complex128]
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
    mask: float
    """Masking threshold"""
    threshold: float
    """Threshold for stopping the loop"""
    start_iter: int = 0
    """Starting iteration"""
    update_mask: bool = True
    """Update the mask after each iteration"""


def shift_rmsf(
    rmsf_spectrum: NDArray[np.complex128],
    fdf_index: int,
    n_phi_pad: int,
    max_rmsf_index: int,
) -> NDArray[np.complex128]:
    """Roll the double-length RMSF so its peak sits at FDF channel `fdf_index`,
    then clip to FDF length.

    Shared shift-and-subtract primitive (Hogbom and multiscale); assumes integer
    shifts and a symmetric RMSF.

    Args:
        rmsf_spectrum (NDArray[np.complex128]): Double-length RMSF.
        fdf_index (int): Target FDF channel for the RMSF peak.
        n_phi_pad (int): Half the length difference between the double and FDF axes.
        max_rmsf_index (int): Index of the RMSF peak.

    Returns:
        NDArray[np.complex128]: RMSF shifted to `fdf_index`, clipped to FDF length.
    """
    return np.asarray(
        np.roll(rmsf_spectrum, fdf_index + n_phi_pad - max_rmsf_index)[
            n_phi_pad:-n_phi_pad
        ]
    )


def minor_loop(
    minor_loop_arrays: MinorLoopArrays,
    minor_loop_options: MinorLoopOptions,
) -> MinorLoopResults:
    """Hogbom minor loop: shift-subtract the RMSF at each residual peak within the
    mask until the peak drops below `threshold` (or max_iter).

    Args:
        minor_loop_arrays (MinorLoopArrays): Masked residual, phi axes, RMSF, FWHM.
        minor_loop_options (MinorLoopOptions): Gain, mask, threshold, max_iter, mask update.

    Returns:
        MinorLoopResults: Clean, residual, model, masked residual, and iteration count.
    """
    resid_fdf_spectrum_mask = minor_loop_arrays.resid_fdf_spectrum_mask.copy()
    resid_fdf_spectrum = minor_loop_arrays.resid_fdf_spectrum_mask.data.copy()
    model_fdf_spectrum = np.zeros_like(resid_fdf_spectrum)
    clean_fdf_spectrum = np.zeros_like(resid_fdf_spectrum)
    model_rmsf_spectrum = np.zeros_like(resid_fdf_spectrum)
    rmsf_spectrum = minor_loop_arrays.rmsf_spectrum.copy()
    phi_arr_radm2 = minor_loop_arrays.phi_arr_radm2.copy()
    mask_arr = ~resid_fdf_spectrum_mask.mask.copy()
    mask_arr_original = mask_arr.copy()
    iter_count = int(minor_loop_options.start_iter)

    # Find the index of the peak of the RMSF
    max_rmsf_index = np.nanargmax(np.abs(rmsf_spectrum))

    # Calculate the padding in the sampled RMSF
    # Assumes only integer shifts and symmetric RMSF
    n_phi_pad = int(
        (len(minor_loop_arrays.phi_double_arr_radm2) - len(phi_arr_radm2)) / 2
    )

    logger.info(f"Starting minor loop... {mask_arr.sum()} pixels in the mask")
    for iter_count in range(
        minor_loop_options.start_iter, minor_loop_options.max_iter + 1
    ):
        if resid_fdf_spectrum_mask.mask.all():
            logger.warning(
                f"All channels masked. Exiting loop...performed {iter_count} iterations"
            )
            break
        if iter_count == minor_loop_options.max_iter:
            logger.warning(
                f"Max iterations reached. Exiting loop...performed {iter_count} iterations"
            )
            break
        if np.ma.max(np.ma.abs(resid_fdf_spectrum_mask)) < minor_loop_options.threshold:
            logger.info(
                f"Threshold reached. Exiting loop...performed {iter_count} iterations"
            )
            break
        # Get the absolute peak channel, values and Faraday depth
        peak_fdf_index = np.ma.argmax(np.abs(resid_fdf_spectrum_mask))
        peak_fdf = resid_fdf_spectrum_mask[peak_fdf_index]
        peak_rm = phi_arr_radm2[peak_fdf_index]

        # A clean component is "loop-gain * peak_fdf
        clean_component = minor_loop_options.gain * peak_fdf
        model_fdf_spectrum[peak_fdf_index] += clean_component

        # Shift the RMSF & clip so that its peak is centred above this clean_component
        shifted_rmsf_spectrum = shift_rmsf(
            rmsf_spectrum, int(peak_fdf_index), n_phi_pad, int(max_rmsf_index)
        )
        model_rmsf_spectrum += clean_component * shifted_rmsf_spectrum

        # Subtract the product of the clean_component shifted RMSF from the residual FDF
        resid_fdf_spectrum -= clean_component * shifted_rmsf_spectrum

        # Restore the clean_component * a Gaussian to the cleaned FDF
        clean_fdf_spectrum += gaussian(
            x=phi_arr_radm2,
            amplitude=clean_component,
            mean=float(peak_rm),
            fwhm=minor_loop_arrays.rmsf_fwhm,
        )
        # Remake masked residual FDF
        if minor_loop_options.update_mask:
            mask_arr = np.abs(resid_fdf_spectrum) > minor_loop_options.mask
            # Mask anything that was previously masked
            mask_arr = mask_arr & mask_arr_original
        resid_fdf_spectrum_mask = np.ma.array(resid_fdf_spectrum, mask=~mask_arr)

    return MinorLoopResults(
        clean_fdf_spectrum=clean_fdf_spectrum,
        resid_fdf_spectrum=resid_fdf_spectrum,
        resid_fdf_spectrum_mask=resid_fdf_spectrum_mask,
        model_fdf_spectrum=model_fdf_spectrum,
        model_rmsf_spectrum=model_rmsf_spectrum,
        iter_count=iter_count,
    )


class RMSynthArrays(NamedTuple):
    """Arrays for RM-synthesis"""

    dirty_fdf_arr: NDArray[np.complex128]
    """Dirty Faraday dispersion function array"""
    phi_arr_radm2: NDArray[np.float64]
    """Faraday depth array in rad/m^2"""
    phi_double_arr_radm2: NDArray[np.float64]
    """Double-length Faraday depth array in rad/m^2"""
    rmsf_arr: NDArray[np.complex128]
    """RMSF array"""
    fwhm_rmsf_arr: NDArray[np.float64]
    """FWHM of the RMSF array"""
    fdf_mask_arr: NDArray[np.bool_] | None = None
    """Mask of pixels to clean"""
    noise_rmsf_arr: NDArray[np.complex128] | None = None
    """w^2-RMSF noise covariance on the double-phi axis; None -> use rmsf_arr
    (exact for uniform weights). Only used by multiscale selection="snr"."""


def rmclean(
    dirty_fdf_arr: NDArray[np.complex128],
    phi_arr_radm2: NDArray[np.float64],
    rmsf_arr: NDArray[np.complex128],
    phi_double_arr_radm2: NDArray[np.float64],
    fwhm_rmsf_arr: NDArray[np.float64],
    mask: float,
    threshold: float,
    max_iter: int = 1000,
    gain: float = 0.1,
    mask_arr: NDArray[np.bool_] | None = None,
    multiscale: bool = False,
    multiscale_scale_bias: float = 0.6,
    multiscale_scales: NDArray[np.float64] | None = None,
    multiscale_n_scales: int | None = None,
    multiscale_kernel: KernelType = "tapered_quad",
    multiscale_max_iter_sub_minor: int = 10_000,
    multiscale_sub_minor_fraction: float = 0.5,
    multiscale_selection: SelectionType = "hybrid",
    multiscale_selection_margin: float = 0.08,
    phi_max_scale_radm2: float | None = None,
    noise_rmsf_arr: NDArray[np.complex128] | None = None,
) -> RMCleanResults:
    """Perform RM-CLEAN on a Faraday dispersion function array.

    Args:
        dirty_fdf_arr (NDArray[np.complex128]): Dirty Faraday dispersion function array
        phi_arr_radm2 (NDArray[np.float64]): Faraday depth array in rad/m^2
        rmsf_arr (NDArray[np.complex128]): RMSF array
        phi_double_arr_radm2 (NDArray[np.float64]): Double-length Faraday depth array in rad/m^2
        fwhm_rmsf_arr (NDArray[np.float64]): FWHM of the RMSF array
        mask (float): Masking threshold - pixels below this value are not cleaned
        threshold (float): Cleaning threshold - stop when all pixels are below this value
        max_iter (int, optional): Maximum clean iterations. Defaults to 1000.
        gain (float, optional): Glean loop gain. Defaults to 0.1.
        mask_arr (NDArray[np.bool_] | None, optional): Additional mask of pixels to avoid. Defaults to None.
        multiscale (bool, optional): Use multiscale RM-CLEAN (recovers Faraday-thick structure). Defaults to False.
        multiscale_scale_bias (float, optional): Scale-bias in (0, 1]; lower favours larger scales more. Defaults to 0.6.
        multiscale_scales (NDArray[np.float64] | None, optional): Explicit scales (RMSF FWHM units); None auto-selects.
        multiscale_n_scales (int | None, optional): Cap on the auto scale count.
        multiscale_kernel ("tapered_quad" | "gaussian", optional): Scale kernel. Defaults to "tapered_quad".
        multiscale_max_iter_sub_minor (int, optional): Max sub-minor iterations per scale. Defaults to 10_000.
        multiscale_sub_minor_fraction (float, optional): Sub-minor re-selection fraction. Defaults to 0.5.
        multiscale_selection ("bias" | "snr" | "hybrid", optional): Scale-selection mode. "bias" = Offringa eq-3 scale-bias; "snr" = matched-filter max|R conv K_s| / sigma_s (uses a finer scale grid); "hybrid" (default) = width-gated snr, engaging extended scales only when the residual peak fits wider than the measured dirty beam and the extended score competes with scale 0. Defaults to "hybrid".
        multiscale_selection_margin (float, optional): SNR selector (and hybrid fallback), parsimony margin in [0, 1). Among scales within this fraction of the best matched-filter score the smallest is chosen, keeping points and marginally resolved sources on the delta scale. Defaults to 0.08.
        phi_max_scale_radm2 (float | None, optional): Largest recoverable Faraday scale (pi / lambda_sq_min); sets the auto scale range. None falls back to the phi window.
        noise_rmsf_arr (NDArray[np.complex128] | None, optional): w^2-RMSF noise covariance (double-phi axis, same shape as rmsf_arr) for the selection="snr" sigma_s. None uses rmsf_arr (exact for uniform weights). Defaults to None.

    Returns:
        RMCleanResults: clean_fdf_arr, model_fdf_arr, clean_iter_arr, resid_fdf_arr, sub_minor_iter_arr
    """
    rm_synth_arrays = RMSynthArrays(
        dirty_fdf_arr=dirty_fdf_arr,
        phi_arr_radm2=phi_arr_radm2,
        rmsf_arr=rmsf_arr,
        phi_double_arr_radm2=phi_double_arr_radm2,
        fwhm_rmsf_arr=fwhm_rmsf_arr,
        fdf_mask_arr=mask_arr,
        noise_rmsf_arr=noise_rmsf_arr,
    )
    clean_options = RMCleanOptions(
        mask=mask,
        threshold=threshold,
        max_iter=max_iter,
        gain=gain,
    )
    multiscale_options = (
        MultiscaleOptions(
            scale_bias=multiscale_scale_bias,
            scales=multiscale_scales,
            n_scales=multiscale_n_scales,
            kernel=multiscale_kernel,
            max_iter_sub_minor=multiscale_max_iter_sub_minor,
            sub_minor_fraction=multiscale_sub_minor_fraction,
            selection=multiscale_selection,
            selection_margin=multiscale_selection_margin,
        )
        if multiscale
        else None
    )

    return _rmclean_nd(
        rm_synth_arrays,
        clean_options,
        multiscale=multiscale,
        multiscale_options=multiscale_options,
        phi_max_scale_radm2=phi_max_scale_radm2,
    )


def _rmclean_nd(
    rm_synth_arrays: RMSynthArrays,
    clean_options: RMCleanOptions,
    multiscale: bool = False,
    multiscale_options: MultiscaleOptions | None = None,
    phi_max_scale_radm2: float | None = None,
) -> RMCleanResults:
    """Validate array shapes and CLEAN each pixel (single- or multi-scale).

    Args:
        rm_synth_arrays (RMSynthArrays): Dirty FDF cube, phi axes, RMSF, masks.
        clean_options (RMCleanOptions): Mask, threshold, gain, max_iter.
        multiscale (bool): Use multiscale CLEAN.
        multiscale_options (MultiscaleOptions | None): Multiscale settings if enabled.
        phi_max_scale_radm2 (float | None): Largest recoverable scale for the auto grid.

    Raises:
        ValueError: If the input array shapes are inconsistent.

    Returns:
        RMCleanResults: Per-pixel clean/model/resid cubes and iteration counts.
    """
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
    noise_rmsf_arr_2d = (
        nd_to_two_d(rm_synth_arrays.noise_rmsf_arr)
        if rm_synth_arrays.noise_rmsf_arr is not None
        else None
    )
    iter_count_arr_2d = np.zeros(dirty_fdf_arr_2d.shape[1:], dtype=int)
    sub_minor_iter_arr_2d = np.zeros(dirty_fdf_arr_2d.shape[1:], dtype=int)
    fwhm_rmsf_arr_2d = nd_to_two_d(rm_synth_arrays.fwhm_rmsf_arr)

    # Initialise arrays to hold the residual FDF, clean components, clean FDF
    # Residual is initially copies of dirty FDF, so that pixels that are not
    #  processed get correct values (but will be overridden when processed)
    clean_fdf_spectrum_2d = np.zeros_like(dirty_fdf_arr_2d)
    model_fdf_spectrum_2d = np.zeros(dirty_fdf_arr_2d.shape, dtype=complex)
    resid_fdf_arr_2d = dirty_fdf_arr_2d.copy()

    if multiscale:
        ms_options = multiscale_options or MultiscaleOptions()
        rmsf_fwhm = float(np.nanmedian(np.real(fwhm_rmsf_arr_2d)))
        scales = default_scales(
            rm_synth_arrays.phi_arr_radm2,
            rmsf_fwhm,
            ms_options,
            phi_max_scale_radm2=phi_max_scale_radm2,
        )
        logger.info(f"Multiscale scales (RMSF FWHM units): {scales}")

    # Loop through the pixels containing a polarised signal
    for pix_idx in tqdm(
        range(dirty_fdf_arr_2d.shape[1]),
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
                    clean_options=clean_options,
                    multiscale_options=ms_options,
                    noise_rmsf_spectrum=noise_rmsf_arr_2d[:, pix_idx]
                    if noise_rmsf_arr_2d is not None
                    else None,
                )
            )
            clean_fdf_spectrum_2d[:, pix_idx] = clean_spec
            resid_fdf_arr_2d[:, pix_idx] = resid_spec
            model_fdf_spectrum_2d[:, pix_idx] = model_spec
            iter_count_arr_2d[pix_idx] = iters
            sub_minor_iter_arr_2d[pix_idx] = sub_minor_iters
            continue
        clean_loop_results = minor_cycle(
            rm_synth_1d_arrays=RMSynthArrays(
                dirty_fdf_arr=resid_fdf_arr_2d[:, pix_idx],
                phi_arr_radm2=rm_synth_arrays.phi_arr_radm2,
                rmsf_arr=rmsf_arr_2d[:, pix_idx],
                phi_double_arr_radm2=rm_synth_arrays.phi_double_arr_radm2,
                fwhm_rmsf_arr=fwhm_rmsf_arr_2d,
                fdf_mask_arr=nd_to_two_d(rm_synth_arrays.fdf_mask_arr)[:, pix_idx]
                if rm_synth_arrays.fdf_mask_arr is not None
                else None,
            ),
            clean_options=clean_options,
        )
        clean_fdf_spectrum_2d[:, pix_idx] = clean_loop_results.clean_fdf_spectrum
        resid_fdf_arr_2d[:, pix_idx] = clean_loop_results.resid_fdf_spectrum
        model_fdf_spectrum_2d[:, pix_idx] = clean_loop_results.model_fdf_spectrum
        iter_count_arr_2d[pix_idx] = clean_loop_results.iter_count
        # Single-scale: each minor iteration places one component, so the
        # component count equals the minor-iteration count.
        sub_minor_iter_arr_2d[pix_idx] = clean_loop_results.iter_count

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
    for array in (
        rm_synth_1d_arrays.dirty_fdf_arr,
        rm_synth_1d_arrays.phi_arr_radm2,
        rm_synth_1d_arrays.phi_double_arr_radm2,
    ):
        if array.ndim != 1:
            msg = "Arrays in minor cycle must be 1D."
            raise ValueError(msg)

    # Initialise arrays to hold the residual FDF, clean components, clean FDF
    resid_fdf_spectrum = rm_synth_1d_arrays.dirty_fdf_arr.copy()

    mask_arr = np.abs(rm_synth_1d_arrays.dirty_fdf_arr) > clean_options.mask

    if rm_synth_1d_arrays.fdf_mask_arr is not None:
        assert rm_synth_1d_arrays.fdf_mask_arr.ndim == 1, (
            "Arrays in minor cycle must be 1D."
        )
        mask_arr = np.logical_and(
            mask_arr,
            rm_synth_1d_arrays.fdf_mask_arr.astype(bool),
        )

    resid_fdf_spectrum_mask = np.ma.array(resid_fdf_spectrum, mask=~mask_arr)

    minor_loop_arrays = MinorLoopArrays(
        resid_fdf_spectrum_mask=resid_fdf_spectrum_mask,
        phi_arr_radm2=rm_synth_1d_arrays.phi_arr_radm2,
        phi_double_arr_radm2=rm_synth_1d_arrays.phi_double_arr_radm2,
        rmsf_spectrum=rm_synth_1d_arrays.rmsf_arr,
        rmsf_fwhm=float(rm_synth_1d_arrays.fwhm_rmsf_arr.squeeze()),
    )

    minor_loop_options = MinorLoopOptions(
        max_iter=clean_options.max_iter,
        gain=clean_options.gain,
        mask=clean_options.mask,
        threshold=clean_options.threshold,
        start_iter=0,
        update_mask=True,
    )

    logger.info("Starting initial minor loop...")
    initial_loop_results = minor_loop(
        minor_loop_arrays=minor_loop_arrays,
        minor_loop_options=minor_loop_options,
    )

    # Deep clean
    # Mask where clean components have been added
    mask_arr = np.abs(initial_loop_results.model_fdf_spectrum) > 0
    resid_fdf_spectrum = initial_loop_results.resid_fdf_spectrum
    resid_fdf_spectrum_mask = np.ma.array(resid_fdf_spectrum, mask=~mask_arr)

    logger.info("Initial loop complete. Starting deep clean...")

    deep_loop_results = minor_loop(
        minor_loop_arrays=minor_loop_arrays._replace(
            resid_fdf_spectrum_mask=resid_fdf_spectrum_mask
        ),
        minor_loop_options=dataclasses.replace(
            minor_loop_options,
            start_iter=initial_loop_results.iter_count,
            update_mask=False,
        ),
    )

    clean_fdf_spectrum = np.squeeze(
        deep_loop_results.clean_fdf_spectrum + initial_loop_results.clean_fdf_spectrum
    )
    resid_fdf_spectrum = np.squeeze(deep_loop_results.resid_fdf_spectrum)
    model_fdf_spectrum = np.squeeze(
        deep_loop_results.model_fdf_spectrum + initial_loop_results.model_fdf_spectrum
    )

    return CleanLoopResults(
        clean_fdf_spectrum=clean_fdf_spectrum,
        resid_fdf_spectrum=resid_fdf_spectrum,
        model_fdf_spectrum=model_fdf_spectrum,
        iter_count=deep_loop_results.iter_count,
    )


@dataclass(frozen=True, kw_only=True, slots=True)
class MultiscaleOptions:
    """Options for multiscale RM-CLEAN."""

    scale_bias: float = 0.6
    """Bias selector only: scale-bias in (0, 1] (Offringa & Smirnov 2017, eq. 3).
    Lower favours larger scales; 1 behaves like single-scale. Default 0.6 (WSClean)."""
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
    """Scale selector (default "hybrid"). "bias" = Offringa eq-3 scale-bias; "snr" =
    matched filter max|R conv K_s| / sigma_s (finer grid, no bias tuning needed);
    "hybrid" = width-gated snr: engages extended scales only when the residual
    peak fits wider than the measured dirty beam and the extended score competes
    with scale 0, else behaves as "snr". Plain "snr" scores are near
    scale-degenerate under correlated FDF noise, so with the default margin it
    almost never engages; "hybrid" does while staying point-safe."""
    selection_margin: float = 0.08
    """SNR selector (and the hybrid fallback): relative margin in [0, 1) favouring
    smaller scales. Among scales within this fraction of the best score, the
    smallest wins, which keeps points on the delta scale. 0 = raw argmax."""

    def __post_init__(self) -> None:
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
        if self.scales is not None and len(self.scales) == 0:
            msg = "scales must be non-empty."
            raise ValueError(msg)


@np.vectorize
def _scale_bias_function(scale: float, scale_0: float, scale_bias: float) -> float:
    """Scale-bias function (Offringa & Smirnov 2017, eq. 3)."""
    if scale == 0:
        return 1.0
    return float(scale_bias ** (-1 - np.log2(scale / scale_0)))


def scale_bias_function(
    scales: NDArray[np.float64],
    scale_bias: float,
) -> NDArray[np.float64]:
    """Scale-bias weighting per scale (Offringa & Smirnov 2017, eq. 3)."""
    if len(scales) == 1:
        return np.ones_like(scales)
    first_nonzero = scales[scales > 0].min()
    return np.asarray(
        _scale_bias_function(scales, scale_0=first_nonzero, scale_bias=scale_bias),
        dtype=np.float64,
    )


def make_scales(
    max_scale: float,
    n_scales: int | None = None,
    first_scale: float = 4.0,
) -> NDArray[np.float64]:
    """WSClean-style scale set: 0 then geometric doubling up to `max_scale`.

    `first_scale` defaults to 4 (RMSF FWHM units): WSClean's rule of a window 4x
    the beam FWHM, so the first extended scale is well wider than the RMSF and
    point sources stay on the delta scale.

    Args:
        max_scale (float): Upper bound in RMSF FWHM units (exclusive).
        n_scales (int | None): Cap on the scale count; None keeps all.
        first_scale (float): First extended scale in RMSF FWHM units.

    Returns:
        NDArray[np.float64]: Scales in RMSF FWHM units, starting at 0.
    """
    scales = [0.0]
    scale = first_scale
    while scale < max_scale:
        scales.append(scale)
        scale *= 2
    scale_arr = np.array(scales, dtype=np.float64)
    if n_scales is not None:
        scale_arr = scale_arr[:n_scales]
    return scale_arr


def make_fine_scales(
    max_scale: float,
    n_scales: int | None = None,
) -> NDArray[np.float64]:
    """SNR-selector scale set: 0, 3, then geometric doubling from 6 to `max_scale`.

    Finer than `make_scales`, which the SNR selector can afford (it is point-safe
    via sigma_s). The first anchor is 3, not 1.5: a point's matched-filter score at
    s=1.5 is only ~1% below scale 0, so noise flipped bright points onto it.

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


def convolve_fdf_scale(
    scale: float,
    fwhm: float,
    fdf_arr: NDArray[np.complex128] | NDArray[np.float64],
    phi_double_arr_radm2: NDArray[np.float64],
    kernel: KernelType = "gaussian",
    sum_normalised: bool = True,
) -> NDArray[np.complex128] | NDArray[np.float64]:
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
    kernel_func = KERNEL_FUNCS.get(kernel, gaussian_scale_kernel_function)
    # Sample the kernel on a zero-centred grid of the SAME length as the signal
    # (same d_phi as phi_double). A kernel longer than the signal would produce
    # boundary garbage under scipy's reflect mode; this keeps them matched for
    # both FDF-length and RMSF-length inputs.
    d_phi = float(phi_double_arr_radm2[1] - phi_double_arr_radm2[0])
    n = len(fdf_arr)
    kernel_grid = (np.arange(n) - n // 2) * d_phi
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
    model_fdf_spectrum: NDArray[np.complex128],
    phi_double_arr_radm2: NDArray[np.float64],
    rmsf_fwhm: float,
) -> NDArray[np.complex128]:
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
    model_fdf_spectrum: NDArray[np.complex128],
    rmsf_spectrum: NDArray[np.complex128],
    phi_arr_radm2: NDArray[np.float64],
    phi_double_arr_radm2: NDArray[np.float64],
) -> NDArray[np.complex128]:
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
    rmsf_conv_scale: list[NDArray[np.complex128]]
    """`RMSF conv K_s` per scale: the scale-s response in the full-res residual,
    used for the residual subtraction. On the double phi axis."""
    rmsf_conv_scale_twice: list[NDArray[np.complex128]]
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
    for the matched-filter (SNR) selector: sqrt((K_s conv K_s conv C)(0) / C(0))
    with C the noise-covariance RMSF. sigma_s[0] == 1."""
    point_width: float
    """Measured half-max width of |RMSF| in rad/m^2: the dirty-beam width a
    delta presents in the dirty |FDF|, near-in sidelobe shoulders included.
    Wider than the theoretical fwhm under non-uniform lambda^2 sampling; the
    hybrid selector's width test calibrates against this, never the fwhm."""


def compute_scale_kernels(
    scales: NDArray[np.float64],
    rmsf_spectrum: NDArray[np.complex128],
    rmsf_fwhm: float,
    phi_double_arr_radm2: NDArray[np.float64],
    kernel: KernelType,
    noise_rmsf_spectrum: NDArray[np.complex128] | None = None,
) -> ScaleKernels:
    """Precompute the per-scale RMSF responses for one spectrum.

    Follows Offringa & Smirnov (2017), Section 2.2; see `ScaleKernels`.

    Args:
        scales (NDArray[np.float64]): Scales in RMSF FWHM units.
        rmsf_spectrum (NDArray[np.complex128]): RMSF on the double-phi axis.
        rmsf_fwhm (float): RMSF FWHM in rad/m^2.
        phi_double_arr_radm2 (NDArray[np.float64]): Double-phi axis.
        kernel (KernelType): Scale-kernel shape.
        noise_rmsf_spectrum (NDArray[np.complex128] | None): w^2-RMSF noise
            covariance for sigma_s; None uses `rmsf_spectrum` (exact for uniform
            weights).

    Returns:
        ScaleKernels: Per-scale responses (see that type).
    """
    rmsf_conv_scale: list[NDArray[np.complex128]] = []
    rmsf_conv_scale_twice: list[NDArray[np.complex128]] = []
    peak_response = np.ones_like(scales)
    fwhm_conv_scale_twice = np.full_like(scales, rmsf_fwhm)
    # sigma_s = sqrt((K_s conv K_s conv C)(0) / C(0)): the zero-lag value of the
    # noise covariance C twice-convolved with the scale kernel. C is the w^2-RMSF
    # when supplied, else the ordinary RMSF (exact for uniform weights). "Zero
    # lag" is the RMSF peak channel; the twice-conv response peaks there too.
    cov = rmsf_spectrum if noise_rmsf_spectrum is None else noise_rmsf_spectrum
    cov_peak_index = int(np.nanargmax(np.abs(cov)))
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
        if noise_rmsf_spectrum is None:
            # C == RMSF, so the twice-conv of C is exactly conv_twice; reuse it.
            qform[i] = float(np.real(conv_twice[cov_peak_index]))
        else:
            noise_conv_once = convolve_fdf_scale(
                scale, rmsf_fwhm, cov, phi_double_arr_radm2, kernel
            )
            noise_conv_twice = convolve_fdf_scale(
                scale, rmsf_fwhm, noise_conv_once, phi_double_arr_radm2, kernel
            )
            qform[i] = float(np.real(noise_conv_twice[cov_peak_index]))
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
    resid_fdf_spectrum: NDArray[np.complex128],
    scale_kernels: ScaleKernels,
    scale_bias: float,
    rmsf_fwhm: float,
    phi_double_arr_radm2: NDArray[np.float64],
    kernel: KernelType,
    active: NDArray[np.bool_] | None = None,
    selection: SelectionType = "bias",
    selection_margin: float = 0.0,
    engage_floor: float = 0.0,
) -> int:
    """Index of the most-significant scale (Offringa & Smirnov 2017).

    "bias" scores `max|resid conv K_s| * bias_s` (lower `scale_bias` favours larger
    scales); "snr" scores the matched filter `max|resid conv K_s| / sigma_s`, with
    `selection_margin` preferring the smallest scale within that margin of the best.
    "hybrid" is width-gated snr: engage an extended scale only when the residual
    peak fits wider than the measured dirty beam AND its score competes with
    scale 0, else fall back to "snr" (see `_hybrid_scale_selection`).

    Args:
        resid_fdf_spectrum (NDArray[np.complex128]): Current residual FDF.
        scale_kernels (ScaleKernels): Precomputed per-scale responses.
        scale_bias (float): Bias-selector scale-bias (unused otherwise).
        rmsf_fwhm (float): RMSF FWHM in rad/m^2.
        phi_double_arr_radm2 (NDArray[np.float64]): Double-phi axis.
        kernel (KernelType): Scale-kernel shape.
        active (NDArray[np.bool_] | None): Scales still allowed to be selected.
        selection ("bias" | "snr" | "hybrid"): Scale selector.
        selection_margin (float): SNR parsimony margin (see MultiscaleOptions);
            for "hybrid" it applies to the snr fallback.
        engage_floor (float): Hybrid only: below this residual peak (absolute FDF
            units) always fall back to snr. 0 disables the gate.

    Returns:
        int: Index into `scale_kernels.scales` of the selected scale.
    """
    if selection == "hybrid":
        return _hybrid_scale_selection(
            resid_fdf_spectrum,
            scale_kernels,
            rmsf_fwhm,
            phi_double_arr_radm2,
            kernel,
            active=active,
            selection_margin=selection_margin,
            engage_floor=engage_floor,
        )
    bias = scale_bias_function(scale_kernels.scales, scale_bias)
    scores = np.full_like(scale_kernels.scales, -np.inf)
    for i, scale in enumerate(scale_kernels.scales):
        if active is not None and not active[i]:
            continue
        resid_conv = convolve_fdf_scale(
            scale, rmsf_fwhm, resid_fdf_spectrum, phi_double_arr_radm2, kernel
        )
        peak = float(np.nanmax(np.abs(resid_conv)))
        if selection == "snr":
            scores[i] = peak / scale_kernels.sigma_s[i]
        else:
            scores[i] = peak * bias[i]
    if selection == "snr" and selection_margin > 0:
        best = float(np.max(scores))
        if np.isfinite(best) and best > 0:
            # Scales are sorted ascending, so the first within-margin index is the
            # smallest such scale.
            within = scores >= best * (1 - selection_margin)
            return int(np.argmax(within))
    return int(np.argmax(scores))


def _hybrid_scale_selection(
    resid_fdf_spectrum: NDArray[np.complex128],
    scale_kernels: ScaleKernels,
    rmsf_fwhm: float,
    phi_double_arr_radm2: NDArray[np.float64],
    kernel: KernelType,
    active: NDArray[np.bool_] | None,
    selection_margin: float,
    engage_floor: float,
) -> int:
    """Width-gated snr selection.

    Engage an extended scale only when two independent tests agree, else fall
    back to plain snr selection with `selection_margin`:
    1. Width: bounded half-max fit of the residual peak wider than
       HYBRID_WIDTH_FACTOR x the measured dirty-beam width (a delta fits 1.0x,
       the narrowest genuinely extended source ~2x).
    2. Score: best active extended matched-filter score at least
       HYBRID_SCORE_FACTOR x the scale-0 score (kills residual junk that fits
       wide but scores poorly at wide scales).
    Engagement takes the argmax score among active extended scales. Below
    `engage_floor` (endgame) always falls back.
    """

    def snr_fallback() -> int:
        return find_significant_scale(
            resid_fdf_spectrum,
            scale_kernels,
            1.0,
            rmsf_fwhm,
            phi_double_arr_radm2,
            kernel,
            active=active,
            selection="snr",
            selection_margin=selection_margin,
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
    if (hi - lo + 1) * d_phi < HYBRID_WIDTH_FACTOR * point_width:
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
            float(scale), rmsf_fwhm, resid_fdf_spectrum, phi_double_arr_radm2, kernel
        )
        scores[i] = float(np.nanmax(np.abs(resid_conv))) / float(
            scale_kernels.sigma_s[i]
        )
    best_ext = int(np.argmax(np.where(ext_active, scores, -np.inf)))
    if scores[best_ext] < HYBRID_SCORE_FACTOR * scores[0]:
        return snr_fallback()
    return best_ext


def _multiscale_minor_cycles(
    resid_fdf_spectrum: NDArray[np.complex128],
    model_fdf_spectrum: NDArray[np.complex128],
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
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], int, int]:
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
    scale_bias = multiscale_options.scale_bias
    selection = multiscale_options.selection
    active = np.array([bool(s.any()) for s in allowed_supports])
    support = np.logical_or.reduce(allowed_supports)
    n_iter = 0
    sub_minor_total = 0
    if not support.any():
        return resid_fdf_spectrum, model_fdf_spectrum, n_iter, sub_minor_total

    # Divergence guard: keep the lowest-peak state seen and bail if the residual
    # runs away. The peak is measured over the WHOLE array, not just `support`:
    # an over-large scale can grow a spurious feature outside the clean region.
    best_peak = np.inf
    best_model = model_fdf_spectrum.copy()
    best_resid = resid_fdf_spectrum.copy()
    stall_count = 0

    for n_iter in range(1, max_iter + 1):
        peak = float(np.nanmax(np.abs(resid_fdf_spectrum)))
        support_peak = float(np.nanmax(np.abs(resid_fdf_spectrum[support])))
        if peak > best_peak * DIVERGENCE_FACTOR:
            logger.warning(
                f"Multiscale CLEAN diverging at iter {n_iter} "
                f"(peak {peak:0.3g} > {DIVERGENCE_FACTOR}x best {best_peak:0.3g}); "
                "reverting to best."
            )
            model_fdf_spectrum, resid_fdf_spectrum = best_model, best_resid
            break
        if peak < best_peak:
            stall_count = (
                0
                if peak < best_peak * (1 - STALL_REL_IMPROVEMENT)
                else (stall_count + 1)
            )
            best_peak = peak
            best_model = model_fdf_spectrum.copy()
            best_resid = resid_fdf_spectrum.copy()
        else:
            stall_count += 1
        if support_peak < stop_threshold:
            break
        if stall_count >= STALL_PATIENCE:
            logger.warning(
                f"Multiscale CLEAN stalled at iter {n_iter} "
                f"(peak {peak:0.3g} not improving, threshold {stop_threshold:0.3g}); "
                "stopping at best."
            )
            model_fdf_spectrum, resid_fdf_spectrum = best_model, best_resid
            break

        scale_index = find_significant_scale(
            resid_fdf_spectrum,
            kernels,
            scale_bias,
            rmsf_fwhm,
            phi_double_arr_radm2,
            kernel,
            active=active,
            selection=selection,
            selection_margin=multiscale_options.selection_margin,
            engage_floor=HYBRID_ENGAGE_MASK_FACTOR * clean_options.mask,
        )
        scale = float(scales[scale_index])
        peak_response = float(kernels.peak_response[scale_index])
        sigma_s_i = float(kernels.sigma_s[scale_index])

        # thresh_scale sets the detection threshold in the once-convolved residual,
        # where the noise std scales by sigma_s (= sqrt(peak_response) for uniform
        # weights) relative to scale 0. "snr" uses sigma_s; "bias" keeps the
        # historical peak_response (an amplitude stop, not a noise threshold).
        # Amplitude recovery (dividing by peak_response) is separate, see below.
        thresh_scale = peak_response if selection == "bias" else sigma_s_i

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
                mask=stop_threshold * thresh_scale,
                threshold=threshold_sub,
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
    dirty_fdf_spectrum: NDArray[np.complex128],
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
    if width < HYBRID_WIDTH_FACTOR * kernels.point_width:
        return dataclasses.replace(multiscale_options, selection="snr")
    return multiscale_options


def multiscale_clean_spectrum(
    dirty_fdf_spectrum: NDArray[np.complex128],
    phi_arr_radm2: NDArray[np.float64],
    phi_double_arr_radm2: NDArray[np.float64],
    rmsf_spectrum: NDArray[np.complex128],
    rmsf_fwhm: float,
    scales: NDArray[np.float64],
    clean_options: RMCleanOptions,
    multiscale_options: MultiscaleOptions,
    noise_rmsf_spectrum: NDArray[np.complex128] | None = None,
) -> tuple[
    NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128], int, int
]:
    """Multiscale CLEAN one FDF spectrum (Offringa & Smirnov 2017).

    Two-phase auto-mask (Section 2.2.3): phase 1 cleans to `mask` and records where
    each scale places flux; phase 2 deep-cleans to `threshold`, restricting each
    scale to its dilated phase-1 footprint (stops a scale smearing flux across the
    window). Give a generous phi window, as scale kernels wider than it pick up
    reflect-mode boundary artefacts.

    Args:
        dirty_fdf_spectrum (NDArray[np.complex128]): Dirty FDF on the phi axis.
        phi_arr_radm2 (NDArray[np.float64]): Faraday depth axis in rad/m^2.
        phi_double_arr_radm2 (NDArray[np.float64]): Double-phi axis (RMSF length).
        rmsf_spectrum (NDArray[np.complex128]): RMSF on the double-phi axis.
        rmsf_fwhm (float): RMSF FWHM in rad/m^2.
        scales (NDArray[np.float64]): Scales in RMSF FWHM units.
        clean_options (RMCleanOptions): Mask, threshold, gain, max_iter.
        multiscale_options (MultiscaleOptions): Kernel, selector, sub-minor settings.
        noise_rmsf_spectrum (NDArray[np.complex128] | None): w^2-RMSF noise
            covariance for the SNR sigma_s; None uses `rmsf_spectrum`.

    Returns:
        (clean, resid, model, minor_iters, sub_minor_iters): minor_iters counts
        minor cycles (scale re-selections); sub_minor_iters the per-scale Hogbom
        steps (comparable to single-scale's iteration count).
    """
    mask = clean_options.mask
    threshold = clean_options.threshold
    kernel = multiscale_options.kernel
    resid_fdf_spectrum = dirty_fdf_spectrum.copy()
    model_fdf_spectrum = np.zeros_like(resid_fdf_spectrum)
    kernels = compute_scale_kernels(
        scales,
        rmsf_spectrum,
        rmsf_fwhm,
        phi_double_arr_radm2,
        kernel,
        noise_rmsf_spectrum=noise_rmsf_spectrum,
    )
    multiscale_options = _classify_hybrid_source(
        dirty_fdf_spectrum, phi_double_arr_radm2, kernels, multiscale_options
    )

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
        for scale, peak_response_s in zip(scales, kernels.peak_response, strict=True)
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
    response flattens, its normalisation blows up). "bias" takes the coarse
    `make_scales` grid, "snr" and "hybrid" the finer `make_fine_scales`.

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
    if multiscale_options.selection in ("snr", "hybrid"):
        scales = make_fine_scales(max_scale, n_scales=multiscale_options.n_scales)
        first_scale = 3.0
    else:
        scales = make_scales(max_scale, n_scales=multiscale_options.n_scales)
        first_scale = 4.0
    if len(scales) == 1:
        logger.warning(
            "Multiscale auto grid degenerated to [0.0]: the phi window is narrower "
            f"than 2*first_scale*RMSF FWHM ({2 * first_scale * rmsf_fwhm:0.3g} rad/m^2), "
            "so no extended scale fits and multiscale CLEAN reduces to single-scale. "
            "Pass explicit `multiscale_scales` to force an extended grid."
        )
    return scales
