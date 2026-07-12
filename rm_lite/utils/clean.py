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
    fwhm_to_sigma,
    gaussian,
    gaussian_integrand,
    unit_centred_gaussian,
)
from rm_lite.utils.logging import TqdmToLogger, logger

TQDM_OUT = TqdmToLogger(logger, level=logging.INFO)

DType = TypeVar("DType", bound=np.generic)

KernelType: TypeAlias = Literal["tapered_quad", "gaussian"]

# Bail out of a spectrum's major loop if the residual peak exceeds this factor
# times the best (lowest) peak seen: a runaway-divergence backstop.
DIVERGENCE_FACTOR = 2.0


class RMCleanResults(NamedTuple):
    """Results of the RM-CLEAN calculation"""

    clean_fdf_arr: NDArray[np.complex128]
    """The cleaned Faraday dispersion function cube"""
    model_fdf_arr: NDArray[np.complex128]
    """The clean components cube"""
    clean_iter_arr: NDArray[np.int16]
    """The number of iterations for each pixel"""
    resid_fdf_arr: NDArray[np.complex128]
    """The residual Faraday dispersion function cube"""


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
    peak_find_arr: NDArray[np.float64] | None = None
    """Peak finding array"""


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
    then clip to FDF length. Shared shift-and-subtract primitive (Hogbom and
    multiscale). Assumes integer shifts and a symmetric RMSF.
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
    # Trust nothing
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

    if minor_loop_arrays.peak_find_arr is not None:
        peak_find_arr = minor_loop_arrays.peak_find_arr.copy()
        peak_find_arr_mask = np.ma.array(
            minor_loop_arrays.peak_find_arr.copy(), mask=~mask_arr
        )
    else:
        peak_find_arr_mask = None

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
        if peak_find_arr_mask is not None:
            peak_fdf_index = np.ma.argmax(np.abs(peak_find_arr_mask))
        else:
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
        if minor_loop_arrays.peak_find_arr is not None:
            peak_find_arr -= np.abs(clean_component * shifted_rmsf_spectrum)

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
        if peak_find_arr_mask is not None:
            peak_find_arr_mask = np.ma.array(peak_find_arr, mask=~mask_arr)

    return MinorLoopResults(
        clean_fdf_spectrum=clean_fdf_spectrum,
        resid_fdf_spectrum=resid_fdf_spectrum,
        resid_fdf_spectrum_mask=resid_fdf_spectrum_mask,
        model_fdf_spectrum=model_fdf_spectrum,
        model_rmsf_spectrum=model_rmsf_spectrum,
        iter_count=iter_count,
    )


def restore_fdf(
    model_fdf_spectrum: NDArray[np.complex128],
    phi_double_arr_radm2: NDArray[np.float64],
    fwhm_rmsf: float,
) -> NDArray[np.complex128]:
    clean_beam = unit_centred_gaussian(
        x=phi_double_arr_radm2,
        fwhm=fwhm_rmsf,
    ) / gaussian_integrand(amplitude=1, fwhm=fwhm_rmsf)
    restored_fdf = np.convolve(
        model_fdf_spectrum.real, clean_beam, mode="valid"
    ) + 1j * np.convolve(model_fdf_spectrum.imag, clean_beam, mode="valid")
    return np.array(restored_fdf[1:-1])


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
    multiscale_scale_bias: float = 0.95,
    multiscale_scales: NDArray[np.float64] | None = None,
    multiscale_n_scales: int | None = None,
    multiscale_kernel: KernelType = "tapered_quad",
    multiscale_max_iter_sub_minor: int = 10_000,
    multiscale_sub_minor_fraction: float = 0.5,
    phi_max_scale_radm2: float | None = None,
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
        multiscale_scale_bias (float, optional): Scale-bias in (0, 1]; lower favours larger scales more. Defaults to 0.95.
        multiscale_scales (NDArray[np.float64] | None, optional): Explicit scales (RMSF FWHM units); None auto-selects.
        multiscale_n_scales (int | None, optional): Cap on the auto scale count.
        multiscale_kernel ("tapered_quad" | "gaussian", optional): Scale kernel. Defaults to "tapered_quad".
        multiscale_max_iter_sub_minor (int, optional): Max sub-minor iterations per scale. Defaults to 10_000.
        multiscale_sub_minor_fraction (float, optional): Sub-minor re-selection fraction. Defaults to 0.5.
        phi_max_scale_radm2 (float | None, optional): Largest recoverable Faraday scale (pi / lambda_sq_min); sets the auto scale range. None falls back to the phi window.

    Returns:
        RMCleanResults: clean_fdf_arr, model_fdf_arr, clean_iter_arr, resid_fdf_arr
    """
    rm_synth_arrays = RMSynthArrays(
        dirty_fdf_arr=dirty_fdf_arr,
        phi_arr_radm2=phi_arr_radm2,
        rmsf_arr=rmsf_arr,
        phi_double_arr_radm2=phi_double_arr_radm2,
        fwhm_rmsf_arr=fwhm_rmsf_arr,
        fdf_mask_arr=mask_arr,
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
    iter_count_arr_2d = np.zeros(dirty_fdf_arr_2d.shape[1:], dtype=int)
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
            (clean_spec, resid_spec, model_spec, iters) = multiscale_clean_spectrum(
                dirty_fdf_spectrum=resid_fdf_arr_2d[:, pix_idx],
                phi_arr_radm2=rm_synth_arrays.phi_arr_radm2,
                phi_double_arr_radm2=rm_synth_arrays.phi_double_arr_radm2,
                rmsf_spectrum=rmsf_arr_2d[:, pix_idx],
                rmsf_fwhm=rmsf_fwhm,
                scales=scales,
                clean_options=clean_options,
                multiscale_options=ms_options,
            )
            clean_fdf_spectrum_2d[:, pix_idx] = clean_spec
            resid_fdf_arr_2d[:, pix_idx] = resid_spec
            model_fdf_spectrum_2d[:, pix_idx] = model_spec
            iter_count_arr_2d[pix_idx] = iters
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
    else:
        iter_count_arr = two_d_to_nd(
            iter_count_arr_2d, rm_synth_arrays.dirty_fdf_arr.shape[1:]
        )
    resid_fdf_arr = two_d_to_nd(resid_fdf_arr_2d, rm_synth_arrays.dirty_fdf_arr.shape)

    return RMCleanResults(
        clean_fdf_spectrum, model_fdf_spectrum, iter_count_arr, resid_fdf_arr
    )


def minor_cycle(
    rm_synth_1d_arrays: RMSynthArrays,
    clean_options: RMCleanOptions,
) -> CleanLoopResults:
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

    scale_bias: float = 0.95
    """Scale-bias in (0, 1] (Offringa & Smirnov 2017, eq. 3). Sets how strongly
    scale selection prefers larger scales: 1 behaves like single-scale CLEAN,
    lower values favour larger scales (and too low over-extends unresolved
    sources). The RMSF is broad relative to the scale kernels, so the per-scale
    peaks are close together and selection is very sensitive to the bias; the
    default 0.95 is near single-scale so a Faraday-thin source stays a delta,
    while still recovering thick structure. WSClean's image-domain default (0.6)
    over-extends here because its PSF is far narrower than the RMSF."""
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

    def __post_init__(self) -> None:
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
    first_scale: float = 1.0,
) -> NDArray[np.float64]:
    """WSClean-style scale set: 0 then geometric doubling up to `max_scale`."""
    scales = [0.0]
    scale = first_scale
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
    """Tapered quadratic scale kernel (Offringa & Smirnov 2017, eq. 2)."""
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
    """Gaussian scale kernel; sigma = (3/16) * scale (Offringa & Smirnov 2017)."""
    rmsf_sigma = fwhm_to_sigma(rmsf_fwhm)
    sigma = (3 / 16) * scale * rmsf_sigma
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

    scipy.ndimage.convolve drops the imaginary part for complex input, so the
    real and imaginary parts are convolved separately.
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


def compute_scale_kernels(
    scales: NDArray[np.float64],
    rmsf_spectrum: NDArray[np.complex128],
    rmsf_fwhm: float,
    phi_double_arr_radm2: NDArray[np.float64],
    kernel: KernelType,
) -> ScaleKernels:
    """Precompute the per-scale RMSF responses for one spectrum.

    See `ScaleKernels`; follows Offringa & Smirnov (2017), Section 2.2.
    """
    rmsf_conv_scale: list[NDArray[np.complex128]] = []
    rmsf_conv_scale_twice: list[NDArray[np.complex128]] = []
    peak_response = np.ones_like(scales)
    fwhm_conv_scale_twice = np.full_like(scales, rmsf_fwhm)
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
    return ScaleKernels(
        scales,
        rmsf_conv_scale,
        rmsf_conv_scale_twice,
        peak_response,
        fwhm_conv_scale_twice,
    )


def find_significant_scale(
    resid_fdf_spectrum: NDArray[np.complex128],
    scale_kernels: ScaleKernels,
    scale_bias: float,
    rmsf_fwhm: float,
    phi_double_arr_radm2: NDArray[np.float64],
    kernel: KernelType,
) -> int:
    """Index of the bias-weighted most-significant scale (Offringa 2017).

    Selection is `max|resid (conv) k_s| * bias_s`. For `scale_bias` < 1 `bias_s`
    is > 1 for larger scales (and grows as `scale_bias` drops), so a lower
    `scale_bias` favours larger scales; `scale_bias` near 1 makes `bias_s` ~ 1
    for all scales, keeping a point source on scale 0.
    """
    bias = scale_bias_function(scale_kernels.scales, scale_bias)
    scores = np.zeros_like(scale_kernels.scales)
    for i, scale in enumerate(scale_kernels.scales):
        resid_conv = convolve_fdf_scale(
            scale, rmsf_fwhm, resid_fdf_spectrum, phi_double_arr_radm2, kernel
        )
        scores[i] = np.nanmax(np.abs(resid_conv)) * bias[i]
    return int(np.argmax(scores))


def multiscale_clean_spectrum(
    dirty_fdf_spectrum: NDArray[np.complex128],
    phi_arr_radm2: NDArray[np.float64],
    phi_double_arr_radm2: NDArray[np.float64],
    rmsf_spectrum: NDArray[np.complex128],
    rmsf_fwhm: float,
    scales: NDArray[np.float64],
    clean_options: RMCleanOptions,
    multiscale_options: MultiscaleOptions,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128], int]:
    """Multiscale CLEAN one FDF spectrum. Returns (clean, resid, model, iters)."""
    mask = clean_options.mask
    threshold = clean_options.threshold
    kernel = multiscale_options.kernel
    resid_fdf_spectrum = dirty_fdf_spectrum.copy()
    model_fdf_spectrum = np.zeros_like(resid_fdf_spectrum)
    kernels = compute_scale_kernels(
        scales, rmsf_spectrum, rmsf_fwhm, phi_double_arr_radm2, kernel
    )

    # Per-scale mask (Offringa & Smirnov 2017 auto-mask): each scale cleans only
    # where its scale-convolved dirty exceeds the mask. A source at the mask
    # amplitude has a scale-convolved peak of `mask * peak_response`, so the scale
    # kernel sets the mask width: scale 0 is the strict >mask region (no deltas on
    # the wings of a broad component), while a broad scale gets a broad mask so a
    # genuinely broad component is not clipped. `support` (their union) is where
    # any scale may clean, used for the loop-level peak and divergence checks.
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
    support = np.logical_or.reduce(scale_supports)

    iter_count = 0
    if not support.any():
        return (
            _restore_multiscale(model_fdf_spectrum, phi_double_arr_radm2, rmsf_fwhm),
            resid_fdf_spectrum,
            model_fdf_spectrum,
            iter_count,
        )

    # Divergence guard: keep the lowest-peak state seen and bail if the residual
    # runs away (a poorly matched scale can over-subtract and blow up).
    best_peak = np.inf
    best_model = model_fdf_spectrum.copy()
    best_resid = resid_fdf_spectrum.copy()

    for iter_count in range(1, clean_options.max_iter + 1):
        peak = float(np.nanmax(np.abs(resid_fdf_spectrum[support])))
        if peak > best_peak * DIVERGENCE_FACTOR:
            logger.warning(
                f"Multiscale CLEAN diverging at iter {iter_count} "
                f"(peak {peak:0.3g} > {DIVERGENCE_FACTOR}x best {best_peak:0.3g}); "
                "reverting to best."
            )
            model_fdf_spectrum, resid_fdf_spectrum = best_model, best_resid
            break
        if peak < best_peak:
            best_peak = peak
            best_model = model_fdf_spectrum.copy()
            best_resid = resid_fdf_spectrum.copy()
        if peak < threshold:
            break

        scale_index = find_significant_scale(
            resid_fdf_spectrum,
            kernels,
            multiscale_options.scale_bias,
            rmsf_fwhm,
            phi_double_arr_radm2,
            kernel,
        )
        scale = float(scales[scale_index])
        peak_response = float(kernels.peak_response[scale_index])

        # Scale-convolved residual; sub-minor cleans in this space, restricted to
        # the support and down to the clean threshold.
        resid_conv = np.asarray(
            convolve_fdf_scale(
                scale,
                rmsf_fwhm,
                resid_fdf_spectrum,
                phi_double_arr_radm2,
                kernel,
            ),
            dtype=np.complex128,
        )
        mask_conv = (np.abs(resid_conv) > threshold * peak_response) & scale_supports[
            scale_index
        ]
        if not mask_conv.any():
            break
        resid_conv_masked = np.ma.array(resid_conv, mask=~mask_conv)

        # Sub-minor cleans this scale only until its peak drops by
        # `sub_minor_fraction`, then re-select a scale. Stops one scale from
        # deep-cleaning to the floor and over-fitting.
        peak_conv = float(np.ma.max(np.ma.abs(resid_conv_masked)))
        threshold_sub = max(
            threshold * peak_response,
            multiscale_options.sub_minor_fraction * peak_conv,
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
                max_iter=multiscale_options.max_iter_sub_minor,
                gain=clean_options.gain,
                mask=threshold * peak_response,
                threshold=threshold_sub,
                update_mask=True,
            ),
        )

        # Undo the peak-response normalisation to recover true k_s amplitudes.
        true_deltas = sub_minor.model_fdf_spectrum / peak_response
        if not np.any(true_deltas):
            break

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

    clean_fdf_spectrum = _restore_multiscale(
        model_fdf_spectrum, phi_double_arr_radm2, rmsf_fwhm
    )
    return clean_fdf_spectrum, resid_fdf_spectrum, model_fdf_spectrum, iter_count


def default_scales(
    phi_arr_radm2: NDArray[np.float64],
    rmsf_fwhm: float,
    multiscale_options: MultiscaleOptions,
    phi_max_scale_radm2: float | None = None,
) -> NDArray[np.float64]:
    """Scales (RMSF FWHM units): explicit if given, else WSClean-style auto.

    The auto max scale uses the largest recoverable Faraday scale
    `phi_max_scale_radm2` (pi / lambda_sq_min) when provided. If it is not (e.g.
    calling on bare arrays without the frequency sampling), it falls back to the
    FDF phi window. Either way the scale-bias weighting and divergence guard
    handle over-large scales, and an explicit `scales` overrides it entirely.
    """
    if multiscale_options.scales is not None:
        return multiscale_options.scales
    if phi_max_scale_radm2 is not None:
        max_scale = phi_max_scale_radm2 / rmsf_fwhm
    else:
        max_scale = float(phi_arr_radm2.max() - phi_arr_radm2.min()) / (2 * rmsf_fwhm)
    return make_scales(max_scale, n_scales=multiscale_options.n_scales)
