"""Multiscale RM-CLEAN utils"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import convolve
from tqdm.auto import tqdm

from rm_lite.utils.clean import (
    MinorLoopArrays,
    MinorLoopOptions,
    RMCleanResults,
    minor_loop,
)
from rm_lite.utils.fitting import fit_rmsf, fwhm_to_sigma, unit_centred_gaussian
from rm_lite.utils.logging import TqdmToLogger, logger
from rm_lite.utils.synthesis import compute_rmsf_params

TQDM_OUT = TqdmToLogger(logger, level=logging.INFO)

KernelType = Literal["tapered_quad", "gaussian"]

# Bail out of a spectrum's major loop if the residual peak exceeds this factor
# times the best (lowest) peak seen: a runaway-divergence backstop.
DIVERGENCE_FACTOR = 2.0


@dataclass(frozen=True, kw_only=True, slots=True)
class MultiscaleOptions:
    """Options for multiscale RM-CLEAN."""

    scale_bias: float = 0.6
    """Scale-bias. Lower favours larger scales more"""
    scales: NDArray[np.float64] | None = None
    """Explicit scales (RMSF FWHM units); None auto-selects WSClean-style"""
    n_scales: int | None = None
    """Cap on the auto-selected scale count (ignored if `scales` given)"""
    kernel: KernelType = "tapered_quad"
    """Scale kernel shape"""
    max_iter: int = 1000
    """Maximum major cycles"""
    max_iter_sub_minor: int = 10_000
    """Maximum sub-minor (per-scale Hogbom) iterations"""
    gain: float = 0.1
    """Clean loop gain"""
    sub_minor_fraction: float = 0.5
    """Re-select a scale once the activated scale's peak drops by this fraction"""

    def __post_init__(self) -> None:
        if not 0 < self.gain <= 1:
            msg = f"gain must be in (0, 1], got {self.gain}."
            raise ValueError(msg)
        if not 0 < self.sub_minor_fraction < 1:
            msg = (
                f"sub_minor_fraction must be in (0, 1), got {self.sub_minor_fraction}."
            )
            raise ValueError(msg)
        if self.max_iter < 1 or self.max_iter_sub_minor < 1:
            msg = "max_iter and max_iter_sub_minor must be >= 1."
            raise ValueError(msg)
        if self.scales is not None and len(self.scales) == 0:
            msg = "scales must be non-empty."
            raise ValueError(msg)


@np.vectorize
def _scale_bias_function(scale: float, scale_0: float, scale_bias: float) -> float:
    """Scale-bias function (Offringa et al. 2017)."""
    if scale == 0:
        return 1.0
    return scale_bias ** (-1 - np.log2(scale / scale_0))


def scale_bias_function(
    scales: NDArray[np.float64],
    scale_bias: float,
) -> NDArray[np.float64]:
    """Scale-bias weighting per scale (Offringa et al. 2017)."""
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
    """Tapered quadratic scale kernel."""
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
    """Gaussian scale kernel."""
    rmsf_sigma = fwhm_to_sigma(rmsf_fwhm)
    sigma = (3 / 16) * scale * rmsf_sigma
    kernel = unit_centred_gaussian(x=phi_double_arr_radm2, stddev=sigma)
    if sum_normalised:
        kernel /= kernel.sum()
    else:
        kernel /= kernel.max()
    return kernel


KERNEL_FUNCS: dict[str, Callable] = {
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

    mode = "reflect"
    if np.iscomplexobj(fdf_arr):
        conv_spec = convolve(fdf_arr.real, kernel_arr, mode=mode) + 1j * convolve(
            fdf_arr.imag, kernel_arr, mode=mode
        )
    else:
        conv_spec = convolve(fdf_arr, kernel_arr, mode=mode)

    assert len(conv_spec) == len(fdf_arr), "Convolved FDF has wrong length."
    return conv_spec


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
        shifted = np.roll(rmsf_spectrum, int(idx) + n_phi_pad - max_rmsf_index)[
            n_phi_pad:-n_phi_pad
        ]
        out += model_fdf_spectrum[idx] * shifted
    return out


class _ScaleKernels:
    """Precomputed per-scale kernels/couplings for one spectrum."""

    __slots__ = ("fwhm_ss", "gamma", "p_s", "p_ss", "scales")

    def __init__(
        self,
        scales: NDArray[np.float64],
        rmsf_spectrum: NDArray[np.complex128],
        rmsf_fwhm: float,
        phi_double_arr_radm2: NDArray[np.float64],
        kernel: KernelType,
    ) -> None:
        self.scales = scales
        self.p_s: list[NDArray[np.complex128]] = []
        self.p_ss: list[NDArray[np.complex128]] = []
        self.gamma = np.ones_like(scales)
        self.fwhm_ss = np.full_like(scales, rmsf_fwhm)
        for i, scale in enumerate(scales):
            p_s = convolve_fdf_scale(
                scale, rmsf_fwhm, rmsf_spectrum, phi_double_arr_radm2, kernel
            )
            p_ss = convolve_fdf_scale(
                scale, rmsf_fwhm, p_s, phi_double_arr_radm2, kernel
            )
            self.p_s.append(np.asarray(p_s, dtype=np.complex128))
            self.p_ss.append(np.asarray(p_ss, dtype=np.complex128))
            self.gamma[i] = float(np.nanmax(np.abs(p_ss)))
            if scale != 0:
                self.fwhm_ss[i] = fit_rmsf(
                    np.abs(p_ss),
                    phi_double_arr_radm2=phi_double_arr_radm2,
                    fwhm_rmsf_radm2=rmsf_fwhm * scale,
                )


def find_significant_scale(
    resid_fdf_spectrum: NDArray[np.complex128],
    scale_kernels: _ScaleKernels,
    scale_bias: float,
    rmsf_fwhm: float,
    phi_double_arr_radm2: NDArray[np.float64],
    kernel: KernelType,
) -> int:
    """Index of the bias-weighted most-significant scale (Offringa 2017).

    Selection is `max|resid (conv) k_s| * bias_s`.
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
    mask: float,
    threshold: float,
    options: MultiscaleOptions,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128], int]:
    """Multiscale CLEAN one FDF spectrum. Returns (clean, resid, model, iters)."""
    resid_fdf_spectrum = dirty_fdf_spectrum.copy()
    model_fdf_spectrum = np.zeros_like(resid_fdf_spectrum)
    kernels = _ScaleKernels(
        scales, rmsf_spectrum, rmsf_fwhm, phi_double_arr_radm2, options.kernel
    )

    # Divergence guard: keep the lowest-peak state seen and bail if the residual
    # runs away (a poorly matched scale can over-subtract and blow up).
    best_peak = np.inf
    best_model = model_fdf_spectrum.copy()
    best_resid = resid_fdf_spectrum.copy()

    iter_count = 0
    for iter_count in range(1, options.max_iter + 1):
        peak = float(np.nanmax(np.abs(resid_fdf_spectrum)))
        if peak > best_peak * DIVERGENCE_FACTOR:
            logger.warning(
                f"Multiscale CLEAN diverging (peak {peak:0.3g} > "
                f"{DIVERGENCE_FACTOR}x best {best_peak:0.3g}); reverting to best."
            )
            model_fdf_spectrum, resid_fdf_spectrum = best_model, best_resid
            break
        if peak < best_peak:
            best_peak = peak
            best_model = model_fdf_spectrum.copy()
            best_resid = resid_fdf_spectrum.copy()
        if peak < threshold:
            break
        if not (np.abs(resid_fdf_spectrum) > mask).any():
            break

        scale_index = find_significant_scale(
            resid_fdf_spectrum,
            kernels,
            options.scale_bias,
            rmsf_fwhm,
            phi_double_arr_radm2,
            options.kernel,
        )
        scale = float(scales[scale_index])
        gamma = float(kernels.gamma[scale_index])

        # Scale-convolved residual; sub-minor cleans in this space.
        resid_conv = np.asarray(
            convolve_fdf_scale(
                scale,
                rmsf_fwhm,
                resid_fdf_spectrum,
                phi_double_arr_radm2,
                options.kernel,
            ),
            dtype=np.complex128,
        )
        mask_conv = np.abs(resid_conv) > mask * gamma
        resid_conv_masked = np.ma.array(resid_conv, mask=~mask_conv)

        # Sub-minor cleans this scale only until its peak drops by
        # `sub_minor_fraction`, then re-select a scale. Stops one scale from
        # deep-cleaning to the floor and over-fitting.
        peak_conv = float(np.ma.max(np.ma.abs(resid_conv_masked)))
        threshold_sub = max(threshold * gamma, options.sub_minor_fraction * peak_conv)

        sub_minor = minor_loop(
            MinorLoopArrays(
                resid_fdf_spectrum_mask=resid_conv_masked,
                phi_arr_radm2=phi_arr_radm2,
                phi_double_arr_radm2=phi_double_arr_radm2,
                rmsf_spectrum=kernels.p_ss[scale_index] / gamma,
                rmsf_fwhm=float(kernels.fwhm_ss[scale_index]),
            ),
            MinorLoopOptions(
                max_iter=options.max_iter_sub_minor,
                gain=options.gain,
                mask=mask * gamma,
                threshold=threshold_sub,
                update_mask=True,
            ),
        )

        # Undo the R_s normalisation to recover true k_s-model amplitudes.
        true_deltas = sub_minor.model_fdf_spectrum / gamma
        if not np.any(true_deltas):
            break

        model_fdf_spectrum = model_fdf_spectrum + np.asarray(
            convolve_fdf_scale(
                scale, rmsf_fwhm, true_deltas, phi_double_arr_radm2, options.kernel
            ),
            dtype=np.complex128,
        )
        # Full-res residual update: dirty footprint = model (conv) RMSF.
        resid_fdf_spectrum = resid_fdf_spectrum - _reconvolve_model(
            true_deltas, kernels.p_s[scale_index], phi_arr_radm2, phi_double_arr_radm2
        )

    # Restore with a unit-peak clean beam, matching single-scale `minor_loop`
    # (so mom0 normalisation in `calc_faraday_moments` recovers component flux).
    clean_beam = unit_centred_gaussian(x=phi_double_arr_radm2, fwhm=rmsf_fwhm)
    clean_fdf_spectrum = np.asarray(
        convolve(model_fdf_spectrum.real, clean_beam, mode="reflect")
        + 1j * convolve(model_fdf_spectrum.imag, clean_beam, mode="reflect"),
        dtype=np.complex128,
    )
    return clean_fdf_spectrum, resid_fdf_spectrum, model_fdf_spectrum, iter_count


def multiscale_rmclean(
    freq_arr_hz: NDArray[np.float64],
    dirty_fdf_arr: NDArray[np.complex128],
    phi_arr_radm2: NDArray[np.float64],
    rmsf_arr: NDArray[np.complex128],
    phi_double_arr_radm2: NDArray[np.float64],
    fwhm_rmsf_arr: NDArray[np.float64],
    mask: float,
    threshold: float,
    multiscale_options: MultiscaleOptions | None = None,
    mask_arr: NDArray[np.bool_] | None = None,
) -> RMCleanResults:
    """Multiscale RM-CLEAN on a 1D/2D/3D FDF array.

    Args:
        freq_arr_hz: Frequencies (Hz), for the max recoverable scale.
        dirty_fdf_arr: Dirty FDF, phi on axis 0.
        phi_arr_radm2: Faraday depth array (rad/m^2).
        rmsf_arr: RMSF, double-length phi on axis 0.
        phi_double_arr_radm2: Double-length Faraday depth array (rad/m^2).
        fwhm_rmsf_arr: Per-pixel RMSF FWHM.
        mask: Masking threshold (FDF amplitude units).
        threshold: Cleaning threshold (FDF amplitude units).
        multiscale_options: Scale/kernel/iteration options.
        mask_arr: Optional spatial mask of pixels to clean.

    Returns:
        RMCleanResults: clean_fdf_arr, model_fdf_arr, clean_iter_arr, resid_fdf_arr.
    """
    options = multiscale_options or MultiscaleOptions()
    _bad_result = RMCleanResults(
        clean_fdf_arr=dirty_fdf_arr,
        model_fdf_arr=np.zeros_like(dirty_fdf_arr),
        clean_iter_arr=np.zeros_like(phi_arr_radm2),
        resid_fdf_arr=dirty_fdf_arr,
    )
    n_phi = phi_arr_radm2.shape[0]
    n_dimension = dirty_fdf_arr.ndim
    if n_phi != dirty_fdf_arr.shape[0]:
        logger.error("'phi_arr_radm2' and 'dirty_fdf_arr' are not the same length.")
        return _bad_result
    if phi_double_arr_radm2.shape[0] != rmsf_arr.shape[0]:
        logger.error("mismatch in 'phi_double_arr_radm2' and 'rmsf_arr' length.")
        return _bad_result
    if phi_double_arr_radm2.shape[0] < 2 * n_phi:
        logger.error("the Faraday depth of the RMSF must be twice the FDF.")
        return _bad_result
    if n_dimension > 3:
        logger.error("FDF array dimensions must be <= 3.")
        return _bad_result
    if n_dimension != rmsf_arr.ndim:
        logger.error("the input RMSF and FDF must have the same number of axes.")
        return _bad_result
    if rmsf_arr.shape[1:] != dirty_fdf_arr.shape[1:]:
        logger.error("the xy dimensions of the RMSF and FDF must match.")
        return _bad_result

    # Reshape everything to (phi, npix) so one loop covers 1D/2D/3D.
    dirty_2d = dirty_fdf_arr.reshape(n_phi, -1)
    rmsf_2d = rmsf_arr.reshape(rmsf_arr.shape[0], -1)
    fwhm_1d = np.asarray(fwhm_rmsf_arr, dtype=np.float64).reshape(-1)
    n_pixels = dirty_2d.shape[1]
    spatial_shape = dirty_fdf_arr.shape[1:]
    if mask_arr is None:
        pixel_mask = np.ones(n_pixels, dtype=bool)
    else:
        if mask_arr.shape != spatial_shape:
            logger.error("pixel mask must match xy dimension of FDF cube.")
            return _bad_result
        pixel_mask = mask_arr.reshape(-1)

    # Max recoverable Faraday scale sets the auto scale set.
    rmsf_params = compute_rmsf_params(
        freq_arr_hz=freq_arr_hz,
        weight_arr=np.ones_like(freq_arr_hz),
    )
    max_scale = rmsf_params.phi_max_scale / rmsf_params.rmsf_fwhm_meas
    logger.info(
        f"Maximum Faraday scale {rmsf_params.phi_max_scale:0.2f} rad/m^2, "
        f"{max_scale:0.2f} RMSF FWHM."
    )
    scales = (
        options.scales
        if options.scales is not None
        else make_scales(max_scale, n_scales=options.n_scales)
    )
    if scales.max() > max_scale:
        logger.warning(
            f"Maximum scale {scales.max():0.2f} exceeds the RMSF max scale "
            f"{max_scale:0.2f}."
        )
    logger.info(f"Using scales (RMSF FWHM units): {scales}")

    # Only clean pixels with signal above the mask.
    abs_fdf = np.abs(np.nan_to_num(dirty_2d))
    to_clean = np.where((abs_fdf.max(axis=0) >= mask) & pixel_mask)[0]
    logger.info(f"Cleaning {len(to_clean)}/{n_pixels} spectra.")

    clean_2d = np.zeros_like(dirty_2d)
    model_2d = np.zeros_like(dirty_2d)
    resid_2d = dirty_2d.copy()
    iter_2d = np.zeros(n_pixels, dtype=int)

    for pix in tqdm(
        to_clean,
        file=TQDM_OUT,
        leave=False,
        disable=not logger.isEnabledFor(logging.INFO),
    ):
        clean, resid, model, iters = multiscale_clean_spectrum(
            dirty_fdf_spectrum=dirty_2d[:, pix],
            phi_arr_radm2=phi_arr_radm2,
            phi_double_arr_radm2=phi_double_arr_radm2,
            rmsf_spectrum=rmsf_2d[:, pix],
            rmsf_fwhm=float(fwhm_1d[pix]),
            scales=scales,
            mask=mask,
            threshold=threshold,
            options=options,
        )
        clean_2d[:, pix] = clean
        resid_2d[:, pix] = resid
        model_2d[:, pix] = model
        iter_2d[pix] = iters

    # Restore residual into the clean FDF (also covers un-cleaned pixels).
    clean_2d += resid_2d

    return RMCleanResults(
        clean_fdf_arr=clean_2d.reshape(dirty_fdf_arr.shape),
        model_fdf_arr=model_2d.reshape(dirty_fdf_arr.shape),
        clean_iter_arr=iter_2d.reshape(spatial_shape),
        resid_fdf_arr=resid_2d.reshape(dirty_fdf_arr.shape),
    )
