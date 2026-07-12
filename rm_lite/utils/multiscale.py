"""Multiscale RM-CLEAN utils"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, NamedTuple, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_dilation, convolve

from rm_lite.utils.clean import (
    MinorLoopArrays,
    MinorLoopOptions,
    minor_loop,
    shift_rmsf,
)
from rm_lite.utils.fitting import fit_rmsf, fwhm_to_sigma, unit_centred_gaussian
from rm_lite.utils.logging import logger

if TYPE_CHECKING:
    from rm_lite.utils.clean import RMCleanOptions

KernelType: TypeAlias = Literal["tapered_quad", "gaussian"]

# Bail out of a spectrum's major loop if the residual peak exceeds this factor
# times the best (lowest) peak seen: a runaway-divergence backstop.
DIVERGENCE_FACTOR = 2.0


@dataclass(frozen=True, kw_only=True, slots=True)
class MultiscaleOptions:
    """Options for multiscale RM-CLEAN."""

    scale_bias: float = 0.8
    """Scale-bias in (0, 1]. Lower favours larger scales more strongly; too low
    over-extends unresolved sources. Default 0.8 keeps a point source on scale 0
    (WSClean uses 0.6, but its image-domain PSF is far narrower than the RMSF is
    here relative to the scale kernels)."""
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


def _restore(
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
    """Precomputed per-scale kernels/couplings for one spectrum."""

    scales: NDArray[np.float64]
    """Scales (RMSF FWHM units)"""
    p_s: list[NDArray[np.complex128]]
    """Scale-convolved RMSF per scale (RMSF conv k_s), on the double phi axis"""
    p_ss: list[NDArray[np.complex128]]
    """Twice scale-convolved RMSF per scale (RMSF conv k_s conv k_s)"""
    gamma: NDArray[np.float64]
    """Coupling max|p_ss| per scale: a scale-s amplitude appears in the
    scale-convolved residual scaled by this"""
    fwhm_ss: NDArray[np.float64]
    """Fitted FWHM of |p_ss| per scale (sub-minor restore width)"""


def compute_scale_kernels(
    scales: NDArray[np.float64],
    rmsf_spectrum: NDArray[np.complex128],
    rmsf_fwhm: float,
    phi_double_arr_radm2: NDArray[np.float64],
    kernel: KernelType,
) -> ScaleKernels:
    """Precompute the per-scale kernels/couplings for one spectrum."""
    p_s_list: list[NDArray[np.complex128]] = []
    p_ss_list: list[NDArray[np.complex128]] = []
    gamma = np.ones_like(scales)
    fwhm_ss = np.full_like(scales, rmsf_fwhm)
    for i, scale in enumerate(scales):
        p_s = convolve_fdf_scale(
            scale, rmsf_fwhm, rmsf_spectrum, phi_double_arr_radm2, kernel
        )
        p_ss = convolve_fdf_scale(scale, rmsf_fwhm, p_s, phi_double_arr_radm2, kernel)
        p_s_list.append(np.asarray(p_s, dtype=np.complex128))
        p_ss_list.append(np.asarray(p_ss, dtype=np.complex128))
        gamma[i] = float(np.nanmax(np.abs(p_ss)))
        if scale != 0:
            fwhm_ss[i] = fit_rmsf(
                np.abs(p_ss),
                phi_double_arr_radm2=phi_double_arr_radm2,
                fwhm_rmsf_radm2=rmsf_fwhm * scale,
            )
    return ScaleKernels(scales, p_s_list, p_ss_list, gamma, fwhm_ss)


def find_significant_scale(
    resid_fdf_spectrum: NDArray[np.complex128],
    scale_kernels: ScaleKernels,
    scale_bias: float,
    rmsf_fwhm: float,
    phi_double_arr_radm2: NDArray[np.float64],
    kernel: KernelType,
) -> int:
    """Index of the bias-weighted most-significant scale (Offringa 2017).

    Selection is `max|resid (conv) k_s| / eta_s * bias_s`. Dividing by the
    point-source response `eta_s = max|RMSF (conv) k_s|` makes an unresolved
    (delta) source score equally across scales, so scale 0 wins for it and
    multiscale does not over-extend a point source; only genuinely extended
    structure beats scale 0.
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

    # Clean within the mask-seeded support but down to `threshold` (single-scale
    # RM-CLEAN reaches the same depth via its deep-clean phase). Dilate the
    # support by ~1 RMSF FWHM so extended structure can spill past the >mask edge.
    d_phi = float(phi_double_arr_radm2[1] - phi_double_arr_radm2[0])
    support = np.abs(dirty_fdf_spectrum) > mask
    support = binary_dilation(support, iterations=max(1, int(rmsf_fwhm / d_phi)))

    iter_count = 0
    if not support.any():
        return (
            _restore(model_fdf_spectrum, phi_double_arr_radm2, rmsf_fwhm),
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
        gamma = float(kernels.gamma[scale_index])

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
        mask_conv = (np.abs(resid_conv) > threshold * gamma) & support
        if not mask_conv.any():
            break
        resid_conv_masked = np.ma.array(resid_conv, mask=~mask_conv)

        # Sub-minor cleans this scale only until its peak drops by
        # `sub_minor_fraction`, then re-select a scale. Stops one scale from
        # deep-cleaning to the floor and over-fitting.
        peak_conv = float(np.ma.max(np.ma.abs(resid_conv_masked)))
        threshold_sub = max(
            threshold * gamma, multiscale_options.sub_minor_fraction * peak_conv
        )

        sub_minor = minor_loop(
            MinorLoopArrays(
                resid_fdf_spectrum_mask=resid_conv_masked,
                phi_arr_radm2=phi_arr_radm2,
                phi_double_arr_radm2=phi_double_arr_radm2,
                rmsf_spectrum=kernels.p_ss[scale_index] / gamma,
                rmsf_fwhm=float(kernels.fwhm_ss[scale_index]),
            ),
            MinorLoopOptions(
                max_iter=multiscale_options.max_iter_sub_minor,
                gain=clean_options.gain,
                mask=threshold * gamma,
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
                scale, rmsf_fwhm, true_deltas, phi_double_arr_radm2, kernel
            ),
            dtype=np.complex128,
        )
        # Full-res residual update: dirty footprint = model (conv) RMSF.
        resid_fdf_spectrum = resid_fdf_spectrum - _reconvolve_model(
            true_deltas, kernels.p_s[scale_index], phi_arr_radm2, phi_double_arr_radm2
        )

    clean_fdf_spectrum = _restore(model_fdf_spectrum, phi_double_arr_radm2, rmsf_fwhm)
    return clean_fdf_spectrum, resid_fdf_spectrum, model_fdf_spectrum, iter_count


def default_scales(
    phi_arr_radm2: NDArray[np.float64],
    rmsf_fwhm: float,
    multiscale_options: MultiscaleOptions,
) -> NDArray[np.float64]:
    """Scales (RMSF FWHM units): explicit if given, else WSClean-style auto.

    The auto max scale is set from the FDF phi window (freq-free) so the largest
    kernel still fits. The physical limit pi/lambda_sq_min needs frequencies;
    this geometric cap plus the scale-bias weighting and divergence guard handle
    over-large scales, and an explicit `scales` overrides it entirely.
    """
    if multiscale_options.scales is not None:
        return multiscale_options.scales
    max_scale = float(phi_arr_radm2.max() - phi_arr_radm2.min()) / (2 * rmsf_fwhm)
    return make_scales(max_scale, n_scales=multiscale_options.n_scales)
