"""RM-synthesis utils"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, TypeVar, cast

import finufft
import numpy as np
import polars as pl
from astropy.constants import c as speed_of_light
from astropy.stats import mad_std
from numpy.typing import NDArray
from scipy import ndimage
from tqdm.auto import trange

from rm_lite.utils.arrays import arange, nd_to_two_d, two_d_to_nd
from rm_lite.utils.fitting import (
    FitResult,
    StokesIFitOptions,
    fit_fdf,
    fit_rmsf,
    fit_stokes_i_model,
    gaussian_integrand,
    sample_model_error,
)
from rm_lite.utils.logging import logger

# Ricean polarisation-bias correction: debiased P = sqrt(P^2 - factor * sigma^2)
# (POSSUM report 11).
POLARISATION_BIAS_FACTOR = 2.3


class FWHM(NamedTuple):
    fwhm_rmsf_radm2: float
    """The FWHM of the RMSF main lobe"""
    d_lambda_sq_max_m2: float
    """The maximum difference in lambda^2 values"""
    lambda_sq_range_m2: float
    """The range of lambda^2 values"""


class RMsynthResults(NamedTuple):
    """Results of the RM-synthesis calculation"""

    fdf_dirty_cube: NDArray[np.float64]
    """The Faraday dispersion function cube"""
    lam_sq_0_m2: float
    """The reference lambda^2 value"""


class RMSFResults(NamedTuple):
    """Results of the RMSF calculation"""

    rmsf_cube: NDArray[np.float64]
    """The RMSF cube"""
    phi_double_arr_radm2: NDArray[np.float64]
    """The (double length) Faraday depth array"""
    fwhm_rmsf_arr: NDArray[np.float64]
    """The FWHM of the RMSF main lobe"""
    fit_status_arr: NDArray[np.float64]
    """The status of the RMSF fit"""


class StokesData(NamedTuple):
    """Stokes parameters and errors"""

    complex_pol_arr: NDArray[np.complex128]
    """ Stokes Q and U array """
    complex_pol_error: NDArray[np.complex128]
    """ Stokes Q and U error array """
    freq_arr_hz: NDArray[np.float64]
    """ Frequency array in Hz """
    stokes_i_arr: NDArray[np.float64] | None = None
    """ Stokes I array """
    stokes_i_error_arr: NDArray[np.float64] | None = None
    """ Stokes I error array """
    stokes_i_model_arr: NDArray[np.float64] | None = None
    """ Stokes I model array """
    stokes_i_model_error: NDArray[np.float64] | None = None
    """ Stokes I model error array """


class FractionalSpectra(NamedTuple):
    stokes_data: StokesData
    fit_result: FitResult | None
    no_nan_idx: NDArray[np.bool_]


class TheoreticalNoise(NamedTuple):
    """Theoretical noise of the FDF"""

    fdf_error_noise: float
    """Theoretical noise of the FDF"""
    fdf_q_noise: float
    """Theoretical noise of the real FDF"""
    fdf_u_noise: float
    """Theoretical noise of the imaginary FDF"""


WeightType = Literal["variance", "natural", "uniform", "uniform_lsq", "briggs"]
""" RM-synthesis weighting: `variance`/`natural` (1/sigma^2, equivalent),
`uniform` (equal per channel), `uniform_lsq` (equal per lambda^2 interval,
narrows the RMSF), `briggs` (robust interpolation between natural and
uniform_lsq, needs `robust`). """
WEIGHT_TYPES: tuple[str, ...] = (
    "variance",
    "natural",
    "uniform",
    "uniform_lsq",
    "briggs",
)


@dataclass(frozen=True, kw_only=True, slots=True)
class FDFOptions:
    """Options for RM-synthesis, shared by the 1D and 3D tools"""

    phi_max_radm2: float | None = None
    """ Maximum Faraday depth """
    d_phi_radm2: float | None = None
    """ Faraday depth resolution """
    n_samples: float | None = 10.0
    """ Number of samples """
    weight_type: WeightType = "variance"
    """ Weight type """
    robust: float | None = None
    """ Briggs robust parameter (required for weight_type='briggs') """
    do_fit_rmsf: bool = False
    """ Fit RMSF """
    do_fit_rmsf_real: bool = False
    """ Fit real part of the RMSF """

    def __post_init__(self) -> None:
        if self.weight_type not in WEIGHT_TYPES:
            msg = (
                f"weight_type must be one of {WEIGHT_TYPES}, got {self.weight_type!r}."
            )
            raise ValueError(msg)
        if self.weight_type == "briggs" and self.robust is None:
            msg = "weight_type='briggs' requires a `robust` parameter."
            raise ValueError(msg)
        if self.d_phi_radm2 is None and self.n_samples is None:
            msg = "Either d_phi_radm2 or n_samples must be provided."
            raise ValueError(msg)
        for name in ("phi_max_radm2", "d_phi_radm2", "n_samples"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                msg = f"{name} must be positive, got {value}."
                raise ValueError(msg)


def calc_mom2_fdf(
    complex_fdf_arr: NDArray[np.complex128], phi_arr_radm2: NDArray[np.float64]
) -> float:
    """
    Calculate the 2nd moment of the polarised intensity FDF. Can be applied to
    a clean component spectrum or a standard FDF
    """

    phi_weights = np.sum(np.abs(complex_fdf_arr))
    phi_mean = np.sum(phi_arr_radm2 * np.abs(complex_fdf_arr)) / phi_weights
    return float(
        np.sqrt(
            np.sum(np.power((phi_arr_radm2 - phi_mean), 2.0) * np.abs(complex_fdf_arr))
            / phi_weights
        )
    )


class FaradayMoments(NamedTuple):
    """Moments of the polarised intensity Faraday depth spectrum."""

    mom0: NDArray[np.float64]
    """Zeroth moment: total polarised intensity, in the input FDF amplitude units"""
    mom1: NDArray[np.float64]
    """First moment: intensity-weighted mean Faraday depth in rad/m^2"""
    mom2: NDArray[np.float64]
    """Second moment: intensity-weighted Faraday depth dispersion in rad/m^2"""


def _require_single_chunk_on_axis(arr: Any, axis: int, reason: str) -> None:
    """Raise if `arr` is a dask array with more than one chunk along `axis`.

    A median reduction across the Faraday depth axis is not supported by dask.
    """
    if hasattr(arr, "chunks") and len(arr.chunks[axis]) != 1:
        msg = (
            f"The Faraday depth axis must be a single chunk {reason}. "
            f"Rechunk with e.g. `.rechunk({{{axis}: -1}})`."
        )
        raise ValueError(msg)


def calc_faraday_moments(
    complex_fdf_arr: NDArray[np.complex128] | NDArray[np.float64],
    phi_arr_radm2: NDArray[np.float64],
    fwhm_rmsf_radm2: float | NDArray[np.float64],
    axis: int = 0,
    threshold: float | None = None,
    auto_threshold_sigma: float | None = None,
    debias: bool = False,
    lam_sq_0_m2: float | None = None,
    debias_filter_size: int = 5,
    min_weight_fraction: float | None = None,
) -> FaradayMoments:
    """Compute the zeroth, first, and second moments of a Faraday depth spectrum.

    The FDF amplitude is in units per RMSF (the native RM-synthesis scale). mom0
    is converted to integrated units by dividing the Faraday-depth sum by the
    RMSF area (a Gaussian of FWHM `fwhm_rmsf_radm2`), so an unresolved component
    of peak amplitude P gives `mom0 = P`.

    Complex input is reduced with `np.abs`; real input is used as-is, so the
    signed debiased amplitudes from `debias_fdf` integrate without folding noise
    into a positive floor. `debias=True` applies that debiasing internally (needs
    `lam_sq_0_m2` and a spatial axis), giving unbiased moments with no threshold.

    Works on numpy or dask arrays of any dimensionality: the Faraday depth axis
    is reduced away, the rest preserved. `auto_threshold_sigma` and `debias=True`
    reduce over the Faraday depth axis, so for dask that axis must be one chunk.

    Args:
        complex_fdf_arr (NDArray[np.complex128]): Complex (or real) FDF.
        phi_arr_radm2 (NDArray[np.float64]): Uniformly spaced Faraday depth array in rad/m^2.
        fwhm_rmsf_radm2 (float | NDArray[np.float64]): FWHM of the RMSF main lobe in rad/m^2.
            An array must broadcast against the FDF shape with the Faraday depth axis removed.
        axis (int, optional): Faraday depth axis of `complex_fdf_arr`. Defaults to 0.
        threshold (float | None, optional): Exclude amplitudes below this value
            (in FDF amplitude units). Not supported with `debias=True`. Defaults to None.
        auto_threshold_sigma (float | None, optional): Exclude amplitudes below this
            multiple of the per-spectrum noise (a robust `mad_std` of the real and
            imaginary parts). Mutually exclusive with `threshold`, and not supported
            with `debias=True`. Defaults to None.
        debias (bool, optional): Debias the FDF amplitudes with `debias_fdf`
            before computing the moments. Requires complex input with a spatial
            axis, and `lam_sq_0_m2`. Defaults to False.
        lam_sq_0_m2 (float | None, optional): Reference wavelength^2 of the
            RM-synthesis derotation, passed to `debias_fdf`. Required when
            `debias=True`. Defaults to None.
        debias_filter_size (int, optional): Spatial median filter size passed
            to `debias_fdf`. Defaults to 5.
        min_weight_fraction (float | None, optional): Opt-in guard for signed
            input. When set, mom1/mom2 are NaN wherever the net weight
            `|sum(amplitude)|` is below this fraction of the total absolute
            weight `sum(|amplitude|)`, so near-cancelling noise spectra do not
            yield spurious finite Faraday depths. mom0 is unaffected. Off by
            default (irreversible masking); a mom0 detection cut is the
            alternative. Defaults to None.

    Returns:
        FaradayMoments: mom0 (FDF amplitude units), mom1 (rad/m^2), and mom2
            (dispersion, rad/m^2). Spectra with no valid amplitude have
            mom0 = 0 and mom1 = mom2 = NaN.
    """
    if threshold is not None and auto_threshold_sigma is not None:
        msg = "`threshold` and `auto_threshold_sigma` are mutually exclusive."
        raise ValueError(msg)

    phi_arr_radm2 = np.asarray(phi_arr_radm2, dtype=np.float64)
    if phi_arr_radm2.ndim != 1 or phi_arr_radm2.shape[0] < 2:
        msg = "`phi_arr_radm2` must be 1D with at least two samples."
        raise ValueError(msg)
    if complex_fdf_arr.shape[axis] != phi_arr_radm2.shape[0]:
        msg = (
            f"Axis {axis} of the FDF has length {complex_fdf_arr.shape[axis]}, "
            f"but `phi_arr_radm2` has length {phi_arr_radm2.shape[0]}."
        )
        raise ValueError(msg)

    if debias:
        if auto_threshold_sigma is not None:
            msg = "`auto_threshold_sigma` is not supported with `debias=True`."
            raise ValueError(msg)
        if threshold is not None:
            msg = (
                "`threshold` is not supported with `debias=True`: a positive cut "
                "on signed debiased amplitudes clips the negative noise samples "
                "that make the bias cancel."
            )
            raise ValueError(msg)
        if lam_sq_0_m2 is None:
            msg = "`lam_sq_0_m2` is required when `debias=True`."
            raise ValueError(msg)
        abs_fdf_arr = debias_fdf(
            cast("NDArray[np.complex128]", complex_fdf_arr),
            phi_arr_radm2=phi_arr_radm2,
            lam_sq_0_m2=lam_sq_0_m2,
            axis=axis,
            filter_size=debias_filter_size,
        )
    elif np.iscomplexobj(complex_fdf_arr):
        abs_fdf_arr = np.abs(complex_fdf_arr)
    else:
        # Real input is taken as-is: signed debiased amplitudes must keep
        # their negative noise samples so the bias cancels in the sums
        abs_fdf_arr = cast("NDArray[np.float64]", complex_fdf_arr)

    if auto_threshold_sigma is not None:
        _require_single_chunk_on_axis(
            complex_fdf_arr, axis, "for the auto-threshold noise estimate"
        )
        # Per-spectrum noise from a robust MAD estimate of the zero-mean real
        # and imaginary components: signal-robust (tolerates <50% occupancy)
        # and the same estimator the rest of the module uses. The median of
        # |FDF| assumed pure-Rayleigh noise and biased high once signal filled
        # many channels.
        if np.iscomplexobj(complex_fdf_arr):
            components = np.concatenate(
                [complex_fdf_arr.real, complex_fdf_arr.imag], axis=axis
            )
        else:
            components = abs_fdf_arr
        noise = np.expand_dims(mad_std(components, axis=axis, ignore_nan=True), axis)
        abs_fdf_arr = np.where(
            abs_fdf_arr >= auto_threshold_sigma * noise, abs_fdf_arr, np.nan
        )
    elif threshold is not None:
        abs_fdf_arr = np.where(abs_fdf_arr >= threshold, abs_fdf_arr, np.nan)

    phi_shape = [1] * complex_fdf_arr.ndim
    phi_shape[axis] = phi_arr_radm2.shape[0]
    phi_nd = phi_arr_radm2.reshape(phi_shape)
    delta_phi = float(np.abs(phi_arr_radm2[1] - phi_arr_radm2[0]))

    weight_sum = np.nansum(abs_fdf_arr, axis=axis, keepdims=True)
    if min_weight_fraction is not None:
        # Signed input can sum to a tiny (positive or negative) net weight in
        # noise regions, giving a finite but meaningless mom1/mom2. Mask where
        # the net weight is a small fraction of the total absolute weight, so
        # near-cancelling spectra become NaN symmetrically. No-op for |FDF|
        # input, whose weights are all positive (ratio == 1).
        total_abs_weight = np.nansum(np.abs(abs_fdf_arr), axis=axis, keepdims=True)
        safe_weight_sum = np.where(
            np.abs(weight_sum) >= min_weight_fraction * total_abs_weight,
            weight_sum,
            np.nan,
        )
    else:
        safe_weight_sum = np.where(weight_sum > 0, weight_sum, np.nan)
    mom1 = np.nansum(abs_fdf_arr * phi_nd, axis=axis, keepdims=True) / safe_weight_sum
    # Signed (debiased) amplitudes can produce a negative variance in noise
    # regions; map it to NaN rather than warn
    mom2_variance = (
        np.nansum(abs_fdf_arr * (phi_nd - mom1) ** 2, axis=axis, keepdims=True)
        / safe_weight_sum
    )
    mom2 = np.sqrt(np.where(mom2_variance >= 0, mom2_variance, np.nan))
    rmsf_area = fwhm_rmsf_radm2 * gaussian_integrand(amplitude=1.0, fwhm=1.0)
    mom0 = np.squeeze(weight_sum, axis=axis) * delta_phi / rmsf_area

    return FaradayMoments(
        mom0=mom0,
        mom1=np.squeeze(mom1, axis=axis),
        mom2=np.squeeze(mom2, axis=axis),
    )


def _debias_fdf_block(
    complex_fdf_arr: NDArray[np.complex128],
    phi_arr_radm2: NDArray[np.float64],
    lam_sq_0_m2: float,
    axis: int,
    filter_size: int,
) -> NDArray[np.float64]:
    n_phi = phi_arr_radm2.shape[0]
    phi_shape = [1] * complex_fdf_arr.ndim
    phi_shape[axis] = n_phi
    phi_nd = phi_arr_radm2.reshape(phi_shape)

    # Derotate the deterministic 2*lam_sq_0*(RM - phi) angle ramp away (see
    # `debias_fdf` docstring) using a per-pixel peak Faraday depth: the
    # amplitude-weighted centroid of the half-max region about the peak, which
    # is less noise-prone than a 3-point parabola on an oversampled RMSF.
    abs_fdf_arr = np.abs(complex_fdf_arr)
    abs_fdf_arr = np.where(np.isfinite(abs_fdf_arr), abs_fdf_arr, 0.0)
    peak_amp = np.max(abs_fdf_arr, axis=axis, keepdims=True)
    lobe_weight = np.where(
        abs_fdf_arr >= 0.5 * peak_amp, abs_fdf_arr - 0.5 * peak_amp, 0.0
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        peak_rm_radm2 = np.where(
            peak_amp > 0,
            np.sum(lobe_weight * phi_nd, axis=axis, keepdims=True)
            / np.sum(lobe_weight, axis=axis, keepdims=True),
            0.0,
        )
    derotated = complex_fdf_arr * np.exp(-2j * lam_sq_0_m2 * (peak_rm_radm2 - phi_nd))

    theta = np.arctan2(derotated.imag, derotated.real)

    # Median-filter the angle via its cos/sin components to avoid the
    # -pi/pi discontinuity (Mueller et al. 2017, Sect. 2)
    footprint_shape = [1] * complex_fdf_arr.ndim
    for dim in range(complex_fdf_arr.ndim):
        if dim != axis:
            footprint_shape[dim] = filter_size
    footprint = np.ones(footprint_shape, dtype=bool)
    footprint_modified = footprint.copy()
    centre = tuple(size // 2 for size in footprint_shape)
    footprint_modified[centre] = False

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    # 'Modified median filter': the centre pixel carries the very bias being
    # corrected, so blend the plain median with a centre-excluded median
    # (1:2 weighting, the empirical choice of Mueller et al. 2017)
    cos_filt = (
        ndimage.median_filter(cos_theta, footprint=footprint)
        + 2.0 * ndimage.median_filter(cos_theta, footprint=footprint_modified)
    ) / 3.0
    sin_filt = (
        ndimage.median_filter(sin_theta, footprint=footprint)
        + 2.0 * ndimage.median_filter(sin_theta, footprint=footprint_modified)
    ) / 3.0
    theta_filt = np.arctan2(sin_filt, cos_filt)

    # Project Q + iU onto the filtered-angle direction (Mueller et al. 2017,
    # Eq. 9); |derotated| == |fdf|, so this is the debiased FDF amplitude
    return cast(
        "NDArray[np.float64]",
        derotated.imag * np.sin(theta_filt) + derotated.real * np.cos(theta_filt),
    )


def debias_fdf(
    complex_fdf_arr: NDArray[np.complex128],
    phi_arr_radm2: NDArray[np.float64],
    lam_sq_0_m2: float,
    axis: int = 0,
    filter_size: int = 5,
) -> NDArray[np.float64]:
    """Compute debiased polarised intensity amplitudes from a complex FDF cube.

    Implements the polarisation de-biasing of Mueller, Beck & Krause (2017,
    A&A 600, A63), adapted to Faraday depth cubes: the polarisation angle in
    each Faraday depth plane is median-filtered over the spatial axes (via
    its cos/sin components, dodging the angle wrap), and the observed Q + iU
    is projected onto the filtered-angle direction,
    `P* = U sin(theta_m) + Q cos(theta_m)`. Unlike `abs(fdf)`, the result is
    noise-like (zero-mean Gaussian) in signal-free regions, at the cost of
    allowing negative values. Summed over Faraday depth (e.g. by
    `calc_faraday_moments`) the noise cancels instead of accumulating a
    positive floor.

    The Mueller et al. method assumes the angle is smooth across the filter
    box, which an FDF plane violates wherever RM varies: its angle is
    `2 psi0 + 2 lam_sq_0 (RM - phi)`. That deterministic ramp is removed
    first, by derotating each spectrum with a per-pixel peak Faraday depth
    estimate (the amplitude-weighted centroid of the half-max region about
    the peak), so only the intrinsic angle `2 psi0`, assumed spatially
    smooth, is filtered.
    Sightlines with multiple components at very different Faraday depths are
    only derotated for the dominant peak; secondary components lose
    `cos(2 lam_sq_0 dRM)` of amplitude in the projection.

    Works on numpy or dask arrays (dask via `map_overlap` with a
    `filter_size // 2` spatial halo; the Faraday depth axis must be a single
    chunk, as produced by `rm_lite.tools_3d`).

    Args:
        complex_fdf_arr (NDArray[np.complex128]): Complex FDF with at least
            one spatial axis (2D or 3D).
        phi_arr_radm2 (NDArray[np.float64]): Uniformly spaced Faraday depth
            array in rad/m^2.
        lam_sq_0_m2 (float): Reference wavelength^2 of the RM-synthesis
            derotation (e.g. `RMSynth3DResults.lam_sq_0_m2`). Pass 0 to skip
            the RM derotation (the original Mueller et al. method, valid only
            for spatially smooth RM).
        axis (int, optional): Faraday depth axis, excluded from the spatial
            median filter (each Faraday depth plane is filtered
            independently). Defaults to 0.
        filter_size (int, optional): Odd spatial median filter box size in
            pixels. Defaults to 5.

    Returns:
        NDArray[np.float64]: Debiased polarised intensity, same shape as the input.
    """
    if filter_size < 3 or filter_size % 2 == 0:
        msg = f"`filter_size` must be an odd integer >= 3. Got {filter_size}."
        raise ValueError(msg)
    if complex_fdf_arr.ndim < 2:
        msg = "`complex_fdf_arr` must have at least one spatial axis to filter over."
        raise ValueError(msg)

    axis = axis % complex_fdf_arr.ndim
    phi_arr_radm2 = np.asarray(phi_arr_radm2, dtype=np.float64)
    if phi_arr_radm2.ndim != 1 or phi_arr_radm2.shape[0] < 2:
        msg = "`phi_arr_radm2` must be 1D with at least two samples."
        raise ValueError(msg)
    if complex_fdf_arr.shape[axis] != phi_arr_radm2.shape[0]:
        msg = (
            f"Axis {axis} of the FDF has length {complex_fdf_arr.shape[axis]}, "
            f"but `phi_arr_radm2` has length {phi_arr_radm2.shape[0]}."
        )
        raise ValueError(msg)

    if hasattr(complex_fdf_arr, "map_overlap"):  # dask array
        if len(complex_fdf_arr.chunks[axis]) != 1:  # type: ignore[attr-defined]
            msg = (
                "The Faraday depth axis must be a single chunk for the "
                "per-pixel peak derotation. Rechunk with e.g. "
                f"`.rechunk({{{axis}: -1}})`."
            )
            raise ValueError(msg)
        halo = filter_size // 2
        depth = {dim: 0 if dim == axis else halo for dim in range(complex_fdf_arr.ndim)}
        return cast(
            "NDArray[np.float64]",
            complex_fdf_arr.map_overlap(
                _debias_fdf_block,
                depth=depth,
                boundary="reflect",
                dtype=np.float64,
                phi_arr_radm2=phi_arr_radm2,
                lam_sq_0_m2=lam_sq_0_m2,
                axis=axis,
                filter_size=filter_size,
            ),
        )

    return _debias_fdf_block(
        complex_fdf_arr,
        phi_arr_radm2=phi_arr_radm2,
        lam_sq_0_m2=lam_sq_0_m2,
        axis=axis,
        filter_size=filter_size,
    )


def get_mask_index(
    stokes_data: StokesData,
) -> NDArray[np.bool_]:
    return (
        np.isfinite(stokes_data.complex_pol_arr)
        & np.isfinite(stokes_data.complex_pol_error)
        & np.isfinite(stokes_data.freq_arr_hz)
    )


def _fractional_with_error(
    num: NDArray[np.float64],
    num_err: NDArray[np.float64],
    den: NDArray[np.float64],
    den_err: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Elementwise num/den with independent-error propagation.

    Same closed form `uncertainties` uses, but in numpy so a degenerate model
    (near-zero denominator, huge covariance) overflows to inf/nan instead of
    raising OverflowError the way python floats do.
    """
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        frac = num / den
        err = np.sqrt((num_err / den) ** 2 + (num * den_err / den**2) ** 2)
    return frac, err


def create_fractional_spectra(
    stokes_data: StokesData,
    ref_freq_hz: float,
    fit_options: StokesIFitOptions,
) -> FractionalSpectra | None:
    no_nan_idx = get_mask_index(stokes_data)

    if (~no_nan_idx).all():
        msg = "All channels have been masked! No fractional polarization will be calculated."
        logger.warning(msg)
        return None

    # If a model is provided, use that to calculate the fractional spectra
    if stokes_data.stokes_i_model_arr is not None:
        logger.info("Using provided Stokes I model to calculate fractional spectra.")
        if stokes_data.stokes_i_model_error is None:
            msg = "If `stokes_i_model_arr` is provided, `stokes_i_model_error` must also be provided."
            raise ValueError(msg)

        stokes_q_frac_arr, stokes_q_frac_error_arr = _fractional_with_error(
            stokes_data.complex_pol_arr.real,
            stokes_data.complex_pol_error.real,
            stokes_data.stokes_i_model_arr,
            stokes_data.stokes_i_model_error,
        )
        stokes_u_frac_arr, stokes_u_frac_error_arr = _fractional_with_error(
            stokes_data.complex_pol_arr.imag,
            stokes_data.complex_pol_error.imag,
            stokes_data.stokes_i_model_arr,
            stokes_data.stokes_i_model_error,
        )

        stokes_qu_frac_arr = stokes_q_frac_arr + 1j * stokes_u_frac_arr
        stokes_qu_frac_error_arr = (
            stokes_q_frac_error_arr + 1j * stokes_u_frac_error_arr
        )

        fractional_stokes_data = stokes_data._replace(
            complex_pol_arr=stokes_qu_frac_arr.astype(np.complex128),
            complex_pol_error=stokes_qu_frac_error_arr.astype(np.complex128),
        )
        return FractionalSpectra(
            stokes_data=fractional_stokes_data,
            fit_result=None,
            no_nan_idx=no_nan_idx,
        )

    logger.info("Fitting Stokes I model to calculate fractional spectra.")
    if stokes_data.stokes_i_arr is None or stokes_data.stokes_i_error_arr is None:
        msg = "If `stokes_i_model_arr` is not provided, `stokes_i_arr` and `stokes_i_error_arr` must also be provided."
        raise ValueError(msg)

    # Flag out NaNs
    no_nan_idx = (
        no_nan_idx
        & np.isfinite(stokes_data.stokes_i_arr)
        & np.isfinite(stokes_data.stokes_i_error_arr)
    )
    logger.debug(f"{ref_freq_hz=}")

    if (~no_nan_idx).all():
        msg = "All channels have been masked!"
        raise ValueError(msg)

    # Apply flagging here since fitting will fail if NaNs are present
    fit_result = fit_stokes_i_model(
        freq_arr_hz=stokes_data.freq_arr_hz[no_nan_idx],
        ref_freq_hz=ref_freq_hz,
        stokes_i_arr=stokes_data.stokes_i_arr[no_nan_idx],
        stokes_i_error_arr=stokes_data.stokes_i_error_arr[no_nan_idx],
        options=fit_options,
    )
    if fit_result is None:
        msg = "Too few finite Stokes I channels to fit; no fractional polarization."
        logger.warning(msg)
        return None

    stokes_i_model_arr, stokes_i_model_error = sample_model_error(
        fit_result, stokes_data.freq_arr_hz / ref_freq_hz, fit_options.n_error_samples
    )
    stokes_q_frac_arr, stokes_q_frac_error_arr = _fractional_with_error(
        stokes_data.complex_pol_arr.real,
        stokes_data.complex_pol_error.real,
        stokes_i_model_arr,
        stokes_i_model_error,
    )
    stokes_u_frac_arr, stokes_u_frac_error_arr = _fractional_with_error(
        stokes_data.complex_pol_arr.imag,
        stokes_data.complex_pol_error.imag,
        stokes_i_model_arr,
        stokes_i_model_error,
    )

    assert len(stokes_data.stokes_i_arr) == len(stokes_q_frac_arr)
    assert len(stokes_data.stokes_i_arr) == len(stokes_u_frac_arr)
    assert len(stokes_data.stokes_i_arr) == len(stokes_q_frac_error_arr)
    assert len(stokes_data.stokes_i_arr) == len(stokes_u_frac_error_arr)

    complex_pol_arr = stokes_q_frac_arr + 1j * stokes_u_frac_arr
    complex_pol_error = stokes_q_frac_error_arr + 1j * stokes_u_frac_error_arr

    fractional_stokes_data = stokes_data._replace(
        complex_pol_arr=complex_pol_arr,
        complex_pol_error=complex_pol_error,
        stokes_i_model_arr=stokes_i_model_arr,
        stokes_i_model_error=stokes_i_model_error,
    )

    return FractionalSpectra(
        stokes_data=fractional_stokes_data,
        fit_result=fit_result,
        no_nan_idx=no_nan_idx,
    )


T = TypeVar("T", float, NDArray[np.float64])


def freq_to_lambda2(
    freq_hz: T,
) -> T:
    """Convert frequency to lambda^2.

    Args:
        freq_hz (float): Frequency in Hz

    Returns:
        float: Wavelength^2 in m^2
    """
    speed_of_light_m_s = float(speed_of_light.value)
    return (speed_of_light_m_s / freq_hz) ** 2.0  # type: ignore[no-any-return]


def lambda2_to_freq(lambda_sq_m2: T) -> T:
    """Convert lambda^2 to frequency.

    Args:
        lambda_sq_m2 (NDArray[np.float64]): Wavelength^2 in m^2

    Returns:
        NDArray[np.float64]: Frequency in Hz
    """
    speed_of_light_m_s = float(speed_of_light.value)
    return speed_of_light_m_s / np.sqrt(lambda_sq_m2)  # type: ignore[no-any-return]


def compute_theoretical_noise(
    complex_pol_error: NDArray[np.complex128],
    weight_arr: NDArray[np.float64],
) -> TheoreticalNoise:
    weight_arr = np.nan_to_num(weight_arr, nan=0.0, posinf=0.0, neginf=0.0)
    complex_pol_error_flagged = np.nan_to_num(
        complex_pol_error, nan=0.0, posinf=0.0, neginf=0.0
    )
    fdf_complex_noise = np.sqrt(
        np.nansum(weight_arr**2 * complex_pol_error_flagged**2)
        / (np.sum(weight_arr)) ** 2
    )

    fdf_error_noise = (fdf_complex_noise.real + fdf_complex_noise.imag) / 2
    return TheoreticalNoise(
        fdf_error_noise=fdf_error_noise,
        fdf_q_noise=fdf_complex_noise.real,
        fdf_u_noise=fdf_complex_noise.imag,
    )


class RMSynthParams(NamedTuple):
    """Parameters for RM-synthesis calculation"""

    lambda_sq_arr_m2: NDArray[np.float64]
    """ Wavelength^2 values in m^2 """
    lam_sq_0_m2: float
    """ Reference wavelength^2 value """
    phi_arr_radm2: NDArray[np.float64]
    """ Faraday depth values in rad/m^2 """
    weight_arr: NDArray[np.float64]
    """ Weight array """


class SigmaAdd(NamedTuple):
    """Sigma_add complexity metrics"""

    sigma_add: float
    """Sigma_add median value"""
    sigma_add_plus: float
    """Sigma_add upper quartile"""
    sigma_add_minus: float
    """Sigma_add lower quartile"""
    sigma_add_arrays: SigmaAddArrays
    """Sigma_add arrays"""


class StokesSigmaAdd(NamedTuple):
    """Stokes Sigma_add complexity metrics"""

    sigma_add_q: SigmaAdd
    """Sigma_add for Stokes Q"""
    sigma_add_u: SigmaAdd
    """Sigma_add for Stokes U"""
    sigma_add_p: SigmaAdd
    """Sigma_add for polarised intensity"""


class RMSFParams(NamedTuple):
    """RM spread function parameters"""

    rmsf_fwhm_theory: float
    """ Theoretical FWHM of the RMSF """
    rmsf_fwhm_meas: float
    """ Measured FWHM of the RMSF """
    phi_max: float
    """ Maximum Faraday depth """
    phi_max_scale: float
    """ Maximum Faraday depth scale """
    rmsf_results: RMSFResults
    """ Empirical RMSF """


def compute_rmsf_params(
    freq_arr_hz: NDArray[np.float64],
    weight_arr: NDArray[np.float64],
) -> RMSFParams:
    lambda_sq_arr_m2 = freq_to_lambda2(freq_arr_hz)
    # lam_sq_0_m2 is the weighted mean of lambda^2 distribution (B&dB Eqn. 32)
    # Calculate a global lam_sq_0_m2 value, ignoring isolated flagged voxels
    scale_factor = 1.0 / np.nansum(weight_arr)
    lam_sq_0_m2 = float(scale_factor * np.nansum(weight_arr * lambda_sq_arr_m2))
    if not np.isfinite(lam_sq_0_m2):
        lam_sq_0_m2 = float(np.nanmean(lambda_sq_arr_m2))

    lambda_sq_m2_max = np.nanmax(lambda_sq_arr_m2)
    lambda_sq_m2_min = np.nanmin(lambda_sq_arr_m2)
    delta_lambda_sq_m2 = np.median(np.abs(np.diff(lambda_sq_arr_m2)))

    rmsf_fwhm_theory = 3.8 / (lambda_sq_m2_max - lambda_sq_m2_min)
    phi_max = np.sqrt(3.0) / delta_lambda_sq_m2
    phi_max_scale = np.pi / lambda_sq_m2_min
    dphi = float(0.1 * rmsf_fwhm_theory)

    phi_arr_radm2 = make_phi_arr(phi_max * 10 * 2, dphi)

    rmsf_results = get_rmsf_nufft(
        lambda_sq_arr_m2=lambda_sq_arr_m2,
        phi_arr_radm2=phi_arr_radm2,
        weight_arr=weight_arr,
        lam_sq_0_m2=float(lam_sq_0_m2),
    )

    rmsf_fwhm_meas = float(rmsf_results.fwhm_rmsf_arr)

    return RMSFParams(
        rmsf_fwhm_theory=float(rmsf_fwhm_theory),
        rmsf_fwhm_meas=rmsf_fwhm_meas,
        phi_max=phi_max,
        phi_max_scale=float(phi_max_scale),
        rmsf_results=rmsf_results,
    )


def _lambda_sq_density(
    lambda_sq_arr_m2: NDArray[np.float64],
    natural_weight_arr: NDArray[np.float64],
    cell_m2: float,
) -> NDArray[np.float64]:
    """Local lambda^2 sampling density on a virtual grid of cells width cell_m2:
    the total natural weight in each channel's cell, over the cell width.
    Channels sharing a cell share one density, so uniform_lsq (natural/density)
    gives them equal weight and no single channel jumps within a cell; each
    occupied cell then contributes equally (interferometric uniform weighting).
    briggs blends it with the natural weight via robust. The density steps where
    the true sampling density changes (gaps, channelisation changes); this is
    correct inverse-density weighting, not aliasing. Flagged channels (zero
    natural weight) get zero density."""
    density = np.zeros_like(lambda_sq_arr_m2)
    good = natural_weight_arr > 0
    if not good.any():
        return density
    origin = float(lambda_sq_arr_m2[good].min())
    cell_idx = np.floor((lambda_sq_arr_m2[good] - origin) / cell_m2).astype(np.int64)
    occupancy = np.zeros(int(cell_idx.max()) + 1)
    np.add.at(occupancy, cell_idx, natural_weight_arr[good])
    density[good] = occupancy[cell_idx] / cell_m2
    return density


def natural_weight(real_qu_error: NDArray[np.float64]) -> NDArray[np.float64]:
    """Natural (inverse-variance) weights; all ones if no noise is given."""
    if (real_qu_error == 0).all():
        return np.ones_like(real_qu_error)
    return 1.0 / real_qu_error**2


def uniform_lsq_weight(
    lambda_sq_arr_m2: NDArray[np.float64],
    natural_weight_arr: NDArray[np.float64],
    cell_m2: float,
) -> NDArray[np.float64]:
    """Uniform-in-lambda^2 weights: natural weights divided by the virtual-grid
    lambda^2 sampling density, so each occupied cell contributes equally
    regardless of how densely it is sampled, and channels sharing a cell get
    equal weight. This is interferometric uniform weighting on the lambda^2 grid;
    it narrows the RMSF main lobe. The density (and hence the weight) steps where
    the true sampling density changes (gaps, channelisation)."""
    density = _lambda_sq_density(lambda_sq_arr_m2, natural_weight_arr, cell_m2)
    weight = np.zeros_like(natural_weight_arr)
    np.divide(natural_weight_arr, density, out=weight, where=density > 0)
    return weight


def briggs_weight(
    lambda_sq_arr_m2: NDArray[np.float64],
    natural_weight_arr: NDArray[np.float64],
    robust: float,
    cell_m2: float,
) -> NDArray[np.float64]:
    """Briggs robust weights interpolating natural (robust -> +inf) and
    uniform-in-lambda^2 (robust -> -inf). The `f^2` factor is normalised by the
    natural-weighted mean sampling density (CASA convention) so `robust` is
    comparable across datasets with different channel counts."""
    density = _lambda_sq_density(lambda_sq_arr_m2, natural_weight_arr, cell_m2)
    mean_density = float(
        np.sum(natural_weight_arr * density) / np.sum(natural_weight_arr)
    )
    f_sq = (5.0 * 10.0**-robust) ** 2 / mean_density
    weight: NDArray[np.float64] = natural_weight_arr / (1.0 + density * f_sq)
    return weight


def compute_rmsynth_params(
    freq_arr_hz: NDArray[np.float64],
    complex_pol_arr: NDArray[np.complex128],
    complex_pol_error: NDArray[np.complex128],
    fdf_options: FDFOptions,
) -> RMSynthParams:
    """Calculate the parameters for RM-synthesis.

    Args:
        freq_arr_hz (NDArray[np.float64]): Frequency array in Hz
        pol_arr (NDArray[np.complex128]): Complex polarisation array
        real_qu_error (NDArray[np.float64  |  np.float32]): Error in Stokes Q and U (real)
        fdf_options (FDFOptions): Options for RM-synthesis

    Raises:
        ValueError: If d_phi_radm2 is not provided and n_samples is None.

    Returns:
        RMSynthParams: Wavelength^2 values, reference wavelength^2, Faraday depth values, weight array
    """

    real_qu_error = np.abs(complex_pol_error.real + complex_pol_error.imag) / 2.0

    lambda_sq_arr_m2 = freq_to_lambda2(freq_arr_hz)

    fwhm_rmsf_radm2, d_lambda_sq_max_m2, _ = get_fwhm_rmsf(lambda_sq_arr_m2)

    if fdf_options.d_phi_radm2 is None and fdf_options.n_samples is not None:
        d_phi_radm2 = fwhm_rmsf_radm2 / fdf_options.n_samples
    elif fdf_options.d_phi_radm2 is not None:
        d_phi_radm2 = fdf_options.d_phi_radm2
    else:
        msg = "Either d_phi_radm2 or n_samples must be provided."
        raise ValueError(msg)

    if fdf_options.phi_max_radm2 is None:
        phi_max_radm2 = np.sqrt(3.0) / d_lambda_sq_max_m2
        phi_max_radm2 = max(
            phi_max_radm2, fwhm_rmsf_radm2 * 10.0
        )  # Force the minimum phiMax to 10 FWHM
    else:
        phi_max_radm2 = fdf_options.phi_max_radm2

    phi_arr_radm2 = make_phi_arr(phi_max_radm2, d_phi_radm2)

    logger.debug(
        f"phi = {phi_arr_radm2[0]:0.2f} to {phi_arr_radm2[-1]:0.2f} by {d_phi_radm2:0.2f} ({len(phi_arr_radm2)} chans)."
    )

    # lambda^2 gridding cell: caps the per-channel spacing for the lambda^2-based
    # weights so large gaps do not hand runaway weight to gap-edge channels.
    cell_m2 = float(np.sqrt(3.0) / phi_max_radm2)

    logger.debug(f"Weighting type: {fdf_options.weight_type}")
    # Zero flagged channels before the density-based weights so they do not
    # inflate their neighbours' sampling density.
    mask = ~np.isfinite(complex_pol_arr)
    natural_weight_arr = natural_weight(real_qu_error)
    natural_weight_arr[mask] = 0.0
    match fdf_options.weight_type:
        case "variance" | "natural":
            weight_arr = natural_weight_arr
        case "uniform":
            weight_arr = np.ones_like(freq_arr_hz)
        case "uniform_lsq":
            weight_arr = uniform_lsq_weight(
                lambda_sq_arr_m2, natural_weight_arr, cell_m2
            )
        case "briggs":
            if fdf_options.robust is None:
                msg = "Briggs weighting requires a `robust` parameter."
                raise ValueError(msg)
            weight_arr = briggs_weight(
                lambda_sq_arr_m2, natural_weight_arr, fdf_options.robust, cell_m2
            )

    weight_arr[mask] = 0.0

    # lam_sq_0_m2 is the weighted mean of lambda^2 distribution (B&dB Eqn. 32)
    # Calculate a global lam_sq_0_m2 value, ignoring isolated flagged voxels
    scale_factor = 1.0 / np.nansum(weight_arr)
    lam_sq_0_m2 = float(scale_factor * np.nansum(weight_arr * lambda_sq_arr_m2))
    if not np.isfinite(lam_sq_0_m2):
        lam_sq_0_m2 = float(np.nanmean(lambda_sq_arr_m2))

    logger.debug(f"lam_sq_0_m2 = {lam_sq_0_m2:0.2f} m^2")

    return RMSynthParams(
        lambda_sq_arr_m2=lambda_sq_arr_m2,
        lam_sq_0_m2=lam_sq_0_m2,
        phi_arr_radm2=phi_arr_radm2,
        weight_arr=weight_arr,
    )


def make_phi_arr(
    phi_max_radm2: float,
    d_phi_radm2: float,
) -> NDArray[np.float64]:
    """Construct a Faraday depth array.

    Args:
        phi_max_radm2 (float): Maximum Faraday depth in rad/m^2
        d_phi_radm2 (float): Spacing in Faraday depth in rad/m^2

    Returns:
        NDArray[np.float64]: Faraday depth array in rad/m^2
    """
    # Faraday depth sampling. Zero always centred on middle channel
    n_chan_rm = int(np.round(abs((phi_max_radm2 - 0.0) / d_phi_radm2)) * 2.0 + 1.0)
    max_phi_radm2 = (n_chan_rm - 1.0) * d_phi_radm2 / 2.0
    return arange(
        start=-max_phi_radm2, stop=max_phi_radm2, step=d_phi_radm2, include_stop=True
    )


def make_double_phi_arr(
    phi_arr_radm2: NDArray[np.float64],
) -> NDArray[np.float64]:
    d_phi = phi_arr_radm2[1] - phi_arr_radm2[0]
    phi_max_radm2 = np.max(np.abs(phi_arr_radm2))
    return make_phi_arr(
        phi_max_radm2=phi_max_radm2 * 2 + d_phi,
        d_phi_radm2=d_phi,
    )


def get_fwhm_rmsf(
    lambda_sq_arr_m2: NDArray[np.float64],
) -> FWHM:
    """Calculate the FWHM of the RMSF.

    Args:
        lambda_sq_arr_m2 (NDArray[np.float64]): Wavelength^2 values in m^2
        super_resolution (bool, optional): Use Cotton+Rudnick superresolution. Defaults to False.

    Returns:
        fwhm_rmsf_arr: FWHM of the RMSF main lobe, maximum difference in lambda^2 values, range of lambda^2 values
    """
    lambda_sq_range_m2 = float(
        np.nanmax(lambda_sq_arr_m2) - np.nanmin(lambda_sq_arr_m2)
    )
    d_lambda_sq_max_m2 = np.nanmax(np.abs(np.diff(lambda_sq_arr_m2)))

    # Set the Faraday depth range
    fwhm_rmsf_radm2 = float(
        3.8 / lambda_sq_range_m2
    )  # Dickey+2019 theoretical RMSF width
    return FWHM(
        fwhm_rmsf_radm2=fwhm_rmsf_radm2,
        d_lambda_sq_max_m2=d_lambda_sq_max_m2,
        lambda_sq_range_m2=lambda_sq_range_m2,
    )


def rmsynth_nufft(
    complex_pol_arr: NDArray[np.complex128],
    lambda_sq_arr_m2: NDArray[np.float64],
    phi_arr_radm2: NDArray[np.float64],
    weight_arr: NDArray[np.float64],
    lam_sq_0_m2: float,
    eps: float = 1e-6,
    nthreads: int = 0,
) -> NDArray[np.complex128]:
    """Run RM-synthesis on a cube of Stokes Q and U data using the NUFFT method.

    Args:
        complex_pol_arr (NDArray[np.complex128]): Complex polarisation values (Q + iU)
        lambda_sq_arr_m2 (NDArray[np.float64]): Wavelength^2 values in m^2
        phi_arr_radm2 (NDArray[np.float64]): Faraday depth values in rad/m^2
        weight_arr (NDArray[np.float64]): Weight array
        lam_sq_0_m2 (Optional[float], optional): Reference wavelength^2 in m^2. Defaults to None.
        eps (float, optional): NUFFT tolerance. Defaults to 1e-6.
        nthreads (int, optional): finufft OpenMP threads. 0 uses finufft's default
            (all cores). Set to 1 when parallelising across chunks with dask, to
            avoid oversubscription. Defaults to 0.

    Raises:
        ValueError: If the weight and lambda^2 arrays are not the same shape.
        ValueError: If the Stokes Q and U data arrays are not the same shape.
        ValueError: If the data dimensions are > 3.
        ValueError: If the data depth does not match the lambda^2 vector.

    Returns:
        NDArray[np.float64]: Dirty Faraday dispersion function cube
    """
    tick = time.time()
    msg = f"Running RM-synthesis using the NUFFTs over {len(phi_arr_radm2)} Faraday depth channels."
    logger.info(msg)
    flagged_weight_arr = np.nan_to_num(weight_arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Sanity check on array sizes
    if flagged_weight_arr.shape != lambda_sq_arr_m2.shape:
        msg = f"Weight and lambda^2 arrays must be the same shape. Got {weight_arr.shape} and {lambda_sq_arr_m2.shape}"
        raise ValueError(msg)

    n_dims = len(complex_pol_arr.shape)
    if not n_dims <= 3:
        msg = f"Data dimensions must be <= 3. Got {n_dims}"
        raise ValueError(msg)

    if complex_pol_arr.shape[0] != lambda_sq_arr_m2.shape[0]:
        msg = f"Data depth does not match lambda^2 vector ({complex_pol_arr.shape[0]} vs {lambda_sq_arr_m2.shape[0]})."
        raise ValueError(msg)

    if complex_pol_arr.size == 0:
        msg = "No unflagged data remains. Not doing rm-synthesis"
        logger.critical(msg)
        return (
            np.ones_like(phi_arr_radm2) * np.nan
            + 1j * np.ones_like(phi_arr_radm2) * np.nan
        )

    # Reshape the data arrays to 2 dimensions
    if n_dims == 1:
        complex_pol_arr_2d = np.reshape(complex_pol_arr, (complex_pol_arr.shape[0], 1))
    elif n_dims == 3:
        old_data_shape = complex_pol_arr.shape
        complex_pol_arr_2d = np.reshape(
            complex_pol_arr,
            (
                complex_pol_arr.shape[0],
                complex_pol_arr.shape[1] * complex_pol_arr.shape[2],
            ),
        )
    else:
        complex_pol_arr_2d = complex_pol_arr

    # Create a complex polarised cube, B&dB Eqns. (8) and (14)
    # Array has dimensions [nFreq, nY * nX]
    pol_cube = complex_pol_arr_2d * flagged_weight_arr[:, np.newaxis]

    # Check for NaNs (flagged data) in the cube & set to zero
    mask_cube = ~np.isfinite(pol_cube)
    pol_cube = np.nan_to_num(pol_cube, nan=0.0, posinf=0.0, neginf=0.0)

    # If full planes are flagged then set corresponding weights to zero
    mask_planes = np.sum(~mask_cube, axis=1)
    mask_planes = np.where(mask_planes == 0, 0, 1)
    flagged_weight_arr *= mask_planes

    # The K value used to scale each FDF spectrum must take into account
    # flagged voxels data in the datacube and can be position dependent
    weight_cube = np.invert(mask_cube) * flagged_weight_arr[:, np.newaxis]
    with np.errstate(divide="ignore", invalid="ignore"):
        scale_arr = np.true_divide(1.0, np.sum(weight_cube, axis=0))
        scale_arr[scale_arr == np.inf] = 0
        scale_arr = np.nan_to_num(scale_arr)

    # Clean up one cube worth of memory
    del weight_cube

    # Do the RM-synthesis on each plane
    # finufft must have matching dtypes, so complex64 matches float32
    exponent = (lambda_sq_arr_m2 - lam_sq_0_m2).astype(
        f"float{pol_cube.itemsize * 8 / 2:.0f}"
    )
    fdf_dirty_cube = (
        finufft.nufft1d3(
            x=exponent,
            c=np.ascontiguousarray(pol_cube.T),
            s=(phi_arr_radm2 * 2).astype(exponent.dtype),
            eps=eps,
            isign=-1,
            nthreads=nthreads,
        )
        * scale_arr[..., None]
    ).T

    # Check for pixels that have Re(FDF)=Im(FDF)=0. across ALL Faraday depths
    # These pixels will be changed to NaN in the output
    zeromap = np.all(fdf_dirty_cube == 0.0, axis=0)
    fdf_dirty_cube[..., zeromap] = np.nan + 1.0j * np.nan

    # Restore if 3D shape
    if n_dims == 3:
        fdf_dirty_cube = np.reshape(
            fdf_dirty_cube,
            (fdf_dirty_cube.shape[0], old_data_shape[1], old_data_shape[2]),
        )

    # Remove redundant dimensions in the FDF array
    tock = time.time()
    logger.info(f"NUFFT complete in {tock - tick:.3g} seconds.")
    return np.asarray(np.squeeze(fdf_dirty_cube))


def inverse_rmsynth_nufft(
    complex_fdf_arr: NDArray[np.complex128],
    lambda_sq_arr_m2: NDArray[np.float64],
    phi_arr_radm2: NDArray[np.float64],
    lam_sq_0_m2: float,
    eps: float = 1e-6,
    nthreads: int = 0,
) -> NDArray[np.complex128]:
    """Inverse RM-synthesis - FDF to Stokes Q and U in wavelength^2 space.

    Args:
        complex_fdf_arr (NDArray[np.complex128]): Complex polarisation array in Faraday depth space
        lambda_sq_arr_m2 (NDArray[np.float64]): Wavelength^2 values in m^2
        phi_arr_radm2 (NDArray[np.float64]): Faraday depth values in rad/m^2
        lam_sq_0_m2 (float): Reference wavelength^2 value
        eps (float, optional): NUFFT tolerance. Defaults to 1e-6.
        nthreads (int, optional): finufft OpenMP threads. 0 uses finufft's default
            (all cores). Defaults to 0.

    Raises:
        ValueError: If the Stokes Q and U data arrays are not the same shape.
        ValueError: If the data dimensions are > 3.
        ValueError: If the data depth does not match the lambda^2 vector.

    Returns:
        NDArray[np.float64]: Complex polarisation array in wavelength^2 space
    """

    checks: list[tuple[bool, str]] = [
        (
            complex_fdf_arr.ndim <= 3,
            "Data dimensions must be <= 3.",
        ),
        (
            complex_fdf_arr.shape[0] == phi_arr_radm2.shape[0],
            f"Data depth does not match Faraday depth vector ({complex_fdf_arr.shape[0]} vs {phi_arr_radm2.shape[0]}).",
        ),
    ]
    for check, msg in checks:
        if not check:
            raise ValueError(msg)

    fdf_pol_cube_2d = nd_to_two_d(complex_fdf_arr)

    float_size = fdf_pol_cube_2d.itemsize * 8 / 2  # type: ignore[attr-defined,unused-ignore]
    exponent = (lambda_sq_arr_m2 - lam_sq_0_m2).astype(f"float{float_size:.0f}")
    pol_cube_inv = (
        finufft.nufft1d3(
            x=(phi_arr_radm2 * 2).astype(exponent.dtype),
            c=fdf_pol_cube_2d.T.astype(complex),  # type: ignore[attr-defined,unused-ignore]
            s=exponent,
            eps=eps,
            isign=1,
            nthreads=nthreads,
        )
    ).T

    # Restore if 3D shape
    if complex_fdf_arr.ndim == 3:
        pol_cube_inv = two_d_to_nd(pol_cube_inv, original_shape=complex_fdf_arr.shape)

    # Remove redundant dimensions in the FDF array
    return np.asarray(np.squeeze(pol_cube_inv).astype(np.complex128))


def get_rmsf_nufft(
    lambda_sq_arr_m2: NDArray[np.float64],
    phi_arr_radm2: NDArray[np.float64],
    weight_arr: NDArray[np.float64],
    lam_sq_0_m2: float,
    mask_arr: NDArray[np.bool_] | None = None,
    do_fit_rmsf: bool = False,
    do_fit_rmsf_real: bool = False,
    eps: float = 1e-6,
    nthreads: int = 0,
) -> RMSFResults:
    """Compute the RMSF for a given set of lambda^2 values.

    Args:
        lambda_sq_arr_m2 (NDArray[np.float64]): Wavelength^2 values in m^2
        phi_arr_radm2 (NDArray[np.float64]): Faraday depth values in rad/m^2
        weight_arr (NDArray[np.float64]): Weight array
        lam_sq_0_m2 (float): Reference wavelength^2 value
        super_resolution (bool, optional): Use superresolution. Defaults to False.
        mask_arr (Optional[NDArray[np.float64]], optional): Mask array. Defaults to None.
        do_fit_rmsf (bool, optional): Fit the RMSF with a Gaussian. Defaults to False.
        do_fit_rmsf_real (bool, optional): Fit the *real* part of the. Defaults to False.
        eps (float, optional): NUFFT tolerance. Defaults to 1e-6.
        nthreads (int, optional): finufft OpenMP threads. 0 uses finufft's default
            (all cores). Set to 1 when parallelising across chunks with dask, to
            avoid oversubscription. Defaults to 0.

    Raises:
        ValueError: If the wavelength^2 and weight arrays are not the same shape.
        ValueError: If the mask dimensions are > 3.
        ValueError: If the mask depth does not match the lambda^2 vector.

    Returns:
        RMSFResults: rmsf_cube, phi_double_arr_radm2, fwhm_rmsf_arr, fit_status_arr
    """
    phi_double_arr_radm2 = make_double_phi_arr(phi_arr_radm2)
    weight_arr = weight_arr.copy()
    weight_arr = np.nan_to_num(weight_arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Set the mask array (default to 1D, no masked channels)
    if mask_arr is None:
        mask_arr = np.zeros_like(lambda_sq_arr_m2, dtype=bool)
        n_dimension = 1
    else:
        mask_arr = mask_arr.astype(bool)
        n_dimension = len(mask_arr.shape)

    # Sanity checks on array sizes
    if weight_arr.shape != lambda_sq_arr_m2.shape:
        msg = "wavelength^2 and weight arrays must be the same shape."
        raise ValueError(msg)

    if not n_dimension <= 3:
        msg = "mask dimensions must be <= 3."
        raise ValueError(msg)

    if mask_arr.shape[0] != lambda_sq_arr_m2.shape[0]:
        msg = f"Mask depth does not match lambda^2 vector ({mask_arr.shape[0]} vs {lambda_sq_arr_m2.shape[-1]})."
        raise ValueError(msg)

    # Reshape the mask array to 2 dimensions
    if n_dimension == 1:
        mask_arr = np.reshape(mask_arr, (mask_arr.shape[0], 1))
    elif n_dimension == 3:
        old_data_shape = mask_arr.shape
        mask_arr = np.reshape(
            mask_arr, (mask_arr.shape[0], mask_arr.shape[1] * mask_arr.shape[2])
        )
    num_pixels = mask_arr.shape[-1]

    # If full planes are flagged then set corresponding weights to zero
    flag_xy_sum = np.sum(mask_arr, axis=1)
    mskPlanes = np.where(flag_xy_sum == num_pixels, 0, 1)
    weight_arr *= mskPlanes

    fwhm_rmsf_radm2, _, _ = get_fwhm_rmsf(lambda_sq_arr_m2)
    # Calculate the RMSF at each pixel
    # The K value used to scale each RMSF must take into account
    # isolated flagged voxels data in the datacube
    weight_cube = np.invert(mask_arr) * weight_arr[:, np.newaxis]
    with np.errstate(divide="ignore", invalid="ignore"):
        scale_factor_arr = 1.0 / np.sum(weight_cube, axis=0)
        scale_factor_arr = np.nan_to_num(
            scale_factor_arr, nan=0.0, posinf=0.0, neginf=0.0
        )

    # Calculate the RMSF for each plane
    exponent = lambda_sq_arr_m2 - lam_sq_0_m2
    rmsf_cube = (
        finufft.nufft1d3(
            x=exponent,
            c=np.ascontiguousarray(weight_cube.T).astype(complex),
            s=(phi_double_arr_radm2[::-1] * 2).astype(exponent.dtype),
            eps=eps,
            nthreads=nthreads,
        )
        * scale_factor_arr[..., None]
    ).T

    # Clean up one cube worth of memory
    del weight_cube

    # Default to the analytical RMSF
    fwhm_rmsf_arr = np.ones(num_pixels) * fwhm_rmsf_radm2
    fit_status_arr = np.zeros(num_pixels, dtype=bool)

    # Fit the RMSF main lobe
    if do_fit_rmsf:
        logger.info("Fitting main lobe in each RMSF spectrum.")
        for i in trange(
            num_pixels, desc="Fitting RMSF by pixel", disable=num_pixels == 1
        ):
            try:
                fitted_rmsf = fit_rmsf(
                    rmsf_to_fit_arr=(
                        rmsf_cube[:, i].real
                        if do_fit_rmsf_real
                        else np.abs(rmsf_cube[:, i])
                    ),
                    phi_double_arr_radm2=phi_double_arr_radm2,
                    fwhm_rmsf_radm2=fwhm_rmsf_radm2,
                )
                fit_status = True
            except Exception as e:
                logger.error(f"Failed to fit RMSF at pixel {i}: {e}")
                logger.warning("Setting RMSF FWHM to default value.")
                fitted_rmsf = fwhm_rmsf_radm2
                fit_status = False

            fwhm_rmsf_arr[i] = fitted_rmsf
            fit_status_arr[i] = fit_status

    # Remove redundant dimensions
    rmsf_cube = np.squeeze(rmsf_cube)
    fwhm_rmsf_arr = np.squeeze(fwhm_rmsf_arr)
    fit_status_arr = np.squeeze(fit_status_arr)

    # Restore if 3D shape
    if n_dimension == 3:
        rmsf_cube = np.reshape(
            rmsf_cube, (rmsf_cube.shape[0], old_data_shape[1], old_data_shape[2])
        )
        fwhm_rmsf_arr = np.reshape(
            fwhm_rmsf_arr, (old_data_shape[1], old_data_shape[2])
        )
        fit_status_arr = np.reshape(
            fit_status_arr, (old_data_shape[1], old_data_shape[2])
        )

    return RMSFResults(
        rmsf_cube=rmsf_cube,
        phi_double_arr_radm2=phi_double_arr_radm2,
        fwhm_rmsf_arr=fwhm_rmsf_arr,
        fit_status_arr=fit_status_arr,
    )


fdf_params_schema = pl.Schema(
    {
        "fdf_error_mad": pl.Float64,
        "peak_pi_fit": pl.Float64,
        "peak_pi_error": pl.Float64,
        "peak_pi_fit_debias": pl.Float64,
        "peak_pi_fit_snr": pl.Float64,
        "peak_pi_fit_index": pl.Int64,
        "peak_rm_fit": pl.Float64,
        "peak_rm_fit_error": pl.Float64,
        "peak_q_fit": pl.Float64,
        "peak_u_fit": pl.Float64,
        "peak_pa_fit_deg": pl.Float64,
        "peak_pa_fit_deg_error": pl.Float64,
        "peak_pa0_fit_deg": pl.Float64,
        "peak_pa0_fit_deg_error": pl.Float64,
        "fit_function": pl.String,
        "lam_sq_0_m2": pl.Float64,
        "ref_freq_hz": pl.Float64,
        "fwhm_rmsf_radm2": pl.Float64,
        "fdf_error_noise": pl.Float64,
        "fdf_q_noise": pl.Float64,
        "fdf_u_noise": pl.Float64,
        "min_freq_hz": pl.Float64,
        "max_freq_hz": pl.Float64,
        "n_channels": pl.Int64,
        "median_d_freq_hz": pl.Float64,
        "frac_pol": pl.Float64,
        "frac_pol_error": pl.Float64,
        "sigma_add": pl.Float64,
        "sigma_add_minus": pl.Float64,
        "sigma_add_plus": pl.Float64,
        "mom0": pl.Float64,
        "mom0_debias": pl.Float64,
        "mom1_radm2": pl.Float64,
        "mom2_radm2": pl.Float64,
        "moment_threshold_snr": pl.Float64,
    }
)
fdf_params_schema_df = fdf_params_schema.to_frame(eager=True)


def get_fdf_parameters(
    fdf_arr: NDArray[np.complex128],
    phi_arr_radm2: NDArray[np.float64],
    fwhm_rmsf_radm2: float,
    freq_arr_hz: NDArray[np.float64],
    complex_pol_arr: NDArray[np.complex128],
    complex_pol_error: NDArray[np.complex128],
    lambda_sq_arr_m2: NDArray[np.float64],
    lam_sq_0_m2: float,
    stokes_i_reference_flux: float,
    theoretical_noise: TheoreticalNoise,
    fit_function: Literal["log", "linear"],
    bias_correction_snr: float = 5.0,
    moment_threshold_snr: float = 5.0,
) -> pl.DataFrame:
    """
    Measure standard parameters from a complex Faraday Dispersion Function.
    Currently this function assumes that the noise levels in the Stokes Q
    and U spectra are the same.
    Returns a dictionary containing measured parameters.

    Faraday moments (see `calc_faraday_moments`) are computed with amplitudes
    below `moment_threshold_snr` times the theoretical FDF noise excluded.
    `mom0_debias` additionally corrects each amplitude for polarisation bias
    (the same 2.3 sigma^2 correction applied to the fitted peak) before
    integrating.
    """

    abs_fdf_arr = np.abs(fdf_arr)

    if (~np.isfinite(fdf_arr)).all():
        # I hate this, but can happen with bad data
        peak_pi_index = None
    else:
        peak_pi_index = int(np.nanargmax(abs_fdf_arr))

    # Measure the RMS noise in the spectrum after masking the peak
    d_phi = phi_arr_radm2[1] - phi_arr_radm2[0]
    mask = np.ones_like(phi_arr_radm2, dtype=bool)

    if peak_pi_index is not None:
        mask[peak_pi_index] = False
    fwhm_rmsf_arr_pix = fwhm_rmsf_radm2 / d_phi
    for i in np.where(mask)[0]:
        start = int(i - fwhm_rmsf_arr_pix / 2)
        end = int(i + fwhm_rmsf_arr_pix / 2)
        mask[start : end + 2] = False

    # ignore mean of empty slice warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        fdf_error_mad = float(
            mad_std(np.concatenate([fdf_arr[mask].real, fdf_arr[mask].imag]))
        )

    n_good_phi = np.isfinite(fdf_arr).sum()
    lambda_sq_arr_m2_variance = (
        np.sum(lambda_sq_arr_m2**2.0) - np.sum(lambda_sq_arr_m2) ** 2.0 / n_good_phi
    ) / (n_good_phi - 1)

    good_chan_idx = np.isfinite(freq_arr_hz)
    n_good_chan = good_chan_idx.sum()

    if peak_pi_index is None or not (
        peak_pi_index > 0 and peak_pi_index < len(abs_fdf_arr) - 1
    ):
        msg = "Peak index is not within the FDF array. Not fitting."
        logger.critical(msg)
        peak_pi_fit = np.nan
        peak_rm_fit = np.nan
        peak_pi_fit_snr = np.nan
    else:
        peak_pi_fit, peak_rm_fit, _ = fit_fdf(
            fdf_to_fit_arr=abs_fdf_arr,
            phi_arr_radm2=phi_arr_radm2,
            fwhm_fdf_radm2=fwhm_rmsf_radm2,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            peak_pi_fit_snr = peak_pi_fit / theoretical_noise.fdf_error_noise

    # In rare cases, a parabola can be fitted to the edge of the spectrum,
    # producing a unreasonably large RM and polarized intensity.
    # In these cases, everything should get NaN'd out.
    if np.abs(peak_rm_fit) > np.max(np.abs(phi_arr_radm2)):
        peak_rm_fit = np.nan
        peak_pi_fit = np.nan

    # Error on fitted Faraday depth (RM) is same as channel, but using fitted PI
    peak_rm_fit_err = (
        fwhm_rmsf_radm2 * theoretical_noise.fdf_error_noise / (2.0 * peak_pi_fit)
    )

    # Correct the peak for polarisation bias (POSSUM report 11)
    peak_pi_fit_debias = peak_pi_fit
    if peak_pi_fit_snr >= bias_correction_snr:
        peak_pi_fit_debias = np.sqrt(
            peak_pi_fit**2.0
            - POLARISATION_BIAS_FACTOR * theoretical_noise.fdf_error_noise**2.0
        )

    # Calculate the polarisation angle from the fitted peak
    # Uncertainty from Eqn A.12 in Brentjens & De Bruyn 2005
    peak_pi_fit_index = np.interp(
        peak_rm_fit, phi_arr_radm2, np.arange(phi_arr_radm2.shape[-1], dtype="f4")
    )
    peak_u_fit = np.interp(peak_rm_fit, phi_arr_radm2, fdf_arr.imag)
    peak_q_fit = np.interp(peak_rm_fit, phi_arr_radm2, fdf_arr.real)
    peak_pa_fit_deg = 0.5 * np.degrees(np.arctan2(peak_u_fit, peak_q_fit)) % 180
    peak_pa_fit_deg_err = np.degrees(
        theoretical_noise.fdf_error_noise / (2.0 * peak_pi_fit)
    )

    # Calculate the derotated polarisation angle and uncertainty
    # Uncertainty from Eqn A.20 in Brentjens & De Bruyn 2005
    peak_pa0_fit_deg = (
        float(np.degrees(np.radians(peak_pa_fit_deg) - peak_rm_fit * lam_sq_0_m2))
        % 180.0
    )
    peak_pa0_fit_rad_err = np.sqrt(
        theoretical_noise.fdf_error_noise**2.0
        * n_good_phi
        / (4.0 * (n_good_phi - 2.0) * peak_pi_fit**2.0)
        * ((n_good_phi - 1) / n_good_phi + lam_sq_0_m2**2.0 / lambda_sq_arr_m2_variance)
    )
    peak_pa0_fit_deg_err = float(np.degrees(peak_pa0_fit_rad_err))

    moment_threshold = moment_threshold_snr * theoretical_noise.fdf_error_noise
    moments = calc_faraday_moments(
        complex_fdf_arr=fdf_arr,
        phi_arr_radm2=phi_arr_radm2,
        fwhm_rmsf_radm2=fwhm_rmsf_radm2,
        threshold=moment_threshold,
    )
    # Debiased zeroth moment: correct each amplitude for polarisation bias
    # (same Ricean correction as the fitted peak) before integrating. The cut
    # is deliberately on the raw amplitude, not the debiased one, so it selects
    # the same samples as `mom0` above; the debiased value is what gets summed.
    abs_fdf_debias_arr = np.sqrt(
        np.clip(
            abs_fdf_arr**2.0
            - POLARISATION_BIAS_FACTOR * theoretical_noise.fdf_error_noise**2.0,
            0,
            None,
        )
    )
    mom0_debias = float(
        calc_faraday_moments(
            complex_fdf_arr=np.where(
                abs_fdf_arr >= moment_threshold, abs_fdf_debias_arr, np.nan
            ),
            phi_arr_radm2=phi_arr_radm2,
            fwhm_rmsf_radm2=fwhm_rmsf_radm2,
        ).mom0
    )

    stokes_sigma_add = measure_qu_complexity(
        freq_arr_hz=freq_arr_hz,
        complex_pol_arr=complex_pol_arr,
        complex_pol_error=complex_pol_error,
        frac_pol=peak_pi_fit_debias / stokes_i_reference_flux,
        psi0_deg=peak_pa0_fit_deg,
        rm_radm2=peak_rm_fit,
    )

    return fdf_params_schema_df.vstack(
        pl.DataFrame(
            {
                "fdf_error_mad": fdf_error_mad,
                "peak_pi_fit": peak_pi_fit,
                "peak_pi_error": theoretical_noise.fdf_error_noise,
                "peak_pi_fit_debias": peak_pi_fit_debias,
                "peak_pi_fit_snr": peak_pi_fit_snr,
                "peak_pi_fit_index": int(peak_pi_fit_index)
                if np.isfinite(peak_pi_fit_index)
                else -1,
                "peak_rm_fit": peak_rm_fit,
                "peak_rm_fit_error": peak_rm_fit_err,
                "peak_q_fit": peak_q_fit,
                "peak_u_fit": peak_u_fit,
                "peak_pa_fit_deg": peak_pa_fit_deg,
                "peak_pa_fit_deg_error": peak_pa_fit_deg_err,
                "peak_pa0_fit_deg": peak_pa0_fit_deg,
                "peak_pa0_fit_deg_error": peak_pa0_fit_deg_err,
                "fit_function": fit_function,
                "lam_sq_0_m2": lam_sq_0_m2,
                "ref_freq_hz": lambda2_to_freq(lam_sq_0_m2),
                "fwhm_rmsf_radm2": fwhm_rmsf_radm2,
                "fdf_error_noise": theoretical_noise.fdf_error_noise,
                "fdf_q_noise": theoretical_noise.fdf_q_noise,
                "fdf_u_noise": theoretical_noise.fdf_u_noise,
                "min_freq_hz": freq_arr_hz[good_chan_idx].min(),
                "max_freq_hz": freq_arr_hz[good_chan_idx].max(),
                "n_channels": int(n_good_chan),
                "median_d_freq_hz": np.nanmedian(np.diff(freq_arr_hz[good_chan_idx])),
                "frac_pol": peak_pi_fit_debias / stokes_i_reference_flux,
                "frac_pol_error": theoretical_noise.fdf_error_noise
                / stokes_i_reference_flux,
                "sigma_add": stokes_sigma_add.sigma_add_p.sigma_add,
                "sigma_add_minus": stokes_sigma_add.sigma_add_p.sigma_add_minus,
                "sigma_add_plus": stokes_sigma_add.sigma_add_p.sigma_add_plus,
                "mom0": float(moments.mom0),
                "mom0_debias": mom0_debias,
                "mom1_radm2": float(moments.mom1),
                "mom2_radm2": float(moments.mom2),
                "moment_threshold_snr": moment_threshold_snr,
            }
        )
    )


def cdf_percentile(
    values: NDArray[np.float64], cdf: NDArray[np.float64], q: float = 50.0
) -> float:
    """Return the value at a given percentile of a cumulative distribution function

    Args:
        values (NDArray[np.float64]): Array of values
        cdf (NDArray[np.float64]): Cumulative distribution function
        q (float, optional): Percentile. Defaults to 50.0.

    Returns:
        float: Interpolated value at the given percentile
    """
    return float(np.interp(q / 100.0, cdf, values))


class SigmaAddArrays(NamedTuple):
    pdf: NDArray[np.float64]
    """PDF array of the additional noise term"""
    cdf: NDArray[np.float64]
    """CDF array of the additional noise term"""
    sigma_add_arr: NDArray[np.float64]
    """Array of additional noise values"""


def calculate_sigma_add_arr(
    y_arr: NDArray[np.float64],
    dy_arr: NDArray[np.float64],
    median: float | None = None,
    noise: float | None = None,
    n_samples: int = 1000,
) -> SigmaAddArrays:
    # Measure the median and MADFM of the input data if not provided.
    # Used to overplot a normal distribution when debugging.
    if median is None:
        median = float(np.nanmedian(y_arr))
    if noise is None:
        noise = mad_std(y_arr)

    # Sample the PDF of the additional noise term from a limit near zero to
    # a limit of the range of the data, including error bars
    y_range = np.nanmax(y_arr + dy_arr) - np.nanmin(y_arr - dy_arr)
    sigma_add_arr = np.linspace(y_range / n_samples, y_range, n_samples)

    # Model deviation from Gaussian as an additional noise term.
    # Loop through the range of i additional noise samples and calculate
    # chi-squared and sum(ln(sigma_total)), used later to calculate likelihood.
    n_data = len(y_arr)

    # Calculate sigma_sq_tot for all sigma_add values
    sigma_sq_tot = dy_arr**2.0 + sigma_add_arr[:, None] ** 2.0

    # Calculate ln_sigma_sum_arr for all sigma_add values
    ln_sigma_sum_arr = np.nansum(np.log(np.sqrt(sigma_sq_tot)), axis=1)

    # Calculate chi_sq_arr for all sigma_add values
    chi_sq_arr = np.nansum((y_arr - median) ** 2.0 / sigma_sq_tot, axis=1)
    ln_prob_arr = (
        -np.log(sigma_add_arr)
        - n_data * np.log(2.0 * np.pi) / 2.0
        - ln_sigma_sum_arr
        - chi_sq_arr / 2.0
    )
    ln_prob_arr -= np.nanmax(ln_prob_arr)
    prob_arr = np.exp(ln_prob_arr)
    # Normalize the area under the PDF to be 1
    prob_arr /= np.nansum(prob_arr * np.diff(sigma_add_arr)[0])
    # Calculate the CDF
    cdf = np.cumsum(prob_arr) / np.nansum(prob_arr)

    return SigmaAddArrays(
        pdf=prob_arr,
        cdf=cdf,
        sigma_add_arr=sigma_add_arr,
    )


def calculate_sigma_add(
    y_arr: NDArray[np.float64],
    dy_arr: NDArray[np.float64],
    median: float | None = None,
    noise: float | None = None,
    n_samples: int = 1000,
) -> SigmaAdd:
    """Calculate the most likely value of additional scatter, assuming the
    input data is drawn from a normal distribution. The total uncertainty on
    each data point Y_i is modelled as dYtot_i**2 = dY_i**2 + dYadd**2."""

    sigma_add_arrays = calculate_sigma_add_arr(
        y_arr=y_arr,
        dy_arr=dy_arr,
        median=median,
        noise=noise,
        n_samples=n_samples,
    )

    # Calculate the mean of the distribution and the +/- 1-sigma limits
    sigma_add = cdf_percentile(
        values=sigma_add_arrays.sigma_add_arr, cdf=sigma_add_arrays.cdf, q=50.0
    )
    sigma_add_minus = cdf_percentile(
        values=sigma_add_arrays.sigma_add_arr, cdf=sigma_add_arrays.cdf, q=15.72
    )
    sigma_add_plus = cdf_percentile(
        values=sigma_add_arrays.sigma_add_arr, cdf=sigma_add_arrays.cdf, q=84.27
    )

    return SigmaAdd(
        sigma_add=sigma_add,
        sigma_add_minus=sigma_add_minus,
        sigma_add_plus=sigma_add_plus,
        sigma_add_arrays=sigma_add_arrays,
    )


def faraday_simple_spectrum(
    freq_arr_hz: NDArray[np.float64],
    frac_pol: float,
    psi0_deg: float,
    rm_radm2: float,
) -> NDArray[np.complex128]:
    """Create a simple Faraday spectrum with a single component.

    Args:
        freq_arr_hz (NDArray[np.float64]): Frequency array in Hz
        frac_pol (float): Fractional polarization
        psi0_deg (float): Initial polarization angle in degrees
        rm_radm2 (float): RM in rad/m^2

    Returns:
        NDArray[np.float64]: Complex polarization spectrum
    """
    lambda_sq_arr_m2 = freq_to_lambda2(freq_arr_hz)

    return np.asarray(
        frac_pol * np.exp(2j * (np.deg2rad(psi0_deg) + rm_radm2 * lambda_sq_arr_m2))
    )


def measure_qu_complexity(
    freq_arr_hz: NDArray[np.float64],
    complex_pol_arr: NDArray[np.complex128],
    complex_pol_error: NDArray[np.complex128],
    frac_pol: float,
    psi0_deg: float,
    rm_radm2: float,
) -> StokesSigmaAdd:
    # Create a RM-thin model to subtract
    simple_model = faraday_simple_spectrum(
        freq_arr_hz=freq_arr_hz,
        frac_pol=frac_pol,
        psi0_deg=psi0_deg,
        rm_radm2=rm_radm2,
    )

    # Subtract the RM-thin model to create a residual q & u
    residual_qu = complex_pol_arr - simple_model

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        sigma_add_q = calculate_sigma_add(
            y_arr=residual_qu.real / complex_pol_error.real,
            dy_arr=np.ones_like(residual_qu.real),
            median=0.0,
            noise=1.0,
        )
        sigma_add_u = calculate_sigma_add(
            y_arr=residual_qu.imag / complex_pol_error.imag,
            dy_arr=np.ones_like(residual_qu.imag),
            median=0.0,
            noise=1.0,
        )

    sigma_add_p_arr = np.hypot(
        sigma_add_q.sigma_add_arrays.sigma_add_arr,
        sigma_add_u.sigma_add_arrays.sigma_add_arr,
    )
    sigma_add_p_pdf = np.hypot(
        sigma_add_q.sigma_add_arrays.pdf,
        sigma_add_u.sigma_add_arrays.pdf,
    )
    sigma_add_p_cdf = np.cumsum(sigma_add_p_pdf) / np.nansum(sigma_add_p_pdf)
    sigma_add_p_val = cdf_percentile(
        values=sigma_add_p_arr, cdf=sigma_add_p_cdf, q=50.0
    )
    sigma_add_p_minus = cdf_percentile(
        values=sigma_add_p_arr, cdf=sigma_add_p_cdf, q=15.72
    )
    sigma_add_p_plus = cdf_percentile(
        values=sigma_add_p_arr, cdf=sigma_add_p_cdf, q=84.27
    )
    sigma_add_p = SigmaAdd(
        sigma_add=sigma_add_p_val,
        sigma_add_minus=sigma_add_p_minus,
        sigma_add_plus=sigma_add_p_plus,
        sigma_add_arrays=SigmaAddArrays(
            pdf=sigma_add_p_pdf,
            cdf=sigma_add_p_cdf,
            sigma_add_arr=sigma_add_p_arr,
        ),
    )

    return StokesSigmaAdd(
        sigma_add_q=sigma_add_q,
        sigma_add_u=sigma_add_u,
        sigma_add_p=sigma_add_p,
    )


def measure_fdf_complexity(
    phi_arr_radm2: NDArray[np.float64], complex_fdf_arr: NDArray[np.complex128]
) -> float:
    # Second moment of clean component spectrum
    return calc_mom2_fdf(complex_fdf_arr=complex_fdf_arr, phi_arr_radm2=phi_arr_radm2)
