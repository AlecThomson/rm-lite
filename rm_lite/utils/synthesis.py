"""RM-synthesis utils"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, TypeAlias, TypeVar, cast, get_args

import dask.array as da
import finufft
import numpy as np
import polars as pl
from astropy.constants import c as speed_of_light
from astropy.stats import mad_std
from numpy.typing import NDArray
from scipy import ndimage
from tqdm.auto import trange

from rm_lite.utils.arrays import (
    arange,
    broadcast_over_channels,
    float_if_scalar,
    nd_to_two_d,
    two_d_to_nd,
    zero_nonfinite,
)
from rm_lite.utils.fitting import (
    FitResult,
    StokesIFitOptions,
    check_snr_cut_has_error,
    fit_fdf,
    fit_rmsf,
    fit_sampled_peak,
    fit_stokes_i_model,
    flat_fit_result,
    gaussian_integrand,
    model_is_usable,
    sample_model_error,
)
from rm_lite.utils.logging import logger
from rm_lite.utils.spectra import faraday_simple_spectrum

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
    """Theoretical noise of the FDF.

    A float per field for a per-channel weight array; an (ny, nx) map, possibly
    lazy, when the weights vary per pixel.
    """

    fdf_error_noise: float | NDArray[np.float64] | da.Array
    """Theoretical noise of the FDF"""
    fdf_q_noise: float | NDArray[np.float64] | da.Array
    """Theoretical noise of the real FDF"""
    fdf_u_noise: float | NDArray[np.float64] | da.Array
    """Theoretical noise of the imaginary FDF"""


LamSq0Mode: TypeAlias = Literal["auto", "per_pixel"]
""" How to pick the reference lambda^2: one weighted mean for the whole dataset,
or each pixel's own (B&dB 2005 eq. 32, whose criterion is per pixel whenever the
weights are). """

WeightType: TypeAlias = Literal[
    "variance", "natural", "uniform", "uniform_lsq", "briggs"
]
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
    lam_sq_0_m2: float | LamSq0Mode = "auto"
    """ Reference lambda^2 in m^2, or "auto"/"per_pixel" to derive one. The
    Stokes I reference frequency is derived from it, so the phase and flux
    references always match. """

    def __post_init__(self) -> None:
        if isinstance(self.lam_sq_0_m2, str):
            if self.lam_sq_0_m2 not in get_args(LamSq0Mode):
                msg = (
                    "lam_sq_0_m2 must be a value in m^2 or one of "
                    f"{get_args(LamSq0Mode)}, got {self.lam_sq_0_m2!r}."
                )
                raise ValueError(msg)
        elif not np.isfinite(self.lam_sq_0_m2) or self.lam_sq_0_m2 <= 0:
            msg = f"A given lam_sq_0_m2 must be finite and > 0, got {self.lam_sq_0_m2}."
            raise ValueError(msg)
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


def validate_phi_arr(
    complex_fdf_arr: NDArray[np.complex128] | NDArray[np.float64],
    phi_arr_radm2: NDArray[np.float64],
    axis: int,
) -> NDArray[np.float64]:
    """`phi_arr_radm2` as float64, checked against the FDF's Faraday depth axis."""
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
    return phi_arr_radm2


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
    threshold: float | NDArray[np.float64] | None = None,
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

    phi_arr_radm2 = validate_phi_arr(complex_fdf_arr, phi_arr_radm2, axis)

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


class FaradayPeaks(NamedTuple):
    """Peak of the polarised intensity Faraday depth spectrum.

    Every field is shaped like the peak it came from: a scalar for one
    sightline, a map (numpy or dask) for a cube.
    """

    peak_pi: NDArray[np.float64]
    """Peak polarised intensity, in the input FDF amplitude units"""
    peak_pi_debias: NDArray[np.float64]
    """Peak polarised intensity corrected for polarisation bias; NaN without `fdf_error`"""
    peak_pi_error: NDArray[np.float64]
    """1-sigma error on the peak, i.e. `fdf_error` broadcast to the peak's shape.
    A property of the observation, so it is reported even where no peak was
    found. NaN without `fdf_error`"""
    peak_rm_radm2: NDArray[np.float64]
    """Faraday depth of the peak in rad/m^2"""
    peak_rm_error_radm2: NDArray[np.float64]
    """1-sigma error on the peak Faraday depth; NaN without `fdf_error`"""
    peak_pa_deg: NDArray[np.float64]
    """Polarisation angle at the peak in degrees, at the FDF's reference lambda^2"""
    peak_pa_error_deg: NDArray[np.float64]
    """1-sigma error on the polarisation angle; NaN without `fdf_error`"""
    peak_pa0_deg: NDArray[np.float64]
    """Derotated (intrinsic) polarisation angle in degrees; NaN without `lam_sq_0_m2`"""
    peak_pa0_error_deg: NDArray[np.float64]
    """1-sigma error on the intrinsic angle; NaN without `lambda_sq_arr_m2`"""


def calc_peak_stats(
    peak_pi: float | NDArray[np.float64],
    peak_rm_radm2: float | NDArray[np.float64],
    peak_fdf: complex | NDArray[np.complex128],
    fwhm_rmsf_radm2: float | NDArray[np.float64],
    fdf_error: float | NDArray[np.float64] | None = None,
    lam_sq_0_m2: float | NDArray[np.float64] | None = None,
    lambda_sq_arr_m2: NDArray[np.float64] | None = None,
    bias_correction_snr: float = 5.0,
) -> FaradayPeaks:
    """Angles, debiased intensity and errors for an already-located FDF peak.

    Shared by the per-sightline Gaussian fit (`get_fdf_parameters`) and the
    vectorised cube peak finder (`calc_faraday_peaks`), which differ only in how
    they locate the peak. Scalars or arrays (numpy or dask); anything needing an
    argument that was not given comes back NaN.

    Args:
        peak_pi (float | NDArray[np.float64]): Peak polarised intensity, in FDF amplitude units.
        peak_rm_radm2 (float | NDArray[np.float64]): Faraday depth of the peak in rad/m^2.
        peak_fdf (complex | NDArray[np.complex128]): Complex FDF at the peak, for the angle.
        fwhm_rmsf_radm2 (float | NDArray[np.float64]): FWHM of the RMSF main lobe in rad/m^2.
        fdf_error (float | NDArray[np.float64] | None, optional): Theoretical FDF noise, scalar
            or a per-pixel map (`TheoreticalNoise.fdf_error_noise`). Enables the
            errors and the debiased peak. Defaults to None.
        lam_sq_0_m2 (float | NDArray[np.float64] | None, optional): Reference wavelength^2 the
            FDF is derotated to, scalar or a per-pixel map
            (`RMSynth3DResults.lam_sq_0_map`). Enables the intrinsic angle.
            Defaults to None.
        lambda_sq_arr_m2 (NDArray[np.float64] | None, optional): Channel lambda^2 in m^2, for
            the intrinsic-angle error. Defaults to None.
        bias_correction_snr (float, optional): Debias only peaks at or above this
            SNR; below it the raw peak is kept. Defaults to 5.0.

    Returns:
        FaradayPeaks: Peak intensity, Faraday depth and polarisation angles, with
            errors, shaped like `peak_pi`.
    """
    peak_pa_deg = 0.5 * np.degrees(np.arctan2(peak_fdf.imag, peak_fdf.real)) % 180.0
    blank = np.full_like(peak_pi, np.nan, dtype=np.float64)

    if fdf_error is None:
        peak_pi_debias = peak_pi_error = peak_rm_error_radm2 = peak_pa_error_deg = blank
    else:
        # Multiplied out rather than assigned, so a scalar noise becomes a map
        # alongside a map of peaks.
        peak_pi_error = fdf_error * np.ones_like(peak_pi, dtype=np.float64)
        # Ricean correction (POSSUM report 11), only where it is meaningful.
        peak_pi_debias = np.where(
            peak_pi >= bias_correction_snr * peak_pi_error,
            np.sqrt(
                np.clip(
                    peak_pi**2.0 - POLARISATION_BIAS_FACTOR * peak_pi_error**2.0,
                    0,
                    None,
                )
            ),
            peak_pi,
        )
        # Faraday depth error from the RMSF width, angle error from Brentjens &
        # de Bruyn 2005, eq. A.12; both scale as 1/SNR.
        peak_rm_error_radm2 = fwhm_rmsf_radm2 * peak_pi_error / (2.0 * peak_pi)
        peak_pa_error_deg = np.degrees(peak_pi_error / (2.0 * peak_pi))

    if lam_sq_0_m2 is None:
        peak_pa0_deg = blank
    else:
        peak_pa0_deg = (
            np.degrees(np.radians(peak_pa_deg) - peak_rm_radm2 * lam_sq_0_m2) % 180.0
        )

    if lambda_sq_arr_m2 is None or lam_sq_0_m2 is None or fdf_error is None:
        peak_pa0_error_deg = blank
    else:
        # Brentjens & de Bruyn 2005, eq. A.20: the intrinsic angle is an
        # extrapolation to lambda^2 = 0, so its error grows with the lever arm
        # between lam_sq_0_m2 and the spread of the channel lambda^2.
        lam_sq_arr = lambda_sq_arr_m2[np.isfinite(lambda_sq_arr_m2)]
        n_chan = lam_sq_arr.size
        lam_sq_variance = (
            np.sum(lam_sq_arr**2.0) - np.sum(lam_sq_arr) ** 2.0 / n_chan
        ) / (n_chan - 1)
        peak_pa0_error_deg = np.degrees(
            np.sqrt(
                peak_pi_error**2.0
                * n_chan
                / (4.0 * (n_chan - 2.0) * peak_pi**2.0)
                * ((n_chan - 1) / n_chan + lam_sq_0_m2**2.0 / lam_sq_variance)
            )
        )

    return FaradayPeaks(
        peak_pi=cast("NDArray[np.float64]", peak_pi),
        peak_pi_debias=peak_pi_debias,
        peak_pi_error=peak_pi_error,
        peak_rm_radm2=cast("NDArray[np.float64]", peak_rm_radm2),
        peak_rm_error_radm2=peak_rm_error_radm2,
        peak_pa_deg=peak_pa_deg,
        peak_pa_error_deg=peak_pa_error_deg,
        peak_pa0_deg=peak_pa0_deg,
        peak_pa0_error_deg=peak_pa0_error_deg,
    )


def calc_faraday_peaks(
    complex_fdf_arr: NDArray[np.complex128],
    phi_arr_radm2: NDArray[np.float64],
    fwhm_rmsf_radm2: float | NDArray[np.float64],
    axis: int = 0,
    fdf_error: float | NDArray[np.float64] | None = None,
    lam_sq_0_m2: float | NDArray[np.float64] | None = None,
    lambda_sq_arr_m2: NDArray[np.float64] | None = None,
    threshold: float | NDArray[np.float64] | None = None,
    bias_correction_snr: float = 5.0,
) -> FaradayPeaks:
    """Locate the peak of a Faraday depth spectrum and measure it.

    The brightest sample and its two neighbours go to
    `rm_lite.utils.fitting.fit_sampled_peak`, which interpolates the peak
    sub-sample; `calc_peak_stats` then turns that peak into the angles and
    errors, exactly as for the 1D Gaussian fit in `get_fdf_parameters`. Finding
    the peak here is elementwise plus a single reduction over the Faraday depth
    axis, so numpy or dask arrays of any dimensionality work, chunked however
    you like.

    Everything is NaN for a spectrum with no interior maximum (peak on an end
    sample, flat, or no finite samples), and for one whose peak is below
    `threshold`.

    Args:
        complex_fdf_arr (NDArray[np.complex128]): Complex FDF. Real input has no
            polarisation angle, so it is rejected.
        phi_arr_radm2 (NDArray[np.float64]): Uniformly spaced Faraday depth array in rad/m^2.
        fwhm_rmsf_radm2 (float | NDArray[np.float64]): FWHM of the RMSF main lobe in rad/m^2.
            An array must broadcast against the FDF shape with the Faraday depth
            axis removed.
        axis (int, optional): Faraday depth axis of `complex_fdf_arr`. Defaults to 0.
        fdf_error, lam_sq_0_m2, lambda_sq_arr_m2, bias_correction_snr: See
            `calc_peak_stats`.
        threshold (float | NDArray[np.float64] | None, optional): Blank peaks below this
            amplitude (in FDF amplitude units). Defaults to None.

    Returns:
        FaradayPeaks: Peak intensity, Faraday depth and polarisation angles, with
            the Faraday depth axis reduced away.
    """
    phi_arr_radm2 = validate_phi_arr(complex_fdf_arr, phi_arr_radm2, axis)
    if not np.iscomplexobj(complex_fdf_arr):
        msg = "`complex_fdf_arr` must be complex: a real FDF has no polarisation angle."
        raise ValueError(msg)

    n_phi = phi_arr_radm2.shape[0]
    phi_step = float(phi_arr_radm2[1] - phi_arr_radm2[0])
    phi_shape = [1] * complex_fdf_arr.ndim
    phi_shape[axis] = n_phi
    sample_index = np.arange(n_phi).reshape(phi_shape)

    abs_fdf_arr = np.abs(complex_fdf_arr)
    # -inf rather than NaN: argmax has no nan-skipping form that survives an
    # all-NaN spectrum, and those fall out as NaN below anyway.
    peak_index = np.argmax(
        np.where(np.isfinite(abs_fdf_arr), abs_fdf_arr, -np.inf), axis=axis
    )
    peak_index_nd = np.expand_dims(peak_index, axis)

    def sample_offset_from_peak(offset: int) -> NDArray[np.complex128]:
        """The FDF sample `offset` samples along from each spectrum's peak.

        A masked sum rather than a fancy-index gather: it dispatches to dask
        unchanged, and an offset off the end of the axis simply matches nothing.
        """
        picked = np.where(sample_index == peak_index_nd + offset, complex_fdf_arr, 0)
        return cast(
            "NDArray[np.complex128]",
            np.squeeze(np.sum(picked, axis=axis, keepdims=True), axis=axis),
        )

    fdf_below, fdf_at, fdf_above = (sample_offset_from_peak(o) for o in (-1, 0, 1))
    # A brightest sample at either end of the axis has no neighbour on one side,
    # where the gather above returned zero rather than a sample. Blank it, so the
    # fit reports no peak instead of fitting that zero.
    is_interior = (peak_index > 0) & (peak_index < n_phi - 1)
    fdf_at = np.where(is_interior, fdf_at, np.nan)

    peak = fit_sampled_peak(fdf_below, fdf_at, fdf_above)
    peak_pi, peak_offset, peak_fdf = peak.amplitude, peak.offset, peak.value
    if threshold is not None:
        detected = peak_pi >= threshold
        peak_pi = np.where(detected, peak_pi, np.nan)
        peak_offset = np.where(detected, peak_offset, np.nan)
        peak_fdf = np.where(detected, peak_fdf, np.nan)
    # The fit's offset is in samples; the Faraday depth grid is uniform.
    peak_rm_radm2 = phi_arr_radm2[0] + (peak_index + peak_offset) * phi_step

    return calc_peak_stats(
        peak_pi=peak_pi,
        peak_rm_radm2=peak_rm_radm2,
        peak_fdf=peak_fdf,
        fwhm_rmsf_radm2=fwhm_rmsf_radm2,
        fdf_error=fdf_error,
        lam_sq_0_m2=lam_sq_0_m2,
        lambda_sq_arr_m2=lambda_sq_arr_m2,
        bias_correction_snr=bias_correction_snr,
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

    check_snr_cut_has_error(fit_options, stokes_data.stokes_i_error_arr[no_nan_idx])

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

    i_good = stokes_data.stokes_i_arr[no_nan_idx]
    model_good = fit_result.stokes_i_model_func(
        stokes_data.freq_arr_hz[no_nan_idx] / ref_freq_hz,
        *np.asarray(fit_result.popt),
    )
    if not model_is_usable(model_good):
        logger.warning(
            "The fitted Stokes I model cannot safely divide Q/U (see "
            "`rm_lite.utils.fitting.model_is_usable`); falling back to a flat "
            "model at the mean Stokes I, so Q/U get no spectral correction."
        )
        fit_result = flat_fit_result(
            float(np.mean(i_good)),
            len(np.asarray(fit_result.popt)) - 1,
            fit_options.fit_function,
        )

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
    complex_pol_error: NDArray[np.complex128] | da.Array,
    weight_arr: NDArray[np.float64] | da.Array,
) -> TheoreticalNoise:
    """Theoretical FDF noise, reduced over channels only.

    A per-channel weight gives one float; a per-pixel one gives an (ny, nx) map,
    lazily if the weights are lazy. Reducing over the spatial axes too would
    report a noise sqrt(ny*nx) too small for every pixel.
    """
    weight_arr = zero_nonfinite(weight_arr)
    complex_pol_error_flagged = zero_nonfinite(complex_pol_error)
    fdf_complex_noise = np.sqrt(
        np.nansum(weight_arr**2 * complex_pol_error_flagged**2, axis=0)
        / (np.sum(weight_arr, axis=0)) ** 2
    )

    fdf_error_noise = (fdf_complex_noise.real + fdf_complex_noise.imag) / 2
    return TheoreticalNoise(
        fdf_error_noise=float_if_scalar(fdf_error_noise),
        fdf_q_noise=float_if_scalar(fdf_complex_noise.real),
        fdf_u_noise=float_if_scalar(fdf_complex_noise.imag),
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
    cell_m2: float = 0.0
    """ lambda^2 gridding cell, for re-deriving the grid weightings per chunk """


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
    # Vectorised over pixels: the weight may be per-channel (n_freq,) or
    # per-pixel (n_freq, ny, nx), and each pixel gets its own occupancy, since
    # its own flagging decides which cells are occupied. lambda^2 is per-channel
    # either way, so the cell edges are shared and only the origin shifts.
    weight_2d = natural_weight_arr.reshape(natural_weight_arr.shape[0], -1)
    lambda_sq_1d = np.reshape(lambda_sq_arr_m2, -1)
    n_pixel = weight_2d.shape[1]

    density = np.zeros_like(weight_2d)
    good = weight_2d > 0
    if not good.any():
        return density.reshape(natural_weight_arr.shape)

    origin = np.where(good, lambda_sq_1d[:, np.newaxis], np.inf).min(axis=0)
    cell_idx = np.where(
        good, np.floor((lambda_sq_1d[:, np.newaxis] - origin) / cell_m2), 0
    ).astype(np.int64)
    pixel_idx = np.broadcast_to(np.arange(n_pixel), cell_idx.shape)

    # One bincount over flattened (cell, pixel) pairs rather than a scatter-add
    # per pixel: same occupancy, and it stays a single vectorised pass.
    n_cell = int(cell_idx.max()) + 1
    occupancy = np.bincount(
        (cell_idx[good] * n_pixel + pixel_idx[good]),
        weights=weight_2d[good],
        minlength=n_cell * n_pixel,
    ).reshape(n_cell, n_pixel)
    density[good] = occupancy[cell_idx[good], pixel_idx[good]] / cell_m2
    return density.reshape(natural_weight_arr.shape)


def error_from_weight(
    weight_arr: NDArray[np.float64] | da.Array,
) -> NDArray[np.complex128] | da.Array:
    """The complex Q/U error a weight implies, `1/sqrt(weight)`.

    Zero or blank weight means no information, so infinite error.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        real_error = np.where(
            np.isfinite(weight_arr) & (weight_arr > 0),
            1.0 / np.sqrt(np.where(weight_arr > 0, weight_arr, 1.0)),
            np.inf,
        )
    # `real_error * (1 + 1j)`, not `real_error + 1j * real_error`: the latter
    # multiplies inf by a zero real part and turns a blanked channel's error
    # into NaN.
    return cast(
        "NDArray[np.complex128] | da.Array",
        real_error.astype(np.complex128) * (1.0 + 1.0j),
    )


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
    # Reduced over channels, not over everything: with a per-pixel weight array
    # each pixel needs its own natural-weighted mean density, or one pixel's
    # noise level sets `f_sq` for the whole image and `robust` stops meaning the
    # same thing pixel to pixel. Scalars for a per-channel weight, as before.
    total_weight = np.sum(natural_weight_arr, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_density = np.where(
            total_weight > 0,
            np.sum(natural_weight_arr * density, axis=0) / total_weight,
            np.inf,
        )
        f_sq = (5.0 * 10.0**-robust) ** 2 / mean_density
    weight: NDArray[np.float64] = np.where(
        total_weight > 0, natural_weight_arr / (1.0 + density * f_sq), 0.0
    )
    return weight


def _match_channel_mask(
    mask: NDArray[np.bool_], target: NDArray[Any]
) -> NDArray[np.bool_]:
    """Broadcast a per-channel boolean mask to align with `target`'s shape.

    `mask` is 1D (per-channel) when it comes from a per-channel
    `complex_pol_arr`, but `target` (natural_weight_arr / weight_arr) may be
    3D (per-channel, per-pixel). Right-aligned broadcasting can't match a 1D
    mask against a leading channel axis on its own -- reuse the same
    reshape as `broadcast_over_channels`. If `mask` already matches
    `target`'s shape (e.g. a genuinely per-pixel `complex_pol_arr`), leave it
    unchanged.
    """
    if mask.shape == target.shape:
        return mask
    return broadcast_over_channels(mask, target)


def apply_weight_type(
    lambda_sq_arr_m2: NDArray[np.float64],
    real_qu_error: NDArray[np.float64],
    channel_mask: NDArray[np.bool_],
    fdf_options: FDFOptions,
    cell_m2: float,
) -> NDArray[np.float64]:
    """Q/U error -> RM-synthesis weights of the requested type, per pixel.

    `real_qu_error` is per-channel or per-pixel; the result takes its shape.
    Separate from `compute_rmsynth_params` so a chunk can weight its own pixels:
    the grid weightings bin lambda^2 into cells, costing memory per pixel.
    """
    # Zero flagged channels before the density-based weights so they do not
    # inflate their neighbours' sampling density.
    natural_weight_arr = natural_weight(real_qu_error)
    natural_weight_arr = np.where(
        _match_channel_mask(channel_mask, natural_weight_arr), 0.0, natural_weight_arr
    )
    # Broadcast-compatible view of lambda^2 for weight arrays that carry spatial
    # axes beyond the channel axis.
    lambda_sq_arr_m2_b = broadcast_over_channels(lambda_sq_arr_m2, natural_weight_arr)

    weight_arr: NDArray[np.float64]
    match fdf_options.weight_type:
        case "variance" | "natural":
            weight_arr = natural_weight_arr
        case "uniform":
            # Match natural_weight_arr's shape (which may be 3D), not
            # freq_arr_hz (always 1D).
            weight_arr = np.ones_like(natural_weight_arr)
        case "uniform_lsq":
            weight_arr = uniform_lsq_weight(
                lambda_sq_arr_m2_b, natural_weight_arr, cell_m2
            )
        case "briggs":
            if fdf_options.robust is None:
                msg = "Briggs weighting requires a `robust` parameter."
                raise ValueError(msg)
            weight_arr = briggs_weight(
                lambda_sq_arr_m2_b, natural_weight_arr, fdf_options.robust, cell_m2
            )

    return np.where(_match_channel_mask(channel_mask, weight_arr), 0.0, weight_arr)


def weighted_lam_sq_0(
    weight_arr: NDArray[np.float64], lambda_sq_arr_m2: NDArray[np.float64]
) -> float:
    """Weighted mean of lambda^2 over every axis given, B&dB 2005 eq. 32.

    Nulls the RMSF's orthogonal response at phi = 0, keeping the main lobe
    parallel to the polarisation vector at the reference.
    """
    # Scale factor first, matching the order this was computed in before it was
    # given a name, so the default path stays bit-for-bit what it was.
    scale_factor = 1.0 / np.nansum(weight_arr)
    return float(scale_factor * np.nansum(weight_arr * lambda_sq_arr_m2))


def lam_sq_0_per_pixel(
    weight_arr: NDArray[np.float64], lambda_sq_arr_m2: NDArray[np.float64]
) -> NDArray[np.float64]:
    """B&dB eq. 32 per pixel, reducing over channels only.

    Pixels that weight or flag the channels differently each have their own.
    """
    lambda_sq_b = broadcast_over_channels(lambda_sq_arr_m2, weight_arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        return cast(
            "NDArray[np.float64]",
            np.nansum(weight_arr * lambda_sq_b, axis=0) / np.nansum(weight_arr, axis=0),
        )


def derotate_to(
    fdf: NDArray[np.complex128],
    phi_arr_radm2: NDArray[np.float64],
    from_lam_sq_0_m2: float | NDArray[np.float64],
    to_lam_sq_0_m2: float | NDArray[np.float64],
) -> NDArray[np.complex128]:
    """Move an FDF or RMSF between reference lambda^2 values.

    B&dB eq. 25 is a shift theorem, so this is an exact phase ramp: amplitudes
    are untouched and it inverts itself. Either reference may be a per-pixel map.
    Not valid on a *restored* CLEAN cube, whose real restoring beam does not
    commute with the ramp.
    """
    shift = np.asarray(to_lam_sq_0_m2) - np.asarray(from_lam_sq_0_m2)
    phi_b = broadcast_over_channels(phi_arr_radm2, fdf)
    return cast("NDArray[np.complex128]", fdf * np.exp(2j * phi_b * shift))


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
    mask = ~np.isfinite(complex_pol_arr)
    weight_arr = apply_weight_type(
        lambda_sq_arr_m2=lambda_sq_arr_m2,
        real_qu_error=real_qu_error,
        channel_mask=mask,
        fdf_options=fdf_options,
        cell_m2=cell_m2,
    )
    lambda_sq_arr_m2_b = broadcast_over_channels(lambda_sq_arr_m2, weight_arr)

    # lam_sq_0_m2 is the weighted mean of lambda^2 distribution (B&dB Eqn. 32)
    # Calculate a single global lam_sq_0_m2 value (summing over all axes when
    # weight_arr is 3D), ignoring isolated flagged voxels.
    if not isinstance(fdf_options.lam_sq_0_m2, str):
        # Pinned by the caller, e.g. to share a reference with another cube.
        lam_sq_0_m2 = float(fdf_options.lam_sq_0_m2)
    else:
        lam_sq_0_m2 = weighted_lam_sq_0(weight_arr, lambda_sq_arr_m2_b)
        if not np.isfinite(lam_sq_0_m2):
            lam_sq_0_m2 = float(np.nanmean(lambda_sq_arr_m2))

    logger.debug(f"lam_sq_0_m2 = {lam_sq_0_m2:0.2f} m^2")

    return RMSynthParams(
        lambda_sq_arr_m2=lambda_sq_arr_m2,
        lam_sq_0_m2=lam_sq_0_m2,
        phi_arr_radm2=phi_arr_radm2,
        weight_arr=weight_arr,
        cell_m2=cell_m2,
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
        weight_arr (NDArray[np.float64]): Weight array. 1D (per-channel) if
            shared by every pixel, or 3D (matching complex_pol_arr's shape)
            if weights vary spectrally per pixel.
        lam_sq_0_m2 (Optional[float], optional): Reference wavelength^2 in m^2. Defaults to None.
        eps (float, optional): NUFFT tolerance. Defaults to 1e-6.
        nthreads (int, optional): finufft OpenMP threads. 0 uses finufft's default
            (all cores). Set to 1 when parallelising across chunks with dask, to
            avoid oversubscription. Defaults to 0.

    Raises:
        ValueError: If weight_arr is not 1D or the same shape as complex_pol_arr.
        ValueError: If the Stokes Q and U data arrays are not the same shape.
        ValueError: If the data dimensions are > 3.
        ValueError: If the data depth does not match the lambda^2 vector.

    Returns:
        NDArray[np.float64]: Dirty Faraday dispersion function cube
    """
    tick = time.time()
    msg = f"Running RM-synthesis using the NUFFTs over {len(phi_arr_radm2)} Faraday depth channels."
    logger.info(msg)

    n_dims = len(complex_pol_arr.shape)
    if not n_dims <= 3:
        msg = f"Data dimensions must be <= 3. Got {n_dims}"
        raise ValueError(msg)

    if complex_pol_arr.shape[0] != lambda_sq_arr_m2.shape[0]:
        msg = f"Data depth does not match lambda^2 vector ({complex_pol_arr.shape[0]} vs {lambda_sq_arr_m2.shape[0]})."
        raise ValueError(msg)

    # Sanity check on weight_arr: either 1D and per-channel, or 3D and
    # matching complex_pol_arr's full shape (per-channel, per-pixel).
    if weight_arr.ndim == 1:
        if weight_arr.shape[0] != lambda_sq_arr_m2.shape[0]:
            msg = f"Weight and lambda^2 arrays must have matching channel counts. Got {weight_arr.shape} and {lambda_sq_arr_m2.shape}"
            raise ValueError(msg)
    elif weight_arr.ndim == complex_pol_arr.ndim:
        if weight_arr.shape != complex_pol_arr.shape:
            msg = f"3D weight array must match the data shape. Got {weight_arr.shape} and {complex_pol_arr.shape}"
            raise ValueError(msg)
    else:
        msg = f"Weight array must be 1D (per-channel) or match the data's {n_dims}D shape. Got {weight_arr.ndim}D."
        raise ValueError(msg)

    flagged_weight_arr = zero_nonfinite(weight_arr)

    if complex_pol_arr.size == 0:
        msg = "No unflagged data remains. Not doing rm-synthesis"
        logger.critical(msg)
        return (
            np.ones_like(phi_arr_radm2) * np.nan
            + 1j * np.ones_like(phi_arr_radm2) * np.nan
        )

    # Reshape the data array (and, if 3D, the weight array identically) to 2
    # dimensions: (nchan, num_pixels).
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

    if flagged_weight_arr.ndim == 1:
        # Shared per-channel weight: (nchan, 1) broadcasts against every
        # pixel column below.
        flagged_weight_arr = np.reshape(
            flagged_weight_arr, (flagged_weight_arr.shape[0], 1)
        )
    else:
        # Per-pixel weight: flatten the same way complex_pol_arr was.
        flagged_weight_arr = np.reshape(
            flagged_weight_arr, (flagged_weight_arr.shape[0], -1)
        )

    # Create a complex polarised cube, B&dB Eqns. (8) and (14)
    # Array has dimensions [nFreq, nY * nX]
    pol_cube = complex_pol_arr_2d * flagged_weight_arr

    # Check for NaNs (flagged data) in the cube & set to zero
    mask_cube = ~np.isfinite(pol_cube)
    pol_cube = zero_nonfinite(pol_cube)

    # If full planes are flagged then set corresponding weights to zero
    mask_planes = np.sum(~mask_cube, axis=1, keepdims=True)
    mask_planes = np.where(mask_planes == 0, 0, 1)
    flagged_weight_arr = flagged_weight_arr * mask_planes

    # The K value used to scale each FDF spectrum must take into account
    # flagged voxels data in the datacube and can be position dependent
    weight_cube = np.invert(mask_cube) * flagged_weight_arr
    with np.errstate(divide="ignore", invalid="ignore"):
        scale_arr = np.true_divide(1.0, np.sum(weight_cube, axis=0))
        scale_arr[scale_arr == np.inf] = 0
        scale_arr = zero_nonfinite(scale_arr)

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
    reuse_rmsf: bool = True,
) -> RMSFResults:
    """Compute the RMSF for a given set of lambda^2 values.

    Args:
        lambda_sq_arr_m2 (NDArray[np.float64]): Wavelength^2 values in m^2
        phi_arr_radm2 (NDArray[np.float64]): Faraday depth values in rad/m^2
        weight_arr (NDArray[np.float64]): Weight array. 1D (per-channel) if
            every pixel shares the same weighting, or 3D (channel, y, x) if
            weights vary spectrally per pixel (e.g. per-pixel noise
            estimates).
        lam_sq_0_m2 (float): Reference wavelength^2 value
        super_resolution (bool, optional): Use superresolution. Defaults to False.
        mask_arr (Optional[NDArray[np.float64]], optional): Mask array. Defaults to None.
        do_fit_rmsf (bool, optional): Fit the RMSF with a Gaussian. Defaults to False.
        do_fit_rmsf_real (bool, optional): Fit the *real* part of the. Defaults to False.
        eps (float, optional): NUFFT tolerance. Defaults to 1e-6.
        nthreads (int, optional): finufft OpenMP threads. 0 uses finufft's default
            (all cores). Set to 1 when parallelising across chunks with dask, to
            avoid oversubscription. Defaults to 0.
        reuse_rmsf (bool, optional): Compute one RMSF and reuse it for every
            pixel when they all share the same channel flagging and weighting,
            instead of running the NUFFT per pixel for an identical answer. The
            check below gates this regardless, so the flag can only decline a
            saving already known to be safe; it exists to get at the per-pixel
            path for testing and debugging. Defaults to True.

    Raises:
        ValueError: If weight_arr is not 1D or 3D.
        ValueError: If the wavelength^2 and weight arrays don't have matching
            channel counts.
        ValueError: If the mask dimensions are > 3.
        ValueError: If the mask depth does not match the lambda^2 vector.
        ValueError: If a 3D mask and a 3D weight array cover different pixel
            grids.

    Returns:
        RMSFResults: rmsf_cube, phi_double_arr_radm2, fwhm_rmsf_arr, fit_status_arr
    """
    phi_double_arr_radm2 = make_double_phi_arr(phi_arr_radm2)
    weight_arr = np.asarray(weight_arr, dtype=float).copy()
    weight_arr = zero_nonfinite(weight_arr)

    if weight_arr.ndim not in (1, 3):
        msg = "weight array must be 1D (per-channel) or 3D (per-channel, per-pixel)."
        raise ValueError(msg)
    if weight_arr.shape[0] != lambda_sq_arr_m2.shape[0]:
        msg = "wavelength^2 and weight arrays must have matching channel counts."
        raise ValueError(msg)

    # Set the mask array (default to 1D, no masked channels)
    if mask_arr is None:
        mask_arr = np.zeros_like(lambda_sq_arr_m2, dtype=bool)
        n_dimension = 1
    else:
        mask_arr = mask_arr.astype(bool)
        n_dimension = len(mask_arr.shape)

    if not n_dimension <= 3:
        msg = "mask dimensions must be <= 3."
        raise ValueError(msg)

    if mask_arr.shape[0] != lambda_sq_arr_m2.shape[0]:
        msg = f"Mask depth does not match lambda^2 vector ({mask_arr.shape[0]} vs {lambda_sq_arr_m2.shape[-1]})."
        raise ValueError(msg)

    # A 3D mask and a 3D weight array must describe the same pixel grid, or
    # there's no way to know which weight column pairs with which mask column
    # once both are flattened below.
    if (
        n_dimension == 3
        and weight_arr.ndim == 3
        and mask_arr.shape[1:] != weight_arr.shape[1:]
    ):
        msg = (
            "mask and weight arrays cover different pixel grids "
            f"({mask_arr.shape[1:]} vs {weight_arr.shape[1:]})."
        )
        raise ValueError(msg)

    # The true spatial pixel count, from whichever of mask/weight is 3D (they
    # must agree, per the check above, if both are).
    if n_dimension == 3:
        old_data_shape = mask_arr.shape
    elif weight_arr.ndim == 3:
        old_data_shape = weight_arr.shape
    else:
        old_data_shape = None
    num_pixels = (
        old_data_shape[1] * old_data_shape[2] if old_data_shape is not None else 1
    )

    # Reshape the mask array to 2 dimensions: (nchan, 1) if spatially uniform,
    # (nchan, num_pixels) if it varies per pixel.
    if n_dimension == 1:
        mask_arr = np.reshape(mask_arr, (mask_arr.shape[0], 1))
    elif n_dimension == 3:
        mask_arr = np.reshape(mask_arr, (mask_arr.shape[0], num_pixels))

    # Same reshape for the weight array. Kept independently at (nchan, 1) or
    # (nchan, num_pixels) rather than always broadcast out to num_pixels --
    # numpy broadcasting combines mismatched-but-compatible (nchan, 1) and
    # (nchan, num_pixels) arrays below without materialising a full copy.
    if weight_arr.ndim == 1:
        weight_arr = np.reshape(weight_arr, (weight_arr.shape[0], 1))
    else:
        weight_arr = np.reshape(weight_arr, (weight_arr.shape[0], num_pixels))

    # If full planes are flagged then set corresponding weights to zero. This
    # check is relative to the mask's own pixel count (1 if spatially
    # uniform), not the true `num_pixels` -- a uniform mask flagging a
    # channel means it's flagged for every real pixel, by definition.
    flag_xy_sum = np.sum(mask_arr, axis=1)
    mskPlanes = np.where(flag_xy_sum == mask_arr.shape[-1], 0, 1)
    weight_arr = weight_arr * mskPlanes[:, np.newaxis]

    # A pixel's RMSF depends on which channels it has flagged AND how it
    # weights the channels it keeps, so pixels can only share one RMSF if
    # both the flagging and the weighting are uniform across pixels. Real
    # per-channel flagging with per-channel (not per-pixel) weights is the
    # common case, so this stays O(1) in pixel count then -- bit-identical at
    # nthreads=1 (what the dask path uses), and at finufft's default thread
    # count within a few ulp, since it splits a multithreaded type-3
    # differently for one transform than for a batch.
    #
    # Both tests are one pass over their array and are exact. A chunk with
    # any per-pixel variation in either mask or weight fails and takes the
    # per-pixel path, so correctness never rests on `reuse_rmsf` -- the flag
    # can only decline a saving that is already known to be safe.
    mask_uniform = bool(np.array_equal(mask_arr.all(axis=1), mask_arr.any(axis=1)))
    weight_uniform = weight_arr.shape[-1] == 1 or bool(
        np.array_equal(weight_arr, np.broadcast_to(weight_arr[:, :1], weight_arr.shape))
    )
    share_rmsf = reuse_rmsf and num_pixels > 1 and mask_uniform and weight_uniform
    if share_rmsf:
        logger.debug(
            f"All {num_pixels} pixels share the same channel flagging and "
            "weighting; computing one RMSF instead of one per pixel."
        )
    rmsf_mask_arr = mask_arr[:, :1] if share_rmsf else mask_arr
    weight_for_cube = weight_arr[:, :1] if share_rmsf else weight_arr

    fwhm_rmsf_radm2, _, _ = get_fwhm_rmsf(lambda_sq_arr_m2)
    # Calculate the RMSF at each pixel
    # The K value used to scale each RMSF must take into account
    # isolated flagged voxels data in the datacube
    weight_cube = np.invert(rmsf_mask_arr) * weight_for_cube
    with np.errstate(divide="ignore", invalid="ignore"):
        scale_factor_arr = 1.0 / np.sum(weight_cube, axis=0)
        scale_factor_arr = zero_nonfinite(scale_factor_arr)

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
        n_fitted = rmsf_cube.shape[1]
        for i in trange(n_fitted, desc="Fitting RMSF by pixel", disable=n_fitted == 1):
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

        if share_rmsf:
            # One spectrum was fitted, and it is every pixel's RMSF.
            fwhm_rmsf_arr[:] = fwhm_rmsf_arr[0]
            fit_status_arr[:] = fit_status_arr[0]

    if share_rmsf:
        # Fan the one computed spectrum out to every pixel. `repeat`, not
        # `broadcast_to`: the reshape below would have to copy a zero-stride
        # view anyway, and callers get a normal writeable array either way.
        rmsf_cube = np.repeat(rmsf_cube, num_pixels, axis=1)

    # Remove redundant dimensions
    rmsf_cube = np.squeeze(rmsf_cube)
    fwhm_rmsf_arr = np.squeeze(fwhm_rmsf_arr)
    fit_status_arr = np.squeeze(fit_status_arr)

    # Restore if 3D shape -- either the mask or the weight array (or both)
    # carried the original spatial grid.
    if old_data_shape is not None:
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
        "phi_max_scale_radm2": pl.Float64,
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
            peak_pi_fit_snr = peak_pi_fit / float(theoretical_noise.fdf_error_noise)

    # In rare cases, a parabola can be fitted to the edge of the spectrum,
    # producing a unreasonably large RM and polarized intensity.
    # In these cases, everything should get NaN'd out.
    if np.abs(peak_rm_fit) > np.max(np.abs(phi_arr_radm2)):
        peak_rm_fit = np.nan
        peak_pi_fit = np.nan

    # Q and U at the fitted peak, for the polarisation angle
    peak_pi_fit_index = np.interp(
        peak_rm_fit, phi_arr_radm2, np.arange(phi_arr_radm2.shape[-1], dtype="f4")
    )
    peak_u_fit = np.interp(peak_rm_fit, phi_arr_radm2, fdf_arr.imag)
    peak_q_fit = np.interp(peak_rm_fit, phi_arr_radm2, fdf_arr.real)
    # Angles, debiasing and errors are the same for one sightline as for a whole
    # cube, so they come from the shared `calc_peak_stats`.
    peak_stats = calc_peak_stats(
        peak_pi=peak_pi_fit,
        peak_rm_radm2=peak_rm_fit,
        peak_fdf=peak_q_fit + 1j * peak_u_fit,
        fwhm_rmsf_radm2=fwhm_rmsf_radm2,
        fdf_error=theoretical_noise.fdf_error_noise,
        lam_sq_0_m2=lam_sq_0_m2,
        lambda_sq_arr_m2=lambda_sq_arr_m2,
        bias_correction_snr=bias_correction_snr,
    )
    peak_pi_fit_debias = float(peak_stats.peak_pi_debias)

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
        psi0_deg=float(peak_stats.peak_pa0_deg),
        rm_radm2=peak_rm_fit,
    )

    return fdf_params_schema_df.vstack(
        pl.DataFrame(
            {
                "fdf_error_mad": fdf_error_mad,
                "peak_pi_fit": peak_pi_fit,
                "peak_pi_error": float(peak_stats.peak_pi_error),
                "peak_pi_fit_debias": peak_pi_fit_debias,
                "peak_pi_fit_snr": peak_pi_fit_snr,
                "peak_pi_fit_index": int(peak_pi_fit_index)
                if np.isfinite(peak_pi_fit_index)
                else -1,
                "peak_rm_fit": peak_rm_fit,
                "peak_rm_fit_error": float(peak_stats.peak_rm_error_radm2),
                "peak_q_fit": peak_q_fit,
                "peak_u_fit": peak_u_fit,
                "peak_pa_fit_deg": float(peak_stats.peak_pa_deg),
                "peak_pa_fit_deg_error": float(peak_stats.peak_pa_error_deg),
                "peak_pa0_fit_deg": float(peak_stats.peak_pa0_deg),
                "peak_pa0_fit_deg_error": float(peak_stats.peak_pa0_error_deg),
                "fit_function": fit_function,
                "lam_sq_0_m2": lam_sq_0_m2,
                "ref_freq_hz": lambda2_to_freq(lam_sq_0_m2),
                "fwhm_rmsf_radm2": fwhm_rmsf_radm2,
                "phi_max_scale_radm2": float(np.pi / np.nanmin(lambda_sq_arr_m2)),
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
        lambda_sq_arr_m2=freq_to_lambda2(freq_arr_hz),
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
