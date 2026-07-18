"""Synthetic FDF data: the single source of truth for simulated RM-synthesis.

Sources are built in the CHANNEL domain from the physical Burn depolarisation
law (`build_channel_spectrum`): the polarised signal depolarises with lambda^2
about lambda^2 = 0, so |P| is bounded by the intrinsic fraction and the envelope
is anchored at zero wavelength, not at the RM-synthesis reference. Noise is then
injected per frequency channel and transformed by the package RM synthesis, so
the FDF noise is the physically correct correlated complex Gaussian field, not
iid noise painted onto phi.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np
from numpy.typing import NDArray

from rm_lite.utils.fitting import unit_centred_gaussian
from rm_lite.utils.synthesis import (
    compute_theoretical_noise,
    freq_to_lambda2,
    get_fwhm_rmsf,
    get_rmsf_nufft,
    inverse_rmsynth_nufft,
    make_phi_arr,
    rmsynth_nufft,
)

PHI_MAX_FWHM = 40.0  # phi window half-width in RMSF FWHM units (matches _probe)

ComponentKind = Literal["delta", "gauss", "slab"]


class Component(NamedTuple):
    """One source component (Burn model); widths and RM centre in RMSF FWHM units."""

    kind: ComponentKind
    width_fwhm: float = 0.0  # gauss Faraday sigma / slab full width; 0 for delta
    center_fwhm: float = 0.0  # RM (Faraday rotation) in FWHM units
    amp: float = 1.0  # intrinsic polarised fraction |P(lambda^2=0)|


class FDFSpec(NamedTuple):
    """A source as a sum of Burn components, each with |P(0)| = amp."""

    components: tuple[Component, ...]


def delta(center_fwhm: float = 0.0, amp: float = 1.0) -> FDFSpec:
    """Faraday-thin point at RM center_fwhm (flat |P| = amp)."""
    return FDFSpec((Component("delta", 0.0, center_fwhm, amp),))


def gauss(sigma_fwhm: float, center_fwhm: float = 0.0, amp: float = 1.0) -> FDFSpec:
    """External Faraday dispersion; Gaussian FDF of sigma (FWHM units)."""
    return FDFSpec((Component("gauss", sigma_fwhm, center_fwhm, amp),))


def slab(width_fwhm: float, center_fwhm: float = 0.0, amp: float = 1.0) -> FDFSpec:
    """Burn slab of full Faraday width (FWHM units)."""
    return FDFSpec((Component("slab", width_fwhm, center_fwhm, amp),))


def build_channel_spectrum(
    spec: FDFSpec, lambda_sq_arr_m2: NDArray[np.float64], fwhm: float
) -> NDArray[np.complex128]:
    """Channel Q + iU from the physical Burn law, with |P(lambda^2=0)| = amp.

    Each component depolarises with lambda^2 about lambda^2 = 0: a Faraday-thin
    point is flat, external dispersion is a Gaussian in lambda^2, a Burn slab a
    sinc.

    Args:
        spec (FDFSpec): Source as a sum of Burn components.
        lambda_sq_arr_m2 (NDArray[np.float64]): Channel lambda^2 in m^2.
        fwhm (float): RMSF FWHM in rad/m^2 (sets the FWHM-unit scale).

    Returns:
        NDArray[np.complex128]: Channel polarisation Q + iU.
    """
    pol = np.zeros_like(lambda_sq_arr_m2, dtype=np.complex128)
    for comp in spec.components:
        rotation = np.exp(2j * comp.center_fwhm * fwhm * lambda_sq_arr_m2)
        if comp.kind == "gauss":
            sigma_phi = comp.width_fwhm * fwhm
            envelope = np.exp(-2.0 * sigma_phi**2 * lambda_sq_arr_m2**2)
        elif comp.kind == "slab":
            delta_phi = comp.width_fwhm * fwhm
            envelope = np.sinc(delta_phi * lambda_sq_arr_m2 / np.pi)
        else:  # delta: Faraday-thin, no depolarisation
            envelope = np.ones_like(lambda_sq_arr_m2)
        pol += comp.amp * rotation * envelope
    return pol.astype(np.complex128)


def build_model_fdf(
    spec: FDFSpec, phi_arr_radm2: NDArray[np.float64], fwhm: float
) -> NDArray[np.complex128]:
    """Reference true-FDF shape on phi_arr (peak-normalised to amp), for plots only.

    Channel data comes from `build_channel_spectrum`, not from inverting this.

    Args:
        spec (FDFSpec): Source as a sum of Burn components.
        phi_arr_radm2 (NDArray[np.float64]): Faraday depth axis in rad/m^2.
        fwhm (float): RMSF FWHM in rad/m^2.

    Returns:
        NDArray[np.complex128]: Model FDF on `phi_arr_radm2`.
    """
    fdf = np.zeros_like(phi_arr_radm2, dtype=np.complex128)
    for comp in spec.components:
        center = comp.center_fwhm * fwhm
        if comp.kind == "delta":
            idx = int(np.argmin(np.abs(phi_arr_radm2 - center)))
            fdf[idx] += comp.amp
        elif comp.kind == "gauss":
            fdf += comp.amp * unit_centred_gaussian(
                phi_arr_radm2 - center, stddev=comp.width_fwhm * fwhm
            ).astype(np.complex128)
        else:  # slab
            half = comp.width_fwhm * fwhm / 2.0
            fdf += comp.amp * (np.abs(phi_arr_radm2 - center) <= half).astype(
                np.complex128
            )
    return fdf


def model_to_channel(
    model_fdf: NDArray[np.complex128],
    lambda_sq_arr_m2: NDArray[np.float64],
    phi_arr_radm2: NDArray[np.float64],
    lam_sq_0_m2: float,
) -> NDArray[np.complex128]:
    """Transform a model FDF (phi domain) to noise-free channel Q + iU.

    Args:
        model_fdf (NDArray[np.complex128]): Model FDF on the phi axis.
        lambda_sq_arr_m2 (NDArray[np.float64]): Channel lambda^2 in m^2.
        phi_arr_radm2 (NDArray[np.float64]): Faraday depth axis in rad/m^2.
        lam_sq_0_m2 (float): Reference lambda^2 in m^2.

    Returns:
        NDArray[np.complex128]: Noise-free channel Q + iU.
    """
    return inverse_rmsynth_nufft(
        complex_fdf_arr=model_fdf,
        lambda_sq_arr_m2=lambda_sq_arr_m2,
        phi_arr_radm2=phi_arr_radm2,
        lam_sq_0_m2=lam_sq_0_m2,
    )


class NoisyChannels(NamedTuple):
    """Channel Q + iU with per-channel complex noise added, and the error array."""

    complex_pol_arr: NDArray[np.complex128]
    complex_pol_error: NDArray[np.complex128]  # sigma in real and imag per channel


def add_channel_noise(
    complex_pol_arr: NDArray[np.complex128],
    sigma: float,
    rng: np.random.Generator,
) -> NoisyChannels:
    """Add iid complex Gaussian noise (std sigma in Q and in U) per channel.

    Args:
        complex_pol_arr (NDArray[np.complex128]): Noise-free channel Q + iU.
        sigma (float): Noise std per channel, applied to Q and U.
        rng (np.random.Generator): Random generator.

    Returns:
        NoisyChannels: Noisy channels and the per-channel error array.
    """
    n = complex_pol_arr.shape[0]
    noise = rng.normal(scale=sigma, size=n) + 1j * rng.normal(scale=sigma, size=n)
    error = np.full(n, sigma + 1j * sigma, dtype=np.complex128)
    return NoisyChannels((complex_pol_arr + noise).astype(np.complex128), error)


class Geometry(NamedTuple):
    """Coverage geometry: everything the forward model and RMSF need."""

    lambda_sq_arr_m2: NDArray[np.float64]
    weight_arr: NDArray[np.float64]
    lam_sq_0_m2: float
    fwhm: float
    phi_arr_radm2: NDArray[np.float64]
    phi_double_arr_radm2: NDArray[np.float64]
    rmsf_arr: NDArray[np.complex128]


def build_geometry(
    freq_arr_hz: NDArray[np.float64], phi_max_fwhm: float = PHI_MAX_FWHM
) -> Geometry:
    """Uniform-weight geometry and RMSF for a frequency coverage.

    Args:
        freq_arr_hz (NDArray[np.float64]): Channel frequencies in Hz.
        phi_max_fwhm (float): Phi window half-width in RMSF FWHM units.

    Returns:
        Geometry: lambda^2, weights, reference, FWHM, phi axes, and RMSF.
    """
    lam2 = freq_to_lambda2(freq_arr_hz)
    weight = np.ones_like(lam2)
    lam_sq_0 = float(np.average(lam2, weights=weight))
    fwhm = get_fwhm_rmsf(lam2).fwhm_rmsf_radm2
    phi_arr = make_phi_arr(phi_max_radm2=phi_max_fwhm * fwhm, d_phi_radm2=fwhm / 10.0)
    rmsf = get_rmsf_nufft(
        lambda_sq_arr_m2=lam2,
        phi_arr_radm2=phi_arr,
        weight_arr=weight,
        lam_sq_0_m2=lam_sq_0,
    )
    return Geometry(
        lambda_sq_arr_m2=lam2,
        weight_arr=weight,
        lam_sq_0_m2=lam_sq_0,
        fwhm=fwhm,
        phi_arr_radm2=phi_arr,
        phi_double_arr_radm2=rmsf.phi_double_arr_radm2,
        rmsf_arr=np.asarray(rmsf.rmsf_cube, dtype=np.complex128),
    )


class SimResult(NamedTuple):
    """A simulated dirty FDF plus the channel data and geometry behind it."""

    dirty_fdf: NDArray[np.complex128]
    rmsf_arr: NDArray[np.complex128]
    phi_arr_radm2: NDArray[np.float64]
    phi_double_arr_radm2: NDArray[np.float64]
    fwhm: float
    lam_sq_0_m2: float
    fdf_noise: float  # theoretical FDF noise (0 if noise-free)
    complex_pol_arr: NDArray[np.complex128]  # channel Q + iU (with any noise)
    lambda_sq_arr_m2: NDArray[np.float64]  # channel lambda^2


def simulate_fdf(
    spec: FDFSpec,
    freq_arr_hz: NDArray[np.float64],
    rng: np.random.Generator | None = None,
    sigma: float | None = None,
    sn: float | None = None,
    phi_max_fwhm: float = PHI_MAX_FWHM,
    geometry: Geometry | None = None,
) -> SimResult:
    """Forward-model a spec through channel space to a dirty FDF.

    Noise-free unless a channel `sigma` (or `sn`, an S/N against the model channel
    peak) is given. Pass a prebuilt `geometry` to reuse the RMSF.

    Args:
        spec (FDFSpec): Source as a sum of Burn components.
        freq_arr_hz (NDArray[np.float64]): Channel frequencies in Hz.
        rng (np.random.Generator | None): Random generator; required if `sigma` or `sn` is set.
        sigma (float | None): Per-channel noise std. Defaults to None.
        sn (float | None): Target S/N against the model channel peak (sets sigma). Defaults to None.
        phi_max_fwhm (float): Phi window half-width in RMSF FWHM units.
        geometry (Geometry | None): Prebuilt geometry to reuse; None builds one.

    Raises:
        ValueError: If noise is requested (`sigma` or `sn`) without `rng`.

    Returns:
        SimResult: Dirty FDF with the channel data and geometry behind it.
    """
    geom = (
        geometry if geometry is not None else build_geometry(freq_arr_hz, phi_max_fwhm)
    )
    pol = build_channel_spectrum(spec, geom.lambda_sq_arr_m2, geom.fwhm)

    error = np.zeros(pol.shape[0], dtype=np.complex128)
    if sigma is not None or sn is not None:
        if rng is None:
            msg = "rng is required when injecting noise."
            raise ValueError(msg)
        if sigma is None:
            assert sn is not None
            sigma = float(np.nanmax(np.abs(pol))) / sn
        noisy = add_channel_noise(pol, sigma, rng)
        pol, error = noisy.complex_pol_arr, noisy.complex_pol_error

    dirty = rmsynth_nufft(
        complex_pol_arr=pol,
        lambda_sq_arr_m2=geom.lambda_sq_arr_m2,
        phi_arr_radm2=geom.phi_arr_radm2,
        weight_arr=geom.weight_arr,
        lam_sq_0_m2=geom.lam_sq_0_m2,
    )
    fdf_noise = compute_theoretical_noise(error, geom.weight_arr).fdf_error_noise
    return SimResult(
        dirty_fdf=np.asarray(dirty, dtype=np.complex128),
        rmsf_arr=geom.rmsf_arr,
        phi_arr_radm2=geom.phi_arr_radm2,
        phi_double_arr_radm2=geom.phi_double_arr_radm2,
        fwhm=geom.fwhm,
        lam_sq_0_m2=geom.lam_sq_0_m2,
        fdf_noise=float(fdf_noise),
        complex_pol_arr=np.asarray(pol, dtype=np.complex128),
        lambda_sq_arr_m2=geom.lambda_sq_arr_m2,
    )


def demo() -> None:
    """Self-check: noise-free peak ~ delta amp; FDF noise positive, scales with sigma."""
    freq = np.linspace(744e6, 1032e6, 288)
    rng = np.random.default_rng(0)

    clean = simulate_fdf(delta(), freq)
    peak = float(np.nanmax(np.abs(clean.dirty_fdf)))
    assert 0.99 <= peak <= 1.01, f"noise-free delta peak {peak} != 1"
    assert clean.fdf_noise == 0.0, "noise-free result must report zero FDF noise"

    # Extended sources obey the physical law: |P| <= amp, and the Gaussian
    # depolarisation envelope is anchored at lambda^2 = 0 (monotone across a band
    # of positive lambda^2), not centred on the RM-synthesis reference.
    geom = build_geometry(freq)
    for spec in (gauss(4.0, amp=0.7), slab(4.0, amp=0.7)):
        pol = build_channel_spectrum(spec, geom.lambda_sq_arr_m2, geom.fwhm)
        assert np.abs(pol).max() <= 0.7 + 1e-6, f"channel |P| {np.abs(pol).max()} > amp"
    order = np.argsort(geom.lambda_sq_arr_m2)
    gpol = build_channel_spectrum(gauss(4.0, amp=0.7), geom.lambda_sq_arr_m2, geom.fwhm)
    assert np.all(np.diff(np.abs(gpol)[order]) <= 1e-9), "gauss envelope not monotone"

    n1 = simulate_fdf(delta(), freq, rng=rng, sigma=0.1).fdf_noise
    n2 = simulate_fdf(delta(), freq, rng=rng, sigma=0.2).fdf_noise
    assert n1 > 0, "noisy FDF noise must be positive"
    assert abs(n2 / n1 - 2.0) < 1e-6, f"FDF noise must scale with sigma: {n2 / n1}"

    print("simulate.demo OK")  # noqa: T201


if __name__ == "__main__":
    demo()
