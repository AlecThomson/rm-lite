"""Physical Burn channel spectra: rad/m^2 units with a polarisation angle."""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

ComponentKind = Literal["delta", "gauss", "slab", "turbulent"]


def _burn_envelope(
    kind: ComponentKind,
    lambda_sq_arr_m2: NDArray[np.float64],
    width_radm2: float,
    sigma_rm_radm2: float = 0.0,
) -> NDArray[np.complex128]:
    """Burn depolarisation factor in physical rad/m^2 units (no rotation, no amp).

    `width_radm2` is the Gaussian sigma (gauss), the full Faraday depth (slab and
    turbulent), or unused (delta); `sigma_rm_radm2` is the turbulent RM scatter.
    A Faraday-thin point returns unity.
    """
    if kind == "gauss":
        env: NDArray[np.float64] | NDArray[np.complex128] = np.exp(
            -2.0 * width_radm2**2 * lambda_sq_arr_m2**2
        )
    elif kind == "slab":
        env = np.sinc(width_radm2 * lambda_sq_arr_m2 / np.pi)
    elif kind == "turbulent":
        # Sokoloff et al. (1998) eq. 34, symmetrised about the component centre
        # (the e^{-i depth lam^2} factor) to match the slab convention.
        s_arr = (
            2.0 * sigma_rm_radm2**2 * lambda_sq_arr_m2**2
            - 2.0j * width_radm2 * lambda_sq_arr_m2
        )
        small = np.abs(s_arr) < 1e-12
        s_safe = np.where(small, 1.0, s_arr)
        env = np.where(small, 1.0, (1.0 - np.exp(-s_safe)) / s_safe) * np.exp(
            -1j * width_radm2 * lambda_sq_arr_m2
        )
    else:  # delta: Faraday-thin, no depolarisation
        env = np.ones_like(lambda_sq_arr_m2)
    return env.astype(np.complex128)


def _rotation(
    lambda_sq_arr_m2: NDArray[np.float64], psi0_deg: float, rm_radm2: float
) -> NDArray[np.complex128]:
    """Faraday rotation phase e^{2i(psi0 + RM lambda^2)}."""
    return cast(
        NDArray[np.complex128],
        np.exp(2j * (np.deg2rad(psi0_deg) + rm_radm2 * lambda_sq_arr_m2)).astype(
            np.complex128
        ),
    )


def faraday_simple_spectrum(
    lambda_sq_arr_m2: NDArray[np.float64],
    frac_pol: float,
    psi0_deg: float,
    rm_radm2: float,
) -> NDArray[np.complex128]:
    """Faraday-thin channel Q + iU: a single RM, flat polarised fraction.

    Args:
        lambda_sq_arr_m2 (NDArray[np.float64]): Channel lambda^2 in m^2.
        frac_pol (float): Polarised fraction.
        psi0_deg (float): Intrinsic polarisation angle in degrees.
        rm_radm2 (float): RM in rad/m^2.

    Returns:
        NDArray[np.complex128]: Channel Q + iU.
    """
    return (frac_pol * _rotation(lambda_sq_arr_m2, psi0_deg, rm_radm2)).astype(
        np.complex128
    )


def faraday_slab_spectrum(
    lambda_sq_arr_m2: NDArray[np.float64],
    frac_pol: float,
    psi0_deg: float,
    rm_radm2: float,
    delta_rm_radm2: float,
) -> NDArray[np.complex128]:
    """Burn slab channel Q + iU: Faraday-thick, full thickness delta_rm_radm2.

    Args:
        lambda_sq_arr_m2 (NDArray[np.float64]): Channel lambda^2 in m^2.
        frac_pol (float): Polarised fraction.
        psi0_deg (float): Intrinsic polarisation angle in degrees.
        rm_radm2 (float): Central RM in rad/m^2.
        delta_rm_radm2 (float): Full Faraday thickness in rad/m^2.

    Returns:
        NDArray[np.complex128]: Channel Q + iU.
    """
    rotation = _rotation(lambda_sq_arr_m2, psi0_deg, rm_radm2)
    envelope = _burn_envelope("slab", lambda_sq_arr_m2, delta_rm_radm2)
    return (frac_pol * rotation * envelope).astype(np.complex128)


def faraday_gaussian_spectrum(
    lambda_sq_arr_m2: NDArray[np.float64],
    frac_pol: float,
    psi0_deg: float,
    rm_radm2: float,
    sigma_rm_radm2: float,
) -> NDArray[np.complex128]:
    """External-dispersion channel Q + iU: Gaussian RM scatter sigma_rm_radm2.

    Args:
        lambda_sq_arr_m2 (NDArray[np.float64]): Channel lambda^2 in m^2.
        frac_pol (float): Polarised fraction.
        psi0_deg (float): Intrinsic polarisation angle in degrees.
        rm_radm2 (float): Central RM in rad/m^2.
        sigma_rm_radm2 (float): Faraday dispersion sigma in rad/m^2.

    Returns:
        NDArray[np.complex128]: Channel Q + iU.
    """
    rotation = _rotation(lambda_sq_arr_m2, psi0_deg, rm_radm2)
    envelope = _burn_envelope("gauss", lambda_sq_arr_m2, sigma_rm_radm2)
    return (frac_pol * rotation * envelope).astype(np.complex128)
