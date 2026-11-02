"""Fixtures shared by more than one test module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import dask.array as da
import numpy as np
import pytest
from astropy.io import fits
from rm_lite.utils import fitting

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from numpy.typing import NDArray


@pytest.fixture(scope="session")
def chunked() -> Callable[..., da.Array]:
    """Chunk a cube over its spatial axes, keeping frequency whole."""

    def make(array: NDArray[np.floating], cy: int = 2, cx: int = 2) -> da.Array:
        return da.from_array(array, chunks=(-1, cy, cx))

    return make


@pytest.fixture(scope="session")
def fits_cube() -> Callable[..., Path]:
    """Write a cube to FITS as big-endian float32, with a FREQ WCS if given."""

    def write(
        path: Path,
        data: NDArray[np.floating],
        freq_arr_hz: NDArray[np.float64] | None = None,
    ) -> Path:
        header = None
        if freq_arr_hz is not None:
            header = fits.Header()
            header["CTYPE3"] = "FREQ"
            header["CRVAL3"] = float(freq_arr_hz[0])
            header["CDELT3"] = float(freq_arr_hz[1] - freq_arr_hz[0])
            header["CRPIX3"] = 1
            header["CUNIT3"] = "Hz"
        fits.PrimaryHDU(data.astype(">f4"), header=header).writeto(path, overwrite=True)
        return path

    return write


@pytest.fixture(scope="session")
def scipy_fits_only() -> Callable[[pytest.MonkeyPatch], None]:
    """Patch `_lm_fit` never to converge, so every Python fit takes scipy's
    path. The compiled block fit runs its own `_lm_fit`, which this leaves."""

    def unconverged(
        _x_arr: NDArray[np.float64],
        _stokes_i_arr: NDArray[np.float64],
        _sigma_arr: NDArray[np.float64],
        initial_guess: NDArray[np.float64],
        _settings: fitting._ModelSettings,
    ) -> fitting._LMFit:
        n_param = initial_guess.size
        return fitting._LMFit(initial_guess, np.full((n_param, n_param), np.nan), False)

    def patch(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(fitting, "_lm_fit", unconverged)

    return patch


@pytest.fixture(scope="session")
def python_block_fit() -> Callable[[pytest.MonkeyPatch], None]:
    """Patch the compiled block fit to leave every pixel to the per-pixel
    Python fit."""

    def all_left_to_python(
        spectra: fitting.StokesISpectra, options: fitting.SpectraFitOptions
    ) -> fitting.SpectraFit:
        n_spec, n_freq = spectra.i_rows.shape
        fit = fitting._empty_spectra_fit(n_spec, n_freq, abs(options.fit_order) + 1)
        fit.status[:] = fitting.FitStatus.NEEDS_SCIPY
        return fit

    def patch(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(fitting, "fit_stokes_i_spectra", all_left_to_python)

    return patch
