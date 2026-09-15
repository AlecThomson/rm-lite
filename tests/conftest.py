"""Fixtures shared by more than one test module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import dask.array as da
import pytest
from astropy.io import fits

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import numpy as np
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
