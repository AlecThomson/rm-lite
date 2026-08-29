"""Array utils"""

from __future__ import annotations

import logging
from typing import Any, TypeVar, cast

import dask.array as da
import numpy as np
from numpy.typing import NDArray

from rm_lite.utils.logging import TqdmToLogger, logger

TQDM_OUT = TqdmToLogger(logger, level=logging.INFO)

DType = TypeVar("DType", bound=np.generic)
ArrayT = TypeVar("ArrayT", bound=NDArray[np.floating] | da.Array)


def zero_nonfinite(arr: ArrayT) -> ArrayT:
    """Replace nan/+inf/-inf with 0.0, working for both numpy and dask arrays.

    `np.nan_to_num`'s `nan=`/`posinf=`/`neginf=` kwargs aren't supported by
    dask's implementation, so use `isfinite` + `where` instead -- these
    dispatch identically for numpy and dask inputs.
    """
    return cast(ArrayT, np.where(np.isfinite(arr), arr, 0.0))


def broadcast_over_channels(
    arr_1d: NDArray[DType], target: NDArray[Any]
) -> NDArray[DType]:
    """Reshape a per-channel 1D array to broadcast against `target`.

    `target` has frequency/channel as its leading axis and may carry extra
    trailing spatial axes, e.g. `(nchan,)` or `(nchan, ny, nx)`. Right-aligned
    numpy broadcasting can't match a 1D array against a leading axis on its
    own, so pad with trailing singleton axes when `target` has more than one
    dimension.
    """
    if target.ndim == 1:
        return arr_1d
    return arr_1d.reshape(arr_1d.shape[0], *([1] * (target.ndim - 1)))


def float_if_scalar(value: Any) -> float | NDArray[np.float64] | da.Array:
    """A plain float for a 0-d value; anything else is left alone, and lazy."""
    if np.ndim(value) == 0:
        return float(value)
    return cast("NDArray[np.float64] | da.Array", value)


def format_scalar_or_map(value: float | NDArray[np.float64] | da.Array) -> str:
    """Log-friendly string for a scalar, or the finite range of a map."""
    if np.ndim(value) == 0:
        return f"{float(value):0.3g}"
    values = np.asarray(value)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "all non-finite"
    return f"{finite.min():0.3g} to {finite.max():0.3g} per pixel"


def nd_to_two_d(arr: NDArray[DType]) -> NDArray[DType]:
    """Convert an array to 2D.

    - If arr is 1D, it will be reshaped as a column vector (shape: (N, 1)).
    - If arr is already 2D, it is returned as is.
    - If arr has more than 2 dimensions, the first axis is kept intact
      and all remaining axes are flattened. For example, an array with
      shape (a, b, c, d) will become shape (a, b*c*d).

    Args:
        arr (NDArray[Any]): N-dimensional array.

    Returns:
        NDArray[Any]: 2D array.
    """
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        return arr
    return arr.reshape(arr.shape[0], -1)


def two_d_to_nd(
    arr2d: NDArray[DType], original_shape: tuple[int, ...]
) -> NDArray[DType]:
    """
    Reverse the to_2d operation.

    Parameters:
        arr2d (array-like): the 2D array (result from to_2d).
        original_shape (tuple): the shape of the original array before flattening.

    Returns:
        The array reshaped back to its original shape.

    The function assumes:
        - For an original 1D array, original_shape is (N,). In this case, arr2d is of shape (N, 1)
          and will be flattened back to (N,).
        - For an original 2D array, original_shape is (M, N) and arr2d is already (M, N).
        - For an original N_D array (N_D > 2) with shape (a, b, c, ...), arr2d is assumed to have shape
          (a, b*c*...) and will be reshaped back to (a, b, c, ...).
    """
    arr2d = np.asarray(arr2d)
    # If the original was 1D, simply flatten the second axis.
    if len(original_shape) == 1:
        # (N,1) -> (N,)
        return arr2d.ravel()
    # If the original was 2D, reshape directly.
    if len(original_shape) == 2:
        return arr2d.reshape(original_shape)

    # For N_D arrays (with ndim > 2), the to_2d function preserved the first axis
    # and flattened the remaining dimensions.
    expected_first_dim = original_shape[0]
    expected_rest = int(np.prod(original_shape[1:]))
    if arr2d.shape != (expected_first_dim, expected_rest):
        msg = "The provided original shape is not consistent with the 2D array shape."
        raise ValueError(msg)
    return arr2d.reshape(original_shape)


# from https://stackoverflow.com/questions/50299172/range-or-numpy-arange-with-end-limit-include
def arange(
    start: float | int,
    stop: float | int,
    step: float | int,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    include_start: bool = True,
    include_stop: bool = False,
    **kwargs: Any,
) -> NDArray[np.float64]:
    """
    Combines numpy.arange and numpy.isclose to mimic open, half-open and closed intervals.

    Avoids also floating point rounding errors as with
    >>> np.arange(1, 1.3, 0.1)
    array([1., 1.1, 1.2, 1.3])


    Args:
        start (float | int): Start of the interval.
        stop (float | int): End of the interval.
        step (float | int): Spacing between values.
        rtol (float, optional): if last element of array is within this relative tolerance to stop and include[0]==False, it is skipped. Defaults to 1e-05.
        atol (float, optional): if last element of array is within this relative tolerance to stop and include[1]==False, it is skipped. Defaults to 1e-08.
        include_start (bool, optional): if first element is included in the returned array. Defaults to True.
        include_stop (bool, optional): if last elements are included in the returned array if stop equals last element. Defaults to False.
        kwargs: passed to np.arange

    Returns:
        _type_: as np.arange but eventually with first and last element stripped/added
    """
    arr = np.arange(start, stop, step, **kwargs)
    if not include_start:
        arr = np.delete(arr, 0)

    if include_stop:
        if np.isclose(arr[-1] + step, stop, rtol=rtol, atol=atol):
            arr = np.append(arr, arr[-1] + step)
    elif np.isclose(arr[-1], stop, rtol=rtol, atol=atol):
        arr = np.delete(arr, -1)
    return arr
