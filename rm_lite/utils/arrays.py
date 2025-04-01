"""Array utils"""

from __future__ import annotations

import logging
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from rm_lite.utils.logging import TqdmToLogger, logger

TQDM_OUT = TqdmToLogger(logger, level=logging.INFO)

DType = TypeVar("DType", bound=np.generic)


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
    **kwargs,
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
            # arr = np.c_[arr, arr[-1] + step]
            arr = np.append(arr, arr[-1] + step)
    elif np.isclose(arr[-1], stop, rtol=rtol, atol=atol):
        arr = np.delete(arr, -1)
    return arr
