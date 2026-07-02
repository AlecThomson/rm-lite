from __future__ import annotations

import numpy as np
import pytest
from rm_lite.utils.arrays import arange, nd_to_two_d, two_d_to_nd


def test_nd_to_two_d():
    array_1d = np.arange(288)
    array_2d = array_1d.reshape(36, 8)
    array_3d = array_1d.reshape(36, 2, 4)
    array_4d = array_1d.reshape(36, 2, 2, 2)

    # test shapes
    # First axis is kept intact
    assert nd_to_two_d(array_1d).shape == (288, 1)
    assert nd_to_two_d(array_2d).shape == (36, 8)
    assert nd_to_two_d(array_3d).shape == (36, 8)
    assert nd_to_two_d(array_4d).shape == (36, 8)


def test_two_d_to_nd():
    array_1d = np.arange(12)
    array_2d = array_1d.reshape(3, 4)
    array_3d = array_1d.reshape(2, 3, 2)

    assert two_d_to_nd(nd_to_two_d(array_1d), array_1d.shape).shape == array_1d.shape
    assert two_d_to_nd(nd_to_two_d(array_2d), array_2d.shape).shape == array_2d.shape
    assert two_d_to_nd(nd_to_two_d(array_3d), array_3d.shape).shape == array_3d.shape

    # test round trip
    assert np.array_equal(two_d_to_nd(nd_to_two_d(array_1d), array_1d.shape), array_1d)
    assert np.array_equal(two_d_to_nd(nd_to_two_d(array_2d), array_2d.shape), array_2d)
    assert np.array_equal(two_d_to_nd(nd_to_two_d(array_3d), array_3d.shape), array_3d)


@pytest.mark.parametrize(
    ("start", "stop", "step", "include_start", "include_stop", "res_exp"),
    [
        pytest.param(
            0, 7, 1, True, False, np.array([0, 1, 2, 3, 4, 5, 6]), id="arange simple"
        ),
        pytest.param(
            0,
            6.5,
            1,
            True,
            False,
            np.array([0, 1, 2, 3, 4, 5, 6]),
            id="stop not on grid",
        ),
        pytest.param(
            1, 1.3, 0.1, True, False, np.array([1.0, 1.1, 1.2]), id="stop excl"
        ),
        pytest.param(
            1, 1.3, 0.1, True, True, np.array([1.0, 1.1, 1.2, 1.3]), id="stop incl"
        ),
        pytest.param(
            1, 1.3, 0.1, False, False, np.array([1.1, 1.2]), id="stop excl + start excl"
        ),
        pytest.param(
            1,
            1.3,
            0.1,
            False,
            True,
            np.array([1.1, 1.2, 1.3]),
            id="stop incl + start excl",
        ),
    ],
)
def test_arange(
    start: float,
    stop: float,
    step: float,
    include_start: bool,
    include_stop: bool,
    res_exp: np.ndarray,
):
    res = arange(
        start, stop, step, include_start=include_start, include_stop=include_stop
    )
    assert np.allclose(res, res_exp), f"Unexpected result: {res=}, {res_exp=}"
