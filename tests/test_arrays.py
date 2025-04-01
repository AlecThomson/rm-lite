from __future__ import annotations

import numpy as np
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


def test_arange():
    paras_minimal_working_example = {
        "arange simple": {
            "start": 0,
            "stop": 7,
            "step": 1,
            "include_start": True,
            "include_stop": False,
            "res_exp": np.array([0, 1, 2, 3, 4, 5, 6]),
        },
        "stop not on grid": {
            "start": 0,
            "stop": 6.5,
            "step": 1,
            "include_start": True,
            "include_stop": False,
            "res_exp": np.array([0, 1, 2, 3, 4, 5, 6]),
        },
        "arange failing example: stop excl": {
            "start": 1,
            "stop": 1.3,
            "step": 0.1,
            "include_start": True,
            "include_stop": False,
            "res_exp": np.array([1.0, 1.1, 1.2]),
        },
        "arange failing example: stop incl": {
            "start": 1,
            "stop": 1.3,
            "step": 0.1,
            "include_start": True,
            "include_stop": True,
            "res_exp": np.array([1.0, 1.1, 1.2, 1.3]),
        },
        "arange failing example: stop excl + start excl": {
            "start": 1,
            "stop": 1.3,
            "step": 0.1,
            "include_start": False,
            "include_stop": False,
            "res_exp": np.array([1.1, 1.2]),
        },
        "arange failing example: stop incl + start excl": {
            "start": 1,
            "stop": 1.3,
            "step": 0.1,
            "include_start": False,
            "include_stop": True,
            "res_exp": np.array([1.1, 1.2, 1.3]),
        },
    }
    for desc, paras in paras_minimal_working_example.items():
        start, stop, step, include_start, include_stop, res_exp = paras.values()
        res = arange(
            start, stop, step, include_start=include_start, include_stop=include_stop
        )
        assert np.allclose(res, res_exp), (
            f"Unexpected result in {desc}: {res=}, {res_exp=}"
        )
