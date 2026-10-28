from __future__ import annotations

import numpy as np
import pytest
from rm_lite.utils.arrays import (
    arange,
    nb_median,
    nb_nanargmax_abs,
    nb_partition,
    nd_to_two_d,
    two_d_to_nd,
)


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


@pytest.mark.parametrize("order", ["random", "sorted", "reversed", "smooth", "tied"])
@pytest.mark.parametrize("size", [1, 2, 7, 600, 601, 2302, 8191])
def test_nb_median_matches_numpy(order: str, size: int) -> None:
    rng = np.random.default_rng(size)
    values = {
        "random": rng.normal(size=size),
        "sorted": np.sort(rng.normal(size=size)),
        "reversed": np.sort(rng.normal(size=size))[::-1].copy(),
        # Sidelobe-like ringing, which numba's own median is quadratic on
        "smooth": np.sinc(np.linspace(-20, 20, size)),
        "tied": np.round(rng.normal(size=size), 1),
    }[order]
    assert nb_median(values) == np.median(values)
    assert nb_median(values.astype(np.float32)) == np.median(values.astype(np.float32))


def test_nb_median_of_nan_and_empty() -> None:
    assert np.isnan(nb_median(np.array([1.0, np.nan, 2.0])))
    assert np.isnan(nb_median(np.array([], dtype=float)))


def test_nb_partition_matches_numpy() -> None:
    rng = np.random.default_rng(0)
    for size in (5, 700, 5000):
        values = rng.normal(size=size)
        for kth in (0, size // 3, size - 1):
            partitioned = values.copy()
            nb_partition(partitioned, kth)
            assert partitioned[kth] == np.partition(values, kth)[kth]
            assert (partitioned[:kth] <= partitioned[kth]).all()
            assert (partitioned[kth + 1 :] >= partitioned[kth]).all()


def test_nb_nanargmax_abs_matches_numpy() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(size=(3, 100)) + 1j * rng.normal(size=(3, 100))
    values[0, [3, 50]] = np.nan
    values[2] = np.nan
    expected = [*np.nanargmax(np.abs(values[:2]), axis=-1), -1]
    np.testing.assert_array_equal(nb_nanargmax_abs(values), expected)
    assert nb_nanargmax_abs(values[1]).tolist() == [expected[1]]
    # Ties go to the first, as np.argmax
    assert nb_nanargmax_abs(np.tile(values[1], 3)).tolist() == [expected[1]]
