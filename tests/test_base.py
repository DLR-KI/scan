"""Base class for the pytest testing"""

from __future__ import annotations

import unittest
from typing import Optional

import numpy as np
import pytest


def assert_array_equal(actual: np.ndarray, desired: np.ndarray) -> None:
    """Generalization of np.testing.assert_equal

    Raises an AssertionError if the two objects are NOT equal.

    Right now only numpy arrays are supported, but the idea is that it's going to take more in the future, once needed

    Args:
        actual :  The object to check
        desired : The expected object

    Returns: None. Just raises an exception if actual and desired are not equal.

    """
    np.testing.assert_equal(actual, desired)


def assert_array_not_equal(actual: np.ndarray, desired: np.ndarray) -> None:
    """Invertion of assert_array_equal

    Raises an AssertionError if the two objects ARE equal.

    Right now only numpy arrays are supported, but the idea is that it's going to take more in the future, once needed

    Args:
        actual :  The object to check
        desired : The expected object

    Returns: None. Just raises an exception if actual and desired ARE equal.

    """
    try:
        assert_array_equal(actual, desired)
    except AssertionError:
        assert True
        return None
    assert False


def assert_array_almost_equal(actual: np.ndarray, desired: np.ndarray, decimal: Optional[int] = 11) -> None:
    """Generalization of np.testing.assert_almost_equal

    Raises an AssertionError if the two objects are not equal up to the desired precision.

    Right now only numpy arrays are supported, but the idea is that it's going to take more in the future, once needed

    Args:
        actual :  The object to check
        desired : The expected object
        decimal : Desired precision, default is 11

    Returns: None. Just raises an exception if actual and desired are not equal up to the desired precision.

    """
    np.testing.assert_almost_equal(actual, desired, decimal)


class TestScanBase(unittest.TestCase):
    def reset(self) -> None:
        self.tearDown()
        self.setUp()

    def set_seed(self, seed: int = 0) -> None:
        self.seed = seed
        np.random.seed(self.seed)


class TestHelperFunctions(TestScanBase):
    def test_assert_array_equal(self):
        # NOTE: Quick and dirty testing. If generalize this function for more complicated processes in the future, clean
        #  and expand theses tests
        a = np.array([1])
        b = np.array([2])
        c = np.array([1, 2])
        d = np.array([1.0])

        assert_array_equal(a, a)

        with pytest.raises(AssertionError):
            assert_array_equal(a, b)

        with pytest.raises(AssertionError):
            assert_array_equal(a, c)

        assert_array_equal(a, d)

    def test_assert_array_not_equal(self):
        # NOTE: Quick and dirty testing. If generalize this function for more complicated processes in the future, clean
        #  and expand theses tests
        a = np.array([1])
        b = np.array([2])
        c = np.array([1, 2])
        d = np.array([1.0])

        with pytest.raises(AssertionError):
            assert_array_not_equal(a, a)

        assert_array_not_equal(a, b)

        assert_array_not_equal(a, c)

        with pytest.raises(AssertionError):
            assert_array_not_equal(a, d)

    def test_assert_array_almost_equal(self):
        # NOTE: Quick and dirty testing. If generalize this function for more complicated processes in the future, clean
        #  and expand theses tests
        a = np.array([1.0])
        b = np.array([2])
        c = np.array([1, 2])
        d = np.array([1.0])
        e = np.array([1.0 + 1e-11])
        f = np.array([1.0 + 1e-10])

        assert_array_almost_equal(a, a)

        with pytest.raises(AssertionError):
            assert_array_almost_equal(a, b)

        with pytest.raises(AssertionError):
            assert_array_almost_equal(a, c)

        assert_array_almost_equal(a, d)

        assert_array_almost_equal(a, e)

        with pytest.raises(AssertionError):
            assert_array_almost_equal(a, f)
