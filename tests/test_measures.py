"""Tests if the scan.measures module works as it should"""

import unittest

import numpy as np
import pytest

from scan import measures
from tests.test_base import assert_array_almost_equal, assert_array_equal


class TestMeasures(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def tearDown(self):
        np.random.seed(None)

    def test_rmse(self):
        time_steps = np.random.randint(10, 100)
        dim = np.random.randint(10, 100)

        pred = np.random.random((time_steps, dim))
        meas = np.random.random((time_steps, dim))

        rmse_desired = np.sqrt(((pred - meas) ** 2).sum() / meas.shape[0])

        rmse = measures.rmse(pred, meas)

        assert_array_almost_equal(rmse, rmse_desired)

    def test_rmse_3d_slices_copied_slices(self):
        # time_steps = 3
        # dim = 2
        # slices = 4

        time_steps = np.random.randint(10, 100)
        dim = np.random.randint(10, 100)
        slices = np.random.randint(10, 100)

        # pred = np.array([[i for i in range(time_steps)] for j in range(dim)]).T
        # meas = np.zeros((time_steps, dim))

        pred_0slice = np.random.random((time_steps, dim))
        meas_0slice = np.random.random((time_steps, dim))

        pred = np.repeat(pred_0slice[:, :, np.newaxis], slices, axis=2)
        meas = np.repeat(meas_0slice[:, :, np.newaxis], slices, axis=2)

        rmse_desired = measures.rmse(pred_0slice, meas_0slice)

        rmse = measures.rmse(pred, meas)

        assert_array_almost_equal(rmse, rmse_desired)

    def test_rmse_3d_slices_and_norm_equivalences(self):
        # time_steps = 3
        # dim = 2
        # slices = 4
        # pred_unwrapped = np.array([[i for i in range(time_steps * slices)] for j in range(dim)]).T
        # meas_unwrapped = np.zeros((time_steps * slices, dim))

        time_steps = np.random.randint(10, 100)
        dim = np.random.randint(10, 100)
        slices = np.random.randint(10, 100)
        pred_unwrapped = np.random.random((time_steps * slices, dim))
        meas_unwrapped = np.random.random((time_steps * slices, dim))

        pred = np.reshape(pred_unwrapped.swapaxes(0, 1), newshape=(dim, time_steps, slices), order="F").swapaxes(0, 1)
        meas = np.reshape(meas_unwrapped.swapaxes(0, 1), newshape=(dim, time_steps, slices), order="F").swapaxes(0, 1)

        rmse_desired = np.sqrt(((pred_unwrapped - meas_unwrapped) ** 2).sum() / meas_unwrapped.shape[0])

        rmse = measures.rmse(pred, meas)

        # fmt: off
        rmse_alt01 = np.linalg.norm(pred_unwrapped - meas_unwrapped) / np.sqrt(meas_unwrapped.shape[0])
        rmse_alt02 = np.linalg.norm(np.linalg.norm(pred_unwrapped - meas_unwrapped, axis=0)) / np.sqrt(meas_unwrapped.shape[0])
        rmse_alt03 = np.linalg.norm(np.linalg.norm(pred_unwrapped - meas_unwrapped, axis=1)) / np.sqrt(meas_unwrapped.shape[0])

        rmse_alt04 = np.linalg.norm(pred - meas) / np.sqrt(meas.shape[0] * meas.shape[2])
        rmse_alt05 = np.linalg.norm(np.linalg.norm(pred - meas, axis=(0, 1))) / np.sqrt(meas.shape[0] * meas.shape[2])
        rmse_alt06 = np.linalg.norm(np.linalg.norm(np.linalg.norm(pred - meas, axis=0), axis=0)) / np.sqrt(meas.shape[0] * meas.shape[2])
        rmse_alt07 = np.linalg.norm(np.linalg.norm(np.linalg.norm(pred - meas, axis=0), axis=1)) / np.sqrt(meas.shape[0] * meas.shape[2])
        rmse_alt08 = np.linalg.norm(np.linalg.norm(np.linalg.norm(pred - meas, axis=1), axis=0)) / np.sqrt(meas.shape[0] * meas.shape[2])
        rmse_alt09 = np.linalg.norm(np.linalg.norm(np.linalg.norm(pred - meas, axis=1), axis=1)) / np.sqrt(meas.shape[0] * meas.shape[2])
        rmse_alt10 = np.linalg.norm(np.linalg.norm(np.linalg.norm(pred - meas, axis=2), axis=0)) / np.sqrt(meas.shape[0] * meas.shape[2])
        rmse_alt11 = np.linalg.norm(np.linalg.norm(np.linalg.norm(pred - meas, axis=2), axis=1)) / np.sqrt(meas.shape[0] * meas.shape[2])
        # fmt: on

        assert_array_almost_equal(rmse, rmse_desired)
        assert_array_almost_equal(rmse_alt01, rmse_desired)
        assert_array_almost_equal(rmse_alt02, rmse_desired)
        assert_array_almost_equal(rmse_alt03, rmse_desired)
        assert_array_almost_equal(rmse_alt04, rmse_desired)
        assert_array_almost_equal(rmse_alt05, rmse_desired)
        assert_array_almost_equal(rmse_alt06, rmse_desired)
        assert_array_almost_equal(rmse_alt07, rmse_desired)
        assert_array_almost_equal(rmse_alt08, rmse_desired)
        assert_array_almost_equal(rmse_alt09, rmse_desired)
        assert_array_almost_equal(rmse_alt10, rmse_desired)
        assert_array_almost_equal(rmse_alt11, rmse_desired)

    def test_rmse_normalization_mean(self):
        time_steps = np.random.randint(10, 100)
        dim = np.random.randint(10, 100)

        pred = np.random.random((time_steps, dim))
        meas = np.random.random((time_steps, dim))

        nrmse_desired = np.sqrt(((pred - meas) ** 2).sum() / meas.shape[0]) / np.mean(meas)

        nrmse = measures.rmse(pred, meas, normalization="mean")

        assert_array_almost_equal(nrmse, nrmse_desired)

    def test_rmse_normalization_std_over_time(self):
        time_steps = np.random.randint(10, 100)
        dim = np.random.randint(10, 100)

        pred = np.random.random((time_steps, dim))
        meas = np.random.random((time_steps, dim))

        std = np.std(meas, axis=0)
        mean_std = np.mean(std)
        nrmse_desired = np.sqrt(((pred - meas) ** 2).sum() / meas.shape[0]) / mean_std

        nrmse = measures.rmse(pred, meas, normalization="std_over_time")

        assert_array_almost_equal(nrmse, nrmse_desired)

    def test_rmse_normalization_2norm(self):
        time_steps = np.random.randint(10, 100)
        dim = np.random.randint(10, 100)

        pred = np.random.random((time_steps, dim))
        meas = np.random.random((time_steps, dim))

        nrmse_desired = np.sqrt(((pred - meas) ** 2).sum() / meas.shape[0]) / np.linalg.norm(meas)

        nrmse = measures.rmse(pred, meas, normalization="2norm")

        assert_array_almost_equal(nrmse, nrmse_desired)

    def test_rmse_normalization_maxmin(self):
        time_steps = np.random.randint(10, 100)
        dim = np.random.randint(10, 100)

        pred = np.random.random((time_steps, dim))
        meas = np.random.random((time_steps, dim))

        nrmse_desired = np.sqrt(((pred - meas) ** 2).sum() / meas.shape[0]) / (np.max(meas) - np.min(meas))

        nrmse = measures.rmse(pred, meas, normalization="maxmin")

        assert_array_almost_equal(nrmse, nrmse_desired)

    def test_rmse_wrong_shape(self):
        time_steps = np.random.randint(10, 100)
        dim = np.random.randint(10, 100)
        slices = np.random.randint(10, 100)
        some_fourth_dim = np.random.randint(10, 100)

        pred = np.random.random((time_steps, dim, slices, some_fourth_dim))
        meas = np.random.random((time_steps, dim, slices, some_fourth_dim))

        with pytest.raises(ValueError):
            measures.rmse(pred, meas)

    def test_rmse_unknown_normalization(self):
        time_steps = np.random.randint(10, 100)
        dim = np.random.randint(10, 100)
        slices = np.random.randint(10, 100)

        pred = np.random.random((time_steps, dim, slices))
        meas = np.random.random((time_steps, dim, slices))

        with pytest.raises(ValueError):
            measures.rmse(pred, meas, "some_unknown_normalization")

    def test_rmse_over_time_2d_with_1_step(self):
        time_steps = 1
        dim = np.random.randint(10, 100)

        pred = np.random.random((time_steps, dim))
        meas = np.random.random((time_steps, dim))

        rmse_desired = measures.rmse(pred, meas)
        rmse = measures.rmse_over_time(pred, meas)

        assert_array_equal(rmse, rmse_desired)

    def test_rmse_over_time_2d(self):
        time_steps = np.random.randint(10, 100)
        dim = np.random.randint(10, 100)

        pred = np.random.random((time_steps, dim))
        meas = np.random.random((time_steps, dim))

        rmse_desired = []
        for i in range(time_steps):
            rmse_desired.append(measures.rmse(pred[i : i + 1], meas[i : i + 1]))
        rmse_desired = np.array(rmse_desired)

        rmse = measures.rmse_over_time(pred, meas)

        assert_array_equal(rmse, rmse_desired)

    def test_rmse_over_time_3d(self):
        time_steps = np.random.randint(10, 100)
        dim = np.random.randint(10, 100)
        slices = np.random.randint(10, 100)

        pred = np.random.random((time_steps, dim, slices))
        meas = np.random.random((time_steps, dim, slices))

        rmse_desired = []
        for i in range(time_steps):
            rmse_desired.append(measures.rmse(pred[i : i + 1], meas[i : i + 1]))
        rmse_desired = np.array(rmse_desired)

        rmse = measures.rmse_over_time(pred, meas)

        assert_array_equal(rmse, rmse_desired)

    def test_rmse_over_time_mean(self):
        time_steps = np.random.randint(10, 100)
        dim = np.random.randint(10, 100)
        slices = np.random.randint(10, 100)

        pred = np.random.random((time_steps, dim, slices))
        meas = np.random.random((time_steps, dim, slices))

        rmse_desired = []
        for i in range(time_steps):
            rmse_desired.append(measures.rmse(pred[i : i + 1], meas[i : i + 1]) / np.mean(meas))
        rmse_desired = np.array(rmse_desired)

        rmse = measures.rmse_over_time(pred, meas, normalization="mean")

        assert_array_equal(rmse, rmse_desired)

    def test_rmse_over_time_std_over_time(self):
        time_steps = np.random.randint(10, 100)
        dim = np.random.randint(10, 100)
        slices = np.random.randint(10, 100)

        pred = np.random.random((time_steps, dim, slices))
        meas = np.random.random((time_steps, dim, slices))

        rmse_desired = []
        for i in range(time_steps):
            rmse_desired.append(measures.rmse(pred[i : i + 1], meas[i : i + 1]) / np.mean(np.std(meas, axis=0)))
        rmse_desired = np.array(rmse_desired)

        rmse = measures.rmse_over_time(pred, meas, normalization="std_over_time")

        assert_array_equal(rmse, rmse_desired)

    def test_rmse_over_time_2norm(self):
        time_steps = np.random.randint(10, 100)
        dim = np.random.randint(10, 100)
        slices = np.random.randint(10, 100)

        pred = np.random.random((time_steps, dim, slices))
        meas = np.random.random((time_steps, dim, slices))

        rmse_desired = []
        for i in range(time_steps):
            rmse_desired.append(measures.rmse(pred[i : i + 1], meas[i : i + 1]) / np.linalg.norm(meas[i]))
        rmse_desired = np.array(rmse_desired)

        rmse = measures.rmse_over_time(pred, meas, normalization="2norm")

        assert_array_equal(rmse, rmse_desired)

    def test_rmse_over_time_maxmin(self):
        time_steps = np.random.randint(10, 100)
        dim = np.random.randint(10, 100)
        slices = np.random.randint(10, 100)

        pred = np.random.random((time_steps, dim, slices))
        meas = np.random.random((time_steps, dim, slices))

        rmse_desired = []
        for i in range(time_steps):
            rmse_desired.append(measures.rmse(pred[i : i + 1], meas[i : i + 1]) / (np.max(meas) - np.min(meas)))
        rmse_desired = np.array(rmse_desired)

        rmse = measures.rmse_over_time(pred, meas, normalization="maxmin")

        assert_array_equal(rmse, rmse_desired)


class TestLyapunovExponentCalculation(unittest.TestCase):
    def test_largest_lyapunov_one_d_linear_time_dependent(self):
        dt = 0.1
        a = -0.1

        def iterator_func(x):
            return x + dt * a * x

        return_convergence = False
        starting_point = np.array(1)
        deviation_scale = 1e-10
        steps = 10
        steps_skip = 1
        part_time_steps = 5

        actual = measures.largest_lyapunov_exponent(
            iterator_func,
            starting_point,
            return_convergence=return_convergence,
            deviation_scale=deviation_scale,
            steps=steps,
            steps_skip=steps_skip,
            part_time_steps=part_time_steps,
            dt=dt,
        )
        desired = np.array(a)
        assert_array_almost_equal(actual, desired, 2)

    def test_largest_lyapunov_one_d_linear_time_dependent_return_conv(self):
        dt = 0.1
        a = -0.1

        def iterator_func(x):
            return x + dt * a * x

        return_convergence = True
        starting_point = np.array(1)
        deviation_scale = 1e-10
        steps = 10
        steps_skip = 1
        part_time_steps = 5

        actual = measures.largest_lyapunov_exponent(
            iterator_func,
            starting_point,
            return_convergence=return_convergence,
            deviation_scale=deviation_scale,
            steps=steps,
            steps_skip=steps_skip,
            part_time_steps=part_time_steps,
            dt=dt,
        )
        desired = np.array(
            [
                a,
            ]
            * steps
        )
        assert_array_almost_equal(actual, desired, 2)
