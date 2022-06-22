""" Tests if the scan.utilities module works as it should """

import unittest

import numpy as np
import pytest

from scan import data_processing
from tests.test_base import (
    TestScanBase,
    assert_array_almost_equal,
    assert_array_equal,
    assert_array_not_equal,
)


class TestUtilities(TestScanBase):
    def test_embedding_1d_simple(self):
        embedding_dim = 2
        embedding_delay = 1

        x_data = np.array([0, 1])

        x_data_embedded = data_processing.embedding(
            x_data, embedding_dim=embedding_dim, embedding_delay=embedding_delay
        )

        exp_x_data_embedded = np.array([[1, 0], [np.NAN, 1]])

        assert_array_equal(x_data_embedded, exp_x_data_embedded)

    @pytest.mark.skip(reason="Test TODO")
    def test_embedding_1d(self):
        raise Exception

    def test_embedding_2d_simple(self):
        embedding_dim = 2
        embedding_delay = 1

        x_data = np.array([[0, 2], [1, 3]])

        x_data_embedded = data_processing.embedding(
            x_data, embedding_dim=embedding_dim, embedding_delay=embedding_delay
        )

        exp_x_data_embedded = np.array([[1, 3, 0, 2], [np.NAN, np.NAN, 1, 3]])

        assert_array_equal(x_data_embedded, exp_x_data_embedded)

    def test_embedding_2d_xdim1(self):
        embedding_dim = 2
        embedding_delay = 1

        time_steps = 10

        x_data_1d = np.arange(0, time_steps, 1)
        x_data_2d = x_data_1d.reshape((time_steps, 1))

        x_data_embedded = data_processing.embedding(
            x_data_2d, embedding_dim=embedding_dim, embedding_delay=embedding_delay
        )

        exp_x_data_embedded = data_processing.embedding(
            x_data_1d, embedding_dim=embedding_dim, embedding_delay=embedding_delay
        )

        assert_array_equal(x_data_embedded, exp_x_data_embedded)

    @pytest.mark.skip(reason="Test TODO")
    def test_embedding_2d(self):
        raise Exception

    def test_embedding_3d_xdim1_simple(self):
        embedding_dim = 2
        embedding_delay = 1

        # fmt: off
        x_data = np.array(
            [
                [
                    [0, 4], [2, 6]
                ],
                [
                    [1, 5], [3, 7]
                ]
            ]
        )
        # fmt: on

        x_data_embedded = data_processing.embedding(
            x_data, embedding_dim=embedding_dim, embedding_delay=embedding_delay
        )

        # fmt: off
        exp_x_data_embedded = np.array(
            [
                [
                    [1, 5],           [3, 7],           [0, 4], [2, 6]
                ],
                [
                    [np.NAN, np.NAN], [np.NAN, np.NAN], [1, 5], [3, 7]
                ]
            ]
        )
        # fmt: on

        assert_array_equal(x_data_embedded, exp_x_data_embedded)

    def test_embedding_3d_xdim1(self):
        embedding_dim = 3
        embedding_delay = 2

        time_steps = 6
        x_dim = 1
        slices = 2

        x_train_3d = np.arange(0, time_steps * x_dim * slices, 1).reshape((time_steps, x_dim, slices), order="F")

        x_train_3d_embedded = data_processing.embedding(
            x_train_3d, embedding_dim=embedding_dim, embedding_delay=embedding_delay
        )

        x_dims_after_embedding = x_dim * embedding_dim

        exp_x_train_3d_embedded = np.empty(shape=(time_steps, x_dims_after_embedding, slices)) * np.NAN

        for i in range(x_dims_after_embedding):
            shift = embedding_delay * i
            if shift == 0:
                exp_x_train_3d_embedded[:, x_dims_after_embedding - (i + 1), :] = x_train_3d[:, 0, :]
            else:
                exp_x_train_3d_embedded[:-shift, x_dims_after_embedding - (i + 1), :] = x_train_3d[shift:, 0, :]

        assert_array_equal(exp_x_train_3d_embedded, x_train_3d_embedded)

    def test_embedding_3d_xdim1_embedding_dim1(self):
        # should return the original time series
        embedding_dim = 1
        embedding_delay = 2

        time_steps = 6
        x_dim = 1
        slices = 2

        x_train_3d = np.arange(0, time_steps * x_dim * slices, 1).reshape((time_steps, x_dim, slices), order="F")

        x_train_3d_embedded = data_processing.embedding(
            x_train_3d, embedding_dim=embedding_dim, embedding_delay=embedding_delay
        )

        assert_array_equal(x_train_3d, x_train_3d_embedded)

    def test_embedding_3d_xdim1_embedding_dim0(self):
        embedding_dim = 0
        embedding_delay = 2

        time_steps = 6
        x_dim = 1
        slices = 2

        x_train_3d = np.arange(0, time_steps * x_dim * slices, 1).reshape((time_steps, x_dim, slices), order="F")

        with pytest.raises(ValueError):
            data_processing.embedding(x_train_3d, embedding_dim=embedding_dim, embedding_delay=embedding_delay)

    @pytest.mark.skip(reason="Test TODO")
    def test_embedding_3d(self):
        raise Exception

    def test_downsampling_1d_simple(self):
        x_train = np.array([0, 1, 2, 3])
        downsampling_size = 2
        x_train_downsampled = data_processing.downsampling(x_train, downsampling_size=downsampling_size)

        exp_train_downsampled = np.array([0.5, 2.5])

        assert_array_equal(x_train_downsampled, exp_train_downsampled)

    @pytest.mark.skip(reason="Test TODO")
    def test_downsampling_1d(self):
        raise Exception

    def test_downsampling_2d_simple(self):
        x_train = np.array([[0, 2], [1, 3]])
        downsampling_size = 2
        x_train_downsampled = data_processing.downsampling(x_train, downsampling_size=downsampling_size)

        exp_train_downsampled = np.array([[0.5, 2.5]])

        assert_array_equal(x_train_downsampled, exp_train_downsampled)

    @pytest.mark.skip(reason="Test TODO")
    def test_downsampling_2d_xdim1(self):
        raise Exception

    @pytest.mark.skip(reason="Test TODO")
    def test_downsampling_2d(self):
        raise Exception

    def test_downsampling_3d_xdim1_simple(self):
        downsampling_size = 2

        x_train_3d = np.array([[[0, 1]], [[2, 3]], [[4, 5]], [[6, 7]], [[8, 9]], [[10, 11]]])
        x_train_3d_downsampled = data_processing.downsampling(x_train_3d, downsampling_size=downsampling_size)

        exp_x_train_3d_downsampled = np.array([[[1, 2]], [[5, 6]], [[9, 10]]])

        assert_array_equal(exp_x_train_3d_downsampled, x_train_3d_downsampled)

    def test_downsampling_3d_simple(self):
        downsampling_size = 2

        x_train_3d = np.array([[[0, 2], [4, 6]], [[1, 3], [5, 7]]])
        x_train_3d_downsampled = data_processing.downsampling(x_train_3d, downsampling_size=downsampling_size)

        exp_x_train_3d_downsampled = np.array([[[0.5, 2.5], [4.5, 6.5]]])

        assert_array_equal(exp_x_train_3d_downsampled, x_train_3d_downsampled)

    def test_downsampling_3d(self):
        downsampling_size = 2

        time_steps = 6
        x_dim = 3
        slices = 2

        x_train_3d = np.arange(0, time_steps * x_dim * slices, 1).reshape((time_steps, x_dim, slices), order="F")

        x_train_3d_downsampled = data_processing.downsampling(x_train_3d, downsampling_size=downsampling_size)

        exp_x_train_3d_downsampled = np.arange(0.5, time_steps * x_dim * slices - 0.5, 2).reshape(
            (time_steps // downsampling_size, x_dim, slices), order="F"
        )

        assert_array_equal(exp_x_train_3d_downsampled, x_train_3d_downsampled)

    def test_downsampling_3d_downsampling_size_not_clean_multiple_of_time_steps(self):
        downsampling_size = 3

        time_steps = 7
        x_dim = 3
        slices = 2

        x_train_3d = np.arange(0, time_steps * x_dim * slices, 1).reshape((time_steps, x_dim, slices), order="F")

        with pytest.raises(ValueError):
            data_processing.downsampling(x_train_3d, downsampling_size=downsampling_size)