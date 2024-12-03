"""Tests if the scan.file_handling module works as it should"""

import numpy as np
import pandas as pd
import pytest

import scan
from scan import file_handling
from tests.test_base import assert_array_equal


@pytest.mark.skip(reason="Test TODO")
def test_pytest_tmp_path_write(tmp_path):
    raise Exception


def test_save_and_load_pkl_of_dataframe(tmp_path):
    df = pd.DataFrame(data={"col0": [0, 1]})
    filename = f"{tmp_path}_testfile.pkl"
    filename = file_handling.save_pkl(filename, df)

    df_loaded = file_handling.load_pkl(filename)
    pd.testing.assert_frame_equal(df_loaded, df)


def test_save_and_load_pkl_of_series(tmp_path):
    series = pd.Series(data={"col0": [0, 1]})
    filename = f"{tmp_path}_testfile.pkl"
    filename = file_handling.save_pkl(filename, series)

    series_loaded = file_handling.load_pkl(filename)
    pd.testing.assert_series_equal(series_loaded, series)


def test_save_and_load_pkl_of_nparray(tmp_path):
    arr = np.ndarray([0, 1])
    filename = f"{tmp_path}_testfile.pkl"
    filename = file_handling.save_pkl(filename, arr)

    arr_loaded = file_handling.load_pkl(filename)
    assert_array_equal(arr_loaded, arr)


def test_save_and_load_pkl_of_esn(tmp_path):
    esn = scan.ESN()
    filename = f"{tmp_path}_testfile.pkl"
    filename = file_handling.save_pkl(filename, esn)

    esn_loaded = file_handling.load_pkl(filename)

    assert esn == esn_loaded
    esn.create_network()
    assert esn != esn_loaded


class __MyClass:
    def __init__(self):
        self.some_arr = np.ndarray([0, 1])


def test_save_and_load_of_weird_data_type(tmp_path):
    weird_class = __MyClass()

    filename = f"{tmp_path}_testfile.pkl"
    filename = file_handling.save_pkl(filename, weird_class)

    weird_class_loaded = file_handling.load_pkl(filename)
    assert_array_equal(weird_class_loaded.some_arr, weird_class.some_arr)


def test_save_and_load_pkl_with_correct_expected_type(tmp_path):
    df = pd.DataFrame(data={"col0": [0, 1]})
    filename = f"{tmp_path}_testfile.pkl"
    filename = file_handling.save_pkl(filename, df)

    df_loaded = file_handling.load_pkl(filename, expected_type=pd.DataFrame)
    pd.testing.assert_frame_equal(df_loaded, df)


def test_save_and_load_pkl_with_wrong_expected_type(tmp_path):
    df = pd.DataFrame(data={"col0": [0, 1]})
    filename = f"{tmp_path}_testfile.pkl"
    filename = file_handling.save_pkl(filename, df)

    with pytest.raises(TypeError):
        file_handling.load_pkl(filename, expected_type=int)


def test_load_pkl_of_file_that_doesnt_exist(tmp_path):
    filename = f"{tmp_path}_testfile.pkl"
    with pytest.raises(FileNotFoundError):
        file_handling.load_pkl(filename)


def test_save_and_load_pkl_where_pkl_extension_was_automatically_appended(tmp_path):
    df = pd.DataFrame(data={"col0": [0, 1]})
    filename = f"{tmp_path}_testfile"
    filename = file_handling.save_pkl(filename, df)

    df_loaded = file_handling.load_pkl(filename)
    pd.testing.assert_frame_equal(df_loaded, df)


def test_save_pkl_of_unpickleable_thing(tmp_path):
    class __MyWeirdLocalClass:
        def __init__(self):
            self.some_arr = np.ndarray([0])

    weird_local_class = __MyWeirdLocalClass
    filename = f"{tmp_path}_testfile.pkl"

    with pytest.raises(AttributeError):
        file_handling.save_pkl(filename, weird_local_class)
