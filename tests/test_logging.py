"""Tests if the scan loguru logging works as it should"""

import unittest

import pytest


# NOTE: "pytest is a very common testing framework. The caplog fixture captures logging output so that it can be tested"
class TestLogging(unittest.TestCase):
    @pytest.mark.skip(reason="Test TODO")
    def test_logging_check_if_no_MPI_unsave_logger_is_active_on_startup(self):
        raise Exception

    @pytest.mark.skip(reason="Test TODO")
    def test_logging_check_if_set_logger_returns_an_MPI_save_logger(self):
        raise Exception

    @pytest.mark.skip(reason="Test TODO")
    def test_logging_trace(self):
        raise Exception

    @pytest.mark.skip(reason="Test TODO")
    def test_logging_debug(self):
        raise Exception

    @pytest.mark.skip(reason="Test TODO")
    def test_logging_info(self):
        raise Exception

    @pytest.mark.skip(reason="Test TODO")
    def test_logging_success(self):
        raise Exception

    @pytest.mark.skip(reason="Test TODO")
    def test_logging_warning(self):
        raise Exception

    @pytest.mark.skip(reason="Test TODO")
    def test_logging_error(self):
        raise Exception

    @pytest.mark.skip(reason="Test TODO")
    def test_logging_critical(self):
        raise Exception

    @pytest.mark.skip(reason="Test TODO")
    def test_logging_change_logging_level(self):
        raise Exception
