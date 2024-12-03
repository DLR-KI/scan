"""Tests if the scan.locality_measures module works as it should"""

import unittest

import numpy as np
import pytest

from scan import locality_measures
from tests.test_base import assert_array_almost_equal, assert_array_equal


class TestLocalityMeasures(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def tearDown(self):
        np.random.seed(None)

    @pytest.mark.skip(reason="Test TODO")
    def test_find_local_neighborhoods_cores1_neighbors0(self):
        raise Exception

    @pytest.mark.skip(reason="Test TODO")
    def test_find_local_neighborhoods_cores1_neighbors1_local(self):
        raise Exception

    @pytest.mark.skip(reason="Test TODO")
    def test_find_local_neighborhoods_cores1_neighbors2_local(self):
        raise Exception

    @pytest.mark.skip(reason="Test TODO")
    def test_find_local_neighborhoods_cores1_neighbors7_local(self):
        raise Exception

    @pytest.mark.skip(reason="Test TODO")
    def test_shan_entropy(self):
        raise Exception

    @pytest.mark.skip(reason="Test TODO")
    def test_nmi_ts(self):
        raise Exception

    @pytest.mark.skip(reason="Test TODO")
    def test_nmi_loc(self):
        raise Exception

    @pytest.mark.skip(reason="Test TODO")
    def test_cc_loc(self):
        raise Exception

    @pytest.mark.skip(reason="Test TODO")
    def test_sn_loc(self):
        raise Exception

    # NOTE: No test stubs for clustering here, as it never really worked in the first place. just remove it instead.
