""" Tests if the scan.esn module works as it should """

import copy

import numpy as np
import pytest

import scan
from tests.test_base import TestScanBase, assert_array_almost_equal, assert_array_equal


# Use print(np.array_repr(np_array, max_line_width=120, precision=18)) in the debugger to easily get the copy-pastable
# representation of a numpy array.
class TestNGRC(TestScanBase):
    def setUp(self):
        self.set_seed()
        self.esn = scan.ESN()
        np.set_printoptions(linewidth=120, precision=20)

    def tearDown(self):
        del self.esn
        np.random.seed(None)
        np.set_printoptions()

    def test_initiate_ngrc_base_class_instance(self):
        ngrc = scan.ngrc.NG_RC(k=2, s=1)

        assert isinstance(ngrc, scan.ngrc.NG_RC)
        
    def test_apply_ngrc_base_one_dimensional_data(self):
        ngrc = scan.ngrc.NG_RC(k=2, s=1)
        data = np.ones(10)
        ngrc.fit(data)
        return
        
    def test_bias(self):
        return
        
        
