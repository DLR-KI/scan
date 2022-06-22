""" Tests if the scan modules can be imported as intended """

import importlib
import unittest

import pytest


class TestImport(unittest.TestCase):
    def test_import(self):
        import scan
        import scan.data_processing
        import scan.esn
        import scan.file_handling
        import scan.locality_measures
        import scan.measures
        import scan.mpi
        import scan.simulations
        import scan.utilities

    def test_init_version_warning(self):
        import scan

        # spoof the internal number to trigger the version warning. This is automatically reset after this test
        scan._version.__version__ = "0.0.0"

        # Check if an import warning is raised
        with pytest.warns(ImportWarning):
            importlib.reload(scan)
