""" Tests if the scan.utilities module works as it should """

import copy
import fractions
import unittest

import numpy as np
import pytest

import scan
from scan import utilities
from tests.test_base import TestScanBase, assert_array_almost_equal, assert_array_equal


class TestUtilities(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_train_and_predict_input_setup(self):
        train_sync_steps = 2
        train_steps = 4
        pred_sync_steps = 5
        pred_steps = 6
        some_steps_at_the_end = 13
        total_time_steps = train_sync_steps + train_steps + pred_sync_steps + pred_steps + some_steps_at_the_end

        x_dim = 3
        data = np.random.random((total_time_steps, x_dim))

        x_train_desired = data[: train_sync_steps + train_steps]
        x_pred_desired = data[train_sync_steps + train_steps - 1 : -some_steps_at_the_end]

        x_train, x_pred = utilities.train_and_predict_input_setup(
            data,
            train_sync_steps=train_sync_steps,
            train_steps=train_steps,
            pred_sync_steps=pred_sync_steps,
            pred_steps=pred_steps,
        )

        np.testing.assert_equal(x_train, x_train_desired)
        np.testing.assert_equal(x_pred, x_pred_desired)

    def test_find_nth_substring(self):
        test_str = "0.s0.0fd.0sf.5"

        self.assertEqual(utilities.find_nth_substring(test_str, ".", 0), None)
        self.assertEqual(utilities.find_nth_substring(test_str, ".", 1), 1)
        self.assertEqual(utilities.find_nth_substring(test_str, ".", 2), 4)
        self.assertEqual(utilities.find_nth_substring(test_str, ".", 3), 8)
        self.assertEqual(utilities.find_nth_substring(test_str, ".", 4), 12)
        self.assertEqual(utilities.find_nth_substring(test_str, ".", 5), None)

    def test_synonym_add_synonym_wrong_synonym_type(self):
        synonym_dict = utilities.SynonymDict()
        with pytest.raises(TypeError):
            synonym_dict.add_synonyms(0, scan.ESN())

    def test_synonym_add_int_synonyms(self):
        synonym_dict = utilities.SynonymDict()
        synonym_dict.add_synonyms(0, 0)
        synonym_dict.add_synonyms(0, 1)

        self.assertEqual(0, synonym_dict.get_flag(0))
        self.assertEqual(0, synonym_dict.get_flag(1))

    def test_synonym_add_duplicate_synonyms(self):
        synonym_dict = utilities.SynonymDict()
        synonym_dict.add_synonyms(0, "0")
        synonym_dict.add_synonyms(0, "0")

        self.assertEqual(0, synonym_dict.get_flag(0))

    def test_synonym_add_conflicting_synonyms(self):
        synonym_dict = utilities.SynonymDict()
        synonym_dict.add_synonyms(0, "0")
        with pytest.raises(ValueError):
            synonym_dict.add_synonyms(1, "0")

    def test_synonym_get_flag_from_dict_init(self):
        synonym_dict = utilities.SynonymDict({0: "syn0", 1: ["syn11", "syn12"]})

        self.assertEqual(0, synonym_dict.get_flag("syn0"))
        self.assertEqual(1, synonym_dict.get_flag("syn11"))
        self.assertEqual(1, synonym_dict.get_flag("syn12"))

        with pytest.raises(KeyError):
            synonym_dict.get_flag("not_a_synonym")

    def test_synonym_get_flag_from_add_synonym(self):
        synonym_dict = utilities.SynonymDict()
        synonym_dict.add_synonyms(0, "syn0")
        synonym_dict.add_synonyms(1, ["syn10", "syn11"])
        synonym_dict.add_synonyms(2, (f"syn2{i}" for i in range(2)))

        self.assertEqual(0, synonym_dict.get_flag("syn0"))
        self.assertEqual(1, synonym_dict.get_flag("syn10"))
        self.assertEqual(1, synonym_dict.get_flag("syn11"))
        self.assertEqual(2, synonym_dict.get_flag("syn20"))
        self.assertEqual(2, synonym_dict.get_flag("syn21"))

        with pytest.raises(KeyError):
            synonym_dict.get_flag("not_a_synonym")

    def test_synonym_dict_add_synonym_to_flag_that_already_exists(self):
        synonym_dict = utilities.SynonymDict({0: "syn00"})
        synonym_dict.add_synonyms(0, "syn01")
        synonym_dict.add_synonyms(0, "syn02")

        self.assertEqual(0, synonym_dict.get_flag("syn00"))
        self.assertEqual(0, synonym_dict.get_flag("syn01"))
        self.assertEqual(0, synonym_dict.get_flag("syn02"))

    def test_hash_dict_identity(self):
        dict1 = {"a": 1}
        hash1 = utilities.hash_dict(dict1)
        hash2 = utilities.hash_dict(dict1)
        assert hash1 == hash2

    def test_hash_dict_equality(self):
        dict1 = {"a": 1}
        dict2 = {"a": 1}
        hash1 = utilities.hash_dict(dict1)
        hash2 = utilities.hash_dict(dict2)
        assert dict1 is not dict2
        assert hash1 == hash2

    def test_hash_dict_dont_change_dict_during_hashing(self):
        dict1 = {"a": 1, "b": np.array([1, 2])}
        dict1_copy = copy.deepcopy(dict1)
        utilities.hash_dict(dict1)
        assert dict1_copy["a"] == dict1["a"]
        assert_array_equal(dict1_copy["b"], dict1["b"])

    def test_hash_dict_reverse_order(self):
        dict1 = {"a": 1, "b": 2}
        dict2 = {"b": 2, "a": 1}
        hash1 = utilities.hash_dict(dict1)
        hash2 = utilities.hash_dict(dict2)
        assert dict1 is not dict2
        assert hash1 == hash2

    def test_hash_dict_numpy_array(self):
        dict1 = {"a": np.array([1, 2])}
        dict2 = {"a": np.array([1, 2])}
        hash1 = utilities.hash_dict(dict1)
        hash2 = utilities.hash_dict(dict2)
        assert dict1 is not dict2
        assert hash1 == hash2

    def test_hash_dict_numpy_array_different_dicts(self):
        dict1 = {"a": np.array([1, 2])}
        dict2 = {"a": np.array([2, 1])}
        hash1 = utilities.hash_dict(dict1)
        hash2 = utilities.hash_dict(dict2)
        assert hash1 != hash2

    @pytest.mark.xfail(reason="Hash not consistent between platforms")
    def test_hash_dict_platform_and_environment_consistency_ints(self):
        dict1 = {"a": 1, "b": np.array([1, 2]), "c": [4, 6]}
        hashed = utilities.hash_dict(dict1)
        assert hashed == 175858106841432378687297766663211925126

    @pytest.mark.xfail(reason="Hash not consistent between platforms")
    def test_hash_dict_platform_and_environment_consistency_floats(self):
        dict1 = {"a": 1.2, "b": np.array([1.32, 2.7]), "c": [4.1, 6.7]}
        hashed = utilities.hash_dict(dict1)
        assert hashed == 101302184554471131163934114195438235681

    def test_get_environment_version_of_scan(self):
        # Note sure how to accurately test this, without just assuming the expected answer
        env_version = utilities.get_environment_version()
        # exp_env_version = scan._version.__version__
        assert env_version != "0.0.0"

    def test_get_environment_version_of_some_pkg_that_doesnt_exist(self):
        env_version = utilities.get_environment_version(package="some_pkg_that_doesnt_exist")
        # This is actually the internal version, so the test isn't quite doing what I want it to
        exp_env_version = "0.0.0"
        assert env_version == exp_env_version

    def test_compare_package_versions_segment_threshold_micro(self):
        version_a = "0.4.6"
        version_b = "0.4.6"
        version_c = "0.4.7"
        assert utilities.compare_package_versions(version_a, version_b, segment_threshold="micro")
        assert not utilities.compare_package_versions(version_a, version_c, segment_threshold="micro")

    def test_compare_package_versions_segment_threshold_minor(self):
        version_a = "0.4.6"
        version_b = "0.4.14"
        version_c = "0.5.6"
        assert utilities.compare_package_versions(version_a, version_b, segment_threshold="minor")
        assert not utilities.compare_package_versions(version_a, version_c, segment_threshold="minor")

    def test_compare_package_versions_segment_threshold_major(self):
        version_a = "0.4.6"
        version_b = "0.123.64"
        version_c = "1.4.6"
        assert utilities.compare_package_versions(version_a, version_b, segment_threshold="major")
        assert not utilities.compare_package_versions(version_a, version_c, segment_threshold="major")

    def test_compare_package_versions_segment_threshold_unknown(self):
        version_a = "0.4.6"
        version_b = "0.123.64"
        with pytest.raises(ValueError):
            utilities.compare_package_versions(version_a, version_b, segment_threshold="seg_threshold_doesnt_exist")

    def test_is_number_int(self):
        assert utilities.is_number(1)

    def test_is_number_float(self):
        assert utilities.is_number(1.3)

    def test_is_number_fraction(self):
        assert utilities.is_number(fractions.Fraction(5, 3))

    def test_is_number_np_array_with_one_element(self):
        assert utilities.is_number(np.array([1]))
