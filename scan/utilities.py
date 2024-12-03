"""Various utility functions for the scan package."""

from __future__ import annotations

import collections
import hashlib
import inspect
import pickle
from typing import Any, Callable, Iterable, Union

import numpy as np
import pkg_resources

import scan._version

# FlagType is the type a user can input to choose between different options for e.g. activation functions
FlagType = Union[int, str]


class SynonymDict:
    def __init__(self, flags_and_synonyms: dict[int, Iterable[FlagType]] | None = None):
        """Custom dictionary wrapper to match synonyms with integer flags

        Internally the corresponding integer flags are used, but they are very much
        not descriptive so with this class one can define (str) synonyms for these
        flags, similar to how matplotlib does it

        Args:
            flags_and_synonyms: Dictionary matching the integer flags, to be used internally, with their string
                synonyms. Idea:
                flags_and_synonyms = {flag1 : list of synonyms of flag1,
                                      flag2 : list of synonyms of flag2,
                                      ...}
        """

        self._synonym_dict: dict[int, list[FlagType]] = {}
        if flags_and_synonyms:
            for int_flag, synonyms in flags_and_synonyms.items():
                self.add_synonyms(int_flag, synonyms)

    def add_synonyms(self, int_flag: int, synonyms: FlagType | Iterable[FlagType]) -> None:
        """Assigns one or more synonyms to the corresponding integer flag

        Args:
            int_flag: integer flag to pair with the synonym(s)
            synonyms: Synonym or iterable of synonyms. Technically any type
                is possible for a synonym but strings are highly recommended

        """

        # Convert the synonym(s) to a list of synonyms
        if isinstance(synonyms, list):
            synonym_list = synonyms
        elif isinstance(synonyms, (str, int)):
            synonym_list = [synonyms]
        elif isinstance(synonyms, Iterable):
            synonym_list = list(iter(synonyms))
        else:
            raise TypeError

        # make sure that the synonyms are not already paired to different flags
        for synonym in synonym_list:
            found_flag = self.find_flag(synonym)
            if int_flag == found_flag:
                synonym_list.remove(synonym)
            elif found_flag is not None:
                raise ValueError(
                    "Tried to add Synonym %s to flag %d but it was already paired to flag %d"
                    % (str(synonym), int_flag, found_flag)
                )

        # add the synonyms
        if int_flag not in self._synonym_dict:
            self._synonym_dict[int_flag] = []
        self._synonym_dict[int_flag].extend(synonym_list)

    def find_flag(self, synonym: FlagType | Iterable[FlagType]) -> int | None:
        """Finds the corresponding flag to a given synonym.

        A flag is always also a synonym for itself.

        Args:
            synonym: Thing to find the synonym for

        Returns:
            flag: int if found, None if not

        """

        flag = None
        if synonym in self._synonym_dict and isinstance(synonym, int):
            flag = synonym
        else:
            for item in self._synonym_dict.items():
                if synonym in item[1]:
                    flag = item[0]

        return flag

    def get_flag(self, synonym: FlagType | Iterable[FlagType]) -> int:
        """Finds the corresponding int_flag to a given synonym. Raises exception if not found

        see :func:`~SynonymDict._find_flag_from_synonym`

        """
        flag = self.find_flag(synonym)
        if flag is None:
            raise KeyError("Flag corresponding to synonym %s not found" % str(synonym))

        return flag


def remove_invalid_args(func: Callable, args_dict: dict[str, Any]) -> dict:
    """Return dictionary of valid args and kwargs with invalid ones removed

    Adjusted from:
    https://stackoverflow.com/questions/196960/can-you-list-the-keyword-arguments-a-function-receives

    Args:
        func: function to check if the arguments are valid or not
        args_dict: dictionary of arguments

    Returns:
        dict: dictionary of valid arguments

    """
    valid_args = inspect.signature(func).parameters
    # valid_args = func.func_code.co_varnames[:func.func_code.co_argcount]
    return {key: value for key, value in args_dict.items() if key in valid_args}


def train_and_predict_input_setup(
    x_data: np.ndarray,
    train_sync_steps: int,
    train_steps: int,
    pred_sync_steps: int = 0,
    pred_steps: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Splits ESN input data for consecutive training and prediction

    This function is useful because there is an unintuitive overlap between x_train and x_pred of 1 time step which
    makes it easy to make mistakes.

    Args:
        x_data: data to be split/setup
        train_sync_steps: steps to sync the reservoir with before training
        train_steps: steps to use for training and fitting w_in
        pred_sync_steps: steps to sync the reservoir with before prediction
        pred_steps: how many steps to predict the evolution for

    Returns:
        2-element tuple containing:

        - **x_train**: input data for the training
        - **x_pred**: input data for the prediction

    """
    x_train = x_data[: train_sync_steps + train_steps]

    if pred_steps is None:
        x_pred = x_data[train_sync_steps + train_steps - 1 :]
    else:
        x_pred = x_data[
            train_sync_steps + train_steps - 1 : train_sync_steps + train_steps + pred_sync_steps + pred_steps
        ]

    return x_train, x_pred


def find_nth_substring(haystack: str, substring: str, n: int) -> int | None:
    """Finds the position of the n-th occurrence of a substring

    Args:
        haystack: Main string to find the occurrences in
        substring: Substring to search for in the haystack string
        n: The occurrence number of the substring to be located. n > 0

    Returns:
        int_or_None: Position index of the n-th substring occurrence in the
        haystack string if found. None if not found.

    """
    parts = haystack.split(substring, n)

    if n <= 0 or len(parts) <= n:
        return None
    else:
        return len(haystack) - len(parts[-1]) - len(substring)


def get_internal_version() -> str:
    """Returns the internal scan version as specified in scan._version

    Returns:
        int_version, internal scan package version

    """
    int_version = scan._version.__version__
    return int_version


def get_environment_version(package: str = "scan") -> str:
    """Returns the package version as specified in the python environment, as e.g. pip would return it

    Args:
        package: The package for which to get the environment version for

    Returns:
       Package version

    """
    try:
        import scan

        env_version = pkg_resources.require(package)[0].version
    except (ImportError, pkg_resources.DistributionNotFound):
        env_version = "0.0.0"
    return env_version


def compare_version_file_vs_env(segment_threshold: str = "minor") -> bool:
    """Compare version file with the version number in the python environment

    Compares the internally defined version number of the scan package
    as specified in scan._version with the version number specified in the
    activate python environment up to the defined component threshold.

    Args:
        segment_threshold: Defines up to which segment of the version
            string the versions are compared. Possible flags are:

                - "major": major version numbers are compared
                - "minor": major and minor version numbers are compared
                - "micro": major, minor and micro version numbers are compared

    Returns:
        bool: True if internal and environemnt versions are the same up to and
            including the specified threshold. False if not.

    """
    int_version = get_internal_version()
    env_version = get_environment_version()

    return compare_package_versions(env_version, int_version, segment_threshold)


def compare_package_versions(version_a: str, version_b: str, segment_threshold: str = "minor") -> bool:
    """Compare two Major.Minor.Micro package version strings

    Args:
        version_a: version string in the format " Major.Minor.Micro", e.g. "0.5.18"
        version_b: version string 2
        segment_threshold: Defines up to which segment of the version
            string the versions are compared. Possible flags are:

                - "major": major version numbers are compared
                - "minor": major and minor version numbers are compared
                - "micro": major, minor and micro version numbers are compared

    Returns:
        True if the versions match up to the specified segment, False if not
    """

    if segment_threshold == "major":
        version_a = version_a[: find_nth_substring(version_a, ".", 1)]
        version_b = version_b[: find_nth_substring(version_b, ".", 1)]
    elif segment_threshold == "minor":
        version_a = version_a[: find_nth_substring(version_a, ".", 2)]
        version_b = version_b[: find_nth_substring(version_b, ".", 2)]
    elif segment_threshold == "micro":
        pass
    else:
        raise ValueError("segment_threshold %s not recognized" % segment_threshold)

    return version_a == version_b


def is_number(s: Any) -> bool:
    """Tests if input is a number

    Args:
        s: Object you want to check

    Returns:
        True if s is a number, False if not
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


def hash_dict(dictionary: dict[str, Any]) -> int:
    """Create hash of Dictionary with string (or at least sortable) keys

    The keys need to be sortable as we need to sort them so that {'a': 1, 'b': 2} and {'b': 2, 'a': 1} will create the
    same hash. The code assumes string keys but in principle other non-string but sortable keys should also work but
    that's not tested.

    The same dict should create the same hash in different sessions, but this consistency is not given on different
    platforms or for different python environments.

    Args:
        dictionary: Dictionary with string keys and arbitrary, pickleable items

    Returns:
        hash
    """
    ordered_dictionary = collections.OrderedDict(sorted(dictionary.items()))
    pkl_str_serilization = pickle.dumps(ordered_dictionary)

    # We do not use the builtin hash function as that one is salted to randomize the hash between sessions. This is
    # important for the security of python, but just annoying for us. Hence we use hashlib to make sure that the hashes
    # actually stay consistent, allowing us to compare them between sessions.
    dhash = hashlib.md5()
    dhash.update(pkl_str_serilization)
    hex_hash = dhash.hexdigest()
    int_hash = int(hex_hash, 16)
    return int_hash
