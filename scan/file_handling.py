"""Methods relating to saving and loading data from a file on disc."""

from __future__ import annotations

import os
import os.path
import pickle
from typing import Any

import pandas as pd


def save_pkl(filepath: str, data: Any) -> str:
    """Pickle (serialized) object to file.

    Args:
        filepath: Filepath where the pickled object will be stored. Expected to end in ".pkl", if not, it will be
            appended automatically.
        data: The object to save. Can be any pickle-able object.

    Returns:
        The filepath that was actually used, possibly appended with a ".pkl".

    """
    # NOTE: We may want to switch to pandas.io.pickle.to_pickle() as that allows the easy choice of compression and
    #  pickle protocol. Similarly for the load_pkl and pd.read_pickle()
    # TODO: This function should have an option to automatically create the directory structure required if any of the
    #  intermittent directories are missing.
    # TODO: this filename addition shouldn't be handled here, but in a general file path handler/saver
    #  function that doesn't exist yet.

    if not filepath.endswith(".pkl"):
        filepath = filepath + ".pkl"

    if isinstance(data, pd.DataFrame):
        data.to_pickle(filepath)
    else:
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    return filepath


def load_pkl(filepath: str, expected_type: type | None = None) -> Any:
    """Load pickled object from file.

    Args:
        filepath: Filepath where the pickled object will be stored.
        expected_type: Optional type to check the loaded data against. Raises a TypeError if the data is not of the
            expected type.

    Returns:
        The loaded data.

    """

    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        if expected_type is None:
            return data
        elif isinstance(data, expected_type):
            # Note: Annoyingly, mypy can't resolve dynamic isinstance checks like this yet, so even though the type of
            # data is clear, you'll have to anotate it wherever you accept the return value
            return data
        else:
            raise TypeError(f"Unpickled file not of expected type: {expected_type}")
    else:
        raise FileNotFoundError("File to load doesn't exist!")
