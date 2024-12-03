"""Measures for the calculation of generalized local neighborhoods."""

from __future__ import annotations

import numpy as np
import sklearn.cluster


def find_local_neighborhoods(
    locality_matrix: np.ndarray,
    neighbors: int,
    cores: int = 1,
    cluster_method: str = "hacky_loc_neighbors",
    cluster_linkage: str = "average",
) -> np.ndarray:
    """Stub for measures used in the  the Generalized Local State Implementation of RC.

    For details and ETA, please write sebastian.baur@dlr.de.
    """


def shan_entropy(x: np.ndarray) -> float:
    """Stub for measures used in the  the Generalized Local State Implementation of RC.

    For details and ETA, please write sebastian.baur@dlr.de.
    """


def nmi_ts(x: np.ndarray, y: np.ndarray, bins: int | None = None) -> float:
    """Stub for measures used in the  the Generalized Local State Implementation of RC.

    For details and ETA, please write sebastian.baur@dlr.de.
    """


def nmi_loc(matrix: np.ndarray, bins: int | None = None, rowvar: bool = False) -> np.ndarray:
    """Stub for measures used in the  the Generalized Local State Implementation of RC.

    For details and ETA, please write sebastian.baur@dlr.de.
    """


def sn_loc(matrix: np.ndarray, rowvar: bool = False) -> np.ndarray:
    """Stub for measures used in the  the Generalized Local State Implementation of RC.

    For details and ETA, please write sebastian.baur@dlr.de.
    """


def cc_loc(matrix: np.ndarray, rowvar: bool = False) -> np.ndarray:
    """Stub for measures used in the  the Generalized Local State Implementation of RC.

    For details and ETA, please write sebastian.baur@dlr.de.
    """
