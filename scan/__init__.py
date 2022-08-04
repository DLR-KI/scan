import warnings

from loguru import logger

from . import (
    _version,
    data_processing,
    esn,
    file_handling,
    locality_measures,
    logging,
    measures,
    mpi,
    simulations,
    utilities,
)
from ._version import __version__
from .data_processing import downsampling, embedding, smooth
from .esn import ESN, ESNGenLoc, ESNWrapper
from .file_handling import load_pkl, save_pkl
from .logging import set_logger
from .mpi import MPIScheduler
from .simulations import simulate_trajectory

set_logger("WARNING")  # Activate our own MPI-save logging handler

# Print a warning if the external and internal version numbers don't match up
if not utilities.compare_version_file_vs_env(segment_threshold="minor"):
    int_version = utilities.get_internal_version()
    env_version = utilities.get_environment_version()

    warn_string = (
        "The internal scan package version '%s' does not match the version specified in the currently active python"
        " environment '%s'.\nPlease update the package by entering the scan repository folder and running: 'pip install"
        " --upgrade -e .'" % (int_version, env_version)
    )

    warnings.filterwarnings("once", category=ImportWarning)
    warnings.warn(warn_string, ImportWarning, stacklevel=2)
    logger.warning(warn_string)
    warnings.resetwarnings()
