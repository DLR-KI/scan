"""MPI Methods."""

from __future__ import annotations

import copy
import os
import os.path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import pandas as pd
from loguru import logger

from .file_handling import load_pkl, save_pkl

JobSpecType = Tuple[int, str, Dict[str, Any]]
_task_object: MPIScheduler
# TODO: This is almost certainly not the correct way to specify MPI.COMM_WORLD variable type, but I'm not sure how to
#  specify that correctly for an optional import like mpi4py. That it's not correct is easily seen from the fact that
#  we need the "# type:ignore" note so that mypy doesn't complain..
COMM_WORLD_TYPE = Optional["mpi4py.MPI.Comm"]  # type:ignore


def get_mpi_specs() -> tuple[COMM_WORLD_TYPE, int, int]:
    """Get specs of the active MPI environment, if any.

    Returns:
        (MPI communicator, rank, size), if the mpi4py module was able to be imported. (None, 0, 1) if not.
    """
    try:
        # We import mpi4py here, and not at the top of the file, so that we can keep it optional.
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    except ModuleNotFoundError:
        comm = None
        rank = 0
        size = 1
    return comm, rank, size


def _default_task_runner(args: Any) -> Any:
    global _task_object
    return _task_object._execute(args)


class MPIScheduler:
    def __init__(self, overwrite_files: bool = False):
        """Scheduler using the Message Passing Interface (MPI) to implement a parallelized map method.

        Args:
            overwrite_files: If overwrite_files is set to True, previously existing files, e.g. the output from
            previous runs will be overwritten when found. If not, they will be read and returned, skipping parts, or all
            of the scheduled calculations.

        """
        self.comm, self.rank, self.size = get_mpi_specs()
        if self.comm is None:
            # If the mpi4py init doesn't work, we just execute all the jobs serially, essentially in a for loop
            logger.info("Defaulting to serial jobs because mpi4py is not available.")
        self.root = 0
        self.overwrite_files = overwrite_files
        self._prefix = f"Task {self.rank:3d}"
        self._function: Callable

    def __enter__(self) -> MPIScheduler:
        logger.debug(f"Enter {self._prefix} at {self}")
        return self

    def __exit__(self, *args: Any) -> None:
        if isinstance(args[1], Exception):
            logger.error(f"Exit failed {self._prefix} at {self} with exeption {args[0]}")
        else:
            logger.debug(f"Exit successful {self._prefix} at {self}")

    def _make_joblist(self, filename: str, list_of_kwargs: Iterable[dict[str, Any]]) -> tuple[list[JobSpecType], str]:
        dirname = filename + ".mpi"
        logger.debug(f"{self._prefix} Creating joblist for {filename} under {dirname}")
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        joblist = [(i, f"{dirname}/job{i:04d}.pkl", kwargs) for i, kwargs in enumerate(list_of_kwargs)]
        return joblist, dirname

    def _execute(self, job_spec: JobSpecType) -> pd.DataFrame:
        index, tmpfilename, kwargs = job_spec
        if not self.overwrite_files and os.path.exists(tmpfilename):
            logger.debug(f"{self._prefix} loads {tmpfilename}")
            data: pd.DataFrame = load_pkl(tmpfilename, expected_type=pd.DataFrame)
        else:
            logger.debug(f"{self._prefix} executes {tmpfilename}")

            out = self._function(**kwargs)

            # This needs to be a deepcopy, as otherwise the input kwargs themselves will be changed
            series_data = copy.deepcopy(kwargs)

            series_data["out"] = out
            data = pd.DataFrame(data=[series_data], index=[index])
            logger.debug(f"{self._prefix} saves {tmpfilename}")
            save_pkl(tmpfilename, data)
        logger.debug(f"{self._prefix} made {data}")
        return data

    def _cleanup_files(self, dirname: str, joblist: list[JobSpecType]) -> None:
        for i, tmp, *_ in joblist:
            logger.debug(f"{self._prefix} removing file {tmp}")
            os.remove(tmp)
        logger.debug(f"{self._prefix} removing directory {dirname}")
        os.rmdir(dirname)

    def map(self, filepath: str, function: Callable, kwargs_iterable: Iterable[dict[str, Any]]) -> pd.DataFrame:
        """Map kwarg_iterable over the function using MPI runners.

        If mpi4py is not available, map defaults to a serial, non-mpi mode, which is effectively just a simple for loop
        over the contents of kwarg_iterable.

        If the class instance attribute overwrite_files is set to True, previously existing files, e.g. the output from
        previous runs will be overwritten when found. If not, they will be read and returned, skipping parts or all
        of the scheduled calculations.

        Args:
            filepath: Name/path where the final .pkl output file will be saved at. Note that the path given may not
                be exactly the path used f the name doesn't end with ".pkl", it will be added automatically.
            function: Function ot map the iterable over. All arguments of this function must be addressable kwargs
                (which is the case for most normally written functions), as that will be what's passed from the MPI
                mapper.
            kwargs_iterable: Iterable of kwargs to pass to the function. Each Key: Value pair in the dict represents
                a kwarg name and value.

        Returns:
            Dataframe containing input kwargs and function output corresponding to those kworgs in each row.

        """
        global _task_object
        self._function = function
        if not self.overwrite_files and os.path.exists(filepath):
            logger.info(f"{self._prefix} File {filepath} already exists. Returning it")
            data: pd.DataFrame = load_pkl(filepath, expected_type=pd.DataFrame)
            return data
        elif self.size == 1:
            logger.info(f"{self._prefix} Running jobs in serial mode")
            joblist, dirname = self._make_joblist(filepath, kwargs_iterable)
            output = pd.concat([self._execute(job_spec) for job_spec in joblist])
            # TODO: This requires the full result to be stored in memory at all times
            #  Bad due to growing, possibly unconstrained amount of memory usage
            #  Just saving it to file, as I already do, and then reading it from there once everything is
            #  done, is better
            save_pkl(filepath, output)
            logger.debug(f"{self._prefix} Data saved into {filepath}")
            self._cleanup_files(dirname, joblist)
            return output
        else:
            try:
                _task_object = self
                import mpi4py.futures

                with mpi4py.futures.MPICommExecutor(self.comm, self.root) as executor:
                    # with mpi4py.futures.MPIPoolExecutor() as executor:
                    logger.debug(f"{self._prefix} Executor is {executor}")
                    # Root process
                    if executor is not None:
                        # logger.debug(f'{self._prefix} Missing {filename}')
                        joblist, dirname = self._make_joblist(filepath, kwargs_iterable)
                        logger.debug(f"{self._prefix} got joblist with {len(joblist)} items")
                        output = pd.concat(list(executor.map(_default_task_runner, joblist)))
                        # TODO: This requires the full result to be stored in memory at all times
                        #  Bad due to growing, possibly unconstrained amount of memory usage
                        #  Just saving it to file, as I already do, and then reading it from there once everything is
                        #  done, is better
                        logger.debug(f"{self._prefix} collected {len(output)} items")
                        save_pkl(filepath, output)
                        logger.debug(f"{self._prefix} Data saved into {filepath}")
                        self._cleanup_files(dirname, joblist)
                    return output
            except Exception as e:
                logger.error(f"{self._prefix} aborts due to exception: {e}")
                raise
