""" Tests if the scan.mpi module works as it should """

import copy
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scan
from scan.mpi import MPIScheduler, get_mpi_specs

# from tests.test_base import TestScanBase, assert_array_equal, assert_array_almost_equal


def test_pytest_tmp_path(tmp_path):
    assert isinstance(tmp_path, Path)


def test_get_mpi_specs():
    # NOTE: Not a big fan of this test, it's just the function code copy and pasted..
    #  The problem is that to write this test in a more conventional way, you need to specify the environment it's
    #  supposed to be loaded in, before it is run. This is probably doable with tox, but I'm not entirely trivial.
    comm, rank, size = get_mpi_specs()
    try:
        from mpi4py import MPI

        exp_comm = MPI.COMM_WORLD
        exp_rank = comm.Get_rank()
        exp_size = comm.Get_size()
    except ModuleNotFoundError:
        exp_comm = None
        exp_rank = 0
        exp_size = 1
    assert exp_comm == comm
    assert exp_rank == rank
    assert exp_size == size


@pytest.mark.mpi
def test_mpi4py_import(tmp_path):
    import mpi4py


def test_mpi_scheduler_one_kwarg_one_out(tmp_path):
    comm, rank, size = get_mpi_specs()

    def temp_func(x):
        return str(x)

    input_kwargs = [{"x": i} for i in range(3)]
    pkl_save_file_path = f"{tmp_path}.pkl"

    with MPIScheduler() as scheduler:
        data = scheduler.map(pkl_save_file_path, temp_func, input_kwargs)

    if rank == 0:
        exp_data = pd.DataFrame()
        for i in range(len(input_kwargs)):
            kwargs = input_kwargs[i]
            out = temp_func(**kwargs)
            exp_data_row = copy.deepcopy(kwargs)
            exp_data_row["out"] = out
            exp_data_row = pd.DataFrame(data=[exp_data_row], index=[i])
            exp_data = pd.concat([exp_data, exp_data_row])

        pd.testing.assert_frame_equal(data, exp_data)


def test_mpi_scheduler_multiple_kwargs_one_out(tmp_path):
    comm, rank, size = get_mpi_specs()

    def temp_func(x, y):
        return str(x * y)

    input_kwargs = [{"x": i, "y": i**2} for i in range(3)]
    pkl_save_file_path = f"{tmp_path}.pkl"

    with MPIScheduler() as scheduler:
        data = scheduler.map(pkl_save_file_path, temp_func, input_kwargs)

    if rank == 0:
        exp_data = pd.DataFrame()
        for i in range(len(input_kwargs)):
            kwargs = input_kwargs[i]
            out = temp_func(**kwargs)
            exp_data_row = copy.deepcopy(kwargs)
            exp_data_row["out"] = out
            exp_data_row = pd.DataFrame(data=[exp_data_row], index=[i])
            exp_data = pd.concat([exp_data, exp_data_row])

        pd.testing.assert_frame_equal(data, exp_data)


def test_mpi_scheduler_multiple_kwargs_multiple_out(tmp_path):
    comm, rank, size = get_mpi_specs()

    def temp_func(x, y):
        return x, y

    input_kwargs = [{"x": i, "y": i**2} for i in range(3)]
    pkl_save_file_path = f"{tmp_path}.pkl"

    with MPIScheduler() as scheduler:
        data = scheduler.map(pkl_save_file_path, temp_func, input_kwargs)

    if rank == 0:
        exp_data = pd.DataFrame()
        for i in range(len(input_kwargs)):
            kwargs = input_kwargs[i]
            out = temp_func(**kwargs)
            exp_data_row = copy.deepcopy(kwargs)
            exp_data_row["out"] = out
            exp_data_row = pd.DataFrame(data=[exp_data_row], index=[i])
            exp_data = pd.concat([exp_data, exp_data_row])

        pd.testing.assert_frame_equal(data, exp_data)


def test_mpi_scheduler_multiple_array_out(tmp_path):
    comm, rank, size = get_mpi_specs()

    def temp_func(x, y):
        out1 = np.array([i for i in range(x)])
        out2 = np.array([i for i in range(y)])
        return out1, out2

    input_kwargs = [{"x": i, "y": i**2} for i in range(3)]
    pkl_save_file_path = f"{tmp_path}.pkl"

    with MPIScheduler() as scheduler:
        data = scheduler.map(pkl_save_file_path, temp_func, input_kwargs)

    if rank == 0:
        exp_data = pd.DataFrame()
        for i in range(len(input_kwargs)):
            kwargs = input_kwargs[i]
            out = temp_func(**kwargs)
            exp_data_row = copy.deepcopy(kwargs)
            exp_data_row["out"] = out
            exp_data_row = pd.DataFrame(data=[exp_data_row], index=[i])
            exp_data = pd.concat([exp_data, exp_data_row])

        pd.testing.assert_frame_equal(data, exp_data)


def test_mpi_scheduler_raise_exception_due_to_wrong_kwargs(tmp_path):
    comm, rank, size = get_mpi_specs()

    def temp_func(x):
        return str(x)

    input_kwargs = [{"ThisArgumentDoesNotExistInTheFunction": None} for i in range(3)]
    pkl_save_file_path = f"{tmp_path}.pkl"

    with pytest.raises(TypeError):
        with MPIScheduler() as scheduler:
            scheduler.map(pkl_save_file_path, temp_func, input_kwargs)

            # This is pretty much a hack to make pytest.raises cooperate with mpi.
            # As it stands, only the root thread, of rank 0, encounters the exception I want to test for, so the other
            # threads/taskers exit succesfully and w/o exception. This is what we want, but the pytest.raises context
            # checks for the expected exception in all mpi threads, which fails the test unless we also raise the
            # expected exception in all those other non-root, rank > 0, threads..
            if rank != 0:
                raise TypeError


def test_mpi_scheduler_save_to_pkl(tmp_path):
    comm, rank, size = get_mpi_specs()

    def temp_func(x):
        return str(x)

    input_kwargs = [{"x": i} for i in range(3)]
    pkl_save_file_path = f"{tmp_path}.pkl"

    with MPIScheduler() as scheduler:
        data = scheduler.map(pkl_save_file_path, temp_func, input_kwargs)

    if rank == 0:
        exp_data = scan.load_pkl(pkl_save_file_path)
        pd.testing.assert_frame_equal(data, exp_data)


def test_mpi_scheduler_load_partial_results_from_disk(tmp_path):
    # TODO: This test is far less general than we would want it to be. The optimal way to test loading partial results
    #  would be to abort the execution of the scheduler halfway though. How to do that from within python in a sensible
    #  way I don't know though.
    comm, rank, size = get_mpi_specs()

    def temp_func(x):
        return str(x)

    input_kwargs = [{"x": i} for i in range(3)]
    pkl_save_file_path = f"{tmp_path}.pkl"

    if rank == 0:
        exp_data = pd.DataFrame()

        tmpdirname = f"{pkl_save_file_path}.mpi"
        if not os.path.exists(tmpdirname):
            os.mkdir(tmpdirname)

        for i in range(len(input_kwargs)):
            # TODO: Very un-general naming here. This job shouldn't fail if the scheduler's internal naming changes, but
            #  as it is written right now, it very much does fail.
            tmpfilename = f"{tmpdirname}/job{i:04d}.pkl"

            kwargs = input_kwargs[i]
            out = temp_func(**kwargs)
            exp_data_row = copy.deepcopy(kwargs)
            exp_data_row["out"] = out
            exp_data_row = pd.DataFrame(data=[exp_data_row], index=[i])
            scan.save_pkl(tmpfilename, exp_data_row)
            exp_data = pd.concat([exp_data, exp_data_row])
    else:
        exp_data = None

    with MPIScheduler() as scheduler:
        data = scheduler.map(pkl_save_file_path, temp_func, input_kwargs)

    if rank == 0:
        pd.testing.assert_frame_equal(data, exp_data)


def test_mpi_scheduler_load_final_pkl_from_disk(tmp_path):
    comm, rank, size = get_mpi_specs()

    def temp_func(x):
        return str(x)

    input_kwargs = [{"x": i} for i in range(3)]
    pkl_save_file_path = f"{tmp_path}.pkl"

    if rank == 0:
        exp_data = pd.DataFrame()
        for i in range(len(input_kwargs)):
            kwargs = input_kwargs[i]
            out = temp_func(**kwargs)
            exp_data_row = copy.deepcopy(kwargs)
            exp_data_row["out"] = out
            exp_data_row = pd.DataFrame(data=[exp_data_row], index=[i])
            exp_data = pd.concat([exp_data, exp_data_row])
        pkl_save_file_path = scan.save_pkl(pkl_save_file_path, exp_data)
    else:
        exp_data = None

    if comm:
        pkl_save_file_path = comm.bcast(pkl_save_file_path)

    with MPIScheduler() as scheduler:
        data = scheduler.map(pkl_save_file_path, temp_func, input_kwargs)

    if rank == 0:
        pd.testing.assert_frame_equal(data, exp_data)
