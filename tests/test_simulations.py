""" Tests if the scan.simulations module works as it should """

import unittest

import numpy as np
import pytest

import scan
from tests.test_base import (
    TestScanBase,
    assert_array_almost_equal,
    assert_array_equal,
    assert_array_not_equal,
)


# Use print(np.array_repr(np_array, max_line_width=120, precision=18)) in the debugger to easily get the copy-pastable
# representation of a numpy array.
class TestSimulations(TestScanBase):
    def test_simulate_trajectory_lorenz63_single_step_trivial_test(self):
        simulation_time_steps = 2
        starting_point = np.array([-14.03020521, -20.88693127, 25.53545])
        sim_data = scan.simulate_trajectory(
            sys_flag="lorenz", dt=2e-2, time_steps=simulation_time_steps, starting_point=starting_point
        )

        exp_sim_data = np.array(
            [[-14.03020521, -20.88693127, 25.53545], [-15.257976883416845, -20.510306180264724, 30.15606333510718]]
        )

        assert_array_equal(sim_data, exp_sim_data)

    def test_simulate_trajectory_lorenz63_default_starting_point_single_step_trivial_test(self):
        simulation_time_steps = 2
        sim_data = scan.simulate_trajectory(sys_flag="lorenz", dt=2e-2, time_steps=simulation_time_steps)

        exp_sim_data = np.array([[1.0, 2.0, 3.0], [1.2275093315399999, 2.5108524200767555, 2.893000748906853]])

        assert_array_equal(sim_data, exp_sim_data)

    def test_simulate_trajectory_lorenz96_single_step_trivial_test(self):
        simulation_time_steps = 2
        lor_dim = 5
        lor_dt = 1.0
        lor_force = 5

        self.set_seed()
        starting_point = lor_force * np.ones(lor_dim) + 1e-2 * np.random.rand(lor_dim)
        sim_data = scan.simulate_trajectory(
            sys_flag="lorenz_96",
            dt=lor_dt,
            time_steps=simulation_time_steps,
            starting_point=starting_point,
            force=lor_force,
        )

        exp_sim_data = np.array(
            [
                [5.005488135039273, 5.007151893663724, 5.006027633760716, 5.005448831829969, 5.004236547993389],
                [4.824002686688245, 4.576702025628189, 4.932128122531685, 5.377384289786162, 5.271235302887315],
            ]
        )

        assert_array_equal(sim_data, exp_sim_data)

    def test_simulate_trajectory_lorenz96_single_step_no_starting_point(self):
        simulation_time_steps = 2
        lor_dt = 1.0
        lor_force = 5

        self.set_seed()
        sim_data = scan.simulate_trajectory(
            sys_flag="lorenz_96",
            dt=lor_dt,
            time_steps=simulation_time_steps,
            starting_point=None,
            force=lor_force,
        )

        exp_sim_data = np.array([[1.0, 2.0, 3.0], [3.5, 3.875, 4.25]])

        assert_array_equal(sim_data, exp_sim_data)

    def test_kuramoto_sivashinski_6d_2l_05t_custom_starting_point_single_step(self):
        ks_sys_flag = "kuramoto_sivashinsky"
        dimensions = 6
        system_size = 2
        dt = 0.5
        sim_data = scan.simulate_trajectory(
            sys_flag=ks_sys_flag, dimensions=dimensions, system_size=system_size, dt=dt, time_steps=3
        )
        # Note that, right now, the KS simulation is the only simulation function that returns the starting point as
        # first point of the prediction, so we have to take that into account here.
        starting_point = sim_data[1]
        exp_sim_data = scan.simulate_trajectory(
            sys_flag=ks_sys_flag,
            dimensions=dimensions,
            system_size=system_size,
            dt=dt,
            time_steps=2,
            starting_point=starting_point,
        )

        # Due to the FFT parts of the KS algorithm, putting in the same real space point as corresponds to some in
        # point in Fourier Space, actually results in a fairly large simulation difference, even after a single
        # simulation step!
        assert_array_almost_equal(sim_data[-1], exp_sim_data[-1], decimal=6)

    def test_kuramoto_sivashinski_custom_6d_2l_05t_custom_starting_point_single_step(self):
        ks_sys_flag = "kuramoto_sivashinsky_custom"
        dimensions = 6
        system_size = 2
        dt = 0.5
        sim_data = scan.simulate_trajectory(
            sys_flag=ks_sys_flag, dimensions=dimensions, system_size=system_size, dt=dt, time_steps=3
        )
        # Note that, right now, the KS simulation is the only simulation function that returns the starting point as
        # first point of the prediction, so we have to take that into account here.
        starting_point = sim_data[1]
        exp_sim_data = scan.simulate_trajectory(
            sys_flag=ks_sys_flag,
            dimensions=dimensions,
            system_size=system_size,
            dt=dt,
            time_steps=2,
            starting_point=starting_point,
        )

        # Due to the FFT parts of the KS algorithm, putting in the same real space point as corresponds to some in
        # point in Fourier Space, actually results in a fairly large simulation difference, even after a single
        # simulation step!
        assert_array_almost_equal(sim_data[-1], exp_sim_data[-1], decimal=6)

    def test_kuramoto_sivashinski_custom_40d_22l_05t_npfft_no_precision_change(self):
        ks_sys_flag = "kuramoto_sivashinsky_custom"
        dimensions = 40
        system_size = 22
        dt = 0.5
        time_steps = 10
        fft_type = "numpy"

        sim_data = scan.simulate_trajectory(
            sys_flag=ks_sys_flag,
            dimensions=dimensions,
            system_size=system_size,
            dt=dt,
            time_steps=time_steps,
            precision=None,
            fft_type=fft_type,
        )

        exp_sim_data = scan.simulate_trajectory(
            sys_flag=ks_sys_flag,
            dimensions=dimensions,
            system_size=system_size,
            dt=dt,
            time_steps=time_steps,
            precision=64,
            fft_type=fft_type,
        )

        assert_array_equal(sim_data, exp_sim_data)

    def test_kuramoto_sivashinski_custom_40d_22l_05t_npfft_128_precision(self):
        ks_sys_flag = "kuramoto_sivashinsky_custom"
        dimensions = 40
        system_size = 22
        dt = 0.5
        time_steps = 10
        fft_type = "numpy"

        sim_data = scan.simulate_trajectory(
            sys_flag=ks_sys_flag,
            dimensions=dimensions,
            system_size=system_size,
            dt=dt,
            time_steps=time_steps,
            precision=128,
            fft_type=fft_type,
        )

        exp_sim_data = scan.simulate_trajectory(
            sys_flag=ks_sys_flag,
            dimensions=dimensions,
            system_size=system_size,
            dt=dt,
            time_steps=time_steps,
            precision=64,
            fft_type=fft_type,
        )
        assert_array_not_equal(sim_data, exp_sim_data)

    def test_kuramoto_sivashinski_custom_40d_22l_05t_unknown_precision(self):
        ks_sys_flag = "kuramoto_sivashinsky_custom"
        dimensions = 40
        system_size = 22
        dt = 0.5
        time_steps = 10
        fft_type = "numpy"

        with pytest.raises(ValueError):
            scan.simulate_trajectory(
                sys_flag=ks_sys_flag,
                dimensions=dimensions,
                system_size=system_size,
                dt=dt,
                time_steps=time_steps,
                precision="this_precision_does_not_exist",
                fft_type=fft_type,
            )

    def test_kuramoto_sivashinski_custom_40d_22l_05t_unknown_fft_type(self):
        ks_sys_flag = "kuramoto_sivashinsky_custom"
        dimensions = 40
        system_size = 22
        dt = 0.5
        time_steps = 10

        with pytest.raises(ValueError):
            scan.simulate_trajectory(
                sys_flag=ks_sys_flag,
                dimensions=dimensions,
                system_size=system_size,
                dt=dt,
                time_steps=time_steps,
                precision=None,
                fft_type="this_ffttype_does_not_exist",
            )

    def test_simulate_trajectory_unknown_flag(self):
        with pytest.raises(ValueError):
            scan.simulate_trajectory(sys_flag="this_flag_doesnt_exist")


class TestKuramotSivashinskiVariantsDivergence40d22l05t(unittest.TestCase):
    def setUp(self):
        self.ks_sys_flag = "kuramoto_sivashinsky_custom"
        self.dimensions = 40
        self.system_size = 22
        self.dt = 0.5
        self.time_steps = 1000

    @pytest.mark.xfail(reason="the KS simulation sometimes diverges using the numpy FFT")
    def test_kuramoto_sivashinski_custom_40d_22l_05t_divergence(self):
        # NOTE: This isn't actually all that great of a test, as it doesn't really test for the actual logic/algorithm
        #  we implemented and only checks if it diverges or not. Refactor by usinge e.g. a more sophisticated Lyapunov
        #  exponent test.
        ks_sys_flag = "kuramoto_sivashinsky"
        dimensions = 40
        system_size = 22
        dt = 0.5
        time_steps = 1000
        sim_data = scan.simulate_trajectory(
            sys_flag=ks_sys_flag, dimensions=dimensions, system_size=system_size, dt=dt, time_steps=time_steps
        )
        assert not np.isnan(sim_data).any()
        assert np.amax(np.abs(sim_data)) < 100

    @pytest.mark.xfail(reason="the KS simulation sometimes diverges using the numpy FFT")
    def test_kuramoto_sivashinski_custom_40d_22l_05t_npfft_64bit_divergence(self):
        # NOTE: This isn't actually all that great of a test, as it doesn't really test for the actual logic/algorithm
        #  we implemented and only checks if it diverges or not. Refactor by usinge e.g. a more sophisticated Lyapunov
        #  exponent test.
        precision = 64
        fft_type = "numpy"
        sim_data = scan.simulate_trajectory(
            sys_flag=self.ks_sys_flag,
            dimensions=self.dimensions,
            system_size=self.system_size,
            dt=self.dt,
            time_steps=self.time_steps,
            precision=precision,
            fft_type=fft_type,
        )
        assert not np.isnan(sim_data).any()
        assert np.amax(np.abs(sim_data)) < 100

    @pytest.mark.xfail(reason="the KS simulation sometimes diverges using the numpy FFT")
    def test_kuramoto_sivashinski_custom_40d_22l_05t_npfft_128_divergence(self):
        # NOTE: This isn't actually all that great of a test, as it doesn't really test for the actual logic/algorithm
        #  we implemented and only checks if it diverges or not. Refactor by usinge e.g. a more sophisticated Lyapunov
        #  exponent test.
        precision = 128
        fft_type = "numpy"
        sim_data = scan.simulate_trajectory(
            sys_flag=self.ks_sys_flag,
            dimensions=self.dimensions,
            system_size=self.system_size,
            dt=self.dt,
            time_steps=self.time_steps,
            precision=precision,
            fft_type=fft_type,
        )
        assert not np.isnan(sim_data).any()
        assert np.amax(np.abs(sim_data)) < 100

    @pytest.mark.xfail(reason="the KS simulation sometimes diverges using the numpy FFT")
    def test_kuramoto_sivashinski_custom_40d_22l_05t_npfft_32_divergence(self):
        # NOTE: This isn't actually all that great of a test, as it doesn't really test for the actual logic/algorithm
        #  we implemented and only checks if it diverges or not. Refactor by usinge e.g. a more sophisticated Lyapunov
        #  exponent test.
        precision = 32
        fft_type = "numpy"
        sim_data = scan.simulate_trajectory(
            sys_flag=self.ks_sys_flag,
            dimensions=self.dimensions,
            system_size=self.system_size,
            dt=self.dt,
            time_steps=self.time_steps,
            precision=precision,
            fft_type=fft_type,
        )
        assert not np.isnan(sim_data).any()
        assert np.amax(np.abs(sim_data)) < 100

    @pytest.mark.xfail(reason="the KS simulation sometimes diverges using the numpy FFT")
    def test_kuramoto_sivashinski_custom_40d_22l_05t_npfft_16_divergence(self):
        # NOTE: This isn't actually all that great of a test, as it doesn't really test for the actual logic/algorithm
        #  we implemented and only checks if it diverges or not. Refactor by usinge e.g. a more sophisticated Lyapunov
        #  exponent test.
        precision = 16
        fft_type = "numpy"
        sim_data = scan.simulate_trajectory(
            sys_flag=self.ks_sys_flag,
            dimensions=self.dimensions,
            system_size=self.system_size,
            dt=self.dt,
            time_steps=self.time_steps,
            precision=precision,
            fft_type=fft_type,
        )
        assert not np.isnan(sim_data).any()
        assert np.amax(np.abs(sim_data)) < 100

    def test_kuramoto_sivashinski_custom_40d_22l_05t_scfft_64bit_divergence(self):
        # NOTE: This isn't actually all that great of a test, as it doesn't really test for the actual logic/algorithm
        #  we implemented and only checks if it diverges or not. Refactor by usinge e.g. a more sophisticated Lyapunov
        #  exponent test.
        pytest.importorskip("scipy")
        precision = 64
        fft_type = "scipy"
        sim_data = scan.simulate_trajectory(
            sys_flag=self.ks_sys_flag,
            dimensions=self.dimensions,
            system_size=self.system_size,
            dt=self.dt,
            time_steps=self.time_steps,
            precision=precision,
            fft_type=fft_type,
        )
        assert not np.isnan(sim_data).any()
        assert np.amax(np.abs(sim_data)) < 100
