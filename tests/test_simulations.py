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

DECIMALS = 14

# Use print(np.array_repr(np_array, max_line_width=120, precision=18)) in the debugger to easily get the copy-pastable
# representation of a numpy array.
class TestSimulations(TestScanBase):
    def test_simulate_trajectory_lorenz63_single_step_trivial_test(self):
        simulation_time_steps = 2
        starting_point = np.array([-14.03020521, -20.88693127, 25.53545])

        sim_data = scan.simulations.Lorenz63(dt=2e-2).simulate(
            time_steps=simulation_time_steps, starting_point=starting_point
        )

        exp_sim_data = np.array(
            [[-14.03020521, -20.88693127, 25.53545], [-15.257976883416845, -20.510306180264724, 30.15606333510718]]
        )

        assert_array_almost_equal(sim_data, exp_sim_data, decimal=14)

    def test_simulate_trajectory_lorenz63_default_starting_point_single_step_trivial_test(self):
        simulation_time_steps = 2
        sim_data = scan.simulations.Lorenz63(dt=2e-2).simulate(time_steps=simulation_time_steps)

        exp_sim_data = np.array(
            [[0.0, -0.01, 9.0], [-1.8168870026079268e-03, -1.0161406771353333e-02, 8.5325756633139491e00]]
        )

        assert_array_almost_equal(sim_data, exp_sim_data, decimal=DECIMALS)

    def test_simulate_trajectory_lorenz63_simulate_instance_twice(self):
        simulation_time_steps = 2

        isntance = scan.simulations.Lorenz63()
        sim_data_1 = isntance.simulate(time_steps=simulation_time_steps)
        sim_data_2 = isntance.simulate(time_steps=simulation_time_steps)
        assert_array_equal(sim_data_1, sim_data_2)

    def test_simulate_trajectory_lorenz96_single_step_trivial_test(self):
        simulation_time_steps = 2
        lor_dim = 5
        lor_dt = 1.0
        lor_force = 5

        self.set_seed()
        starting_point = lor_force * np.ones(lor_dim) + 1e-2 * np.random.rand(lor_dim)

        sim_data = scan.simulations.Lorenz96(sys_dim=lor_dim, force=lor_force, dt=lor_dt).simulate(
            time_steps=simulation_time_steps, starting_point=starting_point
        )

        exp_sim_data = np.array(
            [
                [5.005488135039273, 5.007151893663724, 5.006027633760716, 5.005448831829969, 5.004236547993389],
                [4.824002686688245, 4.576702025628189, 4.932128122531685, 5.377384289786162, 5.271235302887315],
            ]
        )

        assert_array_almost_equal(sim_data, exp_sim_data, decimal=DECIMALS)

    def test_simulate_trajectory_lorenz96_single_step_no_starting_point(self):
        simulation_time_steps = 2

        self.set_seed()

        sim_data = scan.simulations.Lorenz96().simulate(time_steps=simulation_time_steps)

        exp_sim_data = np.array(
            [
                np.sin(np.arange(30)),
                [
                    0.3769009997531504,
                    1.2048911212671174,
                    1.2602415620124692,
                    0.4409127670578584,
                    -0.3558303924604377,
                    -0.5112232787601698,
                    0.07503500757496134,
                    1.005512609974874,
                    1.3583335117552717,
                    0.714030171527682,
                    -0.18099116196016424,
                    -0.5450477863753878,
                    -0.15566505877288883,
                    0.756972305739906,
                    1.3655846500065303,
                    0.968056644989393,
                    0.04054393941189638,
                    -0.5163763982152564,
                    -0.33974865011922695,
                    0.4841133914492217,
                    1.2824544729906988,
                    1.176981382103961,
                    0.2966082002705891,
                    -0.426938493134693,
                    -0.4678259531746986,
                    0.21180527032161653,
                    1.1189395692923816,
                    1.3175211558905413,
                    0.5701852056224413,
                    -0.26074897061166247,
                ],
            ]
        )

        assert_array_almost_equal(sim_data, exp_sim_data, decimal=DECIMALS)

    def test_simulate_trajectory_lorenz96_simulate_instance_twice(self):
        simulation_time_steps = 2

        isntance = scan.simulations.Lorenz96()
        sim_data_1 = isntance.simulate(time_steps=simulation_time_steps)
        sim_data_2 = isntance.simulate(time_steps=simulation_time_steps)
        assert_array_equal(sim_data_1, sim_data_2)

    def test_lorenz96_dimension_missmatch(self):
        sys_dim = 10
        time_steps = 2
        starting_point = np.ones(11)

        with pytest.raises(ValueError):

            scan.simulations.Lorenz96(
                sys_dim=sys_dim,
            ).simulate(time_steps=time_steps, starting_point=starting_point)

    def test_simulate_trajectory_roessler_default_starting_point_single_step_trivial_test(self):
        sim_data = scan.simulations.Roessler().simulate(time_steps=2)

        exp_sim_data = np.array([[-9.0, 0.0, 0.0], [-8.955425338833333, -0.90756655, 0.009868544904350832]])

        assert_array_almost_equal(sim_data, exp_sim_data, decimal=DECIMALS)

    def test_simulate_trajectory_roessler_simulate_instance_twice(self):
        simulation_time_steps = 2

        isntance = scan.simulations.Roessler()
        sim_data_1 = isntance.simulate(time_steps=simulation_time_steps)
        sim_data_2 = isntance.simulate(time_steps=simulation_time_steps)
        assert_array_equal(sim_data_1, sim_data_2)

    def test_simulate_trajectory_complex_butterfly_default_starting_point_single_step_trivial_test(self):
        sim_data = scan.simulations.ComplexButterly().simulate(time_steps=2)

        exp_sim_data = np.array([[0.2, 0.0, 0.0], [0.19458405593782552, 0.0010022759114583332, -0.04013613366536458]])

        assert_array_almost_equal(sim_data, exp_sim_data, decimal=DECIMALS)

    def test_simulate_trajectory_complex_butterfly_simulate_instance_twice(self):
        simulation_time_steps = 2

        isntance = scan.simulations.ComplexButterly()
        sim_data_1 = isntance.simulate(time_steps=simulation_time_steps)
        sim_data_2 = isntance.simulate(time_steps=simulation_time_steps)
        assert_array_equal(sim_data_1, sim_data_2)

    def test_simulate_trajectory_chen_default_starting_point_single_step_trivial_test(self):
        sim_data = scan.simulations.Chen().simulate(time_steps=2)

        exp_sim_data = np.array([[-10.0, 0.0, 37.0], [-6.3845564682890625, 4.1595207530446, 35.746433277481756]])

        assert_array_almost_equal(sim_data, exp_sim_data, decimal=DECIMALS)

    def test_simulate_trajectory_chen_simulate_instance_twice(self):
        simulation_time_steps = 2

        isntance = scan.simulations.Chen()
        sim_data_1 = isntance.simulate(time_steps=simulation_time_steps)
        sim_data_2 = isntance.simulate(time_steps=simulation_time_steps)
        assert_array_equal(sim_data_1, sim_data_2)

    def test_simulate_trajectory_chua_default_starting_point_single_step_trivial_test(self):
        sim_data = scan.simulations.ChuaCircuit().simulate(time_steps=2)

        exp_sim_data = np.array([[0.0, 0.0, 0.6], [0.006776632653061224, 0.029199732142857142, 0.5894738520408163]])

        assert_array_almost_equal(sim_data, exp_sim_data, decimal=DECIMALS)

    def test_simulate_trajectory_chua_simulate_instance_twice(self):
        simulation_time_steps = 2

        isntance = scan.simulations.ChuaCircuit()
        sim_data_1 = isntance.simulate(time_steps=simulation_time_steps)
        sim_data_2 = isntance.simulate(time_steps=simulation_time_steps)
        assert_array_equal(sim_data_1, sim_data_2)

    def test_simulate_trajectory_thomas_default_starting_point_single_step_trivial_test(self):
        sim_data = scan.simulations.Thomas().simulate(time_steps=2)

        exp_sim_data = np.array([[0.1, 0.0, 0.0], [0.0965923504007725, 0.0019260974789302725, 0.019268419498910318]])

        assert_array_almost_equal(sim_data, exp_sim_data, decimal=DECIMALS)

    def test_simulate_trajectory_thomas_simulate_instance_twice(self):
        simulation_time_steps = 2

        isntance = scan.simulations.Thomas()
        sim_data_1 = isntance.simulate(time_steps=simulation_time_steps)
        sim_data_2 = isntance.simulate(time_steps=simulation_time_steps)
        assert_array_equal(sim_data_1, sim_data_2)

    def test_simulate_trajectory_windmi_default_starting_point_single_step_trivial_test(self):
        sim_data = scan.simulations.WindmiAttractor().simulate(time_steps=2)

        exp_sim_data = np.array([[0.0, 0.8, 0.0], [0.08011122410215064, 0.8032831905478534, 0.06347854657651615]])

        assert_array_almost_equal(sim_data, exp_sim_data, decimal=DECIMALS)

    def test_simulate_trajectory_windmi_simulate_instance_twice(self):
        simulation_time_steps = 2

        isntance = scan.simulations.WindmiAttractor()
        sim_data_1 = isntance.simulate(time_steps=simulation_time_steps)
        sim_data_2 = isntance.simulate(time_steps=simulation_time_steps)
        assert_array_equal(sim_data_1, sim_data_2)

    def test_simulate_trajectory_rucklidge_default_starting_point_single_step_trivial_test(self):
        sim_data = scan.simulations.Rucklidge().simulate(time_steps=2)

        exp_sim_data = np.array([[1.0, 0.0, 4.5], [0.9075854977280031, 0.047626990242513025, 4.280570709227276]])

        assert_array_almost_equal(sim_data, exp_sim_data, decimal=DECIMALS)

    def test_simulate_trajectory_rucklidge_simulate_instance_twice(self):
        simulation_time_steps = 2

        isntance = scan.simulations.Rucklidge()
        sim_data_1 = isntance.simulate(time_steps=simulation_time_steps)
        sim_data_2 = isntance.simulate(time_steps=simulation_time_steps)
        assert_array_equal(sim_data_1, sim_data_2)

    def test_simulate_trajectory_simplest_quadratic_default_starting_point_single_step_trivial_test(self):
        sim_data = scan.simulations.SimplestQuadraticChaotic().simulate(time_steps=2)

        exp_sim_data = np.array([[-0.9, 0.0, 0.5], [-0.897517119397917, 0.049492007344059, 0.490173819781445]])

        assert_array_almost_equal(sim_data, exp_sim_data, decimal=DECIMALS)

    def test_simulate_trajectory_simplest_quadratic_simulate_instance_twice(self):
        simulation_time_steps = 2

        isntance = scan.simulations.SimplestQuadraticChaotic()
        sim_data_1 = isntance.simulate(time_steps=simulation_time_steps)
        sim_data_2 = isntance.simulate(time_steps=simulation_time_steps)
        assert_array_equal(sim_data_1, sim_data_2)

    def test_simulate_trajectory_simplest_cubic_default_starting_point_single_step_trivial_test(self):
        sim_data = scan.simulations.SimplestCubicChaotic().simulate(time_steps=2)

        exp_sim_data = np.array(
            [[0.0, 0.96, 0.0], [9.599968640000001e-02, 9.599880919807999e-01, -3.522283392479332e-04]]
        )

        assert_array_almost_equal(sim_data, exp_sim_data, decimal=DECIMALS)

    def test_simulate_trajectory_simplest_cubic_simulate_instance_twice(self):
        simulation_time_steps = 2

        isntance = scan.simulations.SimplestCubicChaotic()
        sim_data_1 = isntance.simulate(time_steps=simulation_time_steps)
        sim_data_2 = isntance.simulate(time_steps=simulation_time_steps)
        assert_array_equal(sim_data_1, sim_data_2)

    def test_simulate_trajectory_simplest_piecewise_default_starting_point_single_step_trivial_test(self):
        sim_data = scan.simulations.SimplestPiecewiseLinearChaotic().simulate(time_steps=2)

        exp_sim_data = np.array([[0.0, -0.7, 0.0], [-0.070046333333333, -0.701354283333333, -0.025639846666667]])

        assert_array_almost_equal(sim_data, exp_sim_data, decimal=DECIMALS)

    def test_simulate_trajectory_simplest_piecewise_simulate_instance_twice(self):
        simulation_time_steps = 2

        isntance = scan.simulations.SimplestPiecewiseLinearChaotic()
        sim_data_1 = isntance.simulate(time_steps=simulation_time_steps)
        sim_data_2 = isntance.simulate(time_steps=simulation_time_steps)
        assert_array_equal(sim_data_1, sim_data_2)

    def test_simulate_trajectory_double_scroll_default_starting_point_single_step_trivial_test(self):
        sim_data = scan.simulations.DoubleScroll().simulate(time_steps=2)

        exp_sim_data = np.array([[0.01, 0.01, 0.0], [0.01112802, 0.013813637333333, 0.0752040608]])

        assert_array_almost_equal(sim_data, exp_sim_data, decimal=DECIMALS)

    def test_simulate_trajectory_double_scroll_simulate_instance_twice(self):
        simulation_time_steps = 2

        isntance = scan.simulations.DoubleScroll()
        sim_data_1 = isntance.simulate(time_steps=simulation_time_steps)
        sim_data_2 = isntance.simulate(time_steps=simulation_time_steps)
        assert_array_equal(sim_data_1, sim_data_2)

    def test_simulate_trajectory_lotka_volterra_default_starting_point_single_step_trivial_test(self):
        sim_data = scan.simulations.LotkaVolterra().simulate(time_steps=2)

        exp_sim_data = np.array([[1.0, 1.0], [1.08452909181137, 0.88802549930702]])

        assert_array_almost_equal(sim_data, exp_sim_data, decimal=DECIMALS)

    def test_simulate_trajectory_lotka_volterra_simulate_instance_twice(self):
        simulation_time_steps = 2

        isntance = scan.simulations.LotkaVolterra()
        sim_data_1 = isntance.simulate(time_steps=simulation_time_steps)
        sim_data_2 = isntance.simulate(time_steps=simulation_time_steps)
        assert_array_equal(sim_data_1, sim_data_2)

    def test_simulate_trajectory_linear_default_starting_point_single_step_trivial_test(self):
        sim_data = scan.simulations.LinearSystem().simulate(time_steps=2)

        exp_sim_data = np.array([[1.0, 1.0], [0.895170833333333, 1.0948375]])

        assert_array_almost_equal(sim_data, exp_sim_data, decimal=DECIMALS)

    def test_simulate_trajectory_linear_simulate_instance_twice(self):
        simulation_time_steps = 2

        isntance = scan.simulations.LinearSystem()
        sim_data_1 = isntance.simulate(time_steps=simulation_time_steps)
        sim_data_2 = isntance.simulate(time_steps=simulation_time_steps)
        assert_array_equal(sim_data_1, sim_data_2)

    def test_linear_dimension_missmatch(self):

        A = np.ones((10, 10))
        time_steps = 2
        starting_point = np.ones(11)

        with pytest.raises(ValueError):
            scan.simulations.LinearSystem(
                A=A,
            ).simulate(time_steps=time_steps, starting_point=starting_point)

    def test_simulate_trajectory_henon_default_starting_point_single_step_trivial_test(self):
        sim_data = scan.simulations.Henon().simulate(time_steps=2)

        exp_sim_data = np.array([[0.0, 0.9], [1.27, 0.0]])

        assert_array_almost_equal(sim_data, exp_sim_data, decimal=DECIMALS)

    def test_simulate_trajectory_henon_simulate_instance_twice(self):
        simulation_time_steps = 2

        isntance = scan.simulations.Henon()
        sim_data_1 = isntance.simulate(time_steps=simulation_time_steps)
        sim_data_2 = isntance.simulate(time_steps=simulation_time_steps)
        assert_array_equal(sim_data_1, sim_data_2)

    def test_simulate_trajectory_logistic_default_starting_point_single_step_trivial_test(self):
        sim_data = scan.simulations.Logistic().simulate(time_steps=2)

        exp_sim_data = np.array(
            [
                [
                    0.1,
                ],
                [
                    0.36000000000000004,
                ],
            ]
        )

        assert_array_almost_equal(sim_data, exp_sim_data, decimal=DECIMALS)

    def test_simulate_trajectory_logistic_simulate_instance_twice(self):
        simulation_time_steps = 2

        isntance = scan.simulations.Logistic()
        sim_data_1 = isntance.simulate(time_steps=simulation_time_steps)
        sim_data_2 = isntance.simulate(time_steps=simulation_time_steps)
        assert_array_equal(sim_data_1, sim_data_2)

    def test_simulate_trajectory_simple_driven_chaotic_default_starting_point_single_step_trivial_test(self):
        sim_data = scan.simulations.SimplestDrivenChaotic().simulate(time_steps=2)

        exp_sim_data = np.array([[0.0, 0.0, 0.0], [0.00031287210159712276, 0.009372350531852047, 0.1]])

        assert_array_almost_equal(sim_data, exp_sim_data, decimal=DECIMALS)

    def test_simulate_trajectory_simple_driven_chaotic_simulate_instance_twice(self):
        simulation_time_steps = 2

        isntance = scan.simulations.SimplestDrivenChaotic()
        sim_data_1 = isntance.simulate(time_steps=simulation_time_steps)
        sim_data_2 = isntance.simulate(time_steps=simulation_time_steps)
        assert_array_equal(sim_data_1, sim_data_2)

    def test_simulate_trajectory_ueda_default_starting_point_single_step_trivial_test(self):
        sim_data = scan.simulations.UedaOscillator().simulate(time_steps=2)

        exp_sim_data = np.array([[2.5, 0.0, 0.0], [2.4807171482576695, -0.7648847867534123, 0.05]])

        assert_array_almost_equal(sim_data, exp_sim_data, decimal=DECIMALS)

    def test_simulate_trajectory_ueda_simulate_instance_twice(self):
        simulation_time_steps = 2

        isntance = scan.simulations.UedaOscillator()
        sim_data_1 = isntance.simulate(time_steps=simulation_time_steps)
        sim_data_2 = isntance.simulate(time_steps=simulation_time_steps)
        assert_array_equal(sim_data_1, sim_data_2)

    def test_kuramoto_sivashinski_6d_2l_05t_custom_starting_point_single_step(self):
        sys_dim = 6
        sys_length = 2
        dt = 0.5

        sim_data = scan.simulations.KuramotoSivashinsky(sys_dim=sys_dim, sys_length=sys_length, dt=dt).simulate(
            time_steps=3
        )

        # Note that, right now, the KS simulation is the only simulation function that returns the starting point as
        # first point of the prediction, so we have to take that into account here.
        starting_point = sim_data[1]

        exp_sim_data = scan.simulations.KuramotoSivashinsky(sys_dim=sys_dim, sys_length=sys_length, dt=dt).simulate(
            time_steps=2, starting_point=starting_point
        )

        # Due to the FFT parts of the KS algorithm, putting in the same real space point as corresponds to some in
        # point in Fourier Space, actually results in a fairly large simulation difference, even after a single
        # simulation step!
        assert_array_almost_equal(sim_data[-1], exp_sim_data[-1], decimal=6)

    def test_simulate_trajectory_kuramoto_sivashinski_simulate_instance_twice(self):
        simulation_time_steps = 2

        isntance = scan.simulations.KuramotoSivashinsky()
        sim_data_1 = isntance.simulate(time_steps=simulation_time_steps)
        sim_data_2 = isntance.simulate(time_steps=simulation_time_steps)
        assert_array_equal(sim_data_1, sim_data_2)

    def test_kuramoto_sivashinski_custom_6d_2l_05t_custom_starting_point_single_step(self):
        sys_dim = 6
        sys_length = 2
        dt = 0.5
        sim_data = scan.simulations.KuramotoSivashinskyCustom(sys_dim=sys_dim, sys_length=sys_length, dt=dt).simulate(
            time_steps=3
        )

        # Note that, right now, the KS simulation is the only simulation function that returns the starting point as
        # first point of the prediction, so we have to take that into account here.
        starting_point = sim_data[1]

        exp_sim_data = scan.simulations.KuramotoSivashinskyCustom(
            sys_dim=sys_dim, sys_length=sys_length, dt=dt
        ).simulate(time_steps=2, starting_point=starting_point)

        # Due to the FFT parts of the KS algorithm, putting in the same real space point as corresponds to some in
        # point in Fourier Space, actually results in a fairly large simulation difference, even after a single
        # simulation step!
        assert_array_almost_equal(sim_data[-1], exp_sim_data[-1], decimal=6)

    def test_simulate_trajectory_kuramoto_sivashinski_custom_simulate_instance_twice(self):
        simulation_time_steps = 2

        isntance = scan.simulations.KuramotoSivashinskyCustom()
        sim_data_1 = isntance.simulate(time_steps=simulation_time_steps)
        sim_data_2 = isntance.simulate(time_steps=simulation_time_steps)
        assert_array_equal(sim_data_1, sim_data_2)

    def test_kuramoto_sivashinski_custom_40d_22l_05t_npfft_no_precision_change(self):
        # ks_sys_flag = "kuramoto_sivashinsky_custom"
        sys_dim = 40
        sys_length = 22
        dt = 0.5
        time_steps = 10
        fft_type = "numpy"

        sim_data = scan.simulations.KuramotoSivashinskyCustom(
            sys_dim=sys_dim, sys_length=sys_length, dt=dt, fft_type=fft_type
        ).simulate(time_steps=time_steps)

        exp_sim_data = scan.simulations.KuramotoSivashinskyCustom(
            sys_dim=sys_dim,
            sys_length=sys_length,
            dt=dt,
            precision=64,
            fft_type=fft_type,
        ).simulate(time_steps=time_steps)

        assert_array_almost_equal(sim_data, exp_sim_data, decimal=DECIMALS)

    @pytest.mark.xfail(reason="Datatype does not exist on all systems.")
    def test_kuramoto_sivashinski_custom_40d_22l_05t_npfft_128_precision(self):
        sys_dim = 40
        sys_length = 22
        dt = 0.5
        time_steps = 10
        fft_type = "numpy"

        sim_data = scan.simulations.KuramotoSivashinskyCustom(
            sys_dim=sys_dim,
            sys_length=sys_length,
            dt=dt,
            precision=128,
            fft_type=fft_type,
        ).simulate(time_steps=time_steps)

        exp_sim_data = scan.simulations.KuramotoSivashinskyCustom(
            sys_dim=sys_dim,
            sys_length=sys_length,
            dt=dt,
            precision=64,
            fft_type=fft_type,
        ).simulate(time_steps=time_steps)

        assert_array_not_equal(sim_data, exp_sim_data)

    def test_kuramoto_sivashinski_dimension_missmatch(self):
        sys_dim = 10
        time_steps = 10
        starting_point = np.ones(11)

        with pytest.raises(ValueError):

            scan.simulations.KuramotoSivashinsky(
                sys_dim=sys_dim,
            ).simulate(time_steps=time_steps, starting_point=starting_point)

    def test_kuramoto_sivashinski_custom_dimension_missmatch(self):
        sys_dim = 10
        time_steps = 10
        starting_point = np.ones(11)

        with pytest.raises(ValueError):

            scan.simulations.KuramotoSivashinskyCustom(
                sys_dim=sys_dim,
            ).simulate(time_steps=time_steps, starting_point=starting_point)

    def test_kuramoto_sivashinski_odd_dimension_error(self):
        dimension = 3
        with pytest.raises(ValueError):
            scan.simulations.KuramotoSivashinsky(sys_dim=dimension)

    def test_kuramoto_sivashinski_custom_odd_dimension_error(self):
        dimension = 3
        with pytest.raises(ValueError):
            scan.simulations.KuramotoSivashinskyCustom(sys_dim=dimension)

    def test_kuramoto_sivashinski_custom_40d_22l_05t_unknown_precision(self):
        sys_dim = 40
        sys_length = 22
        dt = 0.5
        time_steps = 10
        fft_type = "numpy"

        with pytest.raises(ValueError):

            scan.simulations.KuramotoSivashinskyCustom(
                sys_dim=sys_dim,
                sys_length=sys_length,
                dt=dt,
                precision="this_precision_does_not_exist",
                fft_type=fft_type,
            ).simulate(time_steps=time_steps)

    def test_kuramoto_sivashinski_custom_40d_22l_05t_unknown_fft_type(self):
        sys_dim = 40
        sys_length = 22
        dt = 0.5
        time_steps = 10

        with pytest.raises(ValueError):

            scan.simulations.KuramotoSivashinskyCustom(
                sys_dim=sys_dim,
                sys_length=sys_length,
                dt=dt,
                precision=None,
                fft_type="this_ffttype_does_not_exist",
            ).simulate(time_steps=time_steps)


class TestKuramotSivashinskiVariantsDivergence40d22l05t(unittest.TestCase):
    def setUp(self):
        # self.ks_sys_flag = "kuramoto_sivashinsky_custom"
        self.sys_dim = 40
        self.sys_length = 22
        self.dt = 0.5
        self.time_steps = 1000

    @pytest.mark.xfail(reason="the KS simulation sometimes diverges using the numpy FFT")
    def test_kuramoto_sivashinski_custom_40d_22l_05t_divergence(self):
        # NOTE: This isn't actually all that great of a test, as it doesn't really test for the actual logic/algorithm
        #  we implemented and only checks if it diverges or not. Refactor by usinge e.g. a more sophisticated Lyapunov
        #  exponent test.
        sys_dim = 40
        sys_length = 22
        dt = 0.5
        time_steps = 1000

        sim_data = scan.simulations.KuramotoSivashinsky(
            sys_dim=sys_dim,
            sys_length=sys_length,
            dt=dt,
        ).simulate(time_steps=time_steps)

        assert not np.isnan(sim_data).any()
        assert np.amax(np.abs(sim_data)) < 100

    @pytest.mark.xfail(reason="the KS simulation sometimes diverges using the numpy FFT")
    def test_kuramoto_sivashinski_custom_40d_22l_05t_npfft_64bit_divergence(self):
        # NOTE: This isn't actually all that great of a test, as it doesn't really test for the actual logic/algorithm
        #  we implemented and only checks if it diverges or not. Refactor by usinge e.g. a more sophisticated Lyapunov
        #  exponent test.
        precision = 64
        fft_type = "numpy"

        sim_data = scan.simulations.KuramotoSivashinskyCustom(
            sys_dim=self.sys_dim,
            sys_length=self.sys_length,
            dt=self.dt,
            precision=precision,
            fft_type=fft_type,
        ).simulate(time_steps=self.time_steps)

        assert not np.isnan(sim_data).any()
        assert np.amax(np.abs(sim_data)) < 100

    @pytest.mark.xfail(reason="the KS simulation sometimes diverges using the numpy FFT")
    def test_kuramoto_sivashinski_custom_40d_22l_05t_npfft_128_divergence(self):
        # NOTE: This isn't actually all that great of a test, as it doesn't really test for the actual logic/algorithm
        #  we implemented and only checks if it diverges or not. Refactor by usinge e.g. a more sophisticated Lyapunov
        #  exponent test.
        precision = 128
        fft_type = "numpy"

        sim_data = scan.simulations.KuramotoSivashinskyCustom(
            sys_dim=self.sys_dim,
            sys_length=self.sys_length,
            dt=self.dt,
            precision=precision,
            fft_type=fft_type,
        ).simulate(time_steps=self.time_steps)

        assert not np.isnan(sim_data).any()
        assert np.amax(np.abs(sim_data)) < 100

    @pytest.mark.xfail(reason="the KS simulation sometimes diverges using the numpy FFT")
    def test_kuramoto_sivashinski_custom_40d_22l_05t_npfft_32_divergence(self):
        # NOTE: This isn't actually all that great of a test, as it doesn't really test for the actual logic/algorithm
        #  we implemented and only checks if it diverges or not. Refactor by usinge e.g. a more sophisticated Lyapunov
        #  exponent test.
        precision = 32
        fft_type = "numpy"

        sim_data = scan.simulations.KuramotoSivashinskyCustom(
            sys_dim=self.sys_dim,
            sys_length=self.sys_length,
            dt=self.dt,
            precision=precision,
            fft_type=fft_type,
        ).simulate(time_steps=self.time_steps)

        assert not np.isnan(sim_data).any()
        assert np.amax(np.abs(sim_data)) < 100

    @pytest.mark.xfail(reason="the KS simulation sometimes diverges using the numpy FFT")
    def test_kuramoto_sivashinski_custom_40d_22l_05t_npfft_16_divergence(self):
        # NOTE: This isn't actually all that great of a test, as it doesn't really test for the actual logic/algorithm
        #  we implemented and only checks if it diverges or not. Refactor by usinge e.g. a more sophisticated Lyapunov
        #  exponent test.
        precision = 16
        fft_type = "numpy"

        sim_data = scan.simulations.KuramotoSivashinskyCustom(
            sys_dim=self.sys_dim,
            sys_length=self.sys_length,
            dt=self.dt,
            precision=precision,
            fft_type=fft_type,
        ).simulate(time_steps=self.time_steps)

        assert not np.isnan(sim_data).any()
        assert np.amax(np.abs(sim_data)) < 100

    def test_kuramoto_sivashinski_custom_40d_22l_05t_scfft_64bit_divergence(self):
        # NOTE: This isn't actually all that great of a test, as it doesn't really test for the actual logic/algorithm
        #  we implemented and only checks if it diverges or not. Refactor by usinge e.g. a more sophisticated Lyapunov
        #  exponent test.
        pytest.importorskip("scipy")
        precision = 64
        fft_type = "scipy"

        sim_data = scan.simulations.KuramotoSivashinskyCustom(
            sys_dim=self.sys_dim,
            sys_length=self.sys_length,
            dt=self.dt,
            precision=precision,
            fft_type=fft_type,
        ).simulate(time_steps=self.time_steps)

        assert not np.isnan(sim_data).any()
        assert np.amax(np.abs(sim_data)) < 100
