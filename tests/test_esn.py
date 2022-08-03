""" Tests if the scan.esn module works as it should """

import copy

import numpy as np
import pytest

import scan
from tests.test_base import TestScanBase, assert_array_almost_equal, assert_array_equal


# Use print(np.array_repr(np_array, max_line_width=120, precision=18)) in the debugger to easily get the copy-pastable
# representation of a numpy array.
class TestESN(TestScanBase):
    def setUp(self):
        self.set_seed()
        self.esn = scan.ESN()
        np.set_printoptions(linewidth=120, precision=20)

    def tearDown(self):
        del self.esn
        np.random.seed(None)
        np.set_printoptions()

    def create_network_3x3_rand_simple(self, **kwargs):
        n_dim = 3
        n_rad = 0.1
        n_avg_deg = 5.0
        n_type_flag = "erdos_renyi"
        return self.esn.create_network(n_dim=n_dim, n_rad=n_rad, n_avg_deg=n_avg_deg, n_type_flag=n_type_flag, **kwargs)

    def train_simple_3x3_x_train(self, **kwargs):
        sync_steps = 1
        x_train = np.array([[-1, 0, 1], [-1.2, -0.2, 0.8], [-1.4, -0.4, 0.6]])
        self.esn.train(x_train=x_train, sync_steps=sync_steps, **kwargs)

    def train_simple_3x1_x_train(self):
        sync_steps = 1
        x_train = np.array([[1], [1.2], [1.4]])
        self.esn.train(x_train=x_train, sync_steps=sync_steps)

    def test_create_network_random(self):
        # NOTE: This isn't actually all that great of a test, as it doesn't really test for the actual logic/algorithm
        #  we implemented and only checks for consistency with a successfully run example. Refactor if possible.
        csr_network = self.create_network_3x3_rand_simple()
        np_network = csr_network.toarray()
        np_network_desired = np.array(
            [
                [0.0, 0.023520089126284775, -0.03953395373715219],
                [-0.07882747240843854, 0.0, -0.08302550388252539],
                [0.0537971172135872, 0.0708814863105328, 0.0],
            ]
        )
        # Here too, the last bit of the network generation should get flipped occasionally, depending on soft- and
        # hardware config. It seems like the sofware (numpy?) backend changing is enough for that bit to be flipped
        assert_array_almost_equal(np_network_desired, np_network, decimal=15)

    def test_create_network_scale_free(self):
        # NOTE: This isn't actually all that great of a test, as it doesn't really test for the actual logic/algorithm
        #  we implemented and only checks for consistency with a successfully run example. Refactor if possible.
        n_dim = 3
        n_rad = 0.1
        n_avg_deg = 5.0
        n_type_flag = "scale_free"
        csr_network = self.esn.create_network(n_dim=n_dim, n_rad=n_rad, n_avg_deg=n_avg_deg, n_type_flag=n_type_flag)
        np_network = csr_network.toarray()
        np_network_desired = np.array(
            [
                [0.0, 0.06223427902373477, -0.10460704866291423],
                [-0.20857790488217556, 0.0, 0.0],
                [-0.21968591815142674, 0.0, 0.0],
            ]
        )
        # Here too, the last bit of the network generation should get flipped occasionally, depending on soft- and
        # hardware config. It seems like the sofware (numpy?) backend changing is enough for that bit to be flipped
        assert_array_almost_equal(np_network_desired, np_network, decimal=15)

    def test_create_network_small_world(self):
        # NOTE: This isn't actually all that great of a test, as it doesn't really test for the actual logic/algorithm
        #  we implemented and only checks for consistency with a successfully run example. Refactor if possible.
        n_dim = 3
        n_rad = 0.1
        n_avg_deg = 2.0
        n_type_flag = "small_world"
        csr_network = self.esn.create_network(n_dim=n_dim, n_rad=n_rad, n_avg_deg=n_avg_deg, n_type_flag=n_type_flag)
        np_network = csr_network.toarray()
        np_network_desired = np.array(
            [
                [0.0, 0.023520089126284775, -0.03953395373715219],
                [-0.07882747240843854, 0.0, -0.08302550388252539],
                [0.0537971172135872, 0.0708814863105328, 0.0],
            ]
        )

        # Here too, the last bit of the network generation should get flipped occasionally, depending on soft- and
        # hardware config. It seems like the sofware (numpy?) backend changing is enough for that bit to be flipped
        assert_array_almost_equal(np_network_desired, np_network, decimal=15)

    def test_create_network_unknown_n_type_flag(self):
        n_dim = 3
        n_rad = 0.1
        n_avg_deg = 2.0
        n_type_flag = "this_flag_doesnt_exist"

        with pytest.raises(ValueError):
            self.esn.create_network(n_dim=n_dim, n_rad=n_rad, n_avg_deg=n_avg_deg, n_type_flag=n_type_flag)

    def test_train_simple(self):
        # NOTE: This isn't actually all that great of a test, as it doesn't really test for the actual logic/algorithm
        #  we implemented and only checks for consistency with a successfully run example. Refactor if possible.
        self.create_network_3x3_rand_simple()
        self.train_simple_3x3_x_train()

        last_r = self.esn.last_r
        w_out = self.esn.w_out

        exp_last_r = np.array([-0.33315555686551146, 0.25772951756691614, -0.6498301009377943])
        exp_w_out = np.array(
            [
                [0.7777436853182361, -0.6016633992095993, 1.517012840155533],
                [0.22221248151973944, -0.1719038283452853, 0.433432240044438],
                [-0.3333187222781988, 0.25785574251722454, -0.650148360067659],
            ]
        )
        assert_array_almost_equal(exp_last_r, last_r)
        assert_array_almost_equal(exp_w_out, w_out)

    def test_train_x_train_1d(self):
        # NOTE: This isn't actually all that great of a test, as it doesn't really test for the actual logic/algorithm
        #  we implemented and only checks for consistency with a successfully run example. Refactor if possible.
        self.create_network_3x3_rand_simple()

        sync_steps = 1
        x_train = np.array([-1, -1.2, -1.4])
        self.esn.train(x_train=x_train, sync_steps=sync_steps)

        last_r = self.esn.last_r
        w_out = self.esn.w_out

        exp_last_r = np.array([-0.3339303492308946, 0.4725814006660192, 0.8068004882223478])
        exp_w_out = np.array([[0.4742464218594163, -0.6711580388515656, -1.145814525644173]])
        assert_array_almost_equal(exp_last_r, last_r)
        assert_array_almost_equal(exp_w_out, w_out)

    def test_train_x_train_2d(self):
        # NOTE: This isn't actually all that great of a test, as it doesn't really test for the actual logic/algorithm
        #  we implemented and only checks for consistency with a successfully run example. Refactor if possible.
        self.create_network_3x3_rand_simple()

        self.train_simple_3x3_x_train()
        last_r = self.esn.last_r
        w_out = self.esn.w_out

        exp_last_r = np.array([-0.33315555686551146, 0.25772951756691614, -0.6498301009377943])
        exp_w_out = np.array(
            [
                [0.7777436853182361, -0.6016633992095993, 1.517012840155533],
                [0.22221248151973944, -0.1719038283452853, 0.433432240044438],
                [-0.3333187222781988, 0.25785574251722454, -0.650148360067659],
            ]
        )
        assert_array_almost_equal(exp_last_r, last_r)
        assert_array_almost_equal(exp_w_out, w_out)

    def test_train_x_train_2d_loc_nbhd_all2s(self):
        loc_nbhd = np.array([2, 2, 2])
        self.create_network_3x3_rand_simple()
        self.train_simple_3x3_x_train(loc_nbhd=loc_nbhd)
        last_r = self.esn.last_r
        w_out = self.esn.w_out

        self.reset()

        self.create_network_3x3_rand_simple()
        self.train_simple_3x3_x_train()
        exp_last_r = self.esn.last_r
        exp_w_out = self.esn.w_out

        assert_array_almost_equal(exp_last_r, last_r)
        assert_array_almost_equal(exp_w_out, w_out)

    def test_train_x_train_2d_loc_nbhd_fit_to_one_dim(self):
        # NOTE: This isn't actually all that great of a test, as it doesn't really test for the actual logic/algorithm
        #  we implemented and only checks for consistency with a successfully run example. Refactor if possible.
        loc_nbhd = np.array([0, 2, 1])
        self.create_network_3x3_rand_simple()
        self.train_simple_3x3_x_train(loc_nbhd=loc_nbhd)
        last_r = self.esn.last_r
        w_out = self.esn.w_out

        exp_last_r = np.array([-0.33315555686551146, 0.25772951756691614, -0.6498301009377943])
        exp_w_out = np.array([[0.22221248151954512, -0.17190382834531942, 0.433432240044524]])

        assert_array_almost_equal(exp_last_r, last_r)
        assert_array_almost_equal(exp_w_out, w_out)

    def test_train_x_train_3d_single_slice(self):
        sync_steps = 0
        x_train_2d = np.array([[-1, 0, 1], [-1.2, -0.2, 0.8], [-1.4, -0.4, 0.6]])
        x_train_3d = x_train_2d[:, :, np.newaxis]

        self.create_network_3x3_rand_simple()
        self.esn.train(x_train=x_train_2d, sync_steps=sync_steps)
        exp_last_r = self.esn.last_r
        exp_w_out = self.esn.w_out

        self.reset()

        self.create_network_3x3_rand_simple()
        self.esn.train(x_train=x_train_3d, sync_steps=sync_steps)
        last_r = self.esn.last_r
        w_out = self.esn.w_out

        assert_array_almost_equal(exp_last_r, last_r)
        assert_array_almost_equal(exp_w_out, w_out)

    def test_train_x_train_3d_multiple_slices(self):
        x_train_2d = np.array(
            [
                [-1, 0, 1],
                [-1.2, -0.2, 0.8],
                [0.0, 1.0, 2.0],
                [-0.2, 0.8, 1.8],
                [1.5, 2.5, 3.5],
                [1.3, 2.3, 3.3],
                [-2.3, -1.3, -0.3],
                [-2.5, -1.5, -0.5],
                [9, 9, 9],
            ]
        )
        x_train_3d = np.array(
            [
                [[-1.0, 0.0, 1.5, -2.3], [0.0, 1.0, 2.5, -1.3], [1.0, 2.0, 3.5, -0.3]],
                [[-1.2, -0.2, 1.3, -2.5], [-0.2, 0.8, 2.3, -1.5], [0.8, 1.8, 3.3, -0.5]],
                [[0.0, 1.5, -2.3, 9.0], [1.0, 2.5, -1.3, 9.0], [2.0, 3.5, -0.3, 9.0]],
            ]
        )
        sync_steps = 0
        reset_r = False

        self.create_network_3x3_rand_simple()
        self.esn.train(x_train=x_train_3d, sync_steps=sync_steps, reset_r=reset_r)
        last_r = self.esn.last_r
        w_out = self.esn.w_out

        self.reset()

        self.create_network_3x3_rand_simple()
        self.esn.train(x_train=x_train_2d, sync_steps=sync_steps, reset_r=reset_r)
        exp_last_r = self.esn.last_r
        exp_w_out = self.esn.w_out

        assert_array_almost_equal(exp_last_r, last_r)
        assert_array_almost_equal(exp_w_out, w_out)

    @pytest.mark.skip(reason="Test TODO")
    def test_train_x_train_3d_sync_steps(self):
        raise Exception

    def test_train_save_r_x_train_3d_multiple_slices(self):
        x_train_2d = np.array(
            [
                [-1, 0, 1],
                [-1.2, -0.2, 0.8],
                [0.0, 1.0, 2.0],
                [-0.2, 0.8, 1.8],
                [1.5, 2.5, 3.5],
                [1.3, 2.3, 3.3],
                [-2.3, -1.3, -0.3],
                [-2.5, -1.5, -0.5],
                [9, 9, 9],
            ]
        )
        x_train_3d = np.array(
            [
                [[-1.0, 0.0, 1.5, -2.3], [0.0, 1.0, 2.5, -1.3], [1.0, 2.0, 3.5, -0.3]],
                [[-1.2, -0.2, 1.3, -2.5], [-0.2, 0.8, 2.3, -1.5], [0.8, 1.8, 3.3, -0.5]],
                [[0.0, 1.5, -2.3, 9.0], [1.0, 2.5, -1.3, 9.0], [2.0, 3.5, -0.3, 9.0]],
            ]
        )
        sync_steps = 0
        reset_r = False
        save_r = True

        self.create_network_3x3_rand_simple()
        self.esn.train(x_train=x_train_2d, sync_steps=sync_steps, reset_r=reset_r, save_r=save_r)
        exp_last_r = self.esn.last_r
        exp_w_out = self.esn.w_out
        exp_r_train = self.esn.r_train

        self.reset()

        self.create_network_3x3_rand_simple()
        self.esn.train(x_train=x_train_3d, sync_steps=sync_steps, reset_r=reset_r, save_r=save_r)
        last_r = self.esn.last_r
        w_out = self.esn.w_out
        r_train = self.esn.r_train

        assert_array_almost_equal(exp_last_r, last_r)
        assert_array_almost_equal(exp_w_out, w_out)
        assert_array_almost_equal(exp_r_train, r_train)

    def test_train_dense_w_in(self):
        # NOTE: This isn't actually all that great of a test, as it doesn't really test for the actual logic/algorithm
        #  we implemented and only checks for consistency with a successfully run example. Refactor if possible.
        self.create_network_3x3_rand_simple()

        self.train_simple_3x3_x_train(w_in_sparse=False)
        last_r = self.esn.last_r
        w_out = self.esn.w_out

        exp_last_r = np.array([-0.7369901104230075, 0.9437586335830703, -0.2501613493112319])
        exp_w_out = np.array(
            [
                [0.6895005244113604, -0.8829454609648854, 0.2340416500872294],
                [0.19700014983056988, -0.25227013170573065, 0.06686904288016592],
                [-0.2955002247473723, 0.3784051975577635, -0.10030356431891914],
            ]
        )
        assert_array_almost_equal(exp_last_r, last_r)
        assert_array_almost_equal(exp_w_out, w_out)

    @pytest.mark.skip(reason="Test TODO")
    def test_train_dense_w_in_1dim_input(self):
        # NOTE: Should be the same as sparse if the input is 1dim so test for that
        raise Exception

    def test_train_w_in_no_update(self):
        train_sync_steps = np.random.randint(10, 100)
        train_steps = np.random.randint(10, 100)
        dim = np.random.randint(10, 100)
        simulation_time_steps = train_sync_steps + train_steps
        sim_data = np.random.random((simulation_time_steps, dim))

        self.create_network_3x3_rand_simple()

        self.esn.train(x_train=sim_data, sync_steps=train_sync_steps, reset_r=True, w_in_seed=0)
        exp_last_r = self.esn.last_r
        exp_w_out = self.esn.w_out

        self.esn.train(x_train=sim_data, sync_steps=train_sync_steps, reset_r=True, w_in_no_update=True, w_in_seed=1)
        last_r = self.esn.last_r
        w_out = self.esn.w_out
        assert_array_almost_equal(exp_last_r, last_r)
        assert_array_almost_equal(exp_w_out, w_out)

        self.esn.train(x_train=sim_data, sync_steps=train_sync_steps, reset_r=True, w_in_no_update=False, w_in_seed=1)
        changed_last_r = self.esn.last_r
        changed_w_out = self.esn.w_out
        with pytest.raises(AssertionError):
            assert_array_almost_equal(changed_last_r, last_r)
        with pytest.raises(AssertionError):
            assert_array_almost_equal(changed_w_out, w_out)

    def test_train_w_in_no_update_changed_x_dim(self):
        self.create_network_3x3_rand_simple()

        train_sync_steps = np.random.randint(10, 100)
        train_steps = np.random.randint(10, 100)
        dim = np.random.randint(10, 100)

        simulation_time_steps = train_sync_steps + train_steps

        sim_data = np.random.random((simulation_time_steps, dim))

        self.esn.train(x_train=sim_data, sync_steps=train_sync_steps, reset_r=True)

        sim_data_larger_dim = np.random.random((simulation_time_steps, dim + 1))

        with pytest.raises(ValueError):
            self.esn.train(x_train=sim_data_larger_dim, sync_steps=train_sync_steps, reset_r=True, w_in_no_update=True)

    @pytest.mark.skip(reason="Test TODO")
    def test_train_save_r(self):
        raise Exception

    @pytest.mark.skip(reason="Test TODO")
    def test_train_save_input(self):
        raise Exception

    def test_train_w_out_fit_linear_and_square_r(self):
        # NOTE: This isn't actually all that great of a test, as it doesn't really test for the actual logic/algorithm
        #  we implemented and only checks for consistency with a successfully run example. Refactor if possible.
        n_dim = 3
        n_rad = 0.1
        n_avg_deg = 5.0
        n_type_flag = "erdos_renyi"
        self.esn.create_network(n_dim=n_dim, n_rad=n_rad, n_avg_deg=n_avg_deg, n_type_flag=n_type_flag)

        sync_steps = 1
        x_train = np.array([[-1, 0, 1], [-1.2, -0.2, 0.8], [-1.4, -0.4, 0.6]])
        self.esn.train(x_train=x_train, sync_steps=sync_steps, w_out_fit_flag="linear_and_square_r")

        last_r = self.esn.last_r
        w_out = self.esn.w_out

        exp_last_r = np.array([-0.33315555686551146, 0.25772951756691614, -0.6498301009377943])
        exp_w_out = np.array(
            [
                [
                    0.586868002806045,
                    -0.4540017542088528,
                    1.1447039848057772,
                    -0.19551833628249188,
                    -0.11700965308501443,
                    -0.7438631059947339,
                ],
                [
                    0.16767657223184046,
                    -0.1297147869167807,
                    0.3270582813737855,
                    -0.055862381795379404,
                    -0.0334313294534661,
                    -0.21253231599601755,
                ],
                [
                    -0.2515148583458059,
                    0.1945721803736588,
                    -0.4905874220596409,
                    0.08379357269241784,
                    0.05014699417965821,
                    0.3187984739983442,
                ],
            ]
        )

        assert_array_almost_equal(exp_last_r, last_r)
        assert_array_almost_equal(exp_w_out, w_out)

    def test_train_unknown_w_out_fit_flag(self):
        self.create_network_3x3_rand_simple()

        sync_steps = 1
        x_train = np.array([[-1, 0, 1], [-1.2, -0.2, 0.8], [-1.4, -0.4, 0.6]])
        with pytest.raises(ValueError):
            self.esn.train(x_train=x_train, sync_steps=sync_steps, w_out_fit_flag="this_flag_doesnt_exist")

    def test_train_unknown_act_fct_flag(self):
        self.create_network_3x3_rand_simple()

        sync_steps = 1
        x_train = np.array([[-1, 0, 1], [-1.2, -0.2, 0.8], [-1.4, -0.4, 0.6]])
        with pytest.raises(ValueError):
            self.esn.train(x_train=x_train, sync_steps=sync_steps, act_fct_flag="this_flag_doesnt_exist")

    def test_train_loc_nbhd(self):
        # NOTE: This isn't actually all that great of a test, as it doesn't really test for the actual logic/algorithm
        #  we implemented and only checks for consistency with a successfully run example. Refactor if possible.
        self.create_network_3x3_rand_simple()

        sync_steps = 1
        x_train = np.array([[-1, 0, 1], [-1.2, -0.2, 0.8], [-1.4, -0.4, 0.6]])
        loc_nbhd = np.array([2, 1, 0])
        self.esn.train(x_train=x_train, sync_steps=sync_steps, loc_nbhd=loc_nbhd)

        last_r = self.esn.last_r
        w_out = self.esn.w_out

        exp_last_r = np.array([-0.33315555686551146, 0.25772951756691614, -0.6498301009377943])
        exp_w_out = np.array([[0.7777436853148857, -0.6016633992101881, 1.517012840157017]])
        assert_array_almost_equal(exp_last_r, last_r)
        assert_array_almost_equal(exp_w_out, w_out)

    def test_predict_simple(self):
        # NOTE: This isn't actually all that great of a test, as it doesn't really test for the actual logic/algorithm
        #  we implemented and only checks for consistency with a successfully run example. Refactor if possible.
        self.create_network_3x3_rand_simple()
        self.train_simple_3x3_x_train()
        x_pred = np.array([[-0.7, 1, 0.2], [-0.2, 0.3, 0.06], [0.3, -0.5, 0], [-0.6, 0.7, 0.3]])

        y_pred, y_test = self.esn.predict(x_pred, sync_steps=0)
        exp_y_test = x_pred[1:]
        exp_y_pred = np.array(
            [
                [-0.26094129545030204, -0.07455465584319378, 0.11183198376558444],
                [-0.4346607260032235, -0.12418877885804211, 0.18628316828712932],
                [-0.5355423450433661, -0.15301209858379447, 0.22951814787575317],
            ]
        )
        assert_array_equal(exp_y_test, y_test)
        assert_array_almost_equal(exp_y_pred, y_pred)

    def test_predict_simple_reset_r_False(self):
        # NOTE: This isn't actually all that great of a test, as it doesn't really test for the actual logic/algorithm
        #  we implemented and only checks for consistency with a successfully run example. Refactor if possible.
        self.create_network_3x3_rand_simple()
        self.train_simple_3x3_x_train()
        x_pred = np.array([[-0.7, 1, 0.2], [-0.2, 0.3, 0.06], [0.3, -0.5, 0], [-0.6, 0.7, 0.3]])

        y_pred, y_test = self.esn.predict(x_pred, sync_steps=0, reset_r=False)
        exp_y_test = np.array([[-0.2, 0.3, 0.06], [0.3, -0.5, 0.0], [-0.6, 0.7, 0.3]])
        exp_y_pred = np.array(
            [
                [-0.2602107081464973, -0.07434591661351493, 0.111518874921082],
                [-0.42495746674642076, -0.12141641907038478, 0.18212462860564194],
                [-0.5240465897348768, -0.1497275970670838, 0.22459139560068606],
            ]
        )
        assert_array_equal(exp_y_test, y_test)
        assert_array_almost_equal(exp_y_pred, y_pred)

    def test_predict_x_pred_1d(self):
        # NOTE: This isn't actually all that great of a test, as it doesn't really test for the actual logic/algorithm
        #  we implemented and only checks for consistency with a successfully run example. Refactor if possible.
        self.create_network_3x3_rand_simple()

        sync_steps = 1
        x_train = np.array([-1, -1.2, -1.4])
        self.esn.train(x_train=x_train, sync_steps=sync_steps)

        x_pred = np.array([-0.7, -0.2, 0.3, -0.6])

        y_pred, y_test = self.esn.predict(x_pred, sync_steps=0)

        # exp_y_test = np.array([-0.2, 0.3, -0.6])
        exp_y_test = x_pred[1:]
        exp_y_pred = np.array([-0.9482831338803914, -1.195576886629772, -1.3963891404575148])
        assert_array_equal(exp_y_test, y_test)
        assert_array_almost_equal(exp_y_pred, y_pred)

    def test_predict_x_pred_2d(self):
        # NOTE: This isn't actually all that great of a test, as it doesn't really test for the actual logic/algorithm
        #  we implemented and only checks for consistency with a successfully run example. Refactor if possible.
        self.create_network_3x3_rand_simple()
        self.train_simple_3x3_x_train()

        x_pred = np.array([[-1, 0, 1], [-1.2, -0.2, 0.8], [-1.4, -0.4, 0.6]])

        y_pred, y_test = self.esn.predict(x_pred, sync_steps=0)

        exp_y_test = x_pred[1:]
        exp_y_pred = np.array(
            [
                [-1.1777172761799404, -0.3364906503372306, 0.5047359755057956],
                [-1.357642798619048, -0.38789794246251913, 0.5818469136938685],
            ]
        )
        assert_array_equal(exp_y_test, y_test)
        assert_array_almost_equal(exp_y_pred, y_pred)

    def test_predict_x_pred_2d_loc_nbhd_all2s(self):
        loc_nbhd = np.array([2, 2, 2])
        self.create_network_3x3_rand_simple()
        self.train_simple_3x3_x_train(loc_nbhd=loc_nbhd)

        x_pred = np.array([[-1, 0, 1], [-1.2, -0.2, 0.8], [-1.4, -0.4, 0.6]])

        y_pred, y_test = self.esn.predict(x_pred, sync_steps=0)

        self.reset()

        self.create_network_3x3_rand_simple()
        self.train_simple_3x3_x_train()

        x_pred = np.array([[-1, 0, 1], [-1.2, -0.2, 0.8], [-1.4, -0.4, 0.6]])

        exp_y_pred, exp_y_test = self.esn.predict(x_pred, sync_steps=0)

        assert_array_equal(exp_y_test, y_test)
        assert_array_almost_equal(exp_y_pred, y_pred)

    def test_predict_x_pred_2d_loc_nbhd_fit_to_one_dim(self):
        # NOTE: This isn't actually all that great of a test, as it doesn't really test for the actual logic/algorithm
        #  we implemented and only checks for consistency with a successfully run example. Refactor if possible.
        loc_nbhd = np.array([0, 2, 1])
        self.create_network_3x3_rand_simple()
        self.train_simple_3x3_x_train(loc_nbhd=loc_nbhd)

        x_pred = np.array([[-1, 0, 1], [-1.2, -0.2, 0.8], [-1.4, -0.4, 0.6]])

        y_pred, y_test = self.esn.predict(x_pred, sync_steps=0)

        exp_y_test = x_pred[1:]
        exp_y_pred = np.array([[np.nan, -0.3364906503371948, np.nan], [np.nan, np.nan, np.nan]])
        assert_array_equal(exp_y_test, y_test)
        assert_array_almost_equal(exp_y_pred, y_pred)

    def test_predict_x_pred_3d_single_slice(self):
        x_pred_2d = np.array([[-0.7, 1, 0.2], [-0.2, 0.3, 0.06], [0.3, -0.5, 0], [-0.6, 0.7, 0.3]])
        x_pred_3d = x_pred_2d[:, :, np.newaxis]

        self.create_network_3x3_rand_simple()
        self.train_simple_3x3_x_train()
        exp_y_pred, exp_y_test = self.esn.predict(x_pred_2d, sync_steps=0)
        exp_y_pred = exp_y_pred[:, :, np.newaxis]
        exp_y_test = exp_y_test[:, :, np.newaxis]

        self.reset()

        self.create_network_3x3_rand_simple()
        self.train_simple_3x3_x_train()
        y_pred, y_test = self.esn.predict(x_pred_3d, sync_steps=0)

        assert_array_equal(exp_y_test, y_test)
        assert_array_almost_equal(exp_y_pred, y_pred)

    def test_predict_x_pred_3d_multiple_slices(self):
        input_dim = 3  # needs to be 3 to use the self.train_simple_3x3_x_train() below
        x_time_steps = 5
        pred_slices = 4

        sync_steps = 0
        reset_r = True

        x_pred_3d = np.random.rand(x_time_steps, input_dim, pred_slices) - 0.5
        self.set_seed()

        # Not a big fan of explicitly defining the output arrays' dimensions here as, in principle, they could change
        # without effecting the test in such a way that it should fail.
        exp_y_pred = np.zeros(shape=(x_time_steps - sync_steps - 1, input_dim, pred_slices))
        exp_y_test = np.zeros(shape=(x_time_steps - sync_steps - 1, input_dim, pred_slices))

        for slice_nr in range(pred_slices):
            self.create_network_3x3_rand_simple()
            self.train_simple_3x3_x_train()
            exp_y_pred[:, :, slice_nr], exp_y_test[:, :, slice_nr] = self.esn.predict(
                x_pred_3d[:, :, slice_nr], sync_steps=sync_steps, reset_r=reset_r
            )

            self.reset()

        self.create_network_3x3_rand_simple()
        self.train_simple_3x3_x_train()
        y_pred, y_test = self.esn.predict(x_pred_3d, sync_steps=sync_steps, reset_r=reset_r)

        assert_array_equal(exp_y_test, y_test)
        assert_array_almost_equal(exp_y_pred, y_pred)

    def test_predict_x_pred_3d_multiple_slices_1d_input(self):
        # NOTE: This test is kinda superfluous, I think
        input_dim = 1
        x_time_steps = 5
        pred_slices = 4

        sync_steps = 0
        reset_r = True

        x_pred_3d = np.random.rand(x_time_steps, input_dim, pred_slices) - 0.5
        self.set_seed()

        # Not a big fan of explicitly defining the output arrays' dimensions here as, in principle, they could change
        # without effecting the test in such a way that it should fail.
        exp_y_pred = np.zeros(shape=(x_time_steps - sync_steps - 1, input_dim, pred_slices))
        exp_y_test = np.zeros(shape=(x_time_steps - sync_steps - 1, input_dim, pred_slices))

        for slice_nr in range(pred_slices):
            self.create_network_3x3_rand_simple()
            self.train_simple_3x1_x_train()
            exp_y_pred[:, 0, slice_nr], exp_y_test[:, 0, slice_nr] = self.esn.predict(
                x_pred_3d[:, 0, slice_nr], sync_steps=sync_steps, reset_r=reset_r
            )

            self.reset()

        self.create_network_3x3_rand_simple()
        self.train_simple_3x1_x_train()
        y_pred, y_test = self.esn.predict(x_pred_3d, sync_steps=sync_steps, reset_r=reset_r)

        assert_array_equal(exp_y_test, y_test)
        assert_array_almost_equal(exp_y_pred, y_pred)

    def test_predict_x_pred_3d_sync_steps(self):
        input_dim = 3  # needs to be 3 to use the self.train_simple_3x3_x_train() below
        x_time_steps = 5
        pred_slices = 4

        sync_steps = 1
        reset_r = True

        x_pred_3d = np.random.rand(x_time_steps, input_dim, pred_slices) - 0.5
        self.set_seed()

        # Not a big fan of explicitly defining the output arrays' dimensions here as, in principle, they could change
        # without effecting the test in such a way that it should fail.
        exp_y_pred = np.zeros(shape=(x_time_steps - sync_steps - 1, input_dim, pred_slices))
        exp_y_test = np.zeros(shape=(x_time_steps - sync_steps - 1, input_dim, pred_slices))

        for slice_nr in range(pred_slices):
            self.create_network_3x3_rand_simple()
            self.train_simple_3x3_x_train()
            exp_y_pred[:, :, slice_nr], exp_y_test[:, :, slice_nr] = self.esn.predict(
                x_pred_3d[:, :, slice_nr], sync_steps=sync_steps, reset_r=reset_r
            )

            self.reset()

        self.create_network_3x3_rand_simple()
        self.train_simple_3x3_x_train()
        y_pred, y_test = self.esn.predict(x_pred_3d, sync_steps=sync_steps, reset_r=reset_r)

        assert_array_equal(exp_y_test, y_test)
        assert_array_almost_equal(exp_y_pred, y_pred)

    def test_predict_x_pred_3d_sync_steps_linear_and_square_r(self):
        input_dim = 3  # needs to be 3 to use the self.train_simple_3x3_x_train() below
        x_time_steps = 5
        pred_slices = 4

        sync_steps = 1
        reset_r = True

        x_pred_3d = np.random.rand(x_time_steps, input_dim, pred_slices) - 0.5
        self.set_seed()

        # Not a big fan of explicitly defining the output arrays' dimensions here as, in principle, they could change
        # without effecting the test in such a way that it should fail.
        exp_y_pred = np.zeros(shape=(x_time_steps - sync_steps - 1, input_dim, pred_slices))
        exp_y_test = np.zeros(shape=(x_time_steps - sync_steps - 1, input_dim, pred_slices))

        for slice_nr in range(pred_slices):
            self.create_network_3x3_rand_simple()
            self.train_simple_3x3_x_train(w_out_fit_flag="linear_and_square_r")
            exp_y_pred[:, :, slice_nr], exp_y_test[:, :, slice_nr] = self.esn.predict(
                x_pred_3d[:, :, slice_nr], sync_steps=sync_steps, reset_r=reset_r
            )

            self.reset()

        self.create_network_3x3_rand_simple()
        self.train_simple_3x3_x_train(w_out_fit_flag="linear_and_square_r")
        y_pred, y_test = self.esn.predict(x_pred_3d, sync_steps=sync_steps, reset_r=reset_r)

        assert_array_equal(exp_y_test, y_test)
        assert_array_almost_equal(exp_y_pred, y_pred)

    def test_predict_repeated_use_of_same_class_instance(self):
        input_dim = 3  # needs to be 3 to use the self.train_simple_3x3_x_train() below
        x_time_steps = 5
        pred_slices = 4

        sync_steps = 1
        reset_r = True

        x_pred_3d = np.random.rand(x_time_steps, input_dim, pred_slices) - 0.5
        self.set_seed()

        self.create_network_3x3_rand_simple()
        self.train_simple_3x3_x_train(reset_r=reset_r)
        exp_y_pred, exp_y_test = self.esn.predict(x_pred_3d, sync_steps=sync_steps, reset_r=reset_r)

        self.set_seed()
        self.create_network_3x3_rand_simple()
        self.train_simple_3x3_x_train(reset_r=reset_r)
        y_pred, y_test = self.esn.predict(x_pred_3d, sync_steps=sync_steps, reset_r=reset_r)

        assert_array_equal(exp_y_test, y_test)
        assert_array_almost_equal(exp_y_pred, y_pred)

    @pytest.mark.skip(reason="Test TODO")
    def test_predict_save_r(self):
        raise Exception

    def test_predict_save_r_x_pred_3d_multiple_slices(self):
        input_dim = 3  # needs to be 3 to use the self.train_simple_3x3_x_train() below
        x_time_steps = 5
        pred_slices = 4

        sync_steps = 0
        reset_r = True

        x_pred_3d = np.random.rand(x_time_steps, input_dim, pred_slices) - 0.5
        self.set_seed()

        # Not a big fan of explicitly defining the output arrays' dimensions here as, in principle, they could change
        # without effecting the test in such a way that it should fail.
        exp_y_pred = np.zeros(shape=(x_time_steps - sync_steps - 1, input_dim, pred_slices))
        exp_y_test = np.zeros(shape=(x_time_steps - sync_steps - 1, input_dim, pred_slices))
        exp_r_pred = np.zeros(shape=(x_time_steps - sync_steps - 1, 3, pred_slices))

        for slice_nr in range(pred_slices):
            self.create_network_3x3_rand_simple()
            self.train_simple_3x3_x_train()
            exp_y_pred[:, :, slice_nr], exp_y_test[:, :, slice_nr] = self.esn.predict(
                x_pred_3d[:, :, slice_nr], sync_steps=sync_steps, reset_r=reset_r, save_r=True
            )
            exp_r_pred[:, :, slice_nr] = self.esn.r_pred[:, :, 0]

            self.reset()

        self.create_network_3x3_rand_simple()
        self.train_simple_3x3_x_train()
        y_pred, y_test = self.esn.predict(x_pred_3d, sync_steps=sync_steps, reset_r=reset_r, save_r=True)
        r_pred = self.esn.r_pred

        assert_array_equal(exp_y_test, y_test)
        assert_array_almost_equal(exp_y_pred, y_pred)
        assert_array_almost_equal(exp_r_pred, r_pred)

    @pytest.mark.skip(reason="Test TODO")
    def test_predict_save_input(self):
        raise Exception

    def test_predict_sync_steps_longer_than_x_pred(self):
        dim = np.random.randint(10, 100)
        train_sync_steps = np.random.randint(10, 100)
        train_steps = np.random.randint(10, 100)
        pred_sync_steps = 4
        pred_steps = 2

        x_train = np.random.random((train_sync_steps + train_steps, dim))
        x_pred = np.random.random((0 + pred_steps + 1, dim))

        self.create_network_3x3_rand_simple()
        self.esn.train(x_train=x_train, sync_steps=train_sync_steps)
        with pytest.raises(ValueError):
            self.esn.predict(x_pred, sync_steps=pred_sync_steps, pred_steps=pred_steps)

    def test_predict_pred_steps_longer_than_x_pred(self):
        # NOTE: This isn't actually all that great of a test, as it doesn't really test for the actual logic/algorithm
        #  we implemented and only checks for consistency with a successfully run example. Refactor if possible.
        self.create_network_3x3_rand_simple()
        self.train_simple_3x3_x_train()
        x_pred = np.array([[-0.7, 1, 0.2], [-0.2, 0.3, 0.06], [0.3, -0.5, 0], [-0.6, 0.7, 0.3]])

        # this will predict TWO, not one, more steps, than if you were to not specify a pred_steps parameter at all,
        # because: y_test.shape[0] == x_pred.shape[0] - 1
        pred_steps = x_pred.shape[0] + 1

        y_pred, y_test = self.esn.predict(x_pred, pred_steps=pred_steps, sync_steps=0)
        exp_y_test = np.array([[-0.2, 0.3, 0.06], [0.3, -0.5, 0.0], [-0.6, 0.7, 0.3]])
        exp_y_pred = np.array(
            [
                [-0.26094129545030204, -0.07455465584319378, 0.11183198376558444],
                [-0.4346607260032235, -0.12418877885804211, 0.18628316828712932],
                [-0.5355423450433661, -0.15301209858379447, 0.22951814787575317],
                [-0.6547847956853798, -0.18708137019579282, 0.28062205529376166],
                [-0.7899640724085603, -0.22570402068812367, 0.33855603103226833],
            ]
        )
        assert_array_equal(exp_y_test, y_test)
        assert_array_almost_equal(exp_y_pred, y_pred)

    def test_predict_x_pred_longer_than_pred_steps(self):
        dim = np.random.randint(10, 100)
        train_sync_steps = np.random.randint(10, 100)
        train_steps = np.random.randint(10, 100)
        pred_sync_steps = np.random.randint(10, 100)
        pred_steps = np.random.randint(10, 100)

        x_train = np.random.random((train_sync_steps + train_steps, dim))
        x_pred = np.random.random((pred_sync_steps + pred_steps + 1, dim))
        x_pred_longer = np.append(x_pred, x_pred, axis=0)

        self.reset()
        self.create_network_3x3_rand_simple()
        self.esn.train(x_train=x_train, sync_steps=train_sync_steps)
        exp_y_pred, exp_y_test = self.esn.predict(x_pred, sync_steps=pred_sync_steps, pred_steps=pred_steps)

        self.reset()
        self.create_network_3x3_rand_simple()
        self.esn.train(x_train=x_train, sync_steps=train_sync_steps)
        y_pred, y_test = self.esn.predict(x_pred_longer, sync_steps=pred_sync_steps, pred_steps=pred_steps)

        assert_array_almost_equal(y_pred, exp_y_pred)
        assert_array_almost_equal(y_test, exp_y_test)

    @pytest.mark.skip(reason="Test TODO")
    def test_predict_pred_steps_0(self):
        raise Exception

    def test_esn_instance_version(self):
        scan_pkg_version = scan._version.__version__
        esn_scan_version = self.esn.scan_version
        self.assertEqual(scan_pkg_version, esn_scan_version)

    def test_equality_with_identity(self):
        self.create_network_3x3_rand_simple()
        self.train_simple_3x3_x_train()

        assert self.esn is self.esn
        assert self.esn == self.esn

    def test_equality_with_different_instance_with_same_content(self):
        self.create_network_3x3_rand_simple()
        self.train_simple_3x3_x_train()

        esn_copy = copy.deepcopy(self.esn)

        self.reset()

        self.create_network_3x3_rand_simple()
        self.train_simple_3x3_x_train()

        assert self.esn is not esn_copy
        assert self.esn == esn_copy

    def test_equality_with_different_instance_with_different_content(self):
        self.create_network_3x3_rand_simple(n_seed=0)
        self.train_simple_3x3_x_train(w_in_seed=0)
        esn0 = copy.deepcopy(self.esn)

        self.create_network_3x3_rand_simple(n_seed=0)
        self.train_simple_3x3_x_train(w_in_seed=1)
        esn1 = copy.deepcopy(self.esn)

        self.create_network_3x3_rand_simple(n_seed=1)
        self.train_simple_3x3_x_train(w_in_seed=0)
        esn2 = copy.deepcopy(self.esn)

        assert esn0 is not esn1
        assert esn0 is not esn2
        assert esn1 is not esn2
        assert esn0 != esn1
        assert esn0 != esn2
        assert esn1 != esn2

    def test_equality_with_non_esn_class(self):
        self.create_network_3x3_rand_simple()
        self.train_simple_3x3_x_train()

        assert self.esn != 5

    def test_independence_from_global_numpy_random_seed(self):
        # NOTE: This isn't actually all that great of a test, as it doesn't really test for the actual logic/algorithm
        #  we implemented and only checks for consistency with a successfully run example. Refactor if possible.
        x_pred = np.array([[-0.7, 1, 0.2], [-0.2, 0.3, 0.06], [0.3, -0.5, 0], [-0.6, 0.7, 0.3]])

        self.set_seed(0)
        self.create_network_3x3_rand_simple()
        self.train_simple_3x3_x_train()
        self.esn.predict(x_pred, sync_steps=0)
        esn0 = copy.deepcopy(self.esn)

        self.reset()
        self.set_seed(1)
        self.create_network_3x3_rand_simple()
        self.train_simple_3x3_x_train()
        self.esn.predict(x_pred, sync_steps=0)
        esn1 = copy.deepcopy(self.esn)

        self.reset()
        self.set_seed(2)
        self.create_network_3x3_rand_simple()
        self.set_seed(3)
        self.train_simple_3x3_x_train()
        self.set_seed(4)
        self.esn.predict(x_pred, sync_steps=0)
        esn2 = copy.deepcopy(self.esn)

        assert esn0 == esn1 == esn2


class TestESNWrapper(TestScanBase):
    def setUp(self):
        self.set_seed()
        self.esn = scan.ESNWrapper()

    def tearDown(self):
        del self.esn
        np.random.seed(None)

    def test_train_and_predict(self):
        train_sync_steps = 2
        train_steps = 5
        pred_steps = 4
        total_time_steps = train_sync_steps + train_steps + pred_steps

        x_dim = 3
        data = np.random.random((total_time_steps, x_dim))

        x_train, x_pred = scan.utilities.train_and_predict_input_setup(
            data,
            train_sync_steps=train_sync_steps,
            train_steps=train_steps,
            pred_steps=pred_steps,
        )

        np.random.seed(1)
        self.esn.create_network()

        self.esn.train(x_train, train_sync_steps, reset_r=False)
        y_pred_desired, y_test_desired = self.esn.predict(x_pred, sync_steps=0, reset_r=False)

        self.reset()

        np.random.seed(1)
        self.esn.create_network()

        y_pred, y_test = self.esn.train_and_predict(
            data,
            train_sync_steps=train_sync_steps,
            train_steps=train_steps,
            pred_steps=pred_steps,
        )

        assert_array_equal(y_test, y_test_desired)
        assert_array_equal(y_pred, y_pred_desired)

    def test_train_and_predict_pred_steps_none(self):
        train_sync_steps = 2
        train_steps = 5
        pred_steps = None
        total_time_steps = train_sync_steps + train_steps + 6

        x_dim = 3
        data = np.random.random((total_time_steps, x_dim))

        x_train, x_pred = scan.utilities.train_and_predict_input_setup(
            data,
            train_sync_steps=train_sync_steps,
            train_steps=train_steps,
            pred_steps=pred_steps,
        )

        np.random.seed(1)
        self.esn.create_network()

        self.esn.train(x_train, train_sync_steps, reset_r=False)
        y_pred_desired, y_test_desired = self.esn.predict(x_pred, sync_steps=0, reset_r=False)

        self.reset()

        np.random.seed(1)
        self.esn.create_network()

        y_pred, y_test = self.esn.train_and_predict(
            data,
            train_sync_steps=train_sync_steps,
            train_steps=train_steps,
            pred_steps=pred_steps,
        )

        assert_array_equal(y_test, y_test_desired)
        assert_array_equal(y_pred, y_pred_desired)

    def test_create_train_and_predict(self):
        n_dim = 3
        n_rad = 0.1
        n_avg_deg = 5.0
        n_type_flag = "erdos_renyi"
        self.esn.create_network(n_dim=n_dim, n_rad=n_rad, n_avg_deg=n_avg_deg, n_type_flag=n_type_flag)

        sync_steps = 1
        x_train = np.array([[-1, 0, 1], [-1.2, -0.2, 0.8], [-1.4, -0.4, 0.6]])
        self.esn.train(x_train=x_train, sync_steps=sync_steps)

        x_pred = np.array([[-0.7, 1, 0.2], [-0.2, 0.3, 0.06], [0.3, -0.5, 0], [-0.6, 0.7, 0.3]])
        exp_y_pred, exp_y_test = self.esn.predict(x_pred=x_pred, sync_steps=sync_steps)

        # self.set_seed()  # Only works if reset_r=True
        self.reset()

        y_pred, y_test = self.esn.create_train_and_predict(
            n_dim=n_dim,
            n_rad=n_rad,
            n_avg_deg=n_avg_deg,
            n_type_flag=n_type_flag,
            x_train=x_train,
            sync_steps=sync_steps,
            x_pred=x_pred,
        )

        assert_array_equal(exp_y_test, y_test)
        assert_array_almost_equal(exp_y_pred, y_pred)

    def test_create_input_matrix(self):
        train_sync_steps = 2
        train_steps = 5
        pred_steps = 4
        total_time_steps = train_sync_steps + train_steps + pred_steps

        x_dim = 3
        data = np.random.random((total_time_steps, x_dim))

        x_train, x_pred = scan.utilities.train_and_predict_input_setup(
            data,
            train_sync_steps=train_sync_steps,
            train_steps=train_steps,
            pred_steps=pred_steps,
        )

        # first w_in:
        np.random.seed(1)
        self.esn.create_network()
        self.esn.train(x_train, train_sync_steps)
        w_in_train = self.esn.w_in

        # second w_in
        self.reset()

        np.random.seed(1)
        self.esn.create_network()
        self.esn.create_input_matrix(x_dim)
        self.esn.train(x_train, train_sync_steps, w_in_no_update=True)
        w_in_create_input_matrix = self.esn.w_in

        assert_array_equal(w_in_train, w_in_create_input_matrix)


class TestESNGenLoc(TestScanBase):
    # def setUp(self):
    #     self.set_seed()
    #     self.esn_gls = scan.ESNGenLoc()
    #
    # def tearDown(self):
    #     del self.esn_gls
    #     np.random.seed(None)

    def test_ESNGenLoc_stub_existence(self):
        with pytest.raises(NotImplementedError):
            scan.ESNGenLoc()
