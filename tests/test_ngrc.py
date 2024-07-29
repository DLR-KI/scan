""" Tests if the scan.esn module works as it should """

import copy

import numpy as np
import pytest

import scan
from tests.test_base import TestScanBase, assert_array_almost_equal, assert_array_equal


# Use print(np.array_repr(np_array, max_line_width=120, precision=18)) in the debugger to easily get the copy-pastable
# representation of a numpy array.
class TestNGRC(TestScanBase):
    def setUp(self):
        self.set_seed()
        self.esn = scan.ESN()
        np.set_printoptions(linewidth=120, precision=20)

    def tearDown(self):
        del self.esn
        np.random.seed(None)
        np.set_printoptions()

    def test_initiate_ngrc_base_class_instance(self):
        ngrc = scan.ngrc.NGRC(k=2, s=1)

        assert isinstance(ngrc, scan.ngrc.NGRC)

    def test_NGRC_functionality_ngrc_base_one_dimensional_data(self):
        ngrc = scan.ngrc.NGRC(k=3, s=2, orders=[1, 2])
        data = np.ones(10).reshape(10, 1)
        ngrc.fit(data)

        assert type(ngrc._w_out) != type(None)

    def test_SINDY_functionality_ngrc_base_one_dimensional_data(self):
        ngrc = scan.ngrc.NGRC(k=1, s=1, orders=[1, 2])
        data = np.ones(10).reshape(10, 1)
        ngrc.fit(data)

        assert type(ngrc._w_out) != type(None)

    def test_VAR_functionality_ngrc_base_one_dimensional_data(self):
        ngrc = scan.ngrc.NGRC(k=3, s=2)
        data = np.ones(10).reshape(10, 1)
        ngrc.fit(data)

        assert type(ngrc._w_out) != type(None)

    def test_NGRC_functionality_ngrc_base_two_dimensional_data(self):
        ngrc = scan.ngrc.NGRC(k=3, s=2, orders=[1, 2])
        data = np.ones((10, 3))
        ngrc.fit(data)

        assert type(ngrc._w_out) != type(None)

    def test_SINDY_functionality_ngrc_base_two_dimensional_data(self):
        ngrc = scan.ngrc.NGRC(k=1, s=1, orders=[1, 2])
        data = np.ones((10, 3))
        ngrc.fit(data)

        assert type(ngrc._w_out) != type(None)

    def test_VAR_functionality_ngrc_base_two_dimensional_data(self):
        ngrc = scan.ngrc.NGRC(k=3, s=2)
        data = np.ones((10, 3))
        ngrc.fit(data)

        assert type(ngrc._w_out) != type(None)

    def test_NGRC_functionality_ngrc_base_two_dimensional_data_bias(self):

        data = np.ones((10, 3))

        ngrc_no_bias = scan.ngrc.NGRC(k=3, s=2, orders=[1, 2], bias=False)
        ngrc_no_bias.fit(data)

        ngrc = scan.ngrc.NGRC(k=3, s=2, orders=[1, 2], bias=True)
        ngrc.fit(data)

        assert type(ngrc._w_out) != type(None)
        assert ngrc._w_out.shape[0] == ngrc_no_bias._w_out.shape[0]
        assert ngrc._w_out.shape[1] == ngrc_no_bias._w_out.shape[1] + 1

    def test_SINDY_functionality_ngrc_base_two_dimensional_data_bias(self):

        data = np.ones((10, 3))

        ngrc_no_bias = scan.ngrc.NGRC(k=1, s=1, orders=[1, 2], bias=False)
        ngrc_no_bias.fit(data)

        ngrc = scan.ngrc.NGRC(k=1, s=1, orders=[1, 2], bias=True)
        ngrc.fit(data)

        assert type(ngrc._w_out) != type(None)
        assert ngrc._w_out.shape[0] == ngrc_no_bias._w_out.shape[0]
        assert ngrc._w_out.shape[1] == ngrc_no_bias._w_out.shape[1] + 1

    def test_VAR_functionality_ngrc_base_two_dimensional_data_bias(self):

        data = np.ones((10, 3))

        ngrc_no_bias = scan.ngrc.NGRC(k=3, s=2, bias=False)
        ngrc_no_bias.fit(data)

        ngrc = scan.ngrc.NGRC(k=3, s=2, bias=True)
        ngrc.fit(data)

        assert type(ngrc._w_out) != type(None)
        assert ngrc._w_out.shape[0] == ngrc_no_bias._w_out.shape[0]
        assert ngrc._w_out.shape[1] == ngrc_no_bias._w_out.shape[1] + 1

    def test_NGRC_functionality_ngrc_base_two_dimensional_data_expanding_orders(self):

        data = np.ones((10, 3))

        ngrc_no_expanding_orders = scan.ngrc.NGRC(k=3, s=2, orders=[1, 2], bias=False)
        ngrc_no_expanding_orders.fit(data)

        ngrc = scan.ngrc.NGRC(k=3, s=2, orders=[1, 2], expanding_orders=[1, 2, 3, 4])
        ngrc.fit(data)

        assert type(ngrc._w_out) != type(None)
        assert ngrc._w_out.shape[0] == ngrc_no_expanding_orders._w_out.shape[0]
        assert len(ngrc._expanding_orders) * ngrc_no_expanding_orders._w_out.shape[1] == ngrc._w_out.shape[1]

    def test_SINDY_functionality_ngrc_base_two_dimensional_data_expanding_orders(self):

        data = np.ones((10, 3))

        ngrc_no_expanding_orders = scan.ngrc.NGRC(k=1, s=1, orders=[1, 2])
        ngrc_no_expanding_orders.fit(data)

        ngrc = scan.ngrc.NGRC(k=1, s=1, orders=[1, 2], expanding_orders=[1, 2, 3, 4])
        ngrc.fit(data)

        assert type(ngrc._w_out) != type(None)
        assert ngrc._w_out.shape[0] == ngrc_no_expanding_orders._w_out.shape[0]
        assert len(ngrc._expanding_orders) * ngrc_no_expanding_orders._w_out.shape[1] == ngrc._w_out.shape[1]

    def test_VAR_functionality_ngrc_base_two_dimensional_data_expanding_orders(self):

        data = np.ones((10, 3))

        ngrc_no_expanding_orders = scan.ngrc.NGRC(k=3, s=2, bias=False)
        ngrc_no_expanding_orders.fit(data)

        ngrc = scan.ngrc.NGRC(k=3, s=2, expanding_orders=[1, 2, 3, 4])
        ngrc.fit(data)

        assert type(ngrc._w_out) != type(None)
        assert ngrc._w_out.shape[0] == ngrc_no_expanding_orders._w_out.shape[0]
        assert len(ngrc._expanding_orders) * ngrc_no_expanding_orders._w_out.shape[1] == ngrc._w_out.shape[1]

    def test_NGRC_functionality_ngrc_base_two_dimensional_data_expanding_orders_bias(self):

        data = np.ones((10, 3))

        ngrc_no_expanding_orders = scan.ngrc.NGRC(k=3, s=2, orders=[1, 2], bias=False)
        ngrc_no_expanding_orders.fit(data)

        ngrc = scan.ngrc.NGRC(k=3, s=2, orders=[1, 2], expanding_orders=[1, 2, 3, 4], bias=True)
        ngrc.fit(data)

        assert type(ngrc._w_out) != type(None)
        assert ngrc._w_out.shape[0] == ngrc_no_expanding_orders._w_out.shape[0]
        assert len(ngrc._expanding_orders) * ngrc_no_expanding_orders._w_out.shape[1] + 1 == ngrc._w_out.shape[1]

    def test_SINDY_functionality_ngrc_base_two_dimensional_data_expanding_orders_bias(self):

        data = np.ones((10, 3))

        ngrc_no_expanding_orders = scan.ngrc.NGRC(k=1, s=1, orders=[1, 2])
        ngrc_no_expanding_orders.fit(data)

        ngrc = scan.ngrc.NGRC(k=1, s=1, orders=[1, 2], expanding_orders=[1, 2, 3, 4], bias=True)
        ngrc.fit(data)

        assert type(ngrc._w_out) != type(None)
        assert ngrc._w_out.shape[0] == ngrc_no_expanding_orders._w_out.shape[0]
        assert len(ngrc._expanding_orders) * ngrc_no_expanding_orders._w_out.shape[1] + 1 == ngrc._w_out.shape[1]

    def test_VAR_functionality_ngrc_base_two_dimensional_data_expanding_orders_bias(self):

        ### orders = None
        data = np.ones((10, 3))

        ngrc_no_expanding_orders = scan.ngrc.NGRC(k=3, s=2, bias=False)
        ngrc_no_expanding_orders.fit(data)

        ngrc = scan.ngrc.NGRC(k=3, s=2, expanding_orders=[1, 2, 3, 4], bias=True)
        ngrc.fit(data)

        assert type(ngrc._w_out) != type(None)
        assert ngrc._w_out.shape[0] == ngrc_no_expanding_orders._w_out.shape[0]
        assert len(ngrc._expanding_orders) * ngrc_no_expanding_orders._w_out.shape[1] + 1 == ngrc._w_out.shape[1]

    def test_VAR_functionality_ngrc_base_two_dimensional_data_expanding_orders_bias_order_var(self):

        data = np.ones((10, 3))

        ngrc_no_expanding_orders = scan.ngrc.NGRC(k=3, s=2, bias=False)
        ngrc_no_expanding_orders.fit(data)

        ngrc = scan.ngrc.NGRC(k=3, s=2, orders=[1], expanding_orders=[1, 2, 3, 4], bias=True)
        ngrc.fit(data)

        ngrc.predict(steps=2)

        assert type(ngrc._w_out) != type(None)
        assert ngrc._w_out.shape[0] == ngrc_no_expanding_orders._w_out.shape[0]
        assert len(ngrc._expanding_orders) * ngrc_no_expanding_orders._w_out.shape[1] + 1 == ngrc._w_out.shape[1]

    def test_dictionary_functionality_ngrc_base(self):
        ## created when nonlinear_expansion() is called.
        data = np.ones((10, 2))

        ngrc = scan.ngrc.NGRC(k=2, s=1, orders=[1, 3, 5], expanding_orders=[1, 2], bias=True)
        ngrc.fit(data)

        assert len(ngrc._dictionary) == 80

        ngrc_interactions = scan.ngrc.NGRC(
            k=2, s=1, orders=[1, 2, 3, 4], order_type="interactions", expanding_orders=[1, 2], bias=True
        )
        ngrc_interactions.fit(data)

        assert len(ngrc_interactions._dictionary) == 15

    def test_create_states_ngrc_base(self):

        data = np.ones((10, 2))
        target_data = np.zeros((10, 5))
        ngrc = scan.ngrc.NGRC(k=3, s=2, orders=[1, 2], bias=True, expanding_orders=[1, 2], mode="inference")
        ngrc.fit(data, target_data)
        assert np.array_equal(ngrc._states, ngrc.create_states(data))

        data = np.ones((10, 2))
        ngrc = scan.ngrc.NGRC(k=3, s=2, orders=[1, 2], bias=True, expanding_orders=[1, 2], mode="coordinates")
        ngrc.fit(data)
        assert np.array_equal(ngrc._states, ngrc.create_states(data))

        data = np.ones((10, 2))
        ngrc = scan.ngrc.NGRC(k=3, s=2, orders=[1, 2], bias=True, expanding_orders=[1, 2], mode="differences")
        ngrc.fit(data)
        assert np.array_equal(ngrc._states, ngrc.create_states(data))

    def test_save_functionality_ngrc_base(self):

        data = np.ones((10, 2))

        ngrc = scan.ngrc.NGRC(k=2, s=1, orders=[1, 2], expanding_orders=[1, 2], bias=True, save_states=False)
        ngrc.fit(data)

        assert type(ngrc._linear_states) == type(None)
        assert type(ngrc._nonlinear_states) == type(None)
        assert type(ngrc._expanded_states) == type(None)

        ngrc = scan.ngrc.NGRC(k=2, s=1, orders=[1, 2], save_states=False)
        ngrc.fit(data)
        assert type(ngrc._expanded_states) == type(None)

    def test_prediction_ngrc_base_coordinates_ngrc_max(self):
        #### Max Expansion
        # NGRC
        data = np.ones((10, 2))

        ngrc = scan.ngrc.NGRC(k=2, s=1, orders=[1, 3, 5], expanding_orders=[1, 2], bias=True)
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

    def test_prediction_ngrc_base_coordinates_sindy_max(self):

        data = np.ones((10, 2))
        # SINDY
        ngrc = scan.ngrc.NGRC(k=1, s=1, orders=[1, 3], expanding_orders=[1, 2], bias=True)
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

    def test_prediction_ngrc_base_coordinates_var_max(self):

        data = np.ones((10, 2))
        # VAR
        ngrc = scan.ngrc.NGRC(k=3, s=2, expanding_orders=[1, 2], bias=True)
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

    def test_prediction_ngrc_base_coordinates_ngrc_exp_no_bias(self):

        data = np.ones((10, 2))

        ngrc = scan.ngrc.NGRC(k=2, s=1, orders=[1, 3, 5], expanding_orders=[1, 2])
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

    def test_prediction_ngrc_base_coordinates_ngrc_exp_no_bias(self):

        data = np.ones((10, 2))
        ngrc = scan.ngrc.NGRC(k=1, s=1, orders=[1, 3], expanding_orders=[1, 2])
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

    def test_prediction_ngrc_base_coordinates_var_exp_no_bias(self):

        data = np.ones((10, 2))
        ngrc = scan.ngrc.NGRC(k=3, s=2, expanding_orders=[1, 2])
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

    def test_prediction_ngrc_base_coordinates_ngrc_bias(self):

        data = np.ones((10, 2))

        ngrc = scan.ngrc.NGRC(k=2, s=1, orders=[1, 3, 5], bias=True)
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

    def test_prediction_ngrc_base_coordinates_sindy_bias(self):

        data = np.ones((10, 2))
        ngrc = scan.ngrc.NGRC(k=1, s=1, orders=[1, 3], bias=True)
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

    def test_prediction_ngrc_base_coordinates_var_bias(self):

        data = np.ones((10, 2))
        ngrc = scan.ngrc.NGRC(k=3, s=2, bias=True)
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

    def test_prediction_ngrc_base_coordinates_ngrcs(self):

        data = np.ones((10, 2))

        ngrc = scan.ngrc.NGRC(k=2, s=1, orders=[1, 3, 5])
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

    def test_prediction_ngrc_base_coordinates_sindy(self):

        data = np.ones((10, 2))
        ngrc = scan.ngrc.NGRC(k=1, s=1, orders=[1, 3])
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

    def test_prediction_ngrc_base_coordinates_var(self):

        data = np.ones((10, 2))
        ngrc = scan.ngrc.NGRC(k=3, s=2)
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

    def test_prediction_ngrc_base_differences_ngrc_max(self):
        #### Max Expansion
        # NGRC - Max Expansion
        data = np.ones((10, 2))

        ngrc = scan.ngrc.NGRC(k=2, s=1, orders=[1, 3, 5], expanding_orders=[1, 2], bias=True, mode="differences")
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

    def test_prediction_ngrc_base_differences_sindy_max(self):

        data = np.ones((10, 2))
        ngrc = scan.ngrc.NGRC(k=1, s=1, orders=[1, 3], expanding_orders=[1, 2], bias=True, mode="differences")
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

    def test_prediction_ngrc_base_differences_var_max(self):

        data = np.ones((10, 2))
        ngrc = scan.ngrc.NGRC(k=3, s=2, expanding_orders=[1, 2], bias=True, mode="differences")
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

    def test_prediction_ngrc_base_differences_ngrc_exp_no_bias(self):

        data = np.ones((10, 2))

        ngrc = scan.ngrc.NGRC(k=2, s=1, orders=[1, 3, 5], expanding_orders=[1, 2], mode="differences")
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

    def test_prediction_ngrc_base_differences_sindy_exp_no_bias(self):

        data = np.ones((10, 2))
        ngrc = scan.ngrc.NGRC(k=1, s=1, orders=[1, 3], expanding_orders=[1, 2], mode="differences")
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

    def test_prediction_ngrc_base_differences_var_exp_no_bias(self):

        data = np.ones((10, 2))
        ngrc = scan.ngrc.NGRC(k=3, s=2, expanding_orders=[1, 2], mode="differences")
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

    def test_prediction_ngrc_base_differences_ngrc_bias(self):

        data = np.ones((10, 2))
        ngrc = scan.ngrc.NGRC(k=2, s=1, orders=[1, 3, 5], bias=True, mode="differences")
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

    def test_prediction_ngrc_base_differences_sindy_bias(self):

        data = np.ones((10, 2))
        ngrc = scan.ngrc.NGRC(k=1, s=1, orders=[1, 3], bias=True, mode="differences")
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

    def test_prediction_ngrc_base_differences_var_bias(self):

        data = np.ones((10, 2))
        ngrc = scan.ngrc.NGRC(k=3, s=2, bias=True, mode="differences")
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

    def test_prediction_ngrc_base_differences_ngrc(self):

        data = np.ones((10, 2))
        ngrc = scan.ngrc.NGRC(k=2, s=1, orders=[1, 3, 5], mode="differences")
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

    def test_prediction_ngrc_base_differences_sindy(self):

        data = np.ones((10, 2))
        ngrc = scan.ngrc.NGRC(k=1, s=1, orders=[1, 3], mode="differences")
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

    def test_prediction_ngrc_base_differences_var(self):

        data = np.ones((10, 2))
        ngrc = scan.ngrc.NGRC(k=3, s=2, mode="differences")
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

    def test_inference_ngrc_base_ngrc_max(self):

        data = np.ones((10, 2))
        target_data = np.zeros((10, 3))

        ngrc = scan.ngrc.NGRC(k=2, s=1, orders=[1, 3, 5], expanding_orders=[1, 2], bias=True, mode="inference")
        ngrc.fit(data, target_data)

        inference = ngrc.inference(data)

        assert inference.shape == (data.shape[0] - (ngrc._k - 1) * ngrc._s, target_data.shape[1])

    def test_inference_ngrc_base_sindy_max(self):

        data = np.ones((10, 2))
        target_data = np.zeros((10, 3))
        ngrc = scan.ngrc.NGRC(k=1, s=1, orders=[1, 3], expanding_orders=[1, 2], bias=True, mode="inference")
        ngrc.fit(data, target_data)

        inference = ngrc.inference(data)

        assert inference.shape == (data.shape[0] - (ngrc._k - 1) * ngrc._s, target_data.shape[1])

    def test_inference_ngrc_base_var_max(self):

        data = np.ones((10, 2))
        target_data = np.zeros((10, 3))
        ngrc = scan.ngrc.NGRC(k=3, s=2, expanding_orders=[1, 2], bias=True, mode="inference")
        ngrc.fit(data, target_data)

        inference = ngrc.inference(data)

        assert inference.shape == (data.shape[0] - (ngrc._k - 1) * ngrc._s, target_data.shape[1])

    def test_inference_ngrc_base_ngrc_exp_no_bias(self):

        data = np.ones((10, 2))
        target_data = np.zeros((10, 3))
        ngrc = scan.ngrc.NGRC(k=2, s=1, orders=[1, 3, 5], expanding_orders=[1, 2], mode="inference")
        ngrc.fit(data, target_data)

        inference = ngrc.inference(data)

        assert inference.shape == (data.shape[0] - (ngrc._k - 1) * ngrc._s, target_data.shape[1])

    def test_inference_ngrc_base_sindy_exp_no_bias(self):

        data = np.ones((10, 2))
        target_data = np.zeros((10, 3))
        ngrc = scan.ngrc.NGRC(k=1, s=1, orders=[1, 3], expanding_orders=[1, 2], mode="inference")
        ngrc.fit(data, target_data)

        inference = ngrc.inference(data)

        assert inference.shape == (data.shape[0] - (ngrc._k - 1) * ngrc._s, target_data.shape[1])

    def test_inference_ngrc_base_var_exp_no_bias(self):

        data = np.ones((10, 2))
        target_data = np.zeros((10, 3))
        ngrc = scan.ngrc.NGRC(k=3, s=2, expanding_orders=[1, 2], mode="inference")
        ngrc.fit(data, target_data)

        inference = ngrc.inference(data)

        assert inference.shape == (data.shape[0] - (ngrc._k - 1) * ngrc._s, target_data.shape[1])

    def test_inference_ngrc_base_ngrc_bias(self):

        data = np.ones((10, 2))
        target_data = np.zeros((10, 3))
        ngrc = scan.ngrc.NGRC(k=2, s=1, orders=[1, 3, 5], bias=True, mode="inference")
        ngrc.fit(data, target_data)

        inference = ngrc.inference(data)

        assert inference.shape == (data.shape[0] - (ngrc._k - 1) * ngrc._s, target_data.shape[1])

    def test_inference_ngrc_base_sindy_bias(self):

        data = np.ones((10, 2))
        target_data = np.zeros((10, 3))
        ngrc = scan.ngrc.NGRC(k=1, s=1, orders=[1, 3], bias=True, mode="inference")
        ngrc.fit(data, target_data)

        inference = ngrc.inference(data)

        assert inference.shape == (data.shape[0] - (ngrc._k - 1) * ngrc._s, target_data.shape[1])

    def test_inference_ngrc_base_var_bias(self):

        data = np.ones((10, 2))
        target_data = np.zeros((10, 3))
        ngrc = scan.ngrc.NGRC(k=3, s=2, bias=True, mode="inference")
        ngrc.fit(data, target_data)

        inference = ngrc.inference(data)

        assert inference.shape == (data.shape[0] - (ngrc._k - 1) * ngrc._s, target_data.shape[1])

    def test_inference_ngrc_base_ngrc(self):

        data = np.ones((10, 2))
        target_data = np.zeros((10, 3))
        ngrc = scan.ngrc.NGRC(k=2, s=1, orders=[1, 3, 5], mode="inference")
        ngrc.fit(data, target_data)

        inference = ngrc.inference(data)

        assert inference.shape == (data.shape[0] - (ngrc._k - 1) * ngrc._s, target_data.shape[1])

    def test_inference_ngrc_base_sindy(self):

        data = np.ones((10, 2))
        target_data = np.zeros((10, 3))
        ngrc = scan.ngrc.NGRC(k=1, s=1, orders=[1, 3], mode="inference")
        ngrc.fit(data, target_data)

        inference = ngrc.inference(data)

        assert inference.shape == (data.shape[0] - (ngrc._k - 1) * ngrc._s, target_data.shape[1])

    def test_inference_ngrc_base_var(self):

        data = np.ones((10, 2))
        target_data = np.zeros((10, 3))
        ngrc = scan.ngrc.NGRC(k=3, s=2, mode="inference")
        ngrc.fit(data, target_data)

        inference = ngrc.inference(data)

        assert inference.shape == (data.shape[0] - (ngrc._k - 1) * ngrc._s, target_data.shape[1])

    def test_prediction_starting_series_ngrc_base(self):

        data = np.ones((10, 2))

        ngrc = scan.ngrc.NGRC(k=2, s=1, orders=[1, 3, 5], expanding_orders=[1, 2], bias=True)
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps, starting_series=data[-2:])

        assert prediction.shape == (data.shape[1], prediction_steps)

    def test_error_prediction_with_external_target_data(self):

        data = np.ones((10, 2))

        ngrc = scan.ngrc.NGRC(k=2, s=1, orders=[1, 3, 5], expanding_orders=[1, 2], bias=True, mode="coordinates")
        with pytest.raises(ValueError):
            ngrc.fit(data, y_target=[1])

        ngrc = scan.ngrc.NGRC(k=2, s=1, orders=[1, 3, 5], expanding_orders=[1, 2], bias=True, mode="differences")

        with pytest.raises(ValueError):
            ngrc.fit(data, y_target=[1])

    def test_error_inference_with_no_external_target_data(self):

        data = np.ones((10, 2))

        ngrc = scan.ngrc.NGRC(k=2, s=1, orders=[1, 3, 5], expanding_orders=[1, 2], bias=True, mode="inference")
        with pytest.raises(ValueError):
            ngrc.fit(data, y_target=None)

    def test_error_mode_incorrectly_specified(self):

        data = np.ones((10, 2))

        ngrc = scan.ngrc.NGRC(k=2, s=1, orders=[1, 3, 5], expanding_orders=[1, 2], bias=True, mode="test")
        with pytest.raises(ValueError):
            ngrc.fit(data, y_target=None)

    def test_error_order_type_incorrectly_specified(self):

        data = np.ones((10, 2))

        ngrc = scan.ngrc.NGRC(k=2, s=1, orders=[1, 3, 5], expanding_orders=[1, 2], bias=True, order_type="test")
        with pytest.raises(ValueError):
            ngrc.fit(data, y_target=None)

    def test_non_return_expanding_orders(self):
        data = np.ones((10, 2))

        ngrc = scan.ngrc.NGRC(k=2, s=1, orders=[1, 3, 5])

        ngrc.fit(data)
        test = ngrc.expanding_states(training=False, input_data=ngrc._states)
        assert test is None

    def test_error_mode_inference_with_prediction_function(self):

        data = np.ones((10, 2))
        target_data = np.zeros((10, 3))
        ngrc = scan.ngrc.NGRC(k=1, s=1, orders=[1, 3], mode="inference")
        ngrc.fit(data, target_data)
        with pytest.raises(ValueError):
            ngrc.predict(steps=2)

    def test_error_mode_short_starting_series_for_prediction(self):

        data = np.ones((10, 2))
        target_data = np.zeros((10, 3))
        ngrc = scan.ngrc.NGRC(k=2, s=1, orders=[1, 3], mode="coordinates")
        ngrc.fit(data)

        with pytest.raises(ValueError):
            ngrc.predict(steps=2, starting_series=data[-1:])

    def test_error_mode_inference_with_short_input_data(self):

        data = np.ones((10, 2))
        target_data = np.zeros((10, 3))
        ngrc = scan.ngrc.NGRC(k=3, s=1, orders=[1, 3], mode="inference")
        ngrc.fit(data, target_data)

        with pytest.raises(ValueError):
            ngrc.inference(x=data[-2:])

    def test_error_mode_prediction_with_inference_function(self):

        data = np.ones((10, 2))
        target_data = np.zeros((10, 3))
        ngrc = scan.ngrc.NGRC(k=1, s=1, orders=[1, 3], mode="coordinates")
        ngrc.fit(data)
        with pytest.raises(ValueError):
            ngrc.inference(x=data)

    def test_error_minimal_input_fit(self):

        data = np.ones((10, 2))
        target_data = np.zeros((10, 3))
        ngrc = scan.ngrc.NGRC(k=4, s=1, orders=[1, 3], mode="coordinates")
        with pytest.raises(ValueError):
            ngrc.fit(data[:3])

        ngrc = scan.ngrc.NGRC(k=4, s=1, orders=[1, 3], mode="inference")
        with pytest.raises(ValueError):
            ngrc.fit(data[:3])

    def test_precision_mid_term(self):

        sp = np.array([-14.03020521, -20.88693127, 25.53545])
        sim_data = scan.simulations.Lorenz63(dt=2e-2).simulate(time_steps=400, starting_point=sp)

        train_steps = 400
        train_data = sim_data[:train_steps]

        ng_rc = scan.ngrc.NGRC(regression_parameter=9 * 10**-2, k=2, s=1, orders=[1, 2], mode="coordinates")

        ng_rc.fit(train_data)

        staser = sim_data[-(ng_rc._k - 1) * ng_rc._s - 1 :]
        lorenz_prediction = ng_rc.predict(steps=100, starting_series=staser)

        test = np.array([-9.3696167086, -1.73024410942, 35.54744294115])
        pred_ = lorenz_prediction[-1]
        assert_array_almost_equal(test, pred_, decimal=7)

    def test_precision_first_pred(self):

        sp = np.array([-14.03020521, -20.88693127, 25.53545])
        sim_data = scan.simulations.Lorenz63(dt=2e-2).simulate(time_steps=400, starting_point=sp)

        train_steps = 400
        train_data = sim_data[:train_steps]

        ng_rc = scan.ngrc.NGRC(regression_parameter=9 * 10**-2, k=1, s=1, orders=[1, 2], mode="differences")

        ng_rc.fit(train_data)

        staser = sim_data[-(ng_rc._k - 1) * ng_rc._s - 1 :]
        lorenz_prediction = ng_rc.predict(steps=1, starting_series=staser)

        test = np.array([-4.82184971595, -2.37195797593, 26.20243990289])
        pred_ = lorenz_prediction[-1]
        assert_array_almost_equal(test, pred_)

    def test_precision_expanding_orders_mid_term(self):
        sp = np.array([-14.03020521, -20.88693127, 25.53545])
        sim_data = scan.simulations.Lorenz63(dt=2e-2).simulate(time_steps=400, starting_point=sp)

        train_steps = 400
        train_data = sim_data[:train_steps]

        ng_rc = scan.ngrc.NGRC(
            regression_parameter=9 * 10**-2,
            k=2,
            s=1,
            orders=[1, 2],
            mode="differences",
            bias=True,
            expanding_orders=[1, 3],
        )

        ng_rc.fit(train_data)

        staser = sim_data[-(ng_rc._k - 1) * ng_rc._s - 1 :]
        lorenz_prediction = ng_rc.predict(steps=100, starting_series=staser)

        test = np.array([-10.1723369755, -2.36401695605, 36.50132478729])

        pred_ = lorenz_prediction[-1]
        assert_array_almost_equal(test, pred_, decimal=7)
