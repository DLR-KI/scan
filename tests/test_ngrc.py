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
        ngrc = scan.ngrc.NG_RC(k=2, s=1)

        assert isinstance(ngrc, scan.ngrc.NG_RC)

    def test_NGRC_functionality_ngrc_base_one_dimensional_data(self):
        ngrc = scan.ngrc.NG_RC(k=3, s=2, orders=[1, 2])
        data = np.ones(10).reshape(10, 1)
        ngrc.fit(data)

        assert type(ngrc._w_out) != type(None)

    def test_SINDY_functionality_ngrc_base_one_dimensional_data(self):
        ngrc = scan.ngrc.NG_RC(k=1, s=1, orders=[1, 2])
        data = np.ones(10).reshape(10, 1)
        ngrc.fit(data)

        assert type(ngrc._w_out) != type(None)

    def test_VAR_functionality_ngrc_base_one_dimensional_data(self):
        ngrc = scan.ngrc.NG_RC(k=3, s=2)
        data = np.ones(10).reshape(10, 1)
        ngrc.fit(data)

        assert type(ngrc._w_out) != type(None)

    def test_NGRC_functionality_ngrc_base_two_dimensional_data(self):
        ngrc = scan.ngrc.NG_RC(k=3, s=2, orders=[1, 2])
        data = np.ones((10, 3))
        ngrc.fit(data)

        assert type(ngrc._w_out) != type(None)

    def test_SINDY_functionality_ngrc_base_two_dimensional_data(self):
        ngrc = scan.ngrc.NG_RC(k=1, s=1, orders=[1, 2])
        data = np.ones((10, 3))
        ngrc.fit(data)

        assert type(ngrc._w_out) != type(None)

    def test_VAR_functionality_ngrc_base_two_dimensional_data(self):
        ngrc = scan.ngrc.NG_RC(k=3, s=2)
        data = np.ones((10, 3))
        ngrc.fit(data)

        assert type(ngrc._w_out) != type(None)

    def test_NGRC_functionality_ngrc_base_two_dimensional_data_bias(self):

        data = np.ones((10, 3))

        ngrc_no_bias = scan.ngrc.NG_RC(k=3, s=2, orders=[1, 2], bias=False)
        ngrc_no_bias.fit(data)

        ngrc = scan.ngrc.NG_RC(k=3, s=2, orders=[1, 2], bias=True)
        ngrc.fit(data)

        assert type(ngrc._w_out) != type(None)
        assert ngrc._w_out.shape[0] == ngrc_no_bias._w_out.shape[0]
        assert ngrc._w_out.shape[1] == ngrc_no_bias._w_out.shape[1] + 1

    def test_SINDY_functionality_ngrc_base_two_dimensional_data_bias(self):

        data = np.ones((10, 3))

        ngrc_no_bias = scan.ngrc.NG_RC(k=1, s=1, orders=[1, 2], bias=False)
        ngrc_no_bias.fit(data)

        ngrc = scan.ngrc.NG_RC(k=1, s=1, orders=[1, 2], bias=True)
        ngrc.fit(data)

        assert type(ngrc._w_out) != type(None)
        assert ngrc._w_out.shape[0] == ngrc_no_bias._w_out.shape[0]
        assert ngrc._w_out.shape[1] == ngrc_no_bias._w_out.shape[1] + 1

    def test_VAR_functionality_ngrc_base_two_dimensional_data_bias(self):

        data = np.ones((10, 3))

        ngrc_no_bias = scan.ngrc.NG_RC(k=3, s=2, bias=False)
        ngrc_no_bias.fit(data)

        ngrc = scan.ngrc.NG_RC(k=3, s=2, bias=True)
        ngrc.fit(data)

        assert type(ngrc._w_out) != type(None)
        assert ngrc._w_out.shape[0] == ngrc_no_bias._w_out.shape[0]
        assert ngrc._w_out.shape[1] == ngrc_no_bias._w_out.shape[1] + 1

    def test_NGRC_functionality_ngrc_base_two_dimensional_data_expanding_orders(self):

        data = np.ones((10, 3))

        ngrc_no_expanding_orders = scan.ngrc.NG_RC(k=3, s=2, orders=[1, 2], bias=False)
        ngrc_no_expanding_orders.fit(data)

        ngrc = scan.ngrc.NG_RC(k=3, s=2, orders=[1, 2], expanding_orders=[1, 2, 3, 4])
        ngrc.fit(data)

        assert type(ngrc._w_out) != type(None)
        assert ngrc._w_out.shape[0] == ngrc_no_expanding_orders._w_out.shape[0]
        assert len(ngrc._expanding_orders) * ngrc_no_expanding_orders._w_out.shape[1] == ngrc._w_out.shape[1]

    def test_SINDY_functionality_ngrc_base_two_dimensional_data_expanding_orders(self):

        data = np.ones((10, 3))

        ngrc_no_expanding_orders = scan.ngrc.NG_RC(k=1, s=1, orders=[1, 2])
        ngrc_no_expanding_orders.fit(data)

        ngrc = scan.ngrc.NG_RC(k=1, s=1, orders=[1, 2], expanding_orders=[1, 2, 3, 4])
        ngrc.fit(data)

        assert type(ngrc._w_out) != type(None)
        assert ngrc._w_out.shape[0] == ngrc_no_expanding_orders._w_out.shape[0]
        assert len(ngrc._expanding_orders) * ngrc_no_expanding_orders._w_out.shape[1] == ngrc._w_out.shape[1]

    def test_VAR_functionality_ngrc_base_two_dimensional_data_expanding_orders(self):

        data = np.ones((10, 3))

        ngrc_no_expanding_orders = scan.ngrc.NG_RC(k=3, s=2, bias=False)
        ngrc_no_expanding_orders.fit(data)

        ngrc = scan.ngrc.NG_RC(k=3, s=2, expanding_orders=[1, 2, 3, 4])
        ngrc.fit(data)

        assert type(ngrc._w_out) != type(None)
        assert ngrc._w_out.shape[0] == ngrc_no_expanding_orders._w_out.shape[0]
        assert len(ngrc._expanding_orders) * ngrc_no_expanding_orders._w_out.shape[1] == ngrc._w_out.shape[1]

    def test_NGRC_functionality_ngrc_base_two_dimensional_data_expanding_orders_bias(self):

        data = np.ones((10, 3))

        ngrc_no_expanding_orders = scan.ngrc.NG_RC(k=3, s=2, orders=[1, 2], bias=False)
        ngrc_no_expanding_orders.fit(data)

        ngrc = scan.ngrc.NG_RC(k=3, s=2, orders=[1, 2], expanding_orders=[1, 2, 3, 4], bias=True)
        ngrc.fit(data)

        assert type(ngrc._w_out) != type(None)
        assert ngrc._w_out.shape[0] == ngrc_no_expanding_orders._w_out.shape[0]
        assert len(ngrc._expanding_orders) * ngrc_no_expanding_orders._w_out.shape[1] + 1 == ngrc._w_out.shape[1]

    def test_SINDY_functionality_ngrc_base_two_dimensional_data_expanding_orders_bias(self):

        data = np.ones((10, 3))

        ngrc_no_expanding_orders = scan.ngrc.NG_RC(k=1, s=1, orders=[1, 2])
        ngrc_no_expanding_orders.fit(data)

        ngrc = scan.ngrc.NG_RC(k=1, s=1, orders=[1, 2], expanding_orders=[1, 2, 3, 4], bias=True)
        ngrc.fit(data)

        assert type(ngrc._w_out) != type(None)
        assert ngrc._w_out.shape[0] == ngrc_no_expanding_orders._w_out.shape[0]
        assert len(ngrc._expanding_orders) * ngrc_no_expanding_orders._w_out.shape[1] + 1 == ngrc._w_out.shape[1]

    def test_VAR_functionality_ngrc_base_two_dimensional_data_expanding_orders_bias(self):

        ### orders = None
        data = np.ones((10, 3))

        ngrc_no_expanding_orders = scan.ngrc.NG_RC(k=3, s=2, bias=False)
        ngrc_no_expanding_orders.fit(data)

        ngrc = scan.ngrc.NG_RC(k=3, s=2, expanding_orders=[1, 2, 3, 4], bias=True)
        ngrc.fit(data)

        assert type(ngrc._w_out) != type(None)
        assert ngrc._w_out.shape[0] == ngrc_no_expanding_orders._w_out.shape[0]
        assert len(ngrc._expanding_orders) * ngrc_no_expanding_orders._w_out.shape[1] + 1 == ngrc._w_out.shape[1]

        ### orders = [1]
        data = np.ones((10, 3))

        ngrc_no_expanding_orders = scan.ngrc.NG_RC(k=3, s=2, bias=False)
        ngrc_no_expanding_orders.fit(data)

        ngrc = scan.ngrc.NG_RC(k=3, s=2, orders=[1], expanding_orders=[1, 2, 3, 4], bias=True)
        ngrc.fit(data)

        assert type(ngrc._w_out) != type(None)
        assert ngrc._w_out.shape[0] == ngrc_no_expanding_orders._w_out.shape[0]
        assert len(ngrc._expanding_orders) * ngrc_no_expanding_orders._w_out.shape[1] + 1 == ngrc._w_out.shape[1]

    def test_dictionary_functionality_ngrc_base(self):
        ## created when nonlinear_expansion() is called.
        data = np.ones((10, 2))

        ngrc = scan.ngrc.NG_RC(k=2, s=1, orders=[1, 3, 5], expanding_orders=[1, 2], bias=True)
        ngrc.fit(data)

        assert len(ngrc._dictionary) == 80

        ngrc_interactions = scan.ngrc.NG_RC(
            k=2, s=1, orders=[1, 2, 3, 4], order_type="interactions", expanding_orders=[1, 2], bias=True
        )
        ngrc_interactions.fit(data)

        assert len(ngrc_interactions._dictionary) == 15

    def test_create_states_ngrc_base(self):

        data = np.ones((10, 2))
        target_data = np.zeros((10, 5))
        ngrc = scan.ngrc.NG_RC(k=3, s=2, orders=[1, 2], bias=True, expanding_orders=[1, 2], mode="inference")
        ngrc.fit(data, target_data)
        assert np.array_equal(ngrc._states, ngrc.create_states(data))

        data = np.ones((10, 2))
        ngrc = scan.ngrc.NG_RC(k=3, s=2, orders=[1, 2], bias=True, expanding_orders=[1, 2], mode="coordinates")
        ngrc.fit(data)
        assert np.array_equal(ngrc._states, ngrc.create_states(data))

        data = np.ones((10, 2))
        ngrc = scan.ngrc.NG_RC(k=3, s=2, orders=[1, 2], bias=True, expanding_orders=[1, 2], mode="differences")
        ngrc.fit(data)
        assert np.array_equal(ngrc._states, ngrc.create_states(data))

    def test_save_functionality_ngrc_base(self):

        data = np.ones((10, 2))

        ngrc = scan.ngrc.NG_RC(k=2, s=1, orders=[1, 2], expanding_orders=[1, 2], bias=True, save_states=False)
        ngrc.fit(data)

        assert type(ngrc._linear_states) == type(None)
        assert type(ngrc._nonlinear_states) == type(None)
        assert type(ngrc._expanded_states) == type(None)

        ngrc = scan.ngrc.NG_RC(k=2, s=1, orders=[1, 2], save_states=False)
        ngrc.fit(data)
        assert type(ngrc._expanded_states) == type(None)

    def test_prediction_ngrc_base_coordinates(self):
        #### Max Expansion
        # NGRC
        data = np.ones((10, 2))

        ngrc = scan.ngrc.NG_RC(k=2, s=1, orders=[1, 3, 5], expanding_orders=[1, 2], bias=True)
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

        # SINDY
        ngrc = scan.ngrc.NG_RC(k=1, s=1, orders=[1, 3], expanding_orders=[1, 2], bias=True)
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

        # VAR
        ngrc = scan.ngrc.NG_RC(k=3, s=2, expanding_orders=[1, 2], bias=True)
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

        ### Expansion - No bias
        # NGRC
        data = np.ones((10, 2))

        ngrc = scan.ngrc.NG_RC(k=2, s=1, orders=[1, 3, 5], expanding_orders=[1, 2])
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

        # SINDY
        ngrc = scan.ngrc.NG_RC(k=1, s=1, orders=[1, 3], expanding_orders=[1, 2])
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

        # VAR
        ngrc = scan.ngrc.NG_RC(k=3, s=2, expanding_orders=[1, 2])
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

        ### No Expansion - bias
        # NGRC
        data = np.ones((10, 2))

        ngrc = scan.ngrc.NG_RC(k=2, s=1, orders=[1, 3, 5], bias=True)
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

        # SINDY
        ngrc = scan.ngrc.NG_RC(k=1, s=1, orders=[1, 3], bias=True)
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

        # VAR
        ngrc = scan.ngrc.NG_RC(k=3, s=2, bias=True)
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

        ### No Expansion - No bias
        # NGRC
        data = np.ones((10, 2))

        ngrc = scan.ngrc.NG_RC(k=2, s=1, orders=[1, 3, 5])
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

        # SINDY
        ngrc = scan.ngrc.NG_RC(k=1, s=1, orders=[1, 3])
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

        # VAR
        ngrc = scan.ngrc.NG_RC(k=3, s=2)
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

    def test_prediction_ngrc_base_differences(self):
        #### Max Expansion
        # NGRC - Max Expansion
        data = np.ones((10, 2))

        ngrc = scan.ngrc.NG_RC(k=2, s=1, orders=[1, 3, 5], expanding_orders=[1, 2], bias=True, mode="differences")
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

        # SINDY - Max Expansion
        ngrc = scan.ngrc.NG_RC(k=1, s=1, orders=[1, 3], expanding_orders=[1, 2], bias=True, mode="differences")
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

        # VAR - Max Expansion
        ngrc = scan.ngrc.NG_RC(k=3, s=2, expanding_orders=[1, 2], bias=True, mode="differences")
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

        ### Expansion - No bias
        # NGRC - Expansion - No bias
        data = np.ones((10, 2))

        ngrc = scan.ngrc.NG_RC(k=2, s=1, orders=[1, 3, 5], expanding_orders=[1, 2], mode="differences")
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

        # SINDY - Expansion - No bias
        ngrc = scan.ngrc.NG_RC(k=1, s=1, orders=[1, 3], expanding_orders=[1, 2], mode="differences")
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

        # VAR - Expansion - No bias
        ngrc = scan.ngrc.NG_RC(k=3, s=2, expanding_orders=[1, 2], mode="differences")
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

        ### No Expansion - bias
        # NGRC - No Expansion - bias
        data = np.ones((10, 2))

        ngrc = scan.ngrc.NG_RC(k=2, s=1, orders=[1, 3, 5], bias=True, mode="differences")
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

        # SINDY - No Expansion - bias
        ngrc = scan.ngrc.NG_RC(k=1, s=1, orders=[1, 3], bias=True, mode="differences")
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

        # VAR - No Expansion - bias
        ngrc = scan.ngrc.NG_RC(k=3, s=2, bias=True, mode="differences")
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

        ### No Expansion - No bias
        # NGRC - No bias
        data = np.ones((10, 2))

        ngrc = scan.ngrc.NG_RC(k=2, s=1, orders=[1, 3, 5], mode="differences")
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

        # SINDY - No bias
        ngrc = scan.ngrc.NG_RC(k=1, s=1, orders=[1, 3], mode="differences")
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

        # VAR - No bias
        ngrc = scan.ngrc.NG_RC(k=3, s=2, mode="differences")
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

    def test_prediction_ngrc_base_differences(self):
        #### Max Expansion
        # NGRC - Max Expansion
        data = np.ones((10, 2))

        ngrc = scan.ngrc.NG_RC(k=2, s=1, orders=[1, 3, 5], expanding_orders=[1, 2], bias=True, mode="differences")
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

        # SINDY - Max Expansion
        ngrc = scan.ngrc.NG_RC(k=1, s=1, orders=[1, 3], expanding_orders=[1, 2], bias=True, mode="differences")
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

        # VAR - Max Expansion orders=None
        ngrc = scan.ngrc.NG_RC(k=3, s=2, expanding_orders=[1, 2], bias=True, mode="differences")
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

        # VAR - Max Expansion orders=[1]
        ngrc = scan.ngrc.NG_RC(k=3, s=2, orders=[1], expanding_orders=[1, 2], bias=True, mode="differences")
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

        ### Expansion - No bias
        # NGRC - Expansion - No bias
        data = np.ones((10, 2))

        ngrc = scan.ngrc.NG_RC(k=2, s=1, orders=[1, 3, 5], expanding_orders=[1, 2], mode="differences")
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

        # SINDY - Expansion - No bias
        ngrc = scan.ngrc.NG_RC(k=1, s=1, orders=[1, 3], expanding_orders=[1, 2], mode="differences")
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

        # VAR - Expansion - No bias
        ngrc = scan.ngrc.NG_RC(k=3, s=2, expanding_orders=[1, 2], mode="differences")
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

        ### No Expansion - bias
        # NGRC - No Expansion - bias
        data = np.ones((10, 2))

        ngrc = scan.ngrc.NG_RC(k=2, s=1, orders=[1, 3, 5], bias=True, mode="differences")
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

        # SINDY - No Expansion - bias
        ngrc = scan.ngrc.NG_RC(k=1, s=1, orders=[1, 3], bias=True, mode="differences")
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

        # VAR - No Expansion - bias
        ngrc = scan.ngrc.NG_RC(k=3, s=2, bias=True, mode="differences")
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

        ### No Expansion - No bias
        # NGRC - No bias
        data = np.ones((10, 2))

        ngrc = scan.ngrc.NG_RC(k=2, s=1, orders=[1, 3, 5], mode="differences")
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

        # SINDY - No bias
        ngrc = scan.ngrc.NG_RC(k=1, s=1, orders=[1, 3], mode="differences")
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

        # VAR - No bias
        ngrc = scan.ngrc.NG_RC(k=3, s=2, mode="differences")
        ngrc.fit(data)

        prediction_steps = 2
        prediction = ngrc.predict(prediction_steps)

        assert prediction.shape == (data.shape[1], prediction_steps)

    def testinference__ngrc_base_differences(self):

        #### Max Expansion
        # NGRC - Max Expansion
        data = np.ones((10, 2))
        target_data = np.zeros((10, 3))

        ngrc = scan.ngrc.NG_RC(k=2, s=1, orders=[1, 3, 5], expanding_orders=[1, 2], bias=True, mode="inference")
        ngrc.fit(data, target_data)

        inference = ngrc.inference(data)

        assert inference.shape == (data.shape[0] - (ngrc._k - 1) * ngrc._s, target_data.shape[1])

        # SINDY - Max Expansion
        ngrc = scan.ngrc.NG_RC(k=1, s=1, orders=[1, 3], expanding_orders=[1, 2], bias=True, mode="inference")
        ngrc.fit(data, target_data)

        inference = ngrc.inference(data)

        assert inference.shape == (data.shape[0] - (ngrc._k - 1) * ngrc._s, target_data.shape[1])

        # VAR - Max Expansion
        ngrc = scan.ngrc.NG_RC(k=3, s=2, expanding_orders=[1, 2], bias=True, mode="inference")
        ngrc.fit(data, target_data)

        inference = ngrc.inference(data)

        assert inference.shape == (data.shape[0] - (ngrc._k - 1) * ngrc._s, target_data.shape[1])

        ### Expansion - No bias
        # NGRC - Expansion - No bias
        data = np.ones((10, 2))

        ngrc = scan.ngrc.NG_RC(k=2, s=1, orders=[1, 3, 5], expanding_orders=[1, 2], mode="inference")
        ngrc.fit(data, target_data)

        inference = ngrc.inference(data)

        assert inference.shape == (data.shape[0] - (ngrc._k - 1) * ngrc._s, target_data.shape[1])

        # SINDY - Expansion - No bias
        ngrc = scan.ngrc.NG_RC(k=1, s=1, orders=[1, 3], expanding_orders=[1, 2], mode="inference")
        ngrc.fit(data, target_data)

        inference = ngrc.inference(data)

        assert inference.shape == (data.shape[0] - (ngrc._k - 1) * ngrc._s, target_data.shape[1])

        # VAR - Expansion - No bias
        ngrc = scan.ngrc.NG_RC(k=3, s=2, expanding_orders=[1, 2], mode="inference")
        ngrc.fit(data, target_data)

        inference = ngrc.inference(data)

        assert inference.shape == (data.shape[0] - (ngrc._k - 1) * ngrc._s, target_data.shape[1])

        ### No Expansion - bias
        # NGRC - No Expansion - bias
        data = np.ones((10, 2))

        ngrc = scan.ngrc.NG_RC(k=2, s=1, orders=[1, 3, 5], bias=True, mode="inference")
        ngrc.fit(data, target_data)

        inference = ngrc.inference(data)

        assert inference.shape == (data.shape[0] - (ngrc._k - 1) * ngrc._s, target_data.shape[1])

        # SINDY - No Expansion - bias
        ngrc = scan.ngrc.NG_RC(k=1, s=1, orders=[1, 3], bias=True, mode="inference")
        ngrc.fit(data, target_data)

        inference = ngrc.inference(data)

        assert inference.shape == (data.shape[0] - (ngrc._k - 1) * ngrc._s, target_data.shape[1])

        # VAR - No Expansion - bias
        ngrc = scan.ngrc.NG_RC(k=3, s=2, bias=True, mode="inference")
        ngrc.fit(data, target_data)

        inference = ngrc.inference(data)

        assert inference.shape == (data.shape[0] - (ngrc._k - 1) * ngrc._s, target_data.shape[1])

        ### No Expansion - No bias
        # NGRC - No bias
        data = np.ones((10, 2))

        ngrc = scan.ngrc.NG_RC(k=2, s=1, orders=[1, 3, 5], mode="inference")
        ngrc.fit(data, target_data)

        inference = ngrc.inference(data)

        assert inference.shape == (data.shape[0] - (ngrc._k - 1) * ngrc._s, target_data.shape[1])

        # SINDY - No bias
        ngrc = scan.ngrc.NG_RC(k=1, s=1, orders=[1, 3], mode="inference")
        ngrc.fit(data, target_data)

        inference = ngrc.inference(data)

        assert inference.shape == (data.shape[0] - (ngrc._k - 1) * ngrc._s, target_data.shape[1])

        # VAR - No bias
        ngrc = scan.ngrc.NG_RC(k=3, s=2, mode="inference")
        ngrc.fit(data, target_data)

        inference = ngrc.inference(data)

        assert inference.shape == (data.shape[0] - (ngrc._k - 1) * ngrc._s, target_data.shape[1])


def test_prediction_starting_series_ngrc_base(self):

    data = np.ones((10, 2))

    ngrc = scan.ngrc.NG_RC(k=2, s=1, orders=[1, 3, 5], expanding_orders=[1, 2], bias=True)
    ngrc.fit(data)

    prediction_steps = 2
    prediction = ngrc.predict(prediction_steps, starting_series=data[-2:])

    assert prediction.shape == (data.shape[1], prediction_steps)
