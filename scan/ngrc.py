"""Implements Next-Generation Reservoir Computing."""

from __future__ import annotations

from copy import deepcopy
from itertools import combinations, combinations_with_replacement
from typing import List, Literal

import numpy as np


class NGRC:

    mode_types = Literal["coordinates", "differences", "inference"]
    order_types = Literal["all", "interactions"]

    def __init__(
        self,
        k: int,
        s: int,
        orders: List[int] | None = None,
        expanding_orders: List[int] | None = None,
        mode: mode_types = "coordinates",
        order_type: order_types = "all",
        bias: bool = False,
        regression_parameter: float = 0.01,
        index: int | None = None,
        features: int | None = None,
        save_states: bool = True,
    ):

        self._k = k  # number of data points used for ngrc.
        self._s = s  # index spacing between the data points.
        self._orders = orders  # orders used to create unique monomials.
        self._expanding_orders = expanding_orders  # orders to expand ngrc states.
        # e.g. expanding_orders=[1,2] -> r' = (r^1,r^2)
        self._bias = bias  # Appends bias term to ngrc states, r -> (1,r)
        self._mode = mode  # Defining on what the ngrc should be trained on.
        self._order_type = order_type  # Specific order_type, 'all' or only 'interaction' monomials

        self._dictionary: list | None = None

        self._input_data: np.ndarray | None = None
        self._target_data: np.ndarray | None = None

        self._states: np.ndarray | None = None

        self._linear_states: np.ndarray | None = None
        self._nonlinear_states: np.ndarray | None = None
        self._expanded_states: np.ndarray | None = None

        self._regression_parameter = regression_parameter  # for Ridge Regression.
        self._w_out: np.ndarray | None = None

        self._initial_prediction_data: np.ndarray | None = None

        self._save_states = save_states  # if save_states, save linear, nonlinear and expanded states.
        self._index = index  # for indexing ngrc, no functionality.
        self._features = features  # for bookkeeping feature dimension, no functionality.

    def create_train_and_target_data(self, input_data: np.ndarray, target_data: np.ndarray | None = None) -> None:
        """Creates and/or assigns the training and target data given
        'mode'= {"coordinates","differences","inference"}. Last part of
        data is saved to self._initial_prediction_data for initial
        prediction, if not otherwise specified.

        Function is called automatically when apply_ngrc() is called. #1

        Args:
            mode: "coordinates" -> Predicts the next coordinate.
                                    No 'target_data' required.
            mode: "differences" -> Predicts the difference between
                                    current input and next coordinate.
                                    No 'target_data' required.
            mode: "inference"   -> Custom target data on which the ngrc
                                    is trained on.
                                    'target_data' required.

        Returns:
            Training data and target data respecting warmup time, stored
            in class.

        """

        x = tuple(input_data)
        self._dimension = input_data.shape[1]

        if self._mode == "coordinates":
            if target_data is not None:
                raise ValueError(
                    "For prediction in \"coordinates\" mode, target_data is not processed, has to be 'None'."
                )
            else:
                self._target_data = np.array(x[(self._k - 1) * self._s + 1 :])
                self._initial_prediction_data = np.array(x[-(self._k - 1) * self._s - 1 :])
                self._input_data = np.array(x[:-1])

        elif self._mode == "differences":
            if target_data is not None:
                raise ValueError(
                    "For prediction in \"differences\" mode, target_data is not processed, has to be 'None'."
                )
            else:
                self._target_data = np.array(x[(self._k - 1) * self._s + 1 :]) - np.array(
                    x[(self._k - 1) * self._s : -1]
                )
                self._initial_prediction_data = np.array(x[-(self._k - 1) * self._s - 1 :])
                self._input_data = np.array(x[:-1])

        elif self._mode == "inference":
            if target_data is None:
                raise ValueError("Target data for inference mode is not specified")
            else:
                self._target_data = np.array(deepcopy(target_data)[(self._k - 1) * self._s :])
                self._input_data = np.array(x)

        else:
            raise ValueError('Mode configuration Error: choose between "coordinates", "differences" and "inference".')

    def linear_expansion(self, training: bool = True, input_data: np.ndarray | None = None) -> None | np.ndarray:
        """Creates the linear states of ngrc. Takes 'k' past data points
        of the data, which are separated by 's' indices and puts them
        into one vector. Repeated for the every index of the input data.

        Function is called automatically when apply_ngrc() is called. #2

        Args:
            training:   True    -> Is set when states are created for
                                    training.
            training:   False   -> Is set when states are created for
                                    prediction and inference.

        Returns:
            training:   True    -> Updates self._states
            training:   True    -> For prediction it returns one linear
                                    state vector. For inference it
                                    returns the linear state vectors of
                                    all of inputs_data.
            linear_states

        """

        if training:

            assert self._input_data is not None

            number_of_states = len(self._input_data) - (self._k - 1) * self._s

            indices_first_linear_state = np.arange(0, self._k * self._s, self._s)
            pre_indices_linear_states = np.arange(number_of_states)[:, np.newaxis]
            indices_linear_states = indices_first_linear_state + pre_indices_linear_states

            pre_linear_states = self._input_data[indices_linear_states]
            linear_states = pre_linear_states.reshape(
                pre_linear_states.shape[0], pre_linear_states.shape[1] * pre_linear_states.shape[2]
            )

            if self._save_states:
                self._linear_states = linear_states

            self._states = linear_states

            return None

        else:

            assert isinstance(input_data, np.ndarray)

            if self._mode == "coordinates" or self._mode == "differences":
                number_of_states = 1

            elif self._mode == "inference":
                number_of_states = len(input_data) - (self._k - 1) * self._s

            indices_first_linear_state = np.arange(0, self._k * self._s, self._s)
            pre_indices_linear_states = np.arange(number_of_states)[:, np.newaxis]
            indices_linear_states = indices_first_linear_state + pre_indices_linear_states

            pre_linear_states = input_data[indices_linear_states]
            linear_states = pre_linear_states.reshape(
                pre_linear_states.shape[0], pre_linear_states.shape[1] * pre_linear_states.shape[2]
            )
            assert isinstance(linear_states, np.ndarray)
            return linear_states

    def nonlinear_expansion(self, training: bool = True, input_data: np.ndarray | None = None) -> None | np.ndarray:
        """Creates the nonlinear states of ngrc. Uses 'orders' to create
        a dictionary with unique monomials of the provided orders.

        Function is called automatically when apply_ngrc() is called. #3

        Args:
            training:   True    -> Is set when states are created for
                                    training.
            training:   False   -> Is set when states are created for
                                    prediction or inference.

        Returns:
            training:   True    -> Updates self._states.
            training:   False   -> For prediction it returns one
                                    nonlinear_state vector.
                                    For inference it returns all
                                    nonlinear_state vectors for
                                    input_data.
            nonlinear_states:   e.g. orders=[1,2] takes first order
                                monomials of linear_states and
                                appends unique second order monomials.
        """

        assert isinstance(self._states, np.ndarray)
        assert isinstance(self._orders, list)

        if training:

            dimension = self._states.shape[1]

            if self._order_type == "all":

                unique_monomials = []
                for order in self._orders:
                    for combination in combinations_with_replacement(range(dimension), order):
                        monomial = list(combination)
                        unique_monomials.append(monomial)

                self._dictionary = unique_monomials

            elif self._order_type == "interactions":

                unique_monomials = []
                for order in self._orders:
                    for combination in combinations(range(dimension), order):
                        monomial = list(combination)
                        unique_monomials.append(monomial)

                self._dictionary = unique_monomials

            else:
                raise ValueError(
                    "specify order_type correctly. 'all' for using all monomial combinations."
                    "'interactions' for only interaction terms in dictionary."
                )

            data_length = len(self._states)
            nonlinear_states = np.empty((data_length, len(self._dictionary)))

            for i, monomial in enumerate(self._dictionary):
                nonlinear_states[:, i] = np.prod(self._states[:, monomial], axis=1)

            if self._save_states:
                self._nonlinear_states = nonlinear_states

            self._states = nonlinear_states

            return None

        else:

            assert isinstance(input_data, np.ndarray)
            assert isinstance(self._dictionary, list)

            if self._orders == [1]:
                return input_data

            dimension = input_data.shape[1]
            data_length = len(input_data)

            nonlinear_states = np.empty((data_length, len(self._dictionary)))

            for i, monomial in enumerate(self._dictionary):
                nonlinear_states[:, i] = np.prod(input_data[:, monomial], axis=1)

            return nonlinear_states

    def expanding_states(self, training: bool = True, input_data: np.ndarray | None = None) -> None | np.ndarray:
        """Creates expanded states of ngrc. This function will update
        the self._states with different functionalities.
        Adding bias to ngrc states r.
        self._bias = True -> r_new -> (1,r_old)

        Expands states with orders of it self.
        e.g. self._expanding_orders = [1,2] -> r_new = (r_old,r_old**2)

        Function is called automatically when apply_ngrc() is called. #4

        Args:
            training:   True    -> Is set when states are created for
                                    training.
            training:   False   -> Is set when states are created
                                    for prediction and inference.

        Returns:
            training:   True    -> Updates self._states.
            training:   False   -> For prediction returns one
                                    expanded_state vector.
                                    For "inference", returns expanded
                                    state vectors of input_data.
            expanded_states:    expands ngrc states r_old.

        """
        assert isinstance(self._states, np.ndarray)

        if training:

            if self._expanding_orders is None:
                if self._bias is True:
                    self._states = np.insert(self._states, 0, 1, axis=1)

                    if self._save_states is True:
                        self._expanded_states = self._states

            else:
                data_length = self._states.shape[0]
                state_dimension = self._states.shape[1]
                expanded_state_dimension = state_dimension * len(self._expanding_orders)

                expanded_states = np.empty((data_length, expanded_state_dimension))

                for i in range(len(self._expanding_orders)):
                    expanded_states[:, i * state_dimension : (i + 1) * state_dimension] = (
                        self._states ** self._expanding_orders[i]
                    )

                if self._bias is True:
                    expanded_states = np.insert(expanded_states, 0, 1, axis=1)

                if self._save_states:
                    self._expanded_states = expanded_states

                self._states = expanded_states

            return None

        else:

            assert isinstance(input_data, np.ndarray)

            if self._expanding_orders is None:
                if self._bias is True:
                    return np.insert(input_data, 0, 1, axis=1)
                else:
                    return None

            else:
                data_length = input_data.shape[0]
                state_dimension = input_data.shape[1]
                expanded_state_dimension = state_dimension * len(self._expanding_orders)

                expanded_states = np.empty((data_length, expanded_state_dimension))

                if self._expanding_orders is not None:

                    for i in range(len(self._expanding_orders)):
                        expanded_states[:, i * state_dimension : (i + 1) * state_dimension] = (
                            input_data ** self._expanding_orders[i]
                        )

                if self._bias is True:
                    expanded_states = np.insert(expanded_states, 0, 1, axis=1)

                return expanded_states

    def apply_ngrc(self, training: bool = True, input_data: np.ndarray | None = None) -> np.ndarray:
        """Creates the ngrc state vectors. Function creates the feature
        space of ngrc (states) when called.

        Function is called automatically when fit() or
        create_states() is called.

        Args:
            training:   True    -> Is set when states are created for
                                    training.
            training:   False   -> Is set when states are created for
                                    prediction or inference.
            input_data:         For prediction or inference,
                                during training input data is None.

        Returns:
            training:   True    -> creates self._states.
            training:   False   -> For prediction returns one state
                                    vector. For inference returns the
                                    state vectors of all input data.

            self._states: ngrc states stored in class.
            state: returned during prediction or inference.


        """
        assert isinstance(self._input_data, np.ndarray)

        if training:

            self.linear_expansion(training=True)

            if self._orders is not None:
                self.nonlinear_expansion(training=True)

            if self._expanding_orders is not None or self._bias is True:
                self.expanding_states(training=True)

            assert isinstance(self._states, np.ndarray)
            return self._states

        else:

            assert isinstance(input_data, np.ndarray)

            state = self.linear_expansion(training=False, input_data=input_data)

            if self._orders is not None:
                state = self.nonlinear_expansion(training=False, input_data=state)

            if self._expanding_orders is not None or self._bias is True:
                state = self.expanding_states(training=False, input_data=state)

            assert isinstance(state, np.ndarray)
            return state

    def create_states(self, x: np.ndarray) -> np.ndarray:
        """Creates the ngrc state vectors given input data 'x'.

        Args:
            x:  -> Creates ngrc states for data 'x'.

        Returns:
            x_states:   -> The ngrc feature space given its
                            hyperparameters and the input data.

        """
        if self._mode == "coordinates" or self._mode == "differences":
            self._input_data = np.array(x[:-1])

        if self._mode == "inference":
            self._input_data = np.array(x)

        x_states = self.apply_ngrc(training=True)

        return x_states

    def fit(self, x: np.ndarray, y_target: np.ndarray | None = None) -> None:
        """Function initiates ngrc on the input data 'x' to create
        ngrc states ('x_states'), which are trained using
        ridge regression onto the target data 'y_target'.
        For prediction, 'y_target' is automatically generated.
        For inference, 'y_target' must be specified in fit().

        Args:
            x:  -> Input data from where ngrc states should be created.
            y_target:   -> Target data on which the input data is
                            trained on for "inference". Must be
                            None for "coordinates" or 'differences'.

        Returns:
            self._w_out:    Learned weights from ridge regression.
                            Stored in class.

        """
        if self._mode == "coordinates" or self._mode == "differences":

            # ngrc warmup time -> (self._k-1)*self._s + 1
            # prediction target -> adds +1 for minimal length
            if len(x) < (self._k - 1) * self._s + 1 + 1:
                raise ValueError(
                    "Minimal time steps in x for prediction must be at least {} steps".format(
                        (self._k - 1) * self._s + 1 + 1
                    )
                )

        else:
            if len(x) < (self._k - 1) * self._s + 1:
                raise ValueError(
                    "Minimal steps in x for inference must be at least {} steps".format((self._k - 1) * self._s + 1 + 1)
                )

        self.create_train_and_target_data(input_data=x, target_data=y_target)

        x_states = self.apply_ngrc(training=True)
        y_target = self._target_data

        # Ridge Regression
        self._w_out = np.linalg.solve(
            x_states.T @ x_states + self._regression_parameter * np.eye(x_states.shape[1]), x_states.T @ y_target
        ).T

    def predict(self, steps: int, starting_series: np.ndarray | None = None) -> np.ndarray:
        """Predicts using trained ngrc. By default initial prediction
        start from self._initial_prediction_data, custom start specified
        with 'starting_series'.
        "mode" must be "coordinates" or "differences" during training.

        Args:
            steps:  -> Number of prediction steps to be made.
            starting_series:    -> Custom starting time series. Minimal
                                   length is warmup time (k-1)*s+1.

        Returns:
            Returns the ngrc prediction.

        """
        if self._mode != "coordinates" and self._mode != "differences":
            raise ValueError('To call predict(), ngrc._mode has to be "coordinates" or "differences".')

        assert self._initial_prediction_data is not None

        if starting_series is not None and starting_series.shape[0] < (self._k - 1) * self._s + 1:
            raise ValueError(
                "Starting series not correctly specified, make sure that it has a least (k-1)*s={} data points.".format(
                    (self._k - 1) * self._s + 1
                )
            )

        if starting_series is not None:
            self._initial_prediction_data = starting_series

        initial_shape = self._initial_prediction_data.shape[1]
        predictions = np.full(shape=(steps, initial_shape), fill_value=np.nan)

        x = self._initial_prediction_data[-self._k * self._s :]

        if self._mode == "coordinates":

            for i in range(steps):

                state = self.apply_ngrc(training=False, input_data=x)
                prediction = self._w_out @ state[0]
                predictions[i] = prediction
                x = np.append(x[1:], [prediction], axis=0)

        elif self._mode == "differences":

            for i in range(steps):

                state = self.apply_ngrc(training=False, input_data=x)
                prediction = x[-1] + self._w_out @ state[0]
                predictions[i] = prediction
                x = np.append(x[1:], [prediction], axis=0)

        return predictions

    def inference(self, x: np.ndarray) -> np.ndarray:
        """
        Uses trained ngrc to make inference on data 'x'.
        'mode' must be "inference" during training.

        Args:
            x:  ->  Apply the trained ngrc on that data for inference.

        Returns:
            Returns the predicted inference.

        """
        if self._mode == "inference":

            if len(x) < (self._k - 1) * self._s + 1:
                raise ValueError(
                    "Data length does not match minimal warmup time required,"
                    "needs to be a least (k-1)*s={} data"
                    " points.".format((self._k - 1) * self._s)
                )

            states = self.apply_ngrc(training=False, input_data=x)
            inference_output: np.ndarray = self._w_out @ states.T

            return inference_output.T

        else:
            raise ValueError('To call inference(), ngrc._mode has to be "inference".')
