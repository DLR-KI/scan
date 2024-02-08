from copy import deepcopy
from itertools import combinations, combinations_with_replacement
from typing import Optional, Union

import numpy as np


class NG_RC:
    def __init__(
        self,
        k: int,
        s: int,
        orders: Optional[list] = None,
        expanding_orders: Optional[list] = None,
        mode: Optional[str] = "coordinates",
        order_type: Optional[str] = "all",
        bias: Optional[int] = None,
        regression_parameter = None,
        index: Optional[int] = None,
        features: Optional[int] = None,
        save_states: Optional[bool] = True,
    ):

        self._k = k  # number of data points used for ngrc.
        self._s = s  # spacing between the data points.
        self._orders = orders  # orders used to create unique monomials, e.g orders=[1,2,4], of linear state vectors.
        self._expanding_orders = expanding_orders  # orders to expand the feature space of ngrc, e.g. expanding_orders= [1,2] -> r' = (r^1,r^2)
        self._bias = bias  # Appends bias term to ngrc state vectors, r -> (1,r)
        self._mode = (
            mode  # Define on what the ngrc should be trained on. Options: 'coordinates,'differences' and 'inference'.
        )
        self._order_type = order_type  # Specific order_type. Options: 'all' for all unique monomials of certain orders. 'interactions' only interacting terms of certain orders.

        self._dictionary = None

        self._linear_states = None
        self._nonlinear_states = None
        self._expanded_states = None

        self._states = None
        self._target = None

        self._regression_parameter = regression_parameter
        self._w_out = None

        self._initial_prediction_data = None

        self._save_states = (
            save_states  # if save_states is True, saves the intermediate states of the ngrc feature space.
        )
        self._index = index  # used for indexing the hyperparameter setup, no functionality.
        self._features = features  # used for book keeping feature dimension, no functionality.

    def create_train_X_y(
        self,
        input_data: np.array,
        target_data: Optional[np.array] = None,
        mode: Optional[str] = None,
    ) -> None:

        """Creates and/or assigns the training and target data given the chosen ngrc 'mode'= {'coordinates','differences','inference'}.
        Further the last part of the data is saved to self._initial_prediction_data such that predictions automatically starts at the end of the
        provided data without further definitions when predict() is called.

        Ths is usually done automatically when apply_NG_RC() is called. #1

        Args:

                mode: 'coordinates' -> Predicts the next coordinate of the data, given the current input.
                mode: 'differences' -> Predicts the difference between the current input and the next coordinate of the data.
                                                           In predict() the prediction is automatically added to current input, to predict the next coordinate of the data.
                mode: 'inference'   -> Allows to define a custom target data on which the ngrc is trained on.

        Returns:

                Training data and target data split with respect to desired warmup time of ngrc given the k and s parameters.
        """

        X = tuple(input_data)
        self._dimension = input_data.shape[1]

        if mode == "coordinates":
            self._target_data = np.array(X[(self._k - 1) * self._s + 1 :])
            self._initial_prediction_data = np.array(X[-(self._k - 1) * self._s - 1 :])
            self._input_data = np.array(X[:-1])
            self._system_states_dimensions = [len(X[(self._k - 1) * self._s + 1 :])]

        elif mode == "differences":
            self._target_data = np.array(X[(self._k - 1) * self._s + 1 :]) - np.array(X[(self._k - 1) * self._s : -1])
            self._initial_prediction_data = np.array(X[-(self._k - 1) * self._s - 1 :])
            self._input_data = np.array(X[:-1])
            self._system_states_dimensions = [len(X[(self._k - 1) * self._s + 1 :])]

        elif mode == "inference":
            if target_data is None:
                raise ValueError("Target data for inference mode is not specified")
            else:

                self._target_data = np.array(deepcopy(target_data)[(self._k - 1) * self._s + 1 :])
                self._input_data = np.array(X)
                self._system_states_dimensions = [len(X[(self._k - 1) * self._s + 1 :])]

        else:
            raise ValueError("Mode configuration Error: choose between coordinates , differences and inference ")

    def linear_expansion(self, functional=False, input_data=None):

        """Creates the linear states of ngrc. Takes k data points of the data, which are separated by s indices and put them into one vector.
        This is repeated for the every index of the input data.

        Ths is usually done automatically when apply_NG_RC() is called. #2

        Args:

                functional: False   -> Automatically set when states are created for training or inference.
                functional: True    -> Automatically set when states are created for prediction.

        Returns:

                If functional is 'False', updates self._states which are later used for training.
                If functional is 'True', returns the linear_state vector given the input data which is later used for prediction.

        """

        if functional is False:

            num_slices = len(self._input_data) - (self._k - 1) * self._s - 1

            if self._mode == "coordinates" or self._mode == "differences":
                num_slices += 1

            indices = np.arange(0, self._k * self._s, self._s) + np.arange(num_slices)[:, np.newaxis]
            pre_expanded_states = self._input_data[indices]
            linear_states = pre_expanded_states.reshape(
                pre_expanded_states.shape[0], pre_expanded_states.shape[1] * pre_expanded_states.shape[2]
            )

            self._linear_states = linear_states
            self._states = linear_states

        if functional:

            if self._mode == "coordinates" or self._mode == "differences":
                num_slices = 1

            elif self._mode == "inference":
                num_slices = len(input_data) - (self._k - 1) * self._s - 1

            indices = np.arange(0, self._k * self._s, self._s) + np.arange(num_slices)[:, np.newaxis]
            pre_expanded_states = input_data[indices]

            linear_states = pre_expanded_states.reshape(
                pre_expanded_states.shape[0], pre_expanded_states.shape[1] * pre_expanded_states.shape[2]
            )

            return linear_states

    def nonlinear_expansion(self, functional=False, input_data=None):

        """Creates the nonlinear states of ngrc. This function takes the orders provided in 'orders' and creates a dictionary with the unique
        monomials of the provided orders. It updates self._states correspondingly.

        Ths is usually done automatically when apply_NG_RC() is called. #3

        Args:

                functional: False   -> Automatically set when states are created for training or inference.
                functional: True    -> Automatically set when states are created for prediction.

        Returns:

                If functional is 'False', updates self._states which are later used for training.
                If functional is 'True', returns the nonlinear_state vector given the input data which is later used for prediction.

        """

        if functional is False:

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
                    "specify order_type correctly. 'all' standard for using all combinations. 'interactions' for only"
                    " interactions terms in dictionary."
                )

            data_length = len(self._states)
            nonlinear_states = np.empty((data_length, len(self._dictionary)))

            for i, monomial in enumerate(self._dictionary):
                nonlinear_states[:, i] = np.prod(self._states[:, monomial], axis=1)

            if self._save_states is True:
                self._nonlinear_states = nonlinear_states

            self._states = nonlinear_states

        if functional:

            if self._orders == [1]:
                return input_data

            dimension = input_data.shape[1]
            data_length = len(input_data)

            nonlinear_states = np.empty((data_length, len(self._dictionary)))

            for i, monomial in enumerate(self._dictionary):
                nonlinear_states[:, i] = np.prod(input_data[:, monomial], axis=1)

            return nonlinear_states

    def expanding_states(self, functional=False, input_data=None):

        """Creates expanded states of ngrc. This function will update the self._states with different functionalities.

        Ths is usually done automatically when apply_NG_RC() is called. #4

        Args:

                functional: False   -> Automatically set when states are created for training or inference.
                functional: True    -> Automatically set when states are created for prediction.

        Returns:

                If functional is 'False', updates self._states which are later used for training.
                If functional is 'True', returns the expanded_state vector given the input data which is later used for prediction.

        """

        if functional is False:

            if self._expanding_orders is None:
                if self._bias is None:
                    if self._save_states is True:
                        self._expanded_states = self._states

                if self._bias is not None:
                    self._states = np.insert(self._states, 0, 1, axis=1)

                    if self._save_states is True:
                        self._expanded_states = self._states

            else:
                data_length = self._states.shape[0]
                state_dimension = self._states.shape[1]
                expanded_states = np.empty((data_length, state_dimension * len(self._expanding_orders)))

                for i in range(len(self._expanding_orders)):
                    expanded_states[:, i * state_dimension : (i + 1) * state_dimension] = (
                        self._states ** self._expanding_orders[i]
                    )

                if self._bias is not None:
                    expanded_states = np.insert(expanded_states, 0, 1, axis=1)

                if self._save_states is True:
                    self._expanded_states = expanded_states

                self._states = expanded_states

        if functional:

            if self._expanding_orders is None:
                if self._bias is None:
                    return input_data

                if self._bias is not None:
                    return np.insert(input_data, 0, 1, axis=1)

            else:
                data_length = input_data.shape[0]
                state_dimension = input_data.shape[1]
                expanded_states = np.empty((data_length, state_dimension * len(self._expanding_orders)))

                if self._expanding_orders is not None:

                    for i in range(len(self._expanding_orders)):
                        expanded_states[:, i * state_dimension : (i + 1) * state_dimension] = (
                            input_data ** self._expanding_orders[i]
                        )

                if self._bias is not None:
                    expanded_states = np.insert(expanded_states, 0, 1, axis=1)

                return expanded_states

    def apply_NG_RC(self, functional=False, input_data=None):

        """Creates the ngrc state vectors. This function will create the feature space of ngrc given the used hyperparameter and the input data.

        Ths is usually done automatically when fit() or create_states() is called.

        Args:

                functional: False   -> Automatically set when states are created for training or inference.
                functional: True    -> Automatically set when states are created for prediction.

        Returns:

                If functional is 'False', creates self._states which are used for training.
                If functional is 'True', returns the linear_state vector given the input data which is later used for prediction.

        """

        if functional == False:

            self.linear_expansion()

            if self._orders is not None:
                self.nonlinear_expansion()

            if self._expanding_orders is not None or self._bias is not None:
                self.expanding_states()

            return self._states

        if functional == True:

            state = self.linear_expansion(functional=True, input_data=input_data)

            if self._orders is not None:
                state = self.nonlinear_expansion(functional=True, input_data=state)

            return state

    def create_states(self, X: np.array) -> np.array:

        """Creates the ngrc state vectors given input data X.

        Args:

                X: np.array   -> Creates ngrc states for data X

        Returns:

                The feature space given the hyperparameter and input data.

        """

        self.create_train_X_y(input_data=X, mode=self._mode)
        X_states = self.apply_NG_RC()

        return X_states

    def fit(self, X: np.array, y_target: Optional[np.array] = None) -> None:

        self.create_train_X_y(input_data=X, target_data=y_target, mode=self._mode)

        X_states = self.apply_NG_RC()
        y_target = self._target_data

        self._w_out = np.linalg.solve(
            X_states.T @ X_states + self._regression_parameter * np.eye(X_states.shape[1]), X_states.T @ y_target
        ).T

    def predict(self, steps: int, starting_series: Optional[np.array] = None) -> np.array:

        """Uses the trained ngrc to make predictions. By default self._initial_prediction_data is used as a starting_series for the prediction,
        but it can be custom assigned. Therefore, 'mode' must be 'coordinates' or 'differences' during training.

        Args:

                steps: int                  -> Specify the number of prediction steps to be made.
                starting_series: np.array   -> Define custom starting series from where to start predicting. Necessary warm_up time from k and s hyperparameter should be taken into account.

        Returns:

                Returns the prediction.
                If error occurs during prediction, e.g. if trajectory diverges, np.nan is returned.

        """

        if starting_series is not None and starting_series.shape[0] < (self._k - 1) * self._s + 1:
            raise ValueError(
                "Starting series not correctly specified, make sure that it has a least (k-1)*s+1={} data points."
                .format((self._k - 1) * self._s + 1)
            )

        if starting_series is not None:
            self._initial_prediction_data = starting_series

        predictions = np.full(shape=(steps, self._initial_prediction_data.shape[1]), fill_value=np.nan)
        X = self._initial_prediction_data[-self._k * self._s :]

        if self._mode == "coordinates":

            for i in range(steps):

                state = self.apply_NG_RC(functional=True, input_data=X)
                prediction = self._w_out @ state[0]
                predictions[i] = prediction
                X = np.append(X[1:], [prediction], axis=0)

        elif self._mode == "differences":

            for i in range(steps):

                state = self.apply_NG_RC(functional=True, input_data=X)
                prediction = X[-1] + self._w_out @ state[0]
                predictions[i] = prediction
                X = np.append(X[1:], [prediction], axis=0)

        return predictions

    def inference(self, data: np.array) -> np.array:

        """Warning: This function is in development as the states created for inference will substitute states created during training.
        However this will not affect the inference.

        Uses the trained ngrc to make inference on the 'data' provided. Therefore, 'mode' must be 'inference' during training.

        Args:

                data:       ->  Applied the trained ngrc on the data provided

        Returns:

                Returns the prediction of the inference.

        """

        if len(data) < (self._k - 1) * self._s:
            raise (
                "Data length does not match minimal warmup time required, needs to be a least (k-1)*s+1={} data points."
                .format((self._k - 1) * self._s + 1)
            )

        linear_states = self.linear_expansion(functional=True, input_data=data)
        nonlinear_states = self.nonlinear_expansion(functional=True, input_data=linear_states)
        expanded_nonlinear_states = self.expanding_states(functional=True, input_data=nonlinear_states)

        inference_output = self._w_out @ expanded_nonlinear_states.T

        return inference_output
