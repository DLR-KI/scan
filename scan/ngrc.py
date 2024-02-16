"""Implements Next-Generation Reservoir Computing."""

from __future__ import annotations

from copy import deepcopy
from itertools import combinations, combinations_with_replacement
from typing import List, Literal

import numpy as np


class NG_RC:

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
        self._s = s  # spacing between the data points.
        self._orders = orders  # orders used to create unique monomials, e.g orders=[1,2,4], of linear state vectors.
        self._expanding_orders = expanding_orders  # orders to expand the feature space of ngrc,
        # e.g. expanding_orders= [1,2] -> r' = (r^1,r^2)
        self._bias = bias  # Appends bias term to ngrc state vectors, r -> (1,r)
        self._mode = mode  # Define on what the ngrc should be trained on. Options:
        # For prediction choose"coordinates","differences".For inference choose "inference".
        self._order_type = order_type  # Specific order_type. Options: "all" for all unique monomials of certain orders.
        # "interactions" only interacting terms of certain orders.

        self._dictionary: list | None = None

        self._input_data: np.ndarray | None = None
        self._target_data: np.ndarray | None = None

        self._states: np.ndarray | None = None

        self._linear_states: np.ndarray | None = None
        self._nonlinear_states: np.ndarray | None = None
        self._expanded_states: np.ndarray | None = None

        self._regression_parameter = regression_parameter  # regression parameter for Ridge Regression.
        self._w_out: np.ndarray | None = None

        self._initial_prediction_data: np.ndarray | None = None

        self._save_states = (
            save_states  # if save_states is True, saves the intermediate states of the ngrc feature space.
        )
        self._index = index  # used for indexing the hyperparameter setup, no functionality.
        self._features = features  # used for book keeping feature dimension, no functionality.

    def create_train_X_y(self, input_data: np.ndarray, target_data: np.ndarray | None = None) -> None:
        """Creates and/or assigns the training and target data given the chosen ngrc
        "mode"= {"coordinates","differences","inference"}. Further the last part of the data is saved to
        self._initial_prediction_data such that predictions, if not further specified, automatically starts from
        the end of the provided data without further definitions when predict() is called.
        This function is called automatically when apply_ngrc() is called. #1

        Args:
            mode: str = "coordinates" -> Predicts the next coordinate of the data, given the current input.
            mode: str = "differences" -> Predicts the difference between the current input and the next coordinate
                                            of the data. In predict() the difference prediction is automatically
                                            added to current input data point, which predicts the coordinates of
                                            the next data point.
            mode: str = "inference"   -> Allows to define a custom target data on which the ngrc is trained on.

        Returns:
            Saves training data and target data with respect to the necessary warmup time of ngrc given its
            parameters the "k" and "s".
            All are stored in class.

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
            raise ValueError('Mode configuration Error: choose between "coordinates" , "differences" and "inference".')

    def linear_expansion(self, functional: bool = False, input_data: np.ndarray | None = None) -> None | np.ndarray:
        """Creates the linear states of ngrc. Takes "k" past data points of the data, which are separated by "s"
        indices and put them into one vector. This is repeated for the every index of the input data.

        This function is called automatically when apply_ngrc() is called. #2

        Args:
            functional: bool = False   -> Automatically set when states are created for training or inference.
            functional: bool = True    -> Automatically set when states are created for prediction.

        Returns:
            If functional is 'False' updates self._states which are later used for training.
            If functional is 'True' for "coordinates" or "differences" it returns one linear_state vector given
                                    the input data which is later used for prediction.
                                    For "inference" it returns the linear_state vectors of the input data at once.
            linear_states: np.ndarray

        """

        if not functional:

            assert self._input_data is not None

            num_slices = len(self._input_data) - (self._k - 1) * self._s

            indices = np.arange(0, self._k * self._s, self._s) + np.arange(num_slices)[:, np.newaxis]
            pre_expanded_states = self._input_data[indices]
            linear_states = pre_expanded_states.reshape(
                pre_expanded_states.shape[0], pre_expanded_states.shape[1] * pre_expanded_states.shape[2]
            )

            if self._save_states:
                self._linear_states = linear_states

            self._states = linear_states

            return None

        else:

            assert isinstance(input_data, np.ndarray)

            if self._mode == "coordinates" or self._mode == "differences":
                num_slices = 1

            elif self._mode == "inference":
                num_slices = len(input_data) - (self._k - 1) * self._s

            indices = np.arange(0, self._k * self._s, self._s) + np.arange(num_slices)[:, np.newaxis]
            pre_expanded_states = input_data[indices]

            linear_states = pre_expanded_states.reshape(
                pre_expanded_states.shape[0], pre_expanded_states.shape[1] * pre_expanded_states.shape[2]
            )
            assert isinstance(linear_states, np.ndarray)
            return linear_states

    def nonlinear_expansion(self, functional: bool = False, input_data: np.ndarray | None = None) -> None | np.ndarray:
        """Creates the nonlinear states of ngrc. This function takes the orders provided in 'orders' and creates
        a dictionary with the unique monomials of the provided orders. It updates self._states correspondingly.

        This function is called automatically when apply_ngrc() is called. #3

        Args:
            functional: bool = False   -> Automatically set when states are created for training or inference.
            functional: bool = True    -> Automatically set when states are created for prediction.

        Returns:
            If functional is 'False' updates self._states which are later used for training.
            If functional is 'True' for "coordinates" or "differences" it returns the nonlinear_state vector given
                                    the input state vector which is later used for prediction.
                                    For "inference" it returns the nonlinear_state vectors of the input state
                                    vectors at once.
            nonlinear_states: np.ndarray
        """

        assert isinstance(self._states, np.ndarray)
        assert isinstance(self._orders, list)

        if not functional:

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
                    "specify order_type correctly. 'all' standard for using all monomial combinations. 'interactions'"
                    " for only interactions terms in dictionary."
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

    def expanding_states(self, functional: bool = False, input_data: np.ndarray | None = None) -> None | np.ndarray:
        """Creates expanded states of ngrc. This function will update the self._states with different functionalities.

        This function is called automatically when apply_ngrc() is called. #4

        Args:
            functional: bool = False   -> Automatically set when states are created for training or inference.
            functional: bool = True    -> Automatically set when states are created for prediction.

        Returns:
            If functional is 'False' updates self._states which are later used for training.
            If functional is 'True' for "coordinates" or "differences" it returns one expanded_state vector given
                                        the input state vector which is later used for prediction.
                                    for "inference" it returns expanded state vectors of the input state vectors.
            expanded_states: np.ndarray

        """
        assert isinstance(self._states, np.ndarray)

        if not functional:

            if self._expanding_orders is None:
                if self._bias is True:
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
                expanded_states = np.empty((data_length, state_dimension * len(self._expanding_orders)))

                if self._expanding_orders is not None:

                    for i in range(len(self._expanding_orders)):
                        expanded_states[:, i * state_dimension : (i + 1) * state_dimension] = (
                            input_data ** self._expanding_orders[i]
                        )

                if self._bias is True:
                    expanded_states = np.insert(expanded_states, 0, 1, axis=1)

                return expanded_states

    def apply_ngrc(self, functional: bool = False, input_data: np.ndarray | None = None) -> np.ndarray:
        """Creates the ngrc state vectors. This function will create the feature space of ngrc given the used
        hyperparameter and the input data.

        This function is called automatically when when fit() or create_states() is called.

        Args:
            functional: bool = False   -> Automatically set when states are created for training or inference.
            functional: bool = True    -> Automatically set when states are created for prediction.
            input_data: np.ndarray | None -> For training, input data is None,

        Returns:
            If functional is 'False' creates self._states which are used for training.
            If functional is 'True' for "coordinates" or "differences" it returns one state vector given the
                                    input data which is used for prediction,
                                    for "inference" it returns the state vectors of the whole input data
                                    which are used for inference.
            state: np.ndarray

        """
        assert isinstance(self._input_data, np.ndarray)

        if not functional:

            self.linear_expansion()

            if self._orders is not None:
                self.nonlinear_expansion()

            if self._expanding_orders is not None or self._bias is True:
                self.expanding_states()

            assert isinstance(self._states, np.ndarray)
            return self._states

        else:

            assert isinstance(input_data, np.ndarray)

            state = self.linear_expansion(functional=True, input_data=input_data)

            if self._orders is not None:
                state = self.nonlinear_expansion(functional=True, input_data=state)

            if self._expanding_orders is not None or self._bias is True:
                state = self.expanding_states(functional=True, input_data=state)

            assert isinstance(state, np.ndarray)
            return state

    def create_states(self, x: np.ndarray) -> np.ndarray:
        """Creates the ngrc state vectors given input data X.

        Args:
            x: np.ndarray   -> Creates ngrc states for data X.

        Returns:
            x_states: np.ndarray    -> The ngrc feature space given its hyperparameters and the input data.

        """
        if self._mode == "coordinates" or self._mode == "differences":
            self._input_data = np.array(x[:-1])

        if self._mode == "inference":
            self._input_data = np.array(x)

        x_states = self.apply_ngrc()

        return x_states

    def fit(self, x: np.ndarray, y_target: np.ndarray | None = None) -> None:
        """This function will initiate ngrc on the input data 'X' to get ngrc states 'X_states', which
        are trained using ridge regression onto the target data 'y_target'.
        For prediction (self._mode = "coordinates" or "differences"), 'y_target' is automatically generated.
        For inference (self._mode = "inference") 'y_target' must be specified in fit().

        Args:
            X: np.ndarray -> input data from where ngrc states should be created
            y_target: np.ndarray | None, optional   -> target data on what in the input data should be trained on.
                                                    None for "mode" is "coordinates" or "differences"
                                                    np.ndarray for "mode" is "inference".

        Returns:
            self._w_out: np.ndarray stored in class. Learned weights from ridge regression.

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
                    "Minimal time steps in x for prediction must be at least {} steps".format(
                        (self._k - 1) * self._s + 1 + 1
                    )
                )

        self.create_train_X_y(input_data=x, target_data=y_target)

        x_states = self.apply_ngrc()
        y_target = self._target_data

        self._w_out = np.linalg.solve(
            x_states.T @ x_states + self._regression_parameter * np.eye(x_states.shape[1]), x_states.T @ y_target
        ).T

    def predict(self, steps: int, starting_series: np.ndarray | None = None) -> np.ndarray:
        """Uses the trained ngrc to make predictions. By default self._initial_prediction_data is used as a
        starting_series for the prediction, but it can be custom assigned wit starting_series.
        "mode" must be "coordinates" or "differences" during training.

        Args:
            steps: int                      -> Specify the number of prediction steps to be made.
            starting_series: np.ndarray     -> Define custom starting series from where to start the prediction.
                                                Necessary minimal warm_up time of "(k-1)*s" from "k" and "s"
                                                hyperparameters should be taken into account.

        Returns:
            Returns the prediction.
            predictions: np.ndarray

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

        predictions = np.full(shape=(steps, self._initial_prediction_data.shape[1]), fill_value=np.nan)
        x = self._initial_prediction_data[-self._k * self._s :]

        if self._mode == "coordinates":

            for i in range(steps):

                state = self.apply_ngrc(functional=True, input_data=x)
                prediction = self._w_out @ state[0]
                predictions[i] = prediction
                x = np.append(x[1:], [prediction], axis=0)

        elif self._mode == "differences":

            for i in range(steps):

                state = self.apply_ngrc(functional=True, input_data=x)
                prediction = x[-1] + self._w_out @ state[0]
                predictions[i] = prediction
                x = np.append(x[1:], [prediction], axis=0)

        return predictions

    def inference(self, x: np.ndarray) -> np.ndarray:
        """
        Uses the trained ngrc to make inference on the provided data 'x'.
        Therefore, "mode" must be "inference" during training.

        Args:
            x: np.ndarray   ->  Applied the trained ngrc on the data provided

        Returns:
            Returns the prediction of the inference.
            inference_output: np.ndarray

        """
        if self._mode == "inference":

            if len(x) < (self._k - 1) * self._s + 1:
                raise ValueError(
                    "Data length does not match minimal warmup time required, needs to be a least (k-1)*s={} data"
                    " points.".format((self._k - 1) * self._s)
                )

            states = self.apply_ngrc(functional=True, input_data=x)
            inference_output: np.ndarray = self._w_out @ states.T

            return inference_output.T

        else:
            raise ValueError('To call inference(), ngrc._mode has to be "inference".')
