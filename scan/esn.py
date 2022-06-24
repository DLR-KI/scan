"""Implements the Echo State Network (ESN) used in Reservoir Computing."""

from __future__ import annotations

import gc
from copy import deepcopy
from typing import Any, Callable

import networkx as nx
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from loguru import logger

from scan import utilities
from scan._version import __version__
from scan.utilities import FlagType

# Synonyms flags for the different w_out fit options
w_out_fit_flag_synonyms = utilities.SynonymDict(
    {
        0: ["linear_r", "simple"],
        1: "linear_and_square_r",
    }
)

# Synonyms flags for the different activation function options
act_fct_flag_synonyms = utilities.SynonymDict(
    {
        0: ["tanh_simple", "simple"],
    }
)

# Synonyms flags for the different network generation options
n_type_flag_synonyms = utilities.SynonymDict(
    {
        0: ["random", "erdos_renyi"],
        1: ["scale_free", "barabasi_albert"],
        2: ["small_world", "watts_strogatz"],
    }
)


class _ESNCore:
    """The non-reducible core of ESN RC training and prediction.

    While technically possible to be used on it's own, this is very much not recommended. Use the child class(es)
    instead.

    """

    def __init__(self) -> None:

        logger.debug("Create _ESNCore instance")

        self.w_in: np.ndarray | None = None

        self.network: np.ndarray | None = None

        self.w_out: np.ndarray | None = None

        self.act_fct: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None

        self.w_out_fit_flag: int | None = None

        self.last_r: np.ndarray | None = None
        self.last_r_gen: np.ndarray | None = None

        self.loc_nbhd: np.ndarray | None = None

        self.reg_param: float | None = None

    def __hash__(self) -> int:
        """Create an MD5 hash of the dictionary of all class attributes using our custom utilities hash function.

        Returns:
            MD5 hash.

        """
        return utilities.hash_dict(self.__dict__)

    def __eq__(self, other: object) -> bool:
        """Compare all class attributes by comparing the class instances' hash sums.

        Args:
            other: The other ESN class instance to compare this instance to.

        Returns:
            True if all attributes equal, False if not.

        """
        if not isinstance(other, _ESNCore):
            return NotImplemented
        else:
            return self.__hash__() == other.__hash__()

    def synchronize(self, x: np.ndarray, save_r: bool = False) -> np.ndarray | None:
        """Synchronize the reservoir state with the input time series.

        This is usually done automatically in the training and prediction functions.

        Needs self.last_r to be set up appropriately, usually via the train() and predict() functions implemented by
        the inheritors of this class.

        Args:
            x: Input data to be used for the synchronization, shape (t, d).
            save_r: If true, saves and returns r.

        Returns:
            All r states if save_r is True, None if False

        """
        assert self.network is not None
        assert self.last_r is not None
        assert self.act_fct is not None

        if save_r:
            r = np.zeros((x.shape[0], self.n_dim))
            r[0] = self.act_fct(x[0], self.last_r)
            for t in np.arange(x.shape[0] - 1):
                r[t + 1] = self.act_fct(x[t + 1], r[t])
            self.last_r = deepcopy(r[-1])
            return r
        else:
            for t in np.arange(x.shape[0]):
                self.last_r = self.act_fct(x[t], self.last_r)
            return None

    def r_to_generalized_r(self, r: np.ndarray) -> np.ndarray:
        """Convert the internal reservoir state r into the generalized r_gen

        r_gen is the (nonlinear) transformation applied to r before the output is calculated. The type of transformation
        is set via the self.w_out_fit_flag parameter, which is typically set during train().

        Args:
            r: The reservoir state to generalize. Shape (t, d) and shape (d,) are supported.

        Returns:
            The generalized reservoir state.

        """
        if self.w_out_fit_flag == 0:
            return r
        elif self.w_out_fit_flag == 1:
            return np.hstack((r, r**2))
        else:
            raise ValueError(f"self.w_out_fit_flag {self.w_out_fit_flag} unknown!")

    def _fit_w_out(self, y_train: np.ndarray, r: np.ndarray) -> None:
        """Fit the output matrix self.w_out after training.

        Uses linear regression and Tikhonov regularization.

        Args:
            y_train: Desired prediction from the reservoir states, shape (t, d).
            r: Reservoir states, shape (t, d).

        """
        logger.debug("Fit _w_out according to method %s" % str(self.w_out_fit_flag))

        assert self.reg_param is not None

        # Note: Memory inefficient, as it seemingly copies r twice! if the hstack is called
        r_gen = self.r_to_generalized_r(r)

        # Note: Slow to calc. Due to the above hstack not the mem bottleneck for T bound applications though
        a = r_gen.T @ r_gen
        # Note: Trivial from a computation time and memory perspective
        a += self.reg_param * np.eye(r_gen.shape[1])

        if self.loc_nbhd is None:
            b = r_gen.T @ y_train
        else:
            # If we are using local states we only want to use the core dimensions for the fit of w_out, i.e. the
            # dimensions where the corresponding locality matrix == 2
            b = r_gen.T @ y_train[:, self.loc_nbhd == 2]

        # Note: Surprisingly fast, even for an 8000 ndim network! (for x_dim=1, anyway)
        #  Obviously the memory bottleneck for n_dim bound applications
        self.w_out = np.linalg.solve(a, b).T

        return None

    def _predict_step(self, x: np.ndarray) -> np.ndarray:
        """Predict a single time step.

        Assumes a synchronized reservoir.
        Changes self.last_r and self.last_r_gen to stay synchronized to the new system state y.

        Args:
            x: input for the d-dim. system, shape (d,).

        Returns:
            The next time step y as predicted from x, using _w_out and _last_r, shape (d,).

        """
        assert self.last_r is not None
        assert self.act_fct is not None
        assert self.w_out is not None

        self.last_r = self.act_fct(x, self.last_r)
        self.last_r_gen = self.r_to_generalized_r(self.last_r)

        # NOTE: Seemingly, type hinting of numpy's matmul is broken, which is why we need the manual type hint below
        y: np.ndarray = self.w_out @ self.last_r_gen

        if self.loc_nbhd is not None:
            temp = np.empty(self.loc_nbhd.shape)
            temp[:] = np.nan
            temp[self.loc_nbhd == 2] = y
            y = temp

        return y


class ESN(_ESNCore):
    """Implements basic RC functionality.

    This class is written such to implement "normal" RC as well as a framework from which to implement local RC, or
    other generalizations, by using this class as a building block.

    """

    def __init__(self) -> None:

        logger.debug("Create ESN instance")

        super().__init__()

        # create_network() assigns values to:
        self.n_dim: int | None = None  # network_dimension
        self.n_rad: float | None = None  # network_spectral_radius
        self.n_avg_deg: float | None = None  # network_average_degree
        self.n_edge_prob: float | None = None
        self.n_type_flag: int | None = None  # network_type
        self.n_seed: int | None = None
        self.network: np.ndarray | None = self.network

        # _create_w_in() which is called from train() assigns values to:
        self.w_in_sparse: bool | None = None
        self.w_in_scale: float | None = None
        self.w_in_seed: int | None = None
        self.w_in: np.ndarray | None = self.w_in

        # set_activation_function assigns values to:
        self.act_fct_flag: int | None = None
        self.act_fct: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = self.act_fct

        # train() assigns values to:
        self.x_dim: int | None = None  # Typically called d
        self.reg_param: float | None = self.reg_param
        self.w_out: np.ndarray | None = self.w_out
        # if save_r is true, train() also assigns values to:
        self.r_train: np.ndarray | None = None

        # predict() assigns values to:
        # if save_r is true, predict() also assigns values to:
        self.r_pred: np.ndarray | None = None

        # Set during instance creation
        self.scan_version: str = __version__

    def _create_w_in(self) -> None:
        """Create the input matrix w_in.

        Not to be called directly, but instead through train().

        """
        logger.debug("Create w_in")

        assert self.x_dim is not None
        assert self.n_dim is not None
        assert self.w_in_scale is not None
        assert self.w_in_sparse is not None
        assert self.w_in_seed is not None

        w_in_rng = np.random.default_rng(self.w_in_seed)

        if self.w_in_sparse:
            self.w_in = np.zeros((self.n_dim, self.x_dim))
            for i in range(self.n_dim):
                random_x_coord = w_in_rng.choice(np.arange(self.x_dim))
                # maps input values to reservoir
                self.w_in[i, random_x_coord] = w_in_rng.uniform(low=-self.w_in_scale, high=self.w_in_scale)
        else:
            self.w_in = w_in_rng.uniform(low=-self.w_in_scale, high=self.w_in_scale, size=(self.n_dim, self.x_dim))

    def create_network(
        self,
        n_dim: int = 500,
        n_rad: float = 0.1,
        n_avg_deg: int | float = 6.0,
        n_type_flag: FlagType = "random",
        n_seed: int = 0,
    ) -> np.ndarray:
        """Create the internal network used as reservoir.

        Args:
            n_dim: Nr. of nodes in the network.
            n_rad: Spectral radius of the network. Must be greater than zero. Should usually be chosen to be smaller
                than one to ensure the echo state property, but depending on network and data spectral radii larger
                than one can deliver good, or even the best, results.
            n_avg_deg: Average node degree (number of connections) the network should have.
            n_type_flag: Type of network to be generated. Possible flags and their synonyms are:

                - "random", "erdos_renyi".
                - "scale_free", "barabasi_albert".
                - "small_world", "watts_strogatz".
            n_seed: Seed for the random parts of the network generation.

        Returns:
            The network used as reservoir.

        """
        logger.debug("Create network")

        self.n_dim = n_dim
        self.n_rad = n_rad
        self.n_avg_deg = n_avg_deg
        self.n_edge_prob = self.n_avg_deg / (self.n_dim - 1)
        self.n_seed = n_seed
        self.n_type_flag = n_type_flag_synonyms.find_flag(n_type_flag)

        self._create_network_connections()
        self._vary_network()

        # TODO: I don't know why we need this type hint assertion here, but mypy complains otherwise.
        assert self.network is not None

        return self.network

    def _create_network_connections(self) -> None:
        """Generate the network connections.

        Specification done via protected members.

        """
        assert self.n_dim is not None
        assert self.n_edge_prob is not None
        assert self.n_seed is not None
        assert self.n_avg_deg is not None

        # We use the old np.random.RandomState instead of the new np.random.default_rng random number generator here as
        # the below networkx functions don't accept the newer generator as input.
        if self.n_type_flag == 0:
            network = nx.fast_gnp_random_graph(self.n_dim, self.n_edge_prob, seed=np.random.RandomState(self.n_seed))
        elif self.n_type_flag == 1:
            network = nx.barabasi_albert_graph(
                self.n_dim, int(self.n_avg_deg / 2), seed=np.random.RandomState(self.n_seed)
            )
        elif self.n_type_flag == 2:
            network = nx.watts_strogatz_graph(
                self.n_dim, k=int(self.n_avg_deg), p=0.1, seed=np.random.RandomState(self.n_seed)
            )
        else:
            raise ValueError("the network type %s is not implemented" % str(self.n_type_flag))

        self.network = nx.to_numpy_array(network)

    def _vary_network(self) -> None:
        """Varies the weights of self.network, while conserving the topology.

        The non-zero elements of the adjacency matrix are uniformly randomized, and the matrix is scaled
        (self.scale_network()) to self.spectral_radius.

        Specification done via protected members.

        """
        assert self.network is not None
        assert self.n_seed is not None

        # contains the tuples of the non-zero elements:
        arg_binary_network = np.argwhere(self.network)

        # uniform entries from [-0.5, 0.5) at all non-zero locations:
        rand_shape = self.network[self.network != 0.0].shape
        self.network[arg_binary_network[:, 0], arg_binary_network[:, 1]] = (
            np.random.default_rng(self.n_seed).random(size=rand_shape) - 0.5
        )
        self._scale_network()

    def _scale_network(self) -> None:
        """Scale self.network, according to desired spectral radius.

        Specification done via protected members.

        """
        assert self.network is not None
        assert self.n_dim is not None
        assert self.n_rad is not None

        self.network = scipy.sparse.csr_matrix(self.network)
        eigenvals = scipy.sparse.linalg.eigs(self.network, k=1, v0=np.ones(self.n_dim), maxiter=int(1e3 * self.n_dim))[
            0
        ]

        maximum = np.absolute(eigenvals).max()
        self.network = (self.n_rad / maximum) * self.network

    def _set_activation_function(self, act_fct_flag: FlagType) -> None:
        """Set the activation function corresponding to the act_fct_flag.

        Args:
            act_fct_flag: flag corresponding to the activation function one wants to use, see :func:`~esn.ESN.train`
                for a list of possible flags.

        """
        logger.debug("Set activation function to flag: %s" % act_fct_flag)

        self.act_fct_flag = act_fct_flag_synonyms.find_flag(act_fct_flag)

        if self.act_fct_flag == 0:
            self.act_fct = self._act_fct_tanh_simple
        else:
            raise ValueError("Activation function flag %s is not implemented!" % str(self.act_fct_flag))

    def _act_fct_tanh_simple(self, x: np.ndarray, r: np.ndarray) -> np.ndarray:
        """Standard activation function of the elementwise np.tanh().

        Args:
            x: x_data, shape (d,).
            r: Network states, shape (n,).

        Returns:
            New reservoir state.

        """
        assert self.w_in is not None
        assert self.network is not None
        # NOTE: Seemingly, type hinting of numpy's matmul is broken, which is why we need the manual type hint below
        out: np.ndarray = np.tanh(self.w_in @ x + self.network @ r)
        return out

    def create_input_matrix(
        self, x_dim: int, w_in_scale: float = 1.0, w_in_sparse: bool = True, w_in_seed: int = 0
    ) -> None:
        """Create the input matrix w_in.

        Can be used to create w_in before train(), otherwise it is called there at the latest.

        Args:
            x_dim: Dimension of the input data, often also called d.
            w_in_scale: Maximum absolute value of the (random) w_in elements.
            w_in_sparse: If true, creates w_in such that one element in each row is non-zero (Lu,Hunt, Ott 2018).
            w_in_seed: Seed for the random parts of the w_in matrix generation.

        """
        self.w_in_scale = w_in_scale
        self.w_in_sparse = w_in_sparse
        self.x_dim = x_dim
        self.w_in_seed = w_in_seed
        self._create_w_in()

    def train(
        self,
        x_train: np.ndarray,
        sync_steps: int,
        reg_param: float = 1e-5,
        w_in_scale: float = 1.0,
        w_in_sparse: bool = True,
        w_in_seed: int = 0,
        w_in_no_update: bool = False,
        act_fct_flag: FlagType = "tanh_simple",
        reset_r: bool = True,
        save_r: bool = False,
        w_out_fit_flag: FlagType = "simple",
        loc_nbhd: np.ndarray | None = None,
    ) -> None:
        """Synchronize and then train the reservoir.

        Args:
            x_train: Input data used to synchronize and then train the reservoir. Shapes (t,), (t, d) and (t, d, s) are
                supported.
            sync_steps: How many steps to use for synchronization before the prediction starts.
            reg_param: Weight for the Tikhonov-regularization term.
            w_in_scale: Maximum absolute value of the (random) w_in elements.
            w_in_sparse: If true, creates w_in such that one element in each row is non-zero (Lu,Hunt, Ott 2018).
            w_in_seed: Seed for the random parts of the w_in matrix generation.
            w_in_no_update: If true and the input matrix w_in does already exist from a previous training run, w_in does
                not get updated, regardless of all other parameters concerning it.
            act_fct_flag: Specifies the activation function to be used during training (and prediction). Possible flags
                and their synonyms are:

                - "tanh_simple", "simple".
            reset_r: Reset the internal reservoir state r to the default value of all zeros before each prediction
                slice. If this is False, the prediction will depend on the order of the prediction slices as, even with
                a reasonable number of synchronization steps, the previous prediction slices will very slightly
                influence the later ones.
            save_r: If true, saves the reservoir state r(t) internally.
            w_out_fit_flag: Type of nonlinear transformation applied to the reservoir states r to be used during the fit
                of the output matrix w_out and future predictions using it.
            loc_nbhd: The local neighborhood used for the (generalized) local states approach. For more information,
                please see the docs.

        """
        logger.debug("Start training")

        assert self.network is not None

        self.reg_param = reg_param
        self.loc_nbhd = loc_nbhd
        self.w_out_fit_flag = w_out_fit_flag_synonyms.find_flag(w_out_fit_flag)

        if x_train.ndim == 1:
            x_train = x_train[:, np.newaxis, np.newaxis]
        elif x_train.ndim == 2:
            x_train = x_train[:, :, np.newaxis]

        if sync_steps == 0:
            x_sync = None
        else:
            x_sync = x_train[:sync_steps]
            x_train = x_train[sync_steps:]

        # The last value of x_train can't be used for the training, as there is nothing to compare the resulting
        # prediction with, hence the "-1" in the train_steps variable definition here.
        train_steps = x_train.shape[0] - 1
        x_dim = x_train.shape[1]
        slices = x_train.shape[2]

        if self.w_in is not None and w_in_no_update:
            if not self.x_dim == x_dim:
                raise ValueError(
                    f"the x_dim specified in create_input_matrix does not match the data x_dim: {self.x_dim} vs {x_dim}"
                )
        else:
            self.create_input_matrix(x_dim, w_in_scale=w_in_scale, w_in_sparse=w_in_sparse, w_in_seed=w_in_seed)

        self._set_activation_function(act_fct_flag=act_fct_flag)

        # r = np.zeros((train_steps, self.n_dim, slices))
        r = np.zeros((train_steps * slices, self.n_dim))
        r_view = r.view()  # shape: (train_steps * slices, self.n_dim)

        r_view.shape = (slices, train_steps, self.n_dim)  # C layout, as dimension speed is x_dim > time > slices
        r_view = r_view.swapaxes(0, 2)  # shape: (train_steps, self.n_dim, slices)
        r_view = r_view.swapaxes(0, 1)  # shape: (slices, self.n_dim, train_steps)

        if self.last_r is None:
            self.last_r = np.zeros(self.n_dim)

        for slice_nr in range(slices):
            logger.debug(f"Start training of slice Nr.{slice_nr:3d}")
            if reset_r:
                # Initialize/reset the current reservoir state to zeros before each new slice. That way training doesn't
                # depend on slice order, only content
                self.last_r = np.zeros(self.n_dim)
            if x_sync is not None:
                self.synchronize(x_sync[:, :, slice_nr])
            # The last value of x_train can't be used for the training, as there is nothing to compare the resulting
            # prediction with. As such, we cut the last time step of x_train here.
            r_view[:, :, slice_nr] = self.synchronize(x_train[:-1, :, slice_nr], save_r=True)

        y_train = x_train[1:]
        # I wouldn't be surprised if there was a smarter way to reshape r and y from 3dim to 2dim correctly but this is
        # the only way I could figure out how to. The problem is that with the currently used meaning for the
        # dimensions of [timestep, x_data-dimension, x_data_slice] the 'fastest changing' index is actually the middle
        # one. For this reason alone, switching the default meaning from of the first two dim from
        # [timestep, x-data-dimension] to [x-data-dimension, timestep] might be a good idea. ALSO, doing so may enable
        # the in-place reshaping of r.shape = (r.shape[0] * r.shape[2], r.shape[1]), which of course would be much
        # smarter from a memory usage perspective.

        y_train = np.reshape(y_train.swapaxes(0, 1), newshape=(x_dim, train_steps * slices), order="F").swapaxes(0, 1)

        # Fit using all the above train segments, now combined into 2dim arrays
        if save_r:
            self.r_train = r

        self._fit_w_out(y_train, r)

    def predict(
        self,
        x_pred: np.ndarray,
        sync_steps: int = 0,
        pred_steps: int | None = None,
        reset_r: bool = True,
        save_r: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Synchronize the reservoir, then predict the system evolution.

        Args:
            x_pred: Input data used to synchronize the reservoir, and, if long enough, used to create the comparison for
                the prediction by being returned as y_test in the output. Shapes (t,), (t,d) and (t,d,s) are supported.
            sync_steps: How many steps to use for synchronization before the prediction starts.
            pred_steps: How many steps to predict.
            reset_r: Reset the internal reservoir state r to the default value of all zeros between each prediction
                slice. If this is False, the prediction will depend on the order of the prediction slices as, even with
                a reasonable number of synchronization steps, the previous prediction slices will very slightly
                influence the later ones.
            save_r: If true, saves the reservoir state r(t) internally.

        Returns:
            2-element tuple containing:

            - **y_pred**: Predicted future states.
            - **y_test**: Data taken from the input to compare the prediction with. If the prediction were "perfect"
                y_pred and y_test would be equal. Be careful though, y_test might be shorter than y_pred, or even None,
                depending on the length of the input data and the prediction.

        """
        logger.debug("Start Prediction")

        assert self.w_in is not None
        assert self.network is not None
        assert self.w_out is not None
        # # Having no last r state in the prediction function should never happen. If it for some reason ever does, we
        # # should probably just assign it the default self.last_r = np.zeros(self.n_dim)
        # assert self.last_r is not None

        if pred_steps is None:
            pred_steps = x_pred.shape[0] - sync_steps - 1

        x_pred_input_original_ndim = x_pred.ndim

        if x_pred.ndim == 1:
            x_pred = x_pred[:, np.newaxis, np.newaxis]
        elif x_pred.ndim == 2:
            x_pred = x_pred[:, :, np.newaxis]

        x_dim = x_pred.shape[1]
        slices = x_pred.shape[2]

        if x_pred.shape[0] > sync_steps + pred_steps + 1:
            x_pred = x_pred[: sync_steps + pred_steps + 1]

        # If the input data is longer than the number of synchronization steps, we generate a y_test to compare the
        # prediction against here.
        if sync_steps == 0:
            x_sync = None
            y_test = x_pred[1:]
        elif sync_steps <= x_pred.shape[0]:
            x_sync = x_pred[:sync_steps]
            y_test = x_pred[sync_steps + 1 :]
        else:
            raise ValueError(
                f"Requested sync_steps of {sync_steps} are longer than the supplied input data of shape {x_pred.shape}"
            )

        y_pred = np.zeros((pred_steps, x_dim, slices))

        if save_r:
            self.r_pred = np.zeros((pred_steps, self.n_dim, slices))

        for slice_nr in range(slices):
            logger.debug(f"Start prediction of slice Nr.{slice_nr:3d}")
            if reset_r:
                # Initialize/reset the current reservoir state to zeros before each new slice. That way training doesn't
                # depend on slice order, only content
                self.last_r = np.zeros(self.n_dim)
            if x_sync is not None:
                self.synchronize(x_sync[:, :, slice_nr])

            y_pred_slice = y_pred[:, :, slice_nr]
            y_pred_slice[0] = self._predict_step(x_pred[sync_steps, :, slice_nr])

            if save_r:
                # TODO: Mypy incorrectly(?) thinks that self.r_pred could be None here, which is why we have to ignore
                #  the types for the two r_pred lines
                self.r_pred[0, :, slice_nr] = self.last_r  # type: ignore
                for t in range(pred_steps - 1):
                    y_pred_slice[t + 1] = self._predict_step(y_pred_slice[t])
                    self.r_pred[t + 1, :, slice_nr] = self.last_r  # type: ignore
            else:
                for t in range(pred_steps - 1):
                    y_pred_slice[t + 1] = self._predict_step(y_pred_slice[t])

        # Match output dimensions to input dimensions
        # This inplace reshaping is only so trivial here because we throw entire dimensions away. If, like in train(),
        # you need to fold higher dimensional arrays into lower dimensional ones, the index order you iterate over
        # becomes very important!
        # Also note that y_test.shape[0] =/= pred_steps in general, as y_test might be shorter than y_pred if pred_steps
        # was specified manually.
        if x_pred_input_original_ndim == 1:
            y_pred.shape = (pred_steps,)
            y_test.shape = (y_test.shape[0],)
        elif x_pred_input_original_ndim == 2:
            y_pred.shape = (pred_steps, x_dim)
            y_test.shape = (y_test.shape[0], x_dim)

        return y_pred, y_test


class ESNWrapper(ESN):
    """Wrapper for the ESN class, implementing convenience functions."""

    def __init__(self) -> None:
        logger.debug("Create ESNWrapper instance")
        super().__init__()

    def train_and_predict(
        self,
        x_data: np.ndarray,
        disc_steps: int,
        train_sync_steps: int,
        train_steps: int,
        pred_sync_steps: int = 0,
        pred_steps: int | None = None,
        reset_r: bool = False,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Train and then predict the evolution directly following the train data.

        Args:
            x_data: Data used for synchronization, training and prediction.
            disc_steps: Nr. of steps to discard completely before training begins.
            train_sync_steps: Nr. of steps to synchronize the reservoir with before the 'real' training begins.
            train_steps: Nr. of steps to use for training and fitting w_in pred_sync_steps: steps to sync the reservoir
                with before prediction.
            pred_sync_steps: Nr. of steps to sync the reservoir with before prediction.
            pred_steps: Nr. of steps to predict the system evolution for.
            reset_r: Reset the internal reservoir state r to the default value of all zeros before each prediction
                slice. This doesn't make all that much sense for this function here, which is why the default is False.
            **kwargs: further arguments passed to :func:`~esn.ESN.train` and :func:`~esn.ESN.predict`.

        Returns:
            2-element tuple containing:

            - **y_pred**: Predicted future states.
            - **y_test**: Data taken from the input to compare the prediction with. If the prediction were "perfect"
                y_pred and y_test would be equal. Be careful though, y_test might be shorter than y_pred, or even None,
                depending on the length of the input data and the prediction.

        """
        x_train, x_pred = utilities.train_and_predict_input_setup(
            x_data,
            disc_steps=disc_steps,
            train_sync_steps=train_sync_steps,
            train_steps=train_steps,
            pred_sync_steps=pred_sync_steps,
            pred_steps=pred_steps,
        )

        train_kwargs = utilities.remove_invalid_args(self.train, kwargs)
        predict_kwargs = utilities.remove_invalid_args(self.predict, kwargs)

        self.train(x_train, train_sync_steps, reset_r=reset_r, **train_kwargs)

        y_pred, y_test = self.predict(
            x_pred, sync_steps=pred_sync_steps, pred_steps=pred_steps, reset_r=reset_r, **predict_kwargs
        )

        return y_pred, y_test

    def create_train_and_predict(self, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        """Convinience method just passing all valid kwargs to the create_network(), train() and predict() functions.

        No direct relation to train_and_predict().

        Args:
            **kwargs: key word arguments to be filtered and then passed to the three main ESN methods create_network(),
                train() and predict().

        Returns:
            2-element tuple containing:

            - **y_pred**: Predicted future states.
            - **y_test**: Data taken from the input to compare the prediction with. If the prediction were "perfect"
                y_pred and y_test would be equal. Be careful though, y_test might be shorter than y_pred, or even None,
                depending on the length of the input data and the prediction.

        """
        create_network_kwargs = utilities.remove_invalid_args(self.create_network, kwargs)
        train_kwargs = utilities.remove_invalid_args(self.train, kwargs)
        predict_kwargs = utilities.remove_invalid_args(self.predict, kwargs)

        self.create_network(**create_network_kwargs)

        self.train(**train_kwargs)

        return self.predict(**predict_kwargs)


class ESNGenLoc:
    """Stub for the Generalized Local State Implementation of RC

    For details and ETA, please write sebastian.baur@dlr.de

    ----

    Idea behind the Local Neighborhoods matrix implementation:
    Human readable and plottable locality neighborhoods where each entry specifies the nature of a corresponding
    dimension. A simple example might look like this:
        [1, 2, 1, 0] \n
        [0, 1, 2, 1] \n
        [2, 1, 1, 2]
    where 0 means "not part of the neighborhood", 1 means "part of the neighborhood, but not a core" and 2 means "core
    dimension" Each row is a "neighborhood" specifying which dimensions are grouped together as input for each ESN
    instance. As such there can only be one 2 per input dimension (column) as otherwise combining the different
    neighborhood's predictions after each prediction step is not possible. In theory there can be no 2 in a
    dimension/column though. In that case the corresponding dimension just doesn't appear in the prediction output.

    The code is written under the assumption that all internal ESNs have the same hyperparameters. This does not
    necessarily need to be the case though, and a more generalized implementation would allow it.
    """

    def __init__(self) -> None:
        raise NotImplementedError("ESNGenLoc class is only a stub and not fully implemented yet")

    def create_network(self, **kwargs: Any) -> None:
        """STUB - Same Idea as for the ESN Class, with one network copied to all the local reservoirs."""

    def train(
        self,
        x_train: np.ndarray,
        sync_steps: int = 0,
        loc_nbhds: np.ndarray | None = None,
        train_core_only: bool = True,
        esn_train_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """STUB - Same Idea as for the ESN Class, but spread internally over multiple local reservoirs."""

    def _get_local_x(self, x: np.ndarray, nbhd_index: int) -> np.ndarray:
        """STUB - Get the local input form the global input for a given nbhd index."""

    def _get_nbhd_cut_off_zeros(self, nbhd_index: int) -> np.ndarray:
        """STUB - Remove zeros from a neighborhood."""

    def _get_nbhd(self, nbhd_index: int) -> np.ndarray:
        """STUB - Get the local nbhd for a given nbhd index."""

    def predict(
        self,
        x_pred: np.ndarray,
        sync_steps: int,
        pred_steps: int | None = None,
        esn_sync_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """STUB - Same Idea as for the ESN Class, but spread internally over multiple local reservoirs."""

    def synchronize(self, x_sync: np.ndarray, **kwargs: Any) -> None:
        """STUB - Same Idea as for the ESN Class, but spread internally over multiple local reservoirs."""

    def _predict_step(self, x: np.ndarray | None = None) -> np.ndarray:
        """STUB - Same Idea as for the ESN Class, but spread internally over multiple local reservoirs."""

    def get_last_r_states(self) -> np.ndarray:
        """STUB - Get last_r states for all internal ESNs."""

    def set_last_r_states(self, r_states: np.ndarray) -> None:
        """STUB - Set last_r states for all internal ESNs."""
