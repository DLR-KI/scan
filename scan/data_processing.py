"""Methods to manipulate data, mainly used during pre- and postprocessing."""

from __future__ import annotations

import numpy as np


def embedding(x_data: np.ndarray, embedding_dim: int, embedding_delay: int) -> np.ndarray:
    """Embed the x_data using a time series embedding.

    WARNING: Adds np.nans at the end!
    WARNING: Also shifts the xdim=0 into the future, so that all other xdim columns only contain information further in
    the past. This avoids accidentally prediction the xdim=0 column with data from the future, the ML algorithm
    shouldn't have access to. Due to this, the training data embedding is best done with the original data before any
    other preprocessing is applied, to avoid accidentally starting your calculations at a different time step than you
    intended.

    Args:
        x_data: Data to embed of shape (t,), (t, d) or (t, d, s).
        embedding_dim: Embedding dimension. Basically just returns the original time series for embedding_dim=1 and
            raises a ValueError for embedding_dim=0.
        embedding_delay: Delay used for the time series embedding. Just copies the original data without shift to be
            higher dimensional for embedding_delay=0. For embedding_delay>0, adds the embedding delay to each embedding
            dimension.

    Returns:
        Embedded time series of shape (t, embedding_dim), (t, d * embedding_dim) or (t, d * embedding_dim, s)
            respectively.

    Turns an input of shape (4, 2), e.g.:
        [[0.0,    4.0],
         [1.0,    5.0],
         [2.0,    6.0],
         [3.0,    7.0]]
    with embedding_dim=2 and embedding_delay=1 into an output of shape (4, 4):
        [[1.0,    5.0,    0.0, 4.0],
         [2.0,    6.0,    1.0, 5.0],
         [3.0,    7.0,    2.0, 6.0],
         [np.nan, np.nan, 3.0, 7.0]]

    """
    if embedding_dim == 0:
        raise ValueError

    x_data_input_original_ndim = x_data.ndim
    if x_data.ndim == 1:
        x_data = x_data[:, np.newaxis, np.newaxis]
    elif x_data.ndim == 2:
        x_data = x_data[:, :, np.newaxis]

    time_steps, x_dim, slices = x_data.shape

    x_dims_after_embedding = x_dim * embedding_dim

    x_data_embedded = np.empty(shape=(time_steps, x_dims_after_embedding, slices)) * np.nan

    for i in range(embedding_dim):
        shift = embedding_delay * i
        x_data_embedded[: -shift or None, -(i + 1) * x_dim : -i * x_dim or None, :] = x_data[shift:, :, :]

    if x_data_input_original_ndim == 1:
        x_data_embedded.shape = (x_data_embedded.shape[0], embedding_dim)
    elif x_data_input_original_ndim == 2:
        x_data_embedded.shape = (x_data_embedded.shape[0], x_data_embedded.shape[1])

    return x_data_embedded


def downsampling(x_data: np.ndarray, downsampling_size: int) -> np.ndarray:
    """Downsample the input data by averaging nearby data points in time.

    Args:
        x_data: Data to downsample of shape (t,), (t, d) or (t, d, s).
        downsampling_size: Nearby average. This must be a clean divider of the x_data time_steps t.

    Returns:
        Downsampled data length of shape (t/downsampling_size,), (t/downsampling_size, d) or
            (t/downsampling_size, d, s) respectively.

    Turns an input of shape (4, 2), e.g.:
        [[0.0, 4.0],
         [1.0, 5.0],
         [2.0, 6.0],
         [3.0, 7.0]]
    with downsampling_size=2 into an output of shape (2, 2):
        [[0.5, 4.4],
         [2.5, 6.5]]

    """
    x_data_input_original_ndim = x_data.ndim
    if x_data.ndim == 1:
        x_data = x_data[:, np.newaxis, np.newaxis]
    elif x_data.ndim == 2:
        x_data = x_data[:, :, np.newaxis]

    time_steps, x_dim, slices = x_data.shape

    x_data_downsampled = x_data
    x_data_downsampled = x_data_downsampled.reshape((time_steps * x_dim * slices), order="F")
    x_data_downsampled = x_data_downsampled.reshape((-1, downsampling_size))
    x_data_downsampled = np.mean(x_data_downsampled, 1)
    x_data_downsampled = x_data_downsampled.reshape((time_steps // downsampling_size, x_dim, slices), order="F")

    if x_data_input_original_ndim == 1:
        x_data_downsampled.shape = (x_data_downsampled.shape[0],)
    elif x_data_input_original_ndim == 2:
        x_data_downsampled.shape = (x_data_downsampled.shape[0], x_data_downsampled.shape[1])

    return x_data_downsampled


def smooth(
    x_data: np.ndarray,
    kernel_length: int = 3,
    kernel_type: str = "mean",
    number_iterations: int = 5,
) -> np.ndarray:
    """Smooth the input data via convolution, preserving data array shape.

    Boundaries are handled to
        - prevent shifts of the resulting curve.
        - minimize value overshooting at the boundaries.
        - not add any NANs, zeros or similar numerical artifacts.

    For highly noisy data increasing the number of iterations instead of the
    kernel length, seems to result in superior results in most cases, due to
    preserving distinct non-noisy features of the signal better.

    WARNING: Even though this method aims to reduce the impact of boundary
    effects, both ends of the new smoothed data array will nevertheless
    be probably of suboptimal quality. So it may be convenient to cut
    them off afterwards.

    Example:
        smooth_data = scan.data_processing.smooth(x_data)

    Args:
        x_data: Data to smooth with shape (t,), (t, d) or
        (t, d, s).
        kernel_length: Number of adjacent values which are taken
            into account for calculating so to speak "a smoothed value".
        kernel_type: The type of convultion kernel to be used.
            Possible types are:

            - 'mean' means mean value of the value and its neigbours in time
              within the range of the kernel_length parameter, i.e. a boxed
              shaped kernel in the language of convolutions.
        number_iterations: Number of times the smoothing is applied.
            A high number of iterations ensures that the underlying signal
            and its frequencies are preserved from getting smoothed out since
            noise usualy as no fixed timescale but a signal has.

    Returns:
        Numpy array with the same shape as the x_data input array, but
        with values smoothed separately in each dimension.
    Raises:
        ValueError: The user must choose from the currently implemented
                    types of kernels.
    """

    # ensure array is (t,d,s) shaped for the algorithm
    x_data_shape = x_data.shape
    smoothed = x_data.copy().astype(float)
    if smoothed.ndim == 1:
        smoothed = smoothed[:, None, None]
    elif smoothed.ndim == 2:
        smoothed = smoothed[:, :, None]

    if kernel_type == "mean":
        lkl = kernel_length // 2  # left_kernel_length
        rkl = kernel_length - lkl  # right_kernel_length
    else:
        raise ValueError(f"argument kernel_type {kernel_type} is not known")
    for _ in range(number_iterations):
        for slice in range(smoothed.shape[2]):
            # smooth for each slice of data separately
            for dim in range(smoothed.shape[1]):
                # smooth for each dimension of data separately
                # smoothed_old is needed to avoid using already
                # smoothed values within the neighborhood of others
                smoothed_old = smoothed[:, dim, slice].copy()
                for pos_kernel in range(lkl):
                    # for the initial boundary problem the values
                    # will be smoothed separately
                    total = 0
                    for pos_in_kernel in range(kernel_length):
                        total += smoothed_old[pos_in_kernel]
                    smoothed[pos_kernel, dim, slice] = total / kernel_length
                for pos_kernel in range(lkl, smoothed.shape[0] - rkl):
                    # smoothing with values around current position
                    total = 0
                    for pos_in_kernel in range(lkl):
                        # the -1 extra value is used in the
                        # first kernel length because if the
                        # kernel_length is an odd number, the
                        # second kernel length will be +1 bigger
                        # than the first kernel length
                        total += smoothed_old[pos_kernel - 1 - pos_in_kernel]
                    for pos_in_kernel in range(rkl):
                        total += smoothed_old[pos_kernel + pos_in_kernel]
                    smoothed[pos_kernel, dim, slice] = total / kernel_length
                for pos_kernel in range(smoothed.shape[0] - rkl, smoothed.shape[0]):
                    # for the end boundary problem the values will
                    # be smoothed separately
                    total = 0
                    for pos_in_kernel in range(kernel_length):
                        total += smoothed_old[-1 - pos_in_kernel]
                    smoothed[pos_kernel, dim, slice] = total / kernel_length
    smoothed = smoothed.reshape(x_data_shape)
    return smoothed
