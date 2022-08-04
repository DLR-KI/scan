"""Methods to manipulate data, mainly used during pre- and postprocessing."""

from __future__ import annotations

import numpy as np


def embedding(x_data: np.ndarray, embedding_dim: int, embedding_delay: int) -> np.ndarray:
    """Embed the x_data using a time series embedding.

    WARNING: Adds np.NANs at the end!
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
         [np.NAN, np.NAN, 3.0, 7.0]]

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

    x_data_embedded = np.empty(shape=(time_steps, x_dims_after_embedding, slices)) * np.NAN

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
    x_data: np.array,
    kernel_length: int = 3,
    append: bool = False,
    kernel_type: str = "mean",
    number_iterations: int = 1,
) -> np.array:
    """This smoothing preserves the shape of original data array.

    The way the boundaries are handeled has shown to have so far the
    best results, i.e. it prevents shifts of the resulting data curve or
    prevents value overshoots at the ends.

    WARNING: Both ends of the new smoothed data array will probably not
    be of good quality. So it may be convenient to cut them off.

    Args:
        x_data (np.array): Data to smooth with shape (t,) or (t, d)
        kernel_length (int): Number of adjacent values which are taken
            into account for calculating so to speak "a smoothed value".
            Defaults to 3.
        append (bool, optional): Appends smoothed data at the end of
            the origninal data, such that the shape is (t, d+d).
            Defaults to False.
        kernel_type (str, optional): Defaults to 'mean'.
            - 'mean' means mean value of the value and its neigbours
              within the range of attr. kernel_length
        number_iterations (int, optional): How often the data should be
            smoothed with the choosen type. Defaults to 1.

    Returns:
        Numpy array with the same shape as the x_data input array, but
        with values smoothed separately in each dimension.
    Raises:
        ValueError: _description_
    """
    if not kernel_type in {"mean"}:
        raise ValueError(f"argument kernel_type {kernel_type} is not known")

    # if (t,) data shape convert in (t,d) shape
    data_is_time_shape = False
    if x_data.ndim == 1:
        data_is_time_shape = True
        x_data = x_data.reshape((x_data.shape[0], 1))
    # if data is not (t,d,s) shape, produce it for common handling
    if x_data.ndim != 3:
        x_data = x_data.reshape(x_data.shape + (1,))

    # initialize
    smoothed_data = x_data.copy()
    if append:
        append_array = np.empty(shape=(x_data.shape[0], 2 * x_data.shape[1], x_data.shape[2]))

    for num_slice, slice in enumerate(
        x_data[
            :,
            :,
        ]
    ):
        # initialize
        smoothed = slice.copy()  # x_data.copy()
        for _ in range(number_iterations):
            if kernel_type == "mean":
                left_kernel_length = kernel_length // 2
                right_kernel_length = kernel_length - left_kernel_length
                for dim in range(x_data.shape[1]):
                    # loop for each dim of data separately
                    for pos_kernel in range(left_kernel_length):
                        # for the inital boundary problem the values
                        # will be smoothed separately
                        total = 0
                        for pos_in_kernel in range(kernel_length):
                            total += smoothed[pos_in_kernel, dim]
                        smoothed[pos_kernel, dim] = total / kernel_length
                    for pos_kernel in range(left_kernel_length, smoothed.shape[0] - right_kernel_length):
                        # ordinary smoothing with backwards values
                        total = 0
                        for pos_in_kernel in range(left_kernel_length):
                            # the -1 extra value is used in the
                            # first kernel length because if the
                            # kernel_length is an odd number, the
                            # second kernel length will be +1 bigger
                            # than the first kernel length
                            total += smoothed[pos_kernel - 1 - pos_in_kernel, dim]
                        for pos_in_kernel in range(right_kernel_length):
                            total += smoothed[pos_kernel + pos_in_kernel, dim]
                        smoothed[pos_kernel, dim] = total / kernel_length
                    for pos_kernel in range(smoothed.shape[0] - right_kernel_length, smoothed.shape[0]):
                        # for the end boundary problem the values will
                        # be smoothed separately
                        total = 0
                        for pos_in_kernel in range(kernel_length):
                            total += smoothed[-1 - pos_in_kernel, dim]
                        smoothed[pos_kernel, dim] = total / kernel_length
        if append:
            # Ignores the fact that slices
            arr = np.empty(shape=(x_data.shape[0], 2 * x_data.shape[1], x_data.shape[2]))
            arr[:, : x_data.shape[1]] = x_data
            arr[:, x_data.shape[1] :] = smoothed
            if data_is_time_shape:
                raise NotImplementedError
            return arr
        else:
            # ignores the fact that slices
            if data_is_time_shape:
                smoothed = smoothed.flatten()
            return smoothed


########################################################################
"""
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    time = np.arange(0,10, 0.1)
    time = time.reshape((time.shape[0], 1))
    x_data = np.sin(time)
    x_data_noisy = x_data + np.random.random(size=x_data.shape) - 0.5
    new_data = smooth(x_data_noisy, 5, number_iterations=4)
    
    assert (new_data - x_data) <= x_data.max()
    
    plt.plot(time, x_data, label="x_data")
    plt.plot(time, x_data_noisy, label='x_data_noisy')
    plt.plot(time, new_data, label='new_data')
    plt.plot(time, new_data - x_data, label='difference')
    plt.legend()
    plt.show()
"""
