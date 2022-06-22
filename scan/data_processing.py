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
