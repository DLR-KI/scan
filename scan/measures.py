"""Measures and other analysis functions useful for RC."""

from __future__ import annotations

import numpy as np

from scan import utilities


def rmse_over_time(
    pred_time_series: np.ndarray, meas_time_series: np.ndarray, normalization: str | int | float | None = None
) -> np.ndarray:
    """Calculates the NRMSE over time.

    For sclies, this averages the rmse over slices (which is of course different to how dimensions are treated)

    Args:
        pred_time_series: predicted/simulated data, shape (t, d, s) or (t, d).
        meas_time_series: observed/measured/real data, shape (t, d, s)  or (t, d).
        normalization: The normalization method to use. Possible choices are:

            - None: Calculates the pure, standard RMSE.
            - "mean": Calulates RMSE divided by the entire, flattened meas_time_series mean.
            - "std_over_time": Calulates RMSE divided by the entire meas_time_series' standard deviation in time of
              dimension. See Vlachas, Pathak et al. (2019) for details.
            - "2norm": Uses the vector  2-norm of the meas_time_series averaged over time normalize the RMSE for each
              time step.
            - "maxmin": Divides the RMSE by (max(meas) - min(meas)).
            - float: Calulates the RMSE, then divides it by the given float.

    Returns:
        RMSE for each time step, shape (t,).

    """
    pred = pred_time_series
    meas = meas_time_series

    if normalization == "mean":
        normalization = np.mean(meas)
    elif normalization == "std_over_time":
        mean_std_over_time = np.mean(np.std(meas, axis=0))
        normalization = mean_std_over_time
    elif normalization == "2norm":
        pass
    elif normalization == "maxmin":
        maxmin = np.max(meas) - np.min(meas)
        normalization = maxmin

    nrmse_over_time = np.empty(shape=(meas.shape[0]))

    for i in range(0, meas.shape[0]):
        nrmse_over_time[i] = rmse(pred[i : i + 1], meas[i : i + 1], normalization)

    return nrmse_over_time


def rmse(
    pred_time_series: np.ndarray, meas_time_series: np.ndarray, normalization: str | int | float | None = None
) -> np.ndarray:
    """Calculates the root mean squared error between two time series.

    The time series must be of equal length and dimension.

    Treats slices as just a longer time series.

    Args:
        pred_time_series: predicted/simulated data, shape (t, d, s) or (t, d).
        meas_time_series: observed/measured/real data, shape (t, d, s)  or (t, d).
        normalization: The normalization method to use. Possible choices are:

            - None: Calculates the pure, standard RMSE.
            - "mean": Calulates RMSE divided by the entire, flattened meas_time_series mean.
            - "std_over_time": Calulates RMSE divided by the entire meas_time_series' standard deviation in time of
              dimension. See Vlachas, Pathak et al. (2019) for details.
            - "2norm": Uses the vector  2-norm of the meas_time_series averaged over time normalize the RMSE for each
              time step.
            - "maxmin": Divides the RMSE by (max(meas) - min(meas)).
            - float: Calulates the RMSE, then divides it by the given float.

    Returns:
        RMSE or NRMSE.

    """
    pred = pred_time_series
    meas = meas_time_series

    # The RMSE is the same as the elementwise 2 norm divided by sqrt(T), which is the same as the frobenious norm
    # divided by sqrt(T).
    # Slices in axis 2 can be treated by just treating them as a longer dataset with more timesteps. This is actually
    # the same, again, as the frobenius norm, but now divided not by sqrt(T) but by the effective time steps of
    # sqrt(T+nr_slices)
    if meas.ndim in [1, 2]:
        time_steps = np.sqrt(meas.shape[0])
    elif meas.ndim == 3:
        time_steps = np.sqrt(meas.shape[0] * meas.shape[2])
    else:
        raise ValueError("Unsupported input dimension for RMSE calculation")

    error: np.ndarray = np.linalg.norm(pred - meas) / time_steps

    if normalization is None:
        error = error
    elif normalization == "mean":
        error = error / np.mean(meas)
    elif normalization == "std_over_time":
        error = error / np.mean(np.std(meas, axis=0))
    elif normalization == "2norm":
        error = error / np.linalg.norm(meas)
    elif normalization == "maxmin":
        error = error / (np.max(meas) - np.min(meas))
    elif utilities.is_number(normalization):
        error = error / normalization
    else:
        raise ValueError("Type of normalization not implemented")

    return error
