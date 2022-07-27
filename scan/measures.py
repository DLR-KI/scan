"""Measures and other analysis functions useful for RC."""

from __future__ import annotations

from typing import Callable
import numpy as np
import scipy.spatial

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


def largest_lyapunov_exponent(
    iterator_func: Callable[[np.ndarray], np.ndarray],
    starting_point: np.ndarray,
    deviation_scale: float = 1e-10,
    N: int = int(1e4),
    part_time_steps: int = 20,
    N_skip: int = 50,
    dt: float = 1.0,
    initial_pert_direction: np.ndarray | None = None,
    return_convergence: bool = False,
) -> float | np.ndarray:
    """Numerically calculate the largest lyapunov exponent given an iterator function.

    See: Sprott, Julien Clinton, and Julien C. Sprott. Chaos and time-series analysis. Vol. 69.
    Oxford: Oxford university press, 2003.

    Args:
        iterator_func: Function to iterate the system to the next time step: x(i+1) = F(x(i))
        starting_point: The starting_point of the main trajectory.
        deviation_scale: The L2-norm of the initial perturbation.
        N: Number of renormalization steps.
        part_time_steps: Time steps between renormalization steps.
        N_skip: Number of renormalization steps to perform, before tracking the log divergence.
                Avoid transients by using N_skip.
        dt: Size of time step.
        initial_pert_direction:
            - If np.ndarray: The direction of the initial perturbation.
            - If None: The direction of the initial perturbation is assumed to be np.ones(..).
        return_convergence: If True, return the convergence of the largest LE; a numpy array of
                            the shape (N, ).

    Returns:
        The largest Lyapunov Exponent. If return_convergence is True: The convergence (np.ndarray),
        else just the float value, which is the last value in the convergence.
    """

    x_dim = starting_point.size

    if initial_pert_direction is None:
        initial_pert_direction = np.ones(x_dim)

    initial_perturbation = initial_pert_direction * (deviation_scale / np.linalg.norm(initial_pert_direction))

    log_divergence = np.zeros(N)

    x = starting_point
    x_pert = starting_point + initial_perturbation

    for i_n in range(N + N_skip):
        for i_t in range(part_time_steps):
            x = iterator_func(x)
            x_pert = iterator_func(x_pert)
        dx = x_pert - x
        norm_dx = np.linalg.norm(dx)
        x_pert = x + dx * (deviation_scale / norm_dx)
        if i_n >= N_skip:
            log_divergence[i_n - N_skip] = np.log(norm_dx / deviation_scale)

    if return_convergence:
        return np.cumsum(log_divergence) / (np.arange(1, N + 1) * dt * part_time_steps)
    else:
        return np.average(log_divergence) / (dt * part_time_steps)


def largest_lyapunov_exponent_old(
    iterator_func: Callable[[np.ndarray], np.ndarray],
    starting_point: np.ndarray,
    deviation_scale: float = 1e-10,
    N: int = int(1e5),
    part_time_steps: int = 10,
    dt: float = 1.0,
    initial_perturbation: np.ndarray | None = None,  # TODO: maybe initial_pert_direction
    return_convergence: bool = False,
) -> float | np.ndarray:
    """Numerically calculate the largest lyapunov exponent given an iterator function.
    # TODO: local seed and random perturbation.

    See: Sprott, Julien Clinton, and Julien C. Sprott. Chaos and time-series analysis. Vol. 69.
    Oxford: Oxford university press, 2003.

    Args:
        iterator_func: Function to iterate the system to the next time step: x(i+1) = F(x(i))
        starting_point: The starting_point of the main trajectory.
        deviation_scale: The L2-norm of the initial perturbation.
        N: Number of renormalization steps.
        part_time_steps: Time steps between renormalization steps.
        dt: Size of time step.
        initial_perturbation:
            - If np.ndarray: The direction of the initial perturbation.
            - If None: The direction of the initial perturbation is assumed to be np.ones(..).
        return_convergence: If True, return the convergence of the largest LE; a numpy array of
                            the shape (N, ).

    Returns:
        The largest Lyapunov Exponent. If return_convergence is True: The convergence (np.ndarray),
        else just the float value.
    """

    x_dim = starting_point.size

    if initial_perturbation is None:
        initial_perturbation = np.ones(x_dim)
        # TODO: add numpy random here

    initial_perturbation *= deviation_scale / np.linalg.norm(initial_perturbation)

    log_divergence = np.zeros(N)

    x = starting_point
    x_pert = starting_point + initial_perturbation

    for i_n in range(N):
        for i_t in range(part_time_steps):
            x = iterator_func(x)
            x_pert = iterator_func(x_pert)
        dx = x_pert - x
        norm_dx = np.linalg.norm(dx)
        log_divergence[i_n] = np.log(norm_dx / deviation_scale)
        x_pert = x + dx * (deviation_scale / norm_dx)

    if return_convergence:
        return np.cumsum(log_divergence) / (np.arange(1, N + 1) * dt * part_time_steps)
    else:
        return np.average(log_divergence) / (dt * part_time_steps)


def largest_lyapunov_exponent_distance(
    iterator_func: Callable[[np.ndarray], np.ndarray],
    starting_point: np.ndarray,
    deviation_scale: float = 1e-10,
    N: int = int(1e5),
    part_time_steps: int = 10,
    dt: float = 1.0,
    initial_perturbation: np.ndarray | None = None,
) -> float:
    """Numerically calculate the largest lyapunov exponent given an iterator function


    Args:
        iterator_func:
        starting_point:
        deviation_scale:
        N:
        part_time_steps:
        dt:
        initial_perturbation:

    Returns:

    """

    x_dim = starting_point.size

    if initial_perturbation is None:
        initial_perturbation = np.ones(x_dim)

    initial_perturbation *= deviation_scale / np.linalg.norm(initial_perturbation)

    log_divergence = np.zeros(N)
    distance = np.zeros((N, part_time_steps))

    x = starting_point
    x_pert = starting_point + initial_perturbation

    for i_n in range(N):
        for i_t in range(part_time_steps):
            x = iterator_func(x)
            x_pert = iterator_func(x_pert)

            distance[i_n, i_t] = np.linalg.norm(x - x_pert)

        dx = x_pert - x
        norm_dx = np.linalg.norm(dx)
        log_divergence[i_n] = np.log(norm_dx / deviation_scale)
        x_pert = x + dx * (deviation_scale / norm_dx)

    return distance, np.average(log_divergence) / (dt * part_time_steps)
    # return np.cumsum(log_divergence)/np.arange(1, N+1)
    # return np.average(log_divergence)/(dt*part_time_steps)


def largest_lyapunov_exponent_skip(
    iterator_func: Callable[[np.ndarray], np.ndarray],
    starting_point: np.ndarray,
    deviation_scale: float = 1e-10,
    N: int = int(1e5),
    part_time_steps: int = 10,
    dt: float = 1.0,
    initial_pert_direction: np.ndarray | None = None,
    seed: int | None = None,
    N_skip: int = 0,
    return_convergence: bool = False,
) -> float | np.ndarray:
    """Numerically calculate the largest lyapunov exponent for a given iterator function.

    See: Sprott, Julien Clinton, and Julien C. Sprott. Chaos and time-series analysis. Vol. 69.
    Oxford: Oxford university press, 2003.

    Args:
        iterator_func: Function to iterate the system to the next time step: x(i+1) = F(x(i))
        starting_point: The starting_point of the main trajectory.
        deviation_scale: The L2-norm of the initial perturbation.
        N: Number of renormalization steps.
        part_time_steps: Time steps between renormalization steps.
        dt: Size of time step.
        initial_pert_direction:
            - If np.ndarray: The direction of the initial perturbation.
            - If None: The direction of the initial perturbation is assumed to be np.ones(..).
        seed: The seed for the random directions. If None, use the global seed.
        N_skip: Number of normalization steps to perform, before tracking the log divergence.
                Avoid transients by using N_skip.
          return_convergence: If True, return the convergence of the largest LE; a numpy array of
                the shape (N, ).


    Returns:
                The largest Lyapunov Exponent. If return_convergence is True: The convergence (np.ndarray),
        else just the float value.
    """

    x_dim = starting_point.size

    if initial_pert_direction is None:
        if seed is None:
            rng = np.random.default_rng(seed)

        initial_pert_direction = np.ones(x_dim)
        initial_pert_direction = np.ones(x_dim)

    initial_perturbation *= deviation_scale / np.linalg.norm(initial_pert_direction)

    log_divergence = np.zeros(N)

    x = starting_point
    x_pert = starting_point + initial_perturbation

    for i_n in range(N + N_skip):
        for i_t in range(part_time_steps):
            x = iterator_func(x)
            x_pert = iterator_func(x_pert)
        dx = x_pert - x
        norm_dx = np.linalg.norm(dx)
        x_pert = x + dx * (deviation_scale / norm_dx)
        if i_n >= N_skip:
            log_divergence[i_n - N_skip] = np.log(norm_dx / deviation_scale)
    return np.cumsum(log_divergence) / (np.arange(1, N + 1) * dt * part_time_steps)
    # return np.average(log_divergence)/(dt*part_time_steps)


def largest_lyapunov_from_data(
    time_series: np.ndarray,
    time_steps: int = 100,
    dt: float = 1.0,
    neighbours_to_check: int = 50,
    min_index_difference: int = 50,
    distance_upper_bound: float = np.inf,
) -> float | np.ndarray:
    nr_points = time_series.shape[0]
    tree = scipy.spatial.cKDTree(time_series)

    # get the neighbour_index for each index
    neighbours = []
    for i in range(nr_points):
        x = time_series[i, :]

        # get the k nearest neighbours (indices)
        potential_neighbours = tree.query(x, k=neighbours_to_check, distance_upper_bound=distance_upper_bound)[1]

        # skip point if no neighbour was found with the given distance_upper_bound
        if potential_neighbours[1] == nr_points:
            continue

        # only keep the closest neighbour that is at least min_index_difference apart
        i_neighbour = potential_neighbours[np.argmax(np.abs(potential_neighbours - i) >= min_index_difference)]

        # if there is no suitable neighbour skip this point:
        if i == i_neighbour:
            continue

        # check if valid neighbour:
        if i + time_steps < nr_points and i_neighbour + time_steps < nr_points:
            neighbours.append((i, i_neighbour))

    # calculate for each point the distance to the neighbour for the next t time_steps
    nr_valid_points = len(neighbours)
    distance = np.zeros((nr_valid_points, time_steps))
    for i, (i_base, i_neigh) in enumerate(neighbours):
        diff = time_series[i_base : i_base + time_steps, :] - time_series[i_neigh : i_neigh + time_steps, :]
        distance[i, :] = np.linalg.norm(diff, axis=-1)

    # debug:
    initial_distance = distance[:, 0]

    # normalize distance array by first distance for each neighbour_pair
    distance = (distance.T / distance[:, 0]).T

    # calculate the log distance
    log_distance = np.log(distance)

    return log_distance / dt, initial_distance

    # return log_distance
