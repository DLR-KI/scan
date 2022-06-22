""" Simulate various chaotic system to generate artificial data """

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from scan import utilities
from scan.utilities import FlagType


def _lorenz_63(x: np.ndarray, sigma: float = 10, rho: float = 28, beta: float = 8 / 3) -> np.ndarray:
    """Calculates (dx/dt, dy/dt, dz/dt) with given (x,y,z) for RK4

    Args:
        x: (x,y,z) coordinates. Needs to have shape (3,)
        sigma: 'sigma' parameter in the Lorenz 63 equations
        rho: 'rho' parameter in the Lorenz 63 equations
        beta: 'beta' parameter in the Lorenz 63 equations

    Returns:
       : (dx/dt, dy/dt, dz/dt) corresponding to input x

    """
    return np.array([sigma * (x[1] - x[0]), x[0] * (rho - x[2]) - x[1], x[0] * x[1] - beta * x[2]])


def _lorenz_96(x: np.ndarray, force: float = 8) -> np.ndarray:
    """Calculates (dx/dt, dy/dt, dz/dt) with given (x,y,z) for RK4

    Args:
        x: d-dim coordinates
        force: force parameter in the Lorenz96 equations

    Returns:
       d-dim time derivative at x

    """
    system_dimension = x.shape[0]
    derivative = np.zeros(system_dimension)

    # Periodic Boundary Conditions for the 3 edge cases i=1,2,system_dimension
    derivative[0] = (x[1] - x[system_dimension - 2]) * x[system_dimension - 1] - x[0]
    derivative[1] = (x[2] - x[system_dimension - 1]) * x[0] - x[1]
    derivative[system_dimension - 1] = (x[0] - x[system_dimension - 3]) * x[system_dimension - 2] - x[
        system_dimension - 1
    ]

    # TODO: Rewrite using numpy vectorization to make faster
    for i in range(2, system_dimension - 1):
        derivative[i] = (x[i + 1] - x[i - 2]) * x[i - 1] - x[i]

    derivative = derivative + force

    return derivative


def _runge_kutta(
    f: Callable[[np.ndarray], np.ndarray], dt: float, y: np.ndarray = np.array([2.2, -3.5, 4.3])
) -> np.ndarray:
    """Simulate one step for ODEs of the form dy/dt = f(t,y), returns y(t + dt)

    Args:
        f: function used to calculate the time derivate at point y
        dt: time step size
        y: d-dim position at time t

    Returns:
       d-dim position at time t+dt

    """
    k1: np.ndarray = dt * f(y)
    k2: np.ndarray = dt * f(y + k1 / 2)
    k3: np.ndarray = dt * f(y + k2 / 2)
    k4: np.ndarray = dt * f(y + k3)
    next_step: np.ndarray = y + 1.0 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return next_step


_sys_flag_synonyms = utilities.SynonymDict(
    {
        2: ["lorenz_63", "normal_lorenz", "lorenz"],
        4: "lorenz_96",
        13: "kuramoto_sivashinsky",
        15: "kuramoto_sivashinsky_custom",
    }
)


def simulate_trajectory(
    sys_flag: FlagType = "lorenz",
    dt: float = 2e-2,
    time_steps: int = int(2e4),
    starting_point: np.ndarray | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """Simulate a complex system trajectory

    Args:
        sys_flag: The system to be simulated. Possible flags, their synonyms and corresponding possible kwargs are:

            - "lorenz_63", "normal_lorenz", "lorenz": The normal, unmodified Lorenz-63 system. Possible kwargs:
                - sigma (float): 'sigma' parameter in the Lorenz 63 equations
                - rho (float): 'rho' parameter in the Lorenz 63 equations
                - beta (float): 'beta' parameter in the Lorenz 63 equations
            - "lorenz_96": The d-dimensional Lorenz-96 System. Possible kwargs:
                - force (float): force parameter in the Lorenz96 equations
            - "kuramoto_sivashinsky". The d-dimensional Lorenz-96 System. Possible kwargs:
                - dimensions: Nr. of dimensions d of the system grid
                - system_size: physical size of the system
                - eps: If non-zero, vary the parameter infront of the y_xx term: (1+eps)*y_xx

        dt: Size of time steps
        time_steps: Number of time steps t to simulate
        starting_point: Starting point of the trajectory of shape (d,)
        **kwargs: Further Arguments passed to the simulating function. See above for a list of possible arguments.

    Returns:
        Trajectory of shape (t, d)

    """
    sys_flag_syn = _sys_flag_synonyms.find_flag(sys_flag)

    if sys_flag_syn == 2:
        if starting_point is None:
            starting_point = np.array([1, 2, 3])
        f = lambda x: _lorenz_63(x, **kwargs)
    elif sys_flag_syn == 4:
        # Starting point is ignored here atm
        if starting_point is None:
            starting_point = np.array([1, 2, 3])
        f = lambda x: _lorenz_96(x, **kwargs)
    elif sys_flag_syn == 13:
        return _kuramoto_sivashinsky(dt=dt, time_steps=time_steps - 1, starting_point=starting_point, **kwargs)
    elif sys_flag_syn == 15:
        return _kuramoto_sivashinsky_custom(dt=dt, time_steps=time_steps - 1, starting_point=starting_point, **kwargs)
    else:
        raise ValueError(f"sys_flag {sys_flag} unknown!")

    traj_size = (time_steps, starting_point.shape[0])
    traj = np.zeros(traj_size)
    y = starting_point

    for t in range(traj_size[0]):
        traj[t] = y
        y = _runge_kutta(f, dt, y=y)
    return traj


def _kuramoto_sivashinsky(
    dimensions: int,
    system_size: int,
    dt: float,
    time_steps: int,
    starting_point: np.ndarray | None,
    eps: float = 0,
) -> np.ndarray:
    """This function simulates the Kuramoto–Sivashinsky PDE

    Even though it doesn't use the RK4 algorithm, it is bundled with the other
    simulation functions in simulate_trajectory() for consistency.

    Reference for the numerical integration:
    "fourth order time stepping for stiff pde-kassam trefethen 2005" at
    https://people.maths.ox.ac.uk/trefethen/publication/PDF/2005_111.pdf

    Python implementation at: https://github.com/E-Renshaw/kuramoto-sivashinsky

    Args:
        dimensions: nr. of dimensions of the system grid
        system_size: physical size of the system
        dt: time step size
        time_steps: nr. of time steps to simulate
        starting_point: starting point for the simulation of shape
            (dimensions, )
        eps: If non-zero, vary the parameter infront of the y_xx term: (1+eps)*y_xx

    Returns:
       Simulated trajectory of shape (time_steps, dimensions)

    """
    # Rename variables to the names used in the paper
    n = dimensions  # No. of grid points in real space (and hence dimensionality of the output)
    size = system_size  # system size
    h = dt  # time step
    nmax = time_steps  # No. of time steps to simulate

    # Define initial conditions and Fourier Transform them
    if starting_point is None:
        # Use the starting point from the Kassam_2005 paper
        x = size * np.transpose(np.conj(np.arange(1, n + 1))) / n
        u = np.cos(2 * np.pi * x / size) * (1 + np.sin(2 * np.pi * x / size))
    else:
        u = starting_point
    v = np.fft.fft(u)

    # Wave numbers
    k = (
        np.transpose(np.conj(np.concatenate((np.arange(0, n / 2), np.array([0]), np.arange(-n / 2 + 1, 0)))))
        * 2
        * np.pi
        / size
    )

    L = (1 + eps) * k**2 - k**4
    E = np.exp(h * L)
    E_2 = np.exp(h * L / 2)
    M = 64  # TODO: Check if the M even makes any sense
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = h * np.transpose(np.repeat([L], M, axis=0)) + np.repeat([r], n, axis=0)
    Q = h * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=1))
    f1 = h * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3, axis=1))
    f2 = h * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, axis=1))
    f3 = h * np.real(np.mean((-4 - 3 * LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, axis=1))

    # List of Real space solutions, later converted to a np.array. Bad because growing memory and all that
    uu = [np.array(u)]

    g = -0.5j * k

    # See paper for details
    for n in range(1, nmax + 1):
        Nv = g * np.fft.fft(np.real(np.fft.ifft(v)) ** 2)
        a = E_2 * v + Q * Nv
        Na = g * np.fft.fft(np.real(np.fft.ifft(a)) ** 2)
        b = E_2 * v + Q * Na
        Nb = g * np.fft.fft(np.real(np.fft.ifft(b)) ** 2)
        c = E_2 * a + Q * (2 * Nb - Nv)
        Nc = g * np.fft.fft(np.real(np.fft.ifft(c)) ** 2)

        v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3

        u = np.real(np.fft.ifft(v))
        uu.append(np.array(u))

    return np.array(uu)


def _kuramoto_sivashinsky_custom(
    dimensions: int,
    system_size: int,
    dt: float,
    time_steps: int,
    starting_point: np.ndarray | None,
    precision: int | None = None,
    fft_type: str | None = None,
) -> np.ndarray:
    """This function simulates the Kuramoto–Sivashinsky PDE with custom precision and fft backend"""

    if precision is None:
        change_precision = False
    elif precision == 128:
        # NOTE: 128 precision is actually the same as longdouble precision on most (all?) 64 bit machines, that is
        #  80 bits of precision, padded with zeros to 128 bits in memory.
        change_precision = True
        f_dtype = "float128"
        c_dtype = "complex256"
    elif precision == 64:
        change_precision = True
        f_dtype = "float64"
        c_dtype = "complex128"
    elif precision == 32:
        change_precision = True
        f_dtype = "float32"
        c_dtype = "complex64"
    elif precision == 16:
        change_precision = True
        f_dtype = "float16"
        c_dtype = "complex32"
    else:
        raise ValueError("specified precision not recognized")

    if fft_type is None or fft_type == "numpy":
        custom_fft = np.fft.fft
        custom_ifft = np.fft.ifft
    elif fft_type == "scipy":
        import scipy
        import scipy.fft

        custom_fft = scipy.fft.fft
        custom_ifft = scipy.fft.ifft
    else:
        raise ValueError("fft_type not recognized")

    # Rename variables to the names used in the paper
    n = dimensions  # No. of grid points in real space (and hence dimensionality of the output)
    size = system_size  # system size
    h = dt  # time step
    nmax = time_steps  # No. of time steps to simulate

    # Define initial conditions and Fourier Transform them
    if starting_point is None:
        # Use the starting point from the Kassam_2005 paper
        x = size * np.transpose(np.conj(np.arange(1, n + 1))) / n
        u = np.cos(2 * np.pi * x / size) * (1 + np.sin(2 * np.pi * x / size))
    else:
        u = starting_point
    if change_precision:
        u = u.astype(f_dtype)
    v = custom_fft(u)

    # Wave numbers
    k = (
        np.transpose(np.conj(np.concatenate((np.arange(0, n / 2), np.array([0]), np.arange(-n / 2 + 1, 0)))))
        * 2
        * np.pi
        / size
    )
    if change_precision:
        k = k.astype(f_dtype)

    L = k**2 - k**4
    E = np.exp(h * L)
    if change_precision:
        E = E.astype(f_dtype)
    E_2 = np.exp(h * L / 2)
    if change_precision:
        E_2 = E_2.astype(f_dtype)
    M = 64  # TODO: Check if the M even makes any sense
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    if change_precision:
        r = r.astype(c_dtype)
    LR = h * np.transpose(np.repeat([L], M, axis=0)) + np.repeat([r], n, axis=0)
    if change_precision:
        LR = LR.astype(c_dtype)
    Q = h * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=1))
    if change_precision:
        Q = Q.astype(c_dtype)
    f1 = h * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3, axis=1))
    if change_precision:
        f1 = f1.astype(c_dtype)
    f2 = h * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, axis=1))
    if change_precision:
        f2 = f2.astype(c_dtype)
    f3 = h * np.real(np.mean((-4 - 3 * LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, axis=1))
    if change_precision:
        f3 = f3.astype(c_dtype)

    # List of Real space solutions, later converted to a np.array. Bad because growing memory and all that
    uu = [np.array(u)]

    g = -0.5j * k
    if change_precision:
        g = g.astype(c_dtype)

    # See paper for details
    for n in range(1, nmax + 1):
        Nv = g * custom_fft(np.real(custom_ifft(v)) ** 2)
        if change_precision:
            Nv = Nv.astype(c_dtype)
        a = E_2 * v + Q * Nv
        if change_precision:
            a = a.astype(c_dtype)
        Na = g * custom_fft(np.real(custom_ifft(a)) ** 2)
        if change_precision:
            Na = Na.astype(c_dtype)
        b = E_2 * v + Q * Na
        if change_precision:
            b = b.astype(c_dtype)
        Nb = g * custom_fft(np.real(custom_ifft(b)) ** 2)
        if change_precision:
            Nb = Nb.astype(c_dtype)
        c = E_2 * a + Q * (2 * Nb - Nv)
        if change_precision:
            c = c.astype(c_dtype)
        Nc = g * custom_fft(np.real(custom_ifft(c)) ** 2)
        if change_precision:
            Nc = Nc.astype(c_dtype)

        v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
        if change_precision:
            v = v.astype(c_dtype)
        u = np.real(custom_ifft(v))
        uu.append(np.array(u))

    return np.array(uu)
