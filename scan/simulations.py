"""Simulate various chaotic system to generate artificial data.

Every dynamical system is represented as a class.
The general syntax for simulating the trajectory is:
trajectory = SystemClass(parameters=<default>).simulate(time_steps, starting_point=<default>)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, TypedDict

import numpy as np


def _runge_kutta(f: Callable[[np.ndarray], np.ndarray], dt: float, x: np.ndarray) -> np.ndarray:
    """Simulate one step for ODEs of the form dx/dt = f(x(t)).

    Args:
        f: function used to calculate the time derivative at point x.
        dt: time step size.
        x: d-dim position at time t.

    Returns:
       d-dim position at time t+dt.

    """
    k1: np.ndarray = dt * f(x)
    k2: np.ndarray = dt * f(x + k1 / 2)
    k3: np.ndarray = dt * f(x + k2 / 2)
    k4: np.ndarray = dt * f(x + k3)
    next_step: np.ndarray = x + 1.0 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return next_step


def _timestep_iterator(
    f: Callable[[np.ndarray], np.ndarray], time_steps: int, starting_point: np.ndarray
) -> np.ndarray:
    """Iterate a function f: x(i+1) = f(x(i)) multiple times to obtain a full trajectory.

    Args:
        f: The iterator function x(i+1) = f(x(i)).
        time_steps: The number of time_steps of the output trajectory.
                    The starting_point is included as the 0th element in the trajectory.
        starting_point: Starting point of the trajectory.

    Returns:
        trajectory: system-state at every simulated timestep.

    """
    starting_point = np.array(starting_point)
    traj_size = (time_steps, starting_point.shape[0])
    traj = np.zeros(traj_size)
    traj[0, :] = starting_point
    for t in range(1, traj_size[0]):
        traj[t] = f(traj[t - 1])
    return traj


class SimBase(ABC):
    """A base class for all the simulation classes."""

    default_starting_point: np.ndarray
    sys_dim: int

    @abstractmethod
    def iterate(self, x: np.ndarray) -> np.ndarray:
        """The abstract iterate function."""

    def simulate(self, time_steps: int, starting_point: np.ndarray | None = None) -> np.ndarray:
        """Simulate the trajectory.

        Args:
            time_steps: Number of time steps t to simulate.
            starting_point: Starting point of the trajectory shape (sys_dim,). If None, take the
                            default starting point.

        Returns:
            Trajectory of shape (t, sys_dim).

        """
        if starting_point is None:
            starting_point = self.default_starting_point
        else:
            if starting_point.size != self.sys_dim:
                raise ValueError(
                    "Provided starting_point has the wrong dimension. "
                    f"{self.sys_dim} was expected and {starting_point.size} "
                    "was given"
                )
        return _timestep_iterator(self.iterate, time_steps, starting_point)


class SimBaseRungeKutta(SimBase):
    dt: float

    @abstractmethod
    def flow(self, x: np.ndarray) -> np.ndarray:
        """The abstract flow function."""

    def iterate(self, x: np.ndarray) -> np.ndarray:
        """Calculates next timestep x(t+dt) with given x(t) using runge kutta.

        Args:
            x: the previous point x(t).

        Returns:
            : x(t+dt) corresponding to input x(t).

        """
        return _runge_kutta(self.flow, self.dt, x)


class Lorenz63(SimBaseRungeKutta):
    """Simulate the 3-dimensional autonomous flow: Lorenz-63 attractor.

    Literature values (Sprott, Julien Clinton, and Julien C. Sprott. Chaos and time-series
    analysis. Vol. 69. Oxford: Oxford university press, 2003.) for default parameters and
    starting_point:
    - Lyapunov exponents: (0.9059, 0.0, -14.5723)
    - Kaplan-Yorke dimension: 2.06215
    - Correlation dimension: 2.068 +- 0.086
    They refer to:
    - Parameters: {"sigma": 10.0, "rho": 28.0, "beta": 8 / 3}
    - Starting point: [0.0, -0.01, 9.0]
    """

    default_parameters = {"sigma": 10.0, "rho": 28.0, "beta": 8 / 3, "dt": 0.05}
    default_starting_point = np.array([0.0, -0.01, 9.0])
    sys_dim = 3

    def __init__(
        self, sigma: float | None = None, rho: float | None = None, beta: float | None = None, dt: float | None = None
    ) -> None:
        """Define the system parameters.

        Args:
            sigma: 'sigma' parameter in the Lorenz 63 equations.
            rho: 'rho' parameter in the Lorenz 63 equations.
            beta: 'beta' parameter in the Lorenz 63 equations.
            dt: Size of time steps.
        """

        self.sigma = sigma or self.default_parameters["sigma"]
        self.rho = rho or self.default_parameters["rho"]
        self.beta = beta or self.default_parameters["beta"]
        self.dt = dt or self.default_parameters["dt"]

    def flow(self, x: np.ndarray) -> np.ndarray:
        """Calculates (dx/dt, dy/dt, dz/dt) with given (x,y,z) for RK4.

        Args:
            x: (x,y,z) coordinates. Needs to have shape (3,).

        Returns:
            : (dx/dt, dy/dt, dz/dt) corresponding to input x.

        """
        return np.array([self.sigma * (x[1] - x[0]), x[0] * (self.rho - x[2]) - x[1], x[0] * x[1] - self.beta * x[2]])


class Roessler(SimBaseRungeKutta):
    """Simulate the 3-dimensional autonomous flow: Roessler attractor.

    Literature values (Sprott, Julien Clinton, and Julien C. Sprott. Chaos and time-series
    analysis. Vol. 69. Oxford: Oxford university press, 2003.) for default parameters and
    starting_point:
    - Lyapunov exponents: (0.0714, 0, -5.3943)
    - Kaplan-Yorke dimension: 2.0132
    - Correlation dimension: 1.991 +- 0.065
    They refer to:
    - Parameters: {"a": 0.2, "b": 0.2, "c": 5.7}
    - Starting point: [-9.0, 0.0, 0.0]
    """

    default_parameters = {"a": 0.2, "b": 0.2, "c": 5.7, "dt": 0.1}
    default_starting_point = np.array([-9.0, 0.0, 0.0])
    sys_dim = 3

    def __init__(
        self, a: float | None = None, b: float | None = None, c: float | None = None, dt: float | None = None
    ) -> None:
        """Define the system parameters

        Args:
            a: 'a' parameter in the Roessler equations.
            b: 'b' parameter in the Roessler equations.
            c: 'c' parameter in the Roessler equations.
            dt: Size of time steps.
        """

        self.a = a or self.default_parameters["a"]
        self.b = b or self.default_parameters["b"]
        self.c = c or self.default_parameters["c"]
        self.dt = dt or self.default_parameters["dt"]

    def flow(self, x: np.ndarray) -> np.ndarray:
        """Calculates (dx/dt, dy/dt, dz/dt) with given (x,y,z) for RK4.

        Args:
            x: (x,y,z) coordinates. Needs to have shape (3,).

        Returns:
            : (dx/dt, dy/dt, dz/dt) corresponding to input x.

        """
        return np.array([-x[1] - x[2], x[0] + self.a * x[1], self.b + x[2] * (x[0] - self.c)])


class ComplexButterly(SimBaseRungeKutta):
    """Simulate the 3-dimensional autonomous flow: Complex butterfly.

    Literature values (Sprott, Julien Clinton, and Julien C. Sprott. Chaos and time-series
    analysis. Vol. 69. Oxford: Oxford university press, 2003.) for default parameters and
    starting_point:
    - Lyapunov exponents: (0.1690, 0.0, -0.7190)
    - Kaplan-Yorke dimension: 2.2350
    - Correlation dimension: 2.491 +- 0.131
    They refer to:
    - Parameters: {"a": 0.55}
    - Starting point: [0.2, 0.0, 0.0]
    """

    default_parameters = {"a": 0.55, "dt": 0.05}
    default_starting_point = np.array([0.2, 0.0, 0.0])
    sys_dim = 3

    def __init__(self, a: float | None = None, dt: float | None = None) -> None:
        """Define the system parameters.

        Args:
            a: 'a' parameter in the Complex butterfly equations.
            dt: Size of time steps.
        """
        self.a = a or self.default_parameters["a"]
        self.dt = dt or self.default_parameters["dt"]

    def flow(self, x: np.ndarray) -> np.ndarray:
        """Calculates (dx/dt, dy/dt, dz/dt) with given (x,y,z) for RK4.

        Args:
            x: (x,y,z) coordinates. Needs to have shape (3,).

        Returns:
            : (dx/dt, dy/dt, dz/dt) corresponding to input x.

        """
        return np.array([self.a * (x[1] - x[0]), -x[2] * np.sign(x[0]), np.abs(x[0]) - 1])


class Chen(SimBaseRungeKutta):
    """Simulate the 3-dimensional autonomous flow: Chen's system.

    Literature values (Sprott, Julien Clinton, and Julien C. Sprott. Chaos and time-series
    analysis. Vol. 69. Oxford: Oxford university press, 2003.) for default parameters and
    starting_point:
    - Lyapunov exponents: (2.0272, 0, -12.0272)
    - Kaplan-Yorke dimension: 2.1686
    - Correlation dimension: 2.147 +- 0.117
    They refer to:
    - Parameters: {"a": 35.0, "b": 3.0, "c": 28.0}
    - Starting point: [-10.0, 0.0, 37.0]
    """

    default_parameters = {"a": 35.0, "b": 3.0, "c": 28.0, "dt": 0.01}
    default_starting_point = np.array([-10.0, 0.0, 37.0])
    sys_dim = 3

    def __init__(
        self, a: float | None = None, b: float | None = None, c: float | None = None, dt: float | None = None
    ) -> None:
        """Define the system parameters.

        Args:
            a: 'a' parameter in the Chen system.
            b: 'b' parameter in the Chen system.
            c: 'c' parameter in the Chen system.
            dt: Size of time steps.
        """
        self.a = a or self.default_parameters["a"]
        self.b = b or self.default_parameters["b"]
        self.c = c or self.default_parameters["c"]
        self.dt = dt or self.default_parameters["dt"]

    def flow(self, x: np.ndarray) -> np.ndarray:
        """Calculates (dx/dt, dy/dt, dz/dt) with given (x,y,z) for RK4.

        Args:
            x: (x,y,z) coordinates. Needs to have shape (3,).

        Returns:
            : (dx/dt, dy/dt, dz/dt) corresponding to input x.

        """
        return np.array(
            [
                self.a * (x[1] - x[0]),
                (self.c - self.a) * x[0] - x[0] * x[2] + self.c * x[1],
                x[0] * x[1] - self.b * x[2],
            ]
        )


class ChuaCircuit(SimBaseRungeKutta):
    """Simulate the 3-dimensional autonomous flow: Chua's circuit.

    Literature values (Sprott, Julien Clinton, and Julien C. Sprott. Chaos and time-series
    analysis. Vol. 69. Oxford: Oxford university press, 2003.) for default parameters and
    starting_point:
    - Lyapunov exponents: (0.3271, 0.0, -2.5197)
    - Kaplan-Yorke dimension: 2.1298
    - Correlation dimension: 2.215 +- 0.098
    They refer to:
    - Parameters: {"alpha": 9.0, "beta": 100 / 7, "a": 8 / 7, "b": 5 / 7}
    - Starting point: [0.0, 0.0, 0.6]
    """

    default_parameters = {"alpha": 9.0, "beta": 100 / 7, "a": 8 / 7, "b": 5 / 7, "dt": 0.05}
    default_starting_point = np.array([0.0, 0.0, 0.6])
    sys_dim = 3

    def __init__(
        self,
        alpha: float | None = None,
        beta: float | None = None,
        a: float | None = None,
        b: float | None = None,
        dt: float | None = None,
    ) -> None:
        """Define the system parameters

        Args:
            alpha: 'alpha' parameter in the Chua equations.
            beta: 'beta' parameter in the Chua equations.
            a: 'a' parameter in the Chua equations.
            b: 'b' parameter in the Chua equations.
            dt: Size of time steps.
        """
        self.alpha = alpha or self.default_parameters["alpha"]
        self.beta = beta or self.default_parameters["beta"]
        self.a = a or self.default_parameters["a"]
        self.b = b or self.default_parameters["b"]
        self.dt = dt or self.default_parameters["dt"]

    def flow(self, x: np.ndarray) -> np.ndarray:
        """Calculates (dx/dt, dy/dt, dz/dt) with given (x,y,z) for RK4.

        Args:
            x: (x,y,z) coordinates. Needs to have shape (3,).

        Returns:
            : (dx/dt, dy/dt, dz/dt) corresponding to input x.

        """
        return np.array(
            [
                self.alpha
                * (x[1] - x[0] + self.b * x[0] + 0.5 * (self.a - self.b) * (np.abs(x[0] + 1) - np.abs(x[0] - 1))),
                x[0] - x[1] + x[2],
                -self.beta * x[1],
            ]
        )


class Thomas(SimBaseRungeKutta):
    """Simulate the 3-dimensional autonomous flow: Thomas' cyclically symmetric attractor.

    Literature values (Sprott, Julien Clinton, and Julien C. Sprott. Chaos and time-series
    analysis. Vol. 69. Oxford: Oxford university press, 2003.) for default parameters and
    starting_point:
    - Lyapunov exponents: (0.0349, 0.0, -0.5749)
    - Kaplan-Yorke dimension: 2.0607
    - Correlation dimension: 1.843 +- 0.075
    They refer to:
    - Parameters: {"b": 0.18}
    - Starting point: [0.1, 0.0, 0.0]
    """

    default_parameters = {"b": 0.18, "dt": 0.2}
    default_starting_point = np.array([0.1, 0.0, 0.0])
    sys_dim = 3

    def __init__(self, b: float | None = None, dt: float | None = None) -> None:
        """Define the system parameters.

        Args:
            b: 'b' parameter of Thomas' cyclically symmetric attractor.
            dt: Size of time steps.
        """
        self.b = b or self.default_parameters["b"]
        self.dt = dt or self.default_parameters["dt"]

    def flow(self, x: np.ndarray) -> np.ndarray:
        """Calculates (dx/dt, dy/dt, dz/dt) with given (x,y,z) for RK4.

        Args:
            x: (x,y,z) coordinates. Needs to have shape (3,).

        Returns:
            : (dx/dt, dy/dt, dz/dt) corresponding to input x.

        """
        return np.array([-self.b * x[0] + np.sin(x[1]), -self.b * x[1] + np.sin(x[2]), -self.b * x[2] + np.sin(x[0])])


class WindmiAttractor(SimBaseRungeKutta):
    """Simulate the 3-dimensional autonomous flow: WINDMI attractor.

    Literature values (Sprott, Julien Clinton, and Julien C. Sprott. Chaos and time-series
    analysis. Vol. 69. Oxford: Oxford university press, 2003.) for default parameters and
    starting_point:
    - Lyapunov exponents: (0.0755, 0, -0.7755)
    - Kaplan-Yorke dimension: 2.0974
    - Correlation dimension: 2.035 +- 0.095
    They refer to:
    - Parameters: {"a": 0.7, "b": 2.5}
    - Starting point: [0.0, 0.8, 0.0]
    """

    default_parameters = {"a": 0.7, "b": 2.5, "dt": 0.1}
    default_starting_point = np.array([0.0, 0.8, 0.0])
    sys_dim = 3

    def __init__(self, a: float | None = None, b: float | None = None, dt: float | None = None) -> None:
        """Define the system parameters.

        Args:
            a: 'a' parameter in the WINDMI equations.
            b: 'b' parameter in the WINDMI equations.
            dt: Size of time steps.
        """
        self.a = a or self.default_parameters["a"]
        self.b = b or self.default_parameters["b"]
        self.dt = dt or self.default_parameters["dt"]

    def flow(self, x: np.ndarray) -> np.ndarray:
        """Calculates (dx/dt, dy/dt, dz/dt) with given (x,y,z) for RK4.

        Args:
            x: (x,y,z) coordinates. Needs to have shape (3,).

        Returns:
            : (dx/dt, dy/dt, dz/dt) corresponding to input x.

        """
        return np.array([x[1], x[2], -self.a * x[2] - x[1] + self.b - np.exp(x[0])])


class Rucklidge(SimBaseRungeKutta):
    """Simulate the 3-dimensional autonomous flow: Rucklidge attractor.

    Literature values for LLE according to (Rusyn, V. "Modeling, analysis and
    control of chaotic Rucklidge system." Journal of Telecommunication, Electronic and Computer
    Engineering (JTEC) 11.1 (2019): 43-47.).
    - Lyapunov Exponents: (0.1877, 0.0, -3.1893)
    They refer to:
    - Parameters: {"kappa": 2.0, "lam": 6.7}
    - Starting point: [1.2, 0.8, 1.4]

    Sprott Literature values are not the same for the same parameters:
    Literature values (Sprott, Julien Clinton, and Julien C. Sprott. Chaos and time-series
    analysis. Vol. 69. Oxford: Oxford university press, 2003.) for default parameters and
    starting_point:
    - Lyapunov exponents: (0.0643, 0.0, -3.0643)
    - Kaplan-Yorke dimension: 2.0210
    - Correlation dimension: 2.108 +- 0.095
    They refer to:
    - Parameters: {"kappa": 2.0, "lam": 6.7}
    - Starting point: [1.0, 0.0, 4.5]
    """

    default_parameters = {"kappa": 2.0, "lam": 6.7, "dt": 0.05}
    default_starting_point = np.array([1.0, 0.0, 4.5])
    sys_dim = 3

    def __init__(self, kappa: float | None = None, lam: float | None = None, dt: float | None = None) -> None:
        """Define the system parameters.

        Args:
            kappa: 'kappa' parameter in the Rucklidge equations.
            lam: 'lambda' parameter in the Rucklidge equations.
        """

        self.kappa = kappa or self.default_parameters["kappa"]
        self.lam = lam or self.default_parameters["lam"]
        self.dt = dt or self.default_parameters["dt"]

    def flow(self, x: np.ndarray) -> np.ndarray:
        """Calculates (dx/dt, dy/dt, dz/dt) with given (x,y,z) for RK4.

        Args:
            x: (x,y,z) coordinates. Needs to have shape (3,).

        Returns:
            : (dx/dt, dy/dt, dz/dt) corresponding to input x.

        """
        return np.array([-self.kappa * x[0] + self.lam * x[1] - x[1] * x[2], x[0], -x[2] + x[1] ** 2])


class SimplestQuadraticChaotic(SimBaseRungeKutta):
    """Simulate the 3-dimensional autonomous flow: Simplest Quadratic Chaotic flow.

    See: Sprott, Julien Clinton, and Julien C. Sprott. Chaos and time-series
    analysis. Vol. 69. Oxford: Oxford university press, 2003.

    Literature values (Sprott, Julien Clinton, and Julien C. Sprott. Chaos and time-series
    analysis. Vol. 69. Oxford: Oxford university press, 2003.) for default parameters and
    starting_point:
    - Lyapunov exponents: (0.0551, 0.0, -2.0721)
    - Kaplan-Yorke dimension: 2.0266
    - Correlation dimension: 2.187 +- 0.075
    They refer to:
    - Parameters: {"a": 2.017}
    - Starting point: [-0.9, 0.0, 0.5]
    """

    default_parameters = {"a": 2.017, "dt": 0.1}
    default_starting_point = np.array([-0.9, 0.0, 0.5])
    sys_dim = 3

    def __init__(self, a: float | None = None, dt: float | None = None) -> None:
        """Define the system parameters.

        Args:
            a: 'a' parameter in the Simplest Quadratic Chaotic flow.
            dt: Size of time steps.
        """
        self.a = a or self.default_parameters["a"]
        self.dt = dt or self.default_parameters["dt"]

    def flow(self, x: np.ndarray) -> np.ndarray:
        """Calculates (dx/dt, dy/dt, dz/dt) with given (x,y,z) for RK4.

        Args:
            x: (x,y,z) coordinates. Needs to have shape (3,).

        Returns:
            : (dx/dt, dy/dt, dz/dt) corresponding to input x.

        """
        return np.array([x[1], x[2], -self.a * x[2] + x[1] ** 2 - x[0]])


class SimplestCubicChaotic(SimBaseRungeKutta):
    """Simulate the 3-dimensional autonomous flow: Simplest Cubic Chaotic flow.

    See: Sprott, Julien Clinton, and Julien C. Sprott. Chaos and time-series
    analysis. Vol. 69. Oxford: Oxford university press, 2003.

    Literature values (Sprott, Julien Clinton, and Julien C. Sprott. Chaos and time-series
    analysis. Vol. 69. Oxford: Oxford university press, 2003.) for default parameters and
    starting_point:
    - Lyapunov exponents: (0.0837, 0.0, -2.1117)
    - Kaplan-Yorke dimension: 2.0396
    - Correlation dimension: 2.174 +- 0.083
    They refer to:
    - Parameters: {"a": 2.028}
    - Starting point: [0.0, 0.96, 0.0]
    """

    default_parameters = {"a": 2.028, "dt": 0.1}
    default_starting_point = np.array([0.0, 0.96, 0.0])
    sys_dim = 3

    def __init__(self, a: float | None = None, dt: float | None = None) -> None:
        """Define the system parameters.

        Args:
            a: 'a' parameter in the Simplest Cubic Chaotic flow.
            dt: Size of time steps.
        """
        self.a = a or self.default_parameters["a"]
        self.dt = dt or self.default_parameters["dt"]

    def flow(self, x: np.ndarray) -> np.ndarray:
        """Calculates (dx/dt, dy/dt, dz/dt) with given (x,y,z) for RK4.

        Args:
            x: (x,y,z) coordinates. Needs to have shape (3,).

        Returns:
            : (dx/dt, dy/dt, dz/dt) corresponding to input x.

        """
        return np.array([x[1], x[2], -self.a * x[2] + x[1] ** 2 * x[0] - x[0]])


class SimplestPiecewiseLinearChaotic(SimBaseRungeKutta):
    """Simulate the 3-dimensional autonomous flow: Simplest Piecewise Linear Chaotic flow.

    See: Sprott, Julien Clinton, and Julien C. Sprott. Chaos and time-series
    analysis. Vol. 69. Oxford: Oxford university press, 2003.

    Literature values (Sprott, Julien Clinton, and Julien C. Sprott. Chaos and time-series
    analysis. Vol. 69. Oxford: Oxford university press, 2003.) for default parameters and
    starting_point:
    - Lyapunov exponents: (0.0362, 0.0, -0.6362)
    - Kaplan-Yorke dimension: 2.0569
    - Correlation dimension: 2.131 +- 0.072
    They refer to:
    - Parameters: {"a": 0.6}
    - Starting point: [0.0, -0.7, 0.0]
    """

    default_parameters = {"a": 0.6, "dt": 0.1}
    default_starting_point = np.array([0.0, -0.7, 0.0])
    sys_dim = 3

    def __init__(self, a: float | None = None, dt: float | None = None) -> None:
        """Define the system parameters.

        Args:
            a: 'a' parameter in the Simplest Piecewise Linear Chaotic flow.
            dt: Size of time steps.
        """
        self.a = a or self.default_parameters["a"]
        self.dt = dt or self.default_parameters["dt"]

    def flow(self, x: np.ndarray) -> np.ndarray:
        """Calculates (dx/dt, dy/dt, dz/dt) with given (x,y,z) for RK4.

        Args:
            x: (x,y,z) coordinates. Needs to have shape (3,).

        Returns:
            : (dx/dt, dy/dt, dz/dt) corresponding to input x.

        """
        return np.array([x[1], x[2], -self.a * x[2] - x[1] + np.abs(x[0]) - 1])


class DoubleScroll(SimBaseRungeKutta):
    """Simulate the 3-dimensional autonomous flow: Double Scroll system

    Literature values (Sprott, Julien Clinton, and Julien C. Sprott. Chaos and time-series
    analysis. Vol. 69. Oxford: Oxford university press, 2003.) for default parameters and
    starting_point:
    - Lyapunov exponents: (0.0497, 0.0, -0.8497)
    - Kaplan-Yorke dimension: 2.0585
    - Correlation dimension: 2.184 +- 0.107
    They refer to:
    - Parameters: {"a": 0.8}
    - Starting point: [0.01, 0.01, 0.0]
    """

    default_parameters = {"a": 0.8, "dt": 0.1}
    default_starting_point = np.array([0.01, 0.01, 0.0])
    sys_dim = 3

    def __init__(self, a: float | None = None, dt: float | None = None) -> None:
        """Define the system parameters.

        Args:
            a: 'a' parameter in Double Scroll system.
            dt: Size of time steps.
        """
        self.a = a or self.default_parameters["a"]
        self.dt = dt or self.default_parameters["dt"]

    def flow(self, x: np.ndarray) -> np.ndarray:
        """Calculates (dx/dt, dy/dt, dz/dt) with given (x,y,z) for RK4.

        Args:
            x: (x,y,z) coordinates. Needs to have shape (3,).

        Returns:
            : (dx/dt, dy/dt, dz/dt) corresponding to input x.

        """
        return np.array([x[1], x[2], -self.a * (x[2] + x[1] + x[0] - np.sign(x[0]))])


class LotkaVolterra(SimBaseRungeKutta):
    """Simulate the 2-dimensional autonomous flow: Lotka-Volterra"""

    default_parameters = {"a": 2.0, "b": 0.4, "c": 3.0, "d": 0.6, "dt": 0.05}
    default_starting_point = np.array([1.0, 1.0])
    sys_dim = 2

    def __init__(
        self,
        a: float | None = None,
        b: float | None = None,
        c: float | None = None,
        d: float | None = None,
        dt: float | None = None,
    ) -> None:
        """Define the system parameters.

        Args:
            a: 'a' parameter in Lotka Volterra System.
            b: 'b' parameter in Lotka Volterra System.
            c: 'c' parameter in Lotka Volterra System.
            d: 'd' parameter in Lotka Volterra System.
            dt: Size of time steps.
        """
        self.a = a or self.default_parameters["a"]
        self.b = b or self.default_parameters["b"]
        self.c = c or self.default_parameters["c"]
        self.d = d or self.default_parameters["d"]
        self.dt = dt or self.default_parameters["dt"]

    def flow(self, x: np.ndarray) -> np.ndarray:
        """Calculates (dx/dt, dy/dt, dz/dt) with given (x,y,z) for RK4.

        Args:
            x: (x,y,z) coordinates. Needs to have shape (3,).

        Returns:
            : (dx/dt, dy/dt, dz/dt) corresponding to input x.

        """
        return np.array([self.a * x[0] - self.b * x[0] * x[1], -self.c * x[1] + self.d * x[0] * x[1]])


class Henon(SimBase):
    """Simulate the 2-dimensional dissipative map: Henon map.

    Literature values (Sprott, Julien Clinton, and Julien C. Sprott. Chaos and time-series
    analysis. Vol. 69. Oxford: Oxford university press, 2003.) for default parameters and
    starting_point:
    - Lyapunov exponents: (0.41922, -1.62319)
    - Kaplan-Yorke dimension: 1.25827
    - Correlation dimension: 1.220 +- 0.036

    They refer to:
    - Parameters: {"a": 1.4, "b": 0.3}
    - Starting point: [0.0, 0.9]
    """

    default_parameters = {"a": 1.4, "b": 0.3}
    default_starting_point = np.array([0.0, 0.9])
    sys_dim = 2

    def __init__(self, a: float | None = None, b: float | None = None) -> None:
        """Define the system parameters.

        Args:
            a: 'a' parameter of Henon map.
            b: 'b' parameter of Henon map.
        """
        self.a = a or self.default_parameters["a"]
        self.b = b or self.default_parameters["b"]

    def iterate(self, x: np.ndarray) -> np.ndarray:
        """Calculates next timestep (x(i+1), y(i+1)) with given (x(i),y(i)).

        Args:
            x: (x,y,z) coordinates. Needs to have shape (2,).

        Returns:
            : (x(i+1), y(i+1)) corresponding to input x.

        """
        return np.array([1 - self.a * x[0] ** 2 + self.b * x[1], x[0]])


class Logistic(SimBase):
    """Simulate the 1-dimensional noninvertable map: Logistic map.

    Literature values (Sprott, Julien Clinton, and Julien C. Sprott. Chaos and time-series
    analysis. Vol. 69. Oxford: Oxford university press, 2003.) for default parameters and
    starting_point:
    - Lyapunov exponents: ln(2) = 0.6931147..
    - Kaplan-Yorke dimension: 1.0
    - Correlation dimension: 1.0
    They refer to:
    - Parameters: {"r": 4.0}
    - Starting point: [0.1]
    """

    default_parameters = {"r": 4.0}
    default_starting_point = np.array([0.1])
    sys_dim = 1

    def __init__(self, r: float | None = None) -> None:
        """Define the system parameters.

        Args:
            r: 'r' parameter of Logistic map.
        """
        self.r = r or self.default_parameters["r"]

    def iterate(self, x: np.ndarray) -> np.ndarray:
        """Calculates next timestep (x(i+1), ) with given (x(i), ).

        Args:
            x: (x, ) coordinates. Needs to have shape (1,).

        Returns:
            : (x(i+1), ) corresponding to input x.

        """
        return np.array(
            [
                self.r * x[0] * (1 - x[0]),
            ]
        )


class SimplestDrivenChaotic(SimBaseRungeKutta):
    """Simulate the 2+1 dim (2 space, 1 time) conservative flow: Simplest Driven Chaotic flow.

    Note: The third dimension is the linear increasing time dimension. Remove that dimension before
          plotting the trajectory.

    See: Sprott, Julien Clinton, and Julien C. Sprott. Chaos and time-series
    analysis. Vol. 69. Oxford: Oxford university press, 2003.

    Literature values (Sprott, Julien Clinton, and Julien C. Sprott. Chaos and time-series
    analysis. Vol. 69. Oxford: Oxford university press, 2003.) for default parameters and
    starting_point:
    - Lyapunov exponents: (0.0971, 0, -0.0971)
    - Kaplan-Yorke dimension: 3.0
    - Correlation dimension: 2.634 +- 0.160
    They refer to:
    - Parameters: {"omega": 1.88}
    - Starting point: [0.0, 0.0, 0.0]
    """

    default_parameters = {"omega": 1.88, "dt": 0.1}
    default_starting_point = np.array([0.0, 0.0, 0.0])
    sys_dim = 3

    def __init__(self, omega: float | None = None, dt: float | None = None) -> None:
        """Define the system parameters.

        Args:
            omega: 'omega' parameter of the Simplest Driven Chaotic flow.
            dt: Size of time steps.
        """
        self.omega = omega or self.default_parameters["omega"]
        self.dt = dt or self.default_parameters["dt"]

    def flow(self, x: np.ndarray) -> np.ndarray:
        """Calculates (dx/dt, dy/dt, dt/dt) with given (x,y,t) for RK4.

        Args:
            x: (x,y,z) coordinates. Needs to have shape (3,).

        Returns:
            : (dx/dt, dy/dt, dt/dt=1) corresponding to input x.

        """
        return np.array([x[1], -(x[0] ** 3) + np.sin(self.omega * x[2]), 1])


class UedaOscillator(SimBaseRungeKutta):
    """Simulate the 2+1 dim (2 space, 1 time) driven dissipative flow: Ueda oscillator.

    Note: The third dimension is the linear increasing time dimension. Remove that dimension before
          plotting the trajectory.

    Literature values (Sprott, Julien Clinton, and Julien C. Sprott. Chaos and time-series
    analysis. Vol. 69. Oxford: Oxford university press, 2003.) for default parameters and
    starting_point:
    - Lyapunov exponents: (0.1034, 0, -0.1534)
    - Kaplan-Yorke dimension: 2.6741
    - Correlation dimension: 2.675 +- 0.132
    They refer to:
    - Parameters: {"b": 0.05, "A": 7.5, "omega": 1.0}
    - Starting point: [2.5, 0.0, 0.0]
    """

    default_parameters = {"b": 0.05, "A": 7.5, "omega": 1.0, "dt": 0.05}
    default_starting_point = np.array([2.5, 0.0, 0.0])
    sys_dim = 3

    def __init__(
        self, b: float | None = None, A: float | None = None, omega: float | None = None, dt: float | None = None
    ) -> None:
        """Define the system parameters.

        Args:
            b: 'b' parameter of Ueda Oscillator.
            A: 'A' parameter of Ueda Oscillator.
            omega: 'omega' parameter of Ueda Oscillator.
            dt: Size of time steps.
        """
        self.b = b or self.default_parameters["b"]
        self.A = A or self.default_parameters["A"]
        self.omega = omega or self.default_parameters["omega"]
        self.dt = dt or self.default_parameters["dt"]

    def flow(self, x: np.ndarray) -> np.ndarray:
        """Calculates (dx/dt, dy/dt, dt/dt) with given (x,y,t) for RK4.

        Args:
            x: (x,y,z) coordinates. Needs to have shape (3,).

        Returns:
            : (dx/dt, dy/dt, dt/dt=1) corresponding to input x.

        """
        return np.array([x[1], -(x[0] ** 3) - self.b * x[1] + self.A * np.sin(self.omega * x[2]), 1])


class KuramotoSivashinsky(SimBase):
    """Simulate the n-dimensional Kuramoto-Sivashinsky PDE.

    Note: dimension must be an even number.

    PDE: y_t = -y*y_x - (1+eps)*y_xx - y_xxxx.

    Reference for the numerical integration:
    "fourth order time stepping for stiff pde-kassam trefethen 2005" at
    https://people.maths.ox.ac.uk/trefethen/publication/PDF/2005_111.pdf

    Python implementation at: https://github.com/E-Renshaw/kuramoto-sivashinsky

    Literature values (EDSON, R., BUNDER, J., MATTNER, T., & ROBERTS, A. (2019). LYAPUNOV EXPONENTS
    OF THE KURAMOTO–SIVASHINSKY PDE. The ANZIAM Journal, 61(3),
    270-285. doi:10.1017/S1446181119000105) for Lyapunov Exponents:
    - lyapunov exponents: (0.080, 0.056, 0.014, 0.003, -0.003 ...)
    They refer to:
    - Parameters: {"sys_length": 36.0, "eps": 0.0}
    """

    class DefaultParameters(TypedDict):
        sys_dim: int
        sys_length: float
        eps: float
        dt: float

    default_parameters: DefaultParameters = {"sys_dim": 50, "sys_length": 36.0, "eps": 0.0, "dt": 0.1}

    def __init__(
        self,
        sys_dim: int | None = None,
        sys_length: float | None = None,
        eps: float | None = None,
        dt: float | None = None,
    ) -> None:
        """

        Args:
            sys_dim: The sys_dim of the KS system. Must be an even number.
            sys_length: The system length of the KS system.
            eps: A parameter in the KS system: y_t = -y*y_x - (1+eps)*y_xx - y_xxxx.
            dt: Size of time steps.
        """
        if sys_dim is not None:
            if sys_dim % 2 != 0:  # check if even number.
                raise ValueError("Parameter dimension must be an even number.")
        self.sys_dim = sys_dim or self.default_parameters["sys_dim"]
        self.sys_length = sys_length or self.default_parameters["sys_length"]
        self.eps = eps or self.default_parameters["eps"]
        self.dt = dt or self.default_parameters["dt"]
        self.set_default_starting_point()
        self._prepare()

    def _prepare(self) -> None:
        """function to calculate auxiliary variables."""
        k = (
            np.transpose(
                np.conj(
                    np.concatenate((np.arange(0, self.sys_dim / 2), np.array([0]), np.arange(-self.sys_dim / 2 + 1, 0)))
                )
            )
            * 2
            * np.pi
            / self.sys_length
        )

        L = (1 + self.eps) * k**2 - k**4

        self.E = np.exp(self.dt * L)
        self.E_2 = np.exp(self.dt * L / 2)
        M = 64
        r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
        LR = self.dt * np.transpose(np.repeat([L], M, axis=0)) + np.repeat([r], self.sys_dim, axis=0)
        self.Q = self.dt * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=1))
        self.f1 = self.dt * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3, axis=1))
        self.f2 = self.dt * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, axis=1))
        self.f3 = self.dt * np.real(np.mean((-4 - 3 * LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, axis=1))

        self.g = -0.5j * k

    def set_default_starting_point(self) -> None:
        """Get the default starting_point of the simulation.

        The starting point is from Kassam_2005 paper.
        """
        x = self.sys_length * np.transpose(np.conj(np.arange(1, self.sys_dim + 1))) / self.sys_dim
        self.default_starting_point = np.array(
            np.cos(2 * np.pi * x / self.sys_length) * (1 + np.sin(2 * np.pi * x / self.sys_length))
        )

    def iterate(self, x: np.ndarray) -> np.ndarray:
        """Calculates next timestep (x_0(i+1),x_1(i+1),..) with given (x_0(i),x_0(i),..) and dt.

        Args:
            x: (x_0(i),x_1(i),..) coordinates. Needs to have shape (self.sys_dim,).

        Returns:
            : (x_0(i+1),x_1(i+1),..) corresponding to input x.

        """

        v = np.fft.fft(x)
        Nv = self.g * np.fft.fft(np.real(np.fft.ifft(v)) ** 2)
        a = self.E_2 * v + self.Q * Nv
        Na = self.g * np.fft.fft(np.real(np.fft.ifft(a)) ** 2)
        b = self.E_2 * v + self.Q * Na
        Nb = self.g * np.fft.fft(np.real(np.fft.ifft(b)) ** 2)
        c = self.E_2 * a + self.Q * (2 * Nb - Nv)
        Nc = self.g * np.fft.fft(np.real(np.fft.ifft(c)) ** 2)
        v = self.E * v + Nv * self.f1 + 2 * (Na + Nb) * self.f2 + Nc * self.f3
        return np.real(np.fft.ifft(v))


class KuramotoSivashinskyCustom(SimBase):
    """Simulate the n-dimensional Kuramoto-Sivashinsky PDE with custom precision and fft backend.
    PDE: y_t = -y*y_x - y_xx - y_xxxx.

    Note: dimension must be an even number.

    Reference for the numerical integration:
    "fourth order time stepping for stiff pde-kassam trefethen 2005" at
    https://people.maths.ox.ac.uk/trefethen/publication/PDF/2005_111.pdf

    Python implementation at: https://github.com/E-Renshaw/kuramoto-sivashinsky

    """

    class DefaultParameters(TypedDict):
        sys_dim: int
        sys_length: float
        dt: float
        precision: None
        fft_type: None

    default_parameters: DefaultParameters = {
        "sys_dim": 50,
        "sys_length": 36.0,
        "dt": 0.1,
        "precision": None,
        "fft_type": None,
    }

    def __init__(
        self,
        sys_dim: int | None = None,
        sys_length: float | None = None,
        dt: float | None = None,
        precision: int | None = None,
        fft_type: str | None = None,
    ) -> None:
        """

        Args:
            sys_dim: The sys_dim of the KS system. Must be an even number.
            sys_length: The system length of the KS system.
            dt: Size of time steps.
            precision: The numerical precision for the simulation:
                    - None: no precision change
                    - 16, 32, 64, or 128 for the corresponding precision
            fft_type: Either "numpy" or "scipy".
        """
        if sys_dim is not None:
            if sys_dim % 2 != 0:  # check if even number.
                raise ValueError("Parameter dimension must be an even number.")

        self.sys_dim = sys_dim or self.default_parameters["sys_dim"]
        self.sys_length = sys_length or self.default_parameters["sys_length"]
        self.dt = dt or self.default_parameters["dt"]
        self.precision = precision or self.default_parameters["precision"]
        self.fft_type = fft_type or self.default_parameters["fft_type"]

        self.set_default_starting_point()
        self._set_precision_and_fft_type()
        self._prepare()

    def _set_precision_and_fft_type(self) -> None:
        if self.precision is None:
            self.change_precision = False
        elif self.precision == 128:
            # NOTE: 128 precision is actually the same as longdouble precision on most (all?)
            # 64 bit machines, that is
            # 80 bits of precision, padded with zeros to 128 bits in memory.
            self.change_precision = True
            self.f_dtype = "float128"
            self.c_dtype = "complex256"
        elif self.precision == 64:
            self.change_precision = True
            self.f_dtype = "float64"
            self.c_dtype = "complex128"
        elif self.precision == 32:
            self.change_precision = True
            self.f_dtype = "float32"
            self.c_dtype = "complex64"
        elif self.precision == 16:
            self.change_precision = True
            self.f_dtype = "float16"
            self.c_dtype = "complex32"
        else:
            raise ValueError("specified precision not recognized")

        if self.fft_type is None or self.fft_type == "numpy":
            self.custom_fft = np.fft.fft
            self.custom_ifft = np.fft.ifft
        elif self.fft_type == "scipy":
            import scipy
            import scipy.fft

            self.custom_fft = scipy.fft.fft
            self.custom_ifft = scipy.fft.ifft
        else:
            raise ValueError("fft_type not recognized")

    def _prepare(self) -> None:
        """function to calculate auxiliary variables."""
        k = (
            np.transpose(
                np.conj(
                    np.concatenate((np.arange(0, self.sys_dim / 2), np.array([0]), np.arange(-self.sys_dim / 2 + 1, 0)))
                )
            )
            * 2
            * np.pi
            / self.sys_length
        )

        if self.change_precision:
            k = k.astype(self.f_dtype)

        L = k**2 - k**4

        self.E = np.exp(self.dt * L)
        if self.change_precision:
            self.E = self.E.astype(self.f_dtype)
        self.E_2 = np.exp(self.dt * L / 2)
        if self.change_precision:
            self.E_2 = self.E_2.astype(self.f_dtype)
        M = 64
        r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
        if self.change_precision:
            r = r.astype(self.c_dtype)
        LR = self.dt * np.transpose(np.repeat([L], M, axis=0)) + np.repeat([r], self.sys_dim, axis=0)
        if self.change_precision:
            LR = LR.astype(self.c_dtype)
        self.Q = self.dt * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=1))
        if self.change_precision:
            self.Q = self.Q.astype(self.c_dtype)
        self.f1 = self.dt * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3, axis=1))
        if self.change_precision:
            self.f1 = self.f1.astype(self.c_dtype)
        self.f2 = self.dt * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, axis=1))
        if self.change_precision:
            self.f2 = self.f2.astype(self.c_dtype)
        self.f3 = self.dt * np.real(np.mean((-4 - 3 * LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, axis=1))
        if self.change_precision:
            self.f3 = self.f3.astype(self.c_dtype)

        self.g = -0.5j * k
        if self.change_precision:
            self.g = self.g.astype(self.c_dtype)

    def set_default_starting_point(self) -> None:
        """Get the default starting_point of the simulation.

        The starting point is from Kassam_2005 paper.
        """
        x = self.sys_length * np.transpose(np.conj(np.arange(1, self.sys_dim + 1))) / self.sys_dim
        self.default_starting_point = np.array(
            np.cos(2 * np.pi * x / self.sys_length) * (1 + np.sin(2 * np.pi * x / self.sys_length))
        )

    def iterate(self, x: np.ndarray) -> np.ndarray:
        """Calculates next timestep (x_0(i+1),x_1(i+1),..) with given (x_0(i),x_0(i),..) and dt.

        Args:
            x: (x_0(i),x_1(i),..) coordinates. Needs to have shape (self.sys_dim,).

        Returns:
            : (x_0(i+1),x_1(i+1),..) corresponding to input x.

        """

        if self.change_precision:
            x = x.astype(self.f_dtype)

        v = self.custom_fft(x)
        Nv = self.g * self.custom_fft(np.real(self.custom_ifft(v)) ** 2)
        if self.change_precision:
            Nv = Nv.astype(self.c_dtype)
        a = self.E_2 * v + self.Q * Nv
        if self.change_precision:
            a = a.astype(self.c_dtype)
        Na = self.g * self.custom_fft(np.real(self.custom_ifft(a)) ** 2)
        if self.change_precision:
            Na = Na.astype(self.c_dtype)
        b = self.E_2 * v + self.Q * Na
        if self.change_precision:
            b = b.astype(self.c_dtype)
        Nb = self.g * self.custom_fft(np.real(self.custom_ifft(b)) ** 2)
        if self.change_precision:
            Nb = Nb.astype(self.c_dtype)
        c = self.E_2 * a + self.Q * (2 * Nb - Nv)
        if self.change_precision:
            c = c.astype(self.c_dtype)
        Nc = self.g * self.custom_fft(np.real(self.custom_ifft(c)) ** 2)
        if self.change_precision:
            Nc = Nc.astype(self.c_dtype)
        v = self.E * v + Nv * self.f1 + 2 * (Na + Nb) * self.f2 + Nc * self.f3
        if self.change_precision:
            v = v.astype(self.c_dtype)
        return np.real(self.custom_ifft(v))


class Lorenz96(SimBaseRungeKutta):
    """Simulate the n-dimensional dynamical system: Lorenz 96 model."""

    class DefaultParameters(TypedDict):
        sys_dim: int
        force: float
        dt: float

    default_parameters: DefaultParameters = {"sys_dim": 30, "force": 8.0, "dt": 0.05}

    def __init__(self, sys_dim: int | None = None, force: float | None = None, dt: float | None = None) -> None:
        """Define the system parameters.

        Args:
            sys_dim: number of sys_dim in Lorenz96 equations.
            force: 'force' parameter in the Lorenz 96 equations.
            dt: Size of time steps.
        """
        self.sys_dim = sys_dim or int(self.default_parameters["sys_dim"])
        self.force = force or self.default_parameters["force"]
        self.dt = dt or self.default_parameters["dt"]

        self.default_starting_point = np.sin(np.arange(self.sys_dim))

    def flow(self, x: np.ndarray) -> np.ndarray:
        """Calculates (dx_0/dt, dx_1/dt, ..) with given (x_0,x_1,..) for RK4.

        Args:
            x: (x_0,x_1,..) coordinates. Adapts automatically to shape (sys_dim, ).

        Returns:
            : (dx_0/dt, dx_1/dt, ..) corresponding to input x.

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

        derivative = derivative + self.force
        return derivative


class LinearSystem(SimBaseRungeKutta):
    """Simulate a generic n-dimensional linear dynamical system x_t = A*x"""

    class DefaultParameters(TypedDict):
        A: np.ndarray
        dt: float

    default_parameters: DefaultParameters = {"A": np.array([[-0.0, -1.0], [1.0, 0.0]]), "dt": 0.1}

    def __init__(self, A: np.ndarray | None = None, dt: float | None = None) -> None:
        """Define the system parameters.

        Args:
            A: The Matrix describing the linear system: x_t = A*x
            dt: Size of time steps.
        """

        self.A = A or np.array(self.default_parameters["A"])
        self.dt = dt or float(self.default_parameters["dt"])

        self.sys_dim = self.A.shape[0]
        self.default_starting_point = np.ones(self.sys_dim)

    def flow(self, x: np.ndarray) -> np.ndarray:
        """Calculates (dx_0/dt, dx_1/dt, ..) with given (x_0,x_1,..) for RK4.

        Args:
            x: (x_0,x_1,..) coordinates. Adapts automatically to shape (sys_dim, ).

        Returns:
            : (dx_0/dt, dx_1/dt, ..) corresponding to input x.

        """

        return np.array(np.dot(self.A, x))
