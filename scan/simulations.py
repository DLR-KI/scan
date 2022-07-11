"""Simulate various chaotic system to generate artificial data.

Every dynamical system is represented as a class.
The syntax for simulating the trajectory is:
trajectory = SystemClass(parameters).simulate(time_steps, starting_point)
"""

from __future__ import annotations

from typing import Callable

import numpy as np

def _runge_kutta(
    f: Callable[[np.ndarray], np.ndarray], dt: float, x: np.ndarray) -> np.ndarray:
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


def _timestep_iterator(f: Callable[[np.ndarray], np.ndarray],
                       time_steps: int,
                       starting_point: np.ndarray) -> np.ndarray:
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
        traj[t] = f(traj[t-1])
    return traj


class Lorenz63:
    """Simulate the 3-dimensional autonomous flow: Lorenz-63 attractor.

    Literature values (Sprott, Julien Clinton, and Julien C. Sprott. Chaos and time-series
    analysis. Vol. 69. Oxford: Oxford university press, 2003.) for default parameters and
    starting_point:
    - Lyapunov exponents: (0.9059, 0.0, -14.5723)
    - Kaplan-Yorke dimension: 2.06215
    - Correlation dimension: 2.068 +- 0.086
    """
    def __init__(self, sigma: float = 10, rho: float = 28, beta: float = 8 / 3,
                 dt: float = 0.05) -> None:
        """Define the system parameters.

        Args:
            sigma: 'sigma' parameter in the Lorenz 63 equations.
            rho: 'rho' parameter in the Lorenz 63 equations.
            beta: 'beta' parameter in the Lorenz 63 equations.
            dt: Size of time steps.
        """
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.dt = dt

    def flow(self, x: np.ndarray) -> np.ndarray:
        """Calculates (dx/dt, dy/dt, dz/dt) with given (x,y,z) for RK4.

        Args:
            x: (x,y,z) coordinates. Needs to have shape (3,).
        Returns:
            : (dx/dt, dy/dt, dz/dt) corresponding to input x.

        """
        return np.array([self.sigma * (x[1] - x[0]),
                         x[0] * (self.rho - x[2]) - x[1],
                         x[0] * x[1] - self.beta * x[2]])

    def iterate(self, x: np.ndarray) -> np.ndarray:
        """Calculates next timestep (x(i+1), y(i+1), z(i+1)) with given (x(i),y(i),z(i)) and dt
        with RK4.

        Args:
            x: (x,y,z) coordinates. Needs to have shape (3,).
        Returns:
            : (x(i+1), y(i+1), z(i+1)) corresponding to input x.

        """
        return _runge_kutta(self.flow, self.dt, x)

    def simulate(self, time_steps: int,
                 starting_point: np.ndarray = np.array([0.0, -0.01, 9.0])) -> np.ndarray:
        """Simulate Lorenz63 trajectory.

        Args:
            time_steps: Number of time steps t to simulate.
            starting_point: Starting point of the trajectory of shape (3,).
        Returns:
            Trajectory of shape (t, 3).

        """
        return  _timestep_iterator(self.iterate, time_steps, starting_point)


class Roessler:
    """Simulate the 3-dimensional autonomous flow: Roessler attractor.

    Literature values (Sprott, Julien Clinton, and Julien C. Sprott. Chaos and time-series
    analysis. Vol. 69. Oxford: Oxford university press, 2003.) for default parameters and
    starting_point:
    - Lyapunov exponents: (0.0714, 0, -5.3943)
    - Kaplan-Yorke dimension: 2.0132
    - Correlation dimension: 1.991 +- 0.065
    """
    def __init__(self, a: float = 0.2, b: float = 0.2, c: float = 5.7, dt: float = 0.1) -> None:
        """Define the system parameters

        Args:
            a: 'a' parameter in the Roessler equations.
            b: 'b' parameter in the Roessler equations.
            c: 'c' parameter in the Roessler equations.
            dt: Size of time steps.
        """
        self.a = a
        self.b = b
        self.c = c
        self.dt = dt

    def flow(self, x: np.ndarray) -> np.ndarray:
        """Calculates (dx/dt, dy/dt, dz/dt) with given (x,y,z) for RK4.

        Args:
            x: (x,y,z) coordinates. Needs to have shape (3,).
        Returns:
            : (dx/dt, dy/dt, dz/dt) corresponding to input x.

        """
        return np.array([-x[1] - x[2], x[0] + self.a * x[1], self.b + x[2] * (x[0] - self.c)])

    def iterate(self, x: np.ndarray) -> np.ndarray:
        """Calculates next timestep (x(i+1), y(i+1), z(i+1)) with given (x(i),y(i),z(i)) and dt
        with RK4.

        Args:
            x: (x,y,z) coordinates. Needs to have shape (3,).
        Returns:
            : (x(i+1), y(i+1), z(i+1)) corresponding to input x.

        """
        return _runge_kutta(self.flow, self.dt, x)

    def simulate(self, time_steps: int,
                 starting_point: np.ndarray = np.array([-9.0, 0.0, 0.0])) -> np.ndarray:
        """Simulate Roessler trajectory.

        Args:
            time_steps: Number of time steps t to simulate.
            starting_point: Starting point of the trajectory of shape (3,).
        Returns:
            Trajectory of shape (t, 3).

        """
        return  _timestep_iterator(self.iterate, time_steps, starting_point)


class ComplexButterly:
    """Simulate the 3-dimensional autonomous flow: Complex butterfly.

    Literature values (Sprott, Julien Clinton, and Julien C. Sprott. Chaos and time-series
    analysis. Vol. 69. Oxford: Oxford university press, 2003.) for default parameters and
    starting_point:
    - Lyapunov exponents: (0.1690, 0.0, -0.7190)
    - Kaplan-Yorke dimension: 2.2350
    - Correlation dimension: 2.491 +- 0.131
    """
    def __init__(self, a: float = 0.55, dt: float = 0.05) -> None:
        """Define the system parameters.

        Args:
            a: 'a' parameter in the Complex butterfly equations.
            dt: Size of time steps.
        """
        self.a = a
        self.dt = dt

    def flow(self, x: np.ndarray) -> np.ndarray:
        """Calculates (dx/dt, dy/dt, dz/dt) with given (x,y,z) for RK4.

        Args:
            x: (x,y,z) coordinates. Needs to have shape (3,).
        Returns:
            : (dx/dt, dy/dt, dz/dt) corresponding to input x.

        """
        return np.array([self.a * (x[1] - x[0]), -x[2] * np.sign(x[0]), np.abs(x[0]) - 1])

    def iterate(self, x: np.ndarray) -> np.ndarray:
        """Calculates next timestep (x(i+1), y(i+1), z(i+1)) with given (x(i),y(i),z(i)) and dt
        with RK4.

        Args:
            x: (x,y,z) coordinates. Needs to have shape (3,).
        Returns:
            : (x(i+1), y(i+1), z(i+1)) corresponding to input x.

        """
        return _runge_kutta(self.flow, self.dt, x)

    def simulate(self, time_steps: int,
                 starting_point: np.ndarray = np.array([0.2, 0.0, 0.0])) -> np.ndarray:
        """Simulate Complex butterfly trajectory.

        Args:
            time_steps: Number of time steps t to simulate.
            starting_point: Starting point of the trajectory of shape (3,).
        Returns:
            Trajectory of shape (t, 3).

        """
        return  _timestep_iterator(self.iterate, time_steps, starting_point)


class Chen:
    """Simulate the 3-dimensional autonomous flow: Chen's system.

    Literature values (Sprott, Julien Clinton, and Julien C. Sprott. Chaos and time-series
    analysis. Vol. 69. Oxford: Oxford university press, 2003.) for default parameters and
    starting_point:
    - Lyapunov exponents: (2.0272, 0, -12.0272)
    - Kaplan-Yorke dimension: 2.1686
    - Correlation dimension: 2.147 +- 0.117
    """
    def __init__(self, a: float = 35, b: float = 3, c: float = 28, dt: float = 0.01) -> None:
        """Define the system parameters.

        Args:
            a: 'a' parameter in the Chen system.
            b: 'b' parameter in the Chen system.
            c: 'c' parameter in the Chen system.
            dt: Size of time steps.
        """
        self.a = a
        self.b = b
        self.c = c
        self.dt = dt

    def flow(self, x: np.ndarray) -> np.ndarray:
        """Calculates (dx/dt, dy/dt, dz/dt) with given (x,y,z) for RK4.

        Args:
            x: (x,y,z) coordinates. Needs to have shape (3,).
        Returns:
            : (dx/dt, dy/dt, dz/dt) corresponding to input x.

        """
        return np.array([self.a * (x[1] - x[0]),
                         (self.c - self.a) * x[0] - x[0] * x[2] + self.c * x[1],
                         x[0] * x[1] - self.b * x[2]])

    def iterate(self, x: np.ndarray) -> np.ndarray:
        """Calculates next timestep (x(i+1), y(i+1), z(i+1)) with given (x(i),y(i),z(i)) and dt
        with RK4.

        Args:
            x: (x,y,z) coordinates. Needs to have shape (3,).
        Returns:
            : (x(i+1), y(i+1), z(i+1)) corresponding to input x.

        """
        return _runge_kutta(self.flow, self.dt, x)

    def simulate(self, time_steps: int,
                 starting_point: np.ndarray = np.array([-10.0, 0.0, 37.0])) -> np.ndarray:
        """Simulate Chen's system trajectory.

        Args:
            time_steps: Number of time steps t to simulate.
            starting_point: Starting point of the trajectory of shape (3,).
        Returns:
            Trajectory of shape (t, 3).

        """
        return  _timestep_iterator(self.iterate, time_steps, starting_point)


class ChuaCircuit:
    """Simulate the 3-dimensional autonomous flow: Chua's circuit.

    Literature values (Sprott, Julien Clinton, and Julien C. Sprott. Chaos and time-series
    analysis. Vol. 69. Oxford: Oxford university press, 2003.) for default parameters and
    starting_point:
    - Lyapunov exponents: (0.3271, 0.0, -2.5197)
    - Kaplan-Yorke dimension: 2.1298
    - Correlation dimension: 2.215 +- 0.098
    """
    def __init__(self, alpha: float = 9, beta: float = 100 / 7, a: float = 8 / 7, b: float = 5 / 7,
                 dt: float = 0.05) -> None:
        """Define the system parameters

        Args:
            alpha: 'alpha' parameter in the Chua equations.
            beta: 'beta' parameter in the Chua equations.
            a: 'a' parameter in the Chua equations.
            b: 'b' parameter in the Chua equations.
            dt: Size of time steps.
        """
        self.alpha = alpha
        self.beta = beta
        self.a = a
        self.b = b
        self.dt = dt

    def flow(self, x: np.ndarray) -> np.ndarray:
        """Calculates (dx/dt, dy/dt, dz/dt) with given (x,y,z) for RK4.

        Args:
            x: (x,y,z) coordinates. Needs to have shape (3,).
        Returns:
            : (dx/dt, dy/dt, dz/dt) corresponding to input x.

        """
        return np.array([self.alpha * (x[1] - x[0] + self.b * x[0] + 0.5 * (self.a - self.b) *
                                       (np.abs(x[0] + 1) - np.abs(x[0] - 1))),
                         x[0] - x[1] + x[2],
                         -self.beta * x[1]])

    def iterate(self, x: np.ndarray) -> np.ndarray:
        """Calculates next timestep (x(i+1), y(i+1), z(i+1)) with given (x(i),y(i),z(i)) and dt
        with RK4.

        Args:
            x: (x,y,z) coordinates. Needs to have shape (3,).
        Returns:
            : (x(i+1), y(i+1), z(i+1)) corresponding to input x.

        """
        return _runge_kutta(self.flow, self.dt, x)

    def simulate(self, time_steps: int,
                 starting_point: np.ndarray = np.array([0.0, 0.0, 0.6])) -> np.ndarray:
        """Simulate Chua's Circuit trajectory.

        Args:
            time_steps: Number of time steps t to simulate.
            starting_point: Starting point of the trajectory of shape (3,).
        Returns:
            Trajectory of shape (t, 3).

        """
        return  _timestep_iterator(self.iterate, time_steps, starting_point)


class Thomas:
    """Simulate the 3-dimensional autonomous flow: Thomas' cyclically symmetric attractor.

    Literature values (Sprott, Julien Clinton, and Julien C. Sprott. Chaos and time-series
    analysis. Vol. 69. Oxford: Oxford university press, 2003.) for default parameters and
    starting_point:
    - Lyapunov exponents: (0.0349, 0.0, -0.5749)
    - Kaplan-Yorke dimension: 2.0607
    - Correlation dimension: 1.843 +- 0.075
    """
    def __init__(self, b: float = 0.18,
                 dt: float = 0.2) -> None:
        """Define the system parameters.

        Args:
            b: 'b' parameter of Thomas' cyclically symmetric attractor.
            dt: Size of time steps.
        """
        self.b = b
        self.dt = dt

    def flow(self, x: np.ndarray) -> np.ndarray:
        """Calculates (dx/dt, dy/dt, dz/dt) with given (x,y,z) for RK4.

        Args:
            x: (x,y,z) coordinates. Needs to have shape (3,).
        Returns:
            : (dx/dt, dy/dt, dz/dt) corresponding to input x.

        """
        return np.array([-self.b * x[0] + np.sin(x[1]), -self.b * x[1] + np.sin(x[2]), -self.b * x[2] + np.sin(x[0])])

    def iterate(self, x: np.ndarray) -> np.ndarray:
        """Calculates next timestep (x(i+1), y(i+1), z(i+1)) with given (x(i),y(i),z(i)) and dt
        with RK4.

        Args:
            x: (x,y,z) coordinates. Needs to have shape (3,).
        Returns:
            : (x(i+1), y(i+1), z(i+1)) corresponding to input x.

        """
        return _runge_kutta(self.flow, self.dt, x)

    def simulate(self, time_steps: int,
                 starting_point: np.ndarray = np.array([0.1, 0.0, 0.0])) -> np.ndarray:
        """Simulate Thomas' cyclically symmetric attractor trajectory.

        Args:
            time_steps: Number of time steps t to simulate.
            starting_point: Starting point of the trajectory of shape (3,).
        Returns:
            Trajectory of shape (t, 3).

        """
        return  _timestep_iterator(self.iterate, time_steps, starting_point)


class WindmiAttractor:
    """Simulate the 3-dimensional autonomous flow: WINDMI attractor.

    Literature values (Sprott, Julien Clinton, and Julien C. Sprott. Chaos and time-series
    analysis. Vol. 69. Oxford: Oxford university press, 2003.) for default parameters and
    starting_point:
    - Lyapunov exponents: (0.0755, 0, -0.7755)
    - Kaplan-Yorke dimension: 2.0974
    - Correlation dimension: 2.035 +- 0.095
    """

    def __init__(self, a: float = 0.7, b: float = 2.5, dt: float = 0.1) -> None:
        """Define the system parameters.

        Args:
            a: 'a' parameter in the WINDMI equations.
            b: 'b' parameter in the WINDMI equations.
            dt: Size of time steps.
        """
        self.a = a
        self.b = b
        self.dt = dt

    def flow(self, x: np.ndarray) -> np.ndarray:
        """Calculates (dx/dt, dy/dt, dz/dt) with given (x,y,z) for RK4.

        Args:
            x: (x,y,z) coordinates. Needs to have shape (3,).
        Returns:
            : (dx/dt, dy/dt, dz/dt) corresponding to input x.

        """
        return np.array([x[1], x[2], -self.a*x[2] - x[1] + self.b - np.exp(x[0])])

    def iterate(self, x: np.ndarray) -> np.ndarray:
        """Calculates next timestep (x(i+1), y(i+1), z(i+1)) with given (x(i),y(i),z(i)) and dt
        with RK4.

        Args:
            x: (x,y,z) coordinates. Needs to have shape (3,).
        Returns:
            : (x(i+1), y(i+1), z(i+1)) corresponding to input x.

        """
        return _runge_kutta(self.flow, self.dt, x)

    def simulate(self, time_steps: int,
                 starting_point: np.ndarray = np.array([0.0, 0.8, 0.0])) -> np.ndarray:
        """Simulate the WINDMI attractor trajectory.

        Args:
            time_steps: Number of time steps t to simulate.
            starting_point: Starting point of the trajectory of shape (3,).
        Returns:
            Trajectory of shape (t, 3).

        """
        return _timestep_iterator(self.iterate, time_steps, starting_point)


class Rucklidge:
    """Simulate the 3-dimensional autonomous flow: Rucklidge attractor.

    Literature values (Sprott, Julien Clinton, and Julien C. Sprott. Chaos and time-series
    analysis. Vol. 69. Oxford: Oxford university press, 2003.) for default parameters and
    starting_point:
    - Lyapunov exponents: (0.0643, 0.0, -3.0643)
    - Kaplan-Yorke dimension: 2.0210
    - Correlation dimension: 2.108 +- 0.095
    """

    def __init__(self, kappa: float = 2, lam: float = 6.7, dt: float = 0.05) -> None:
        """Define the system parameters.

        Args:
            kappa: 'kappa' parameter in the Rucklidge equations.
            lam: 'lambda' parameter in the Rucklidge equations.
        """
        self.kappa = kappa
        self.lam = lam
        self.dt = dt

    def flow(self, x: np.ndarray) -> np.ndarray:
        """Calculates (dx/dt, dy/dt, dz/dt) with given (x,y,z) for RK4.

        Args:
            x: (x,y,z) coordinates. Needs to have shape (3,).
        Returns:
            : (dx/dt, dy/dt, dz/dt) corresponding to input x.

        """
        return np.array([-self.kappa * x[0] + self.lam * x[1] - x[1] * x[2],
                         x[0], -x[2] + x[1] ** 2])

    def iterate(self, x: np.ndarray) -> np.ndarray:
        """Calculates next timestep (x(i+1), y(i+1), z(i+1)) with given (x(i),y(i),z(i)) and dt
        with RK4.

        Args:
            x: (x,y,z) coordinates. Needs to have shape (3,).
        Returns:
            : (x(i+1), y(i+1), z(i+1)) corresponding to input x.

        """
        return _runge_kutta(self.flow, self.dt, x)

    def simulate(self, time_steps: int,
                 starting_point: np.ndarray = np.array([1.0, 0.0, 4.5])) -> np.ndarray:
        """Simulate the Rucklidge attractor trajectory.

        Args:
            time_steps: Number of time steps t to simulate.
            starting_point: Starting point of the trajectory of shape (3,).
        Returns:
            Trajectory of shape (t, 3).

        """
        return _timestep_iterator(self.iterate, time_steps, starting_point)


class Henon:
    """Simulate the 2-dimensional dissipative map: Henon map.

    Literature values (Sprott, Julien Clinton, and Julien C. Sprott. Chaos and time-series
    analysis. Vol. 69. Oxford: Oxford university press, 2003.) for default parameters and
    starting_point:
    - Lyapunov exponents: (0.41922, -1.62319)
    - Kaplan-Yorke dimension: 1.25827
    - Correlation dimension: 1.220 +- 0.036
    """
    def __init__(self, a: float = 1.4, b: float = 0.3) -> None:
        """Define the system parameters.

        Args:
            a: 'a' parameter of Henon map.
            b: 'b' parameter of Henon map.
        """
        self.a = a
        self.b = b

    def iterate(self, x: np.ndarray) -> np.ndarray:
        """Calculates next timestep (x(i+1), y(i+1)) with given (x(i),y(i)).

        Args:
            x: (x,y,z) coordinates. Needs to have shape (2,).
        Returns:
            : (x(i+1), y(i+1)) corresponding to input x.

        """
        return np.array([1 - self.a*x[0]**2 + self.b * x[1], x[0]])

    def simulate(self, time_steps: int,
                 starting_point: np.ndarray = np.array([0.0, 0.9])) -> np.ndarray:
        """Simulate Henon trajectory.

        Args:
            time_steps: Number of time steps t to simulate.
            starting_point: Starting point of the trajectory of shape (2,).
        Returns:
            Trajectory of shape (t, 2).

        """
        return _timestep_iterator(self.iterate, time_steps, starting_point)


class Logistic:
    """Simulate the 1-dimensional noninvertable map: Logistic map.

    Literature values (Sprott, Julien Clinton, and Julien C. Sprott. Chaos and time-series
    analysis. Vol. 69. Oxford: Oxford university press, 2003.) for default parameters and
    starting_point:
    - Lyapunov exponents: ln(2) = 0.6931147..
    - Kaplan-Yorke dimension: 1.0
    - Correlation dimension: 1.0
    """
    def __init__(self, r: float = 4) -> None:
        """Define the system parameters.

        Args:
            r: 'r' parameter of Logistic map.
        """
        self.r = r

    def iterate(self, x: np.ndarray) -> np.ndarray:
        """Calculates next timestep (x(i+1), ) with given (x(i), ).

        Args:
            x: (x, ) coordinates. Needs to have shape (1,).
        Returns:
            : (x(i+1), ) corresponding to input x.

        """
        return np.array([self.r * x[0] * (1 - x[0]), ])

    def simulate(self, time_steps: int,
                 starting_point: np.ndarray = np.array([0.1, ])) -> np.ndarray:
        """Simulate Lorenz63 trajectory.

        Args:
            time_steps: Number of time steps t to simulate.
            starting_point: Starting point of the trajectory of shape (1,).
        Returns:
            Trajectory of shape (t, 1).

        """
        return _timestep_iterator(self.iterate, time_steps, starting_point)


class SimplestDrivenChaotic:
    """Simulate the 2+1 dim (2 space, 1 time) conservative flow: Simplest Driven Chaotic flow.

    Literature values (Sprott, Julien Clinton, and Julien C. Sprott. Chaos and time-series
    analysis. Vol. 69. Oxford: Oxford university press, 2003.) for default parameters and
    starting_point:
    - Lyapunov exponents: (0.0971, 0, -0.0971)
    - Kaplan-Yorke dimension: 3.0
    - Correlation dimension: 2.634 +- 0.160
    """
    def __init__(self, omega: float = 1.88, dt: float = 0.1) -> None:
        """Define the system parameters.

        Args:
            omega: 'omega' parameter of Simplest Driven Chaotic flow.
            dt: Size of time steps.
        """
        self.omega = omega
        self.dt = dt

    def flow(self, x: np.ndarray) -> np.ndarray:
        """Calculates (dx/dt, dy/dt, dt/dt) with given (x,y,t) for RK4.

        Args:
            x: (x,y,z) coordinates. Needs to have shape (3,).
        Returns:
            : (dx/dt, dy/dt, dt/dt=1) corresponding to input x.

        """
        return np.array([x[1], -(x[0] ** 3) + np.sin(self.omega * x[2]), 1])

    def iterate(self, x: np.ndarray) -> np.ndarray:
        """Calculates next timestep (x(i+1), y(i+1), t(i+1)) with given (x(i),y(i),t(i)) and dt.
        with RK4. The third coordinate is time: t(i+1) = t(i) + dt.

        Args:
            x: (x,y,t) coordinates. Needs to have shape (3,).
        Returns:
            : (x(i+1), y(i+1), t(i+1)) corresponding to input x.

        """
        return _runge_kutta(self.flow, self.dt, x)

    def simulate(self, time_steps: int,
                 starting_point: np.ndarray = np.array([0.0, 0.0])) -> np.ndarray:
        """Simulate Simplest Driven Chaotic flow trajectory.

         Args:
             time_steps: Number of time steps t to simulate.
             starting_point: Starting point of the trajectory of shape (2,) (time dimension
             excluded).
         Returns:
             Trajectory of shape (t, 2).

         """
        starting_point = np.hstack((starting_point, 0.0))
        return _timestep_iterator(self.iterate, time_steps, starting_point)[:, :-1]


class UedaOscillator:
    """Simulate the 2+1 dim (2 space, 1 time) driven dissipative flow: Ueda oscillator.

    Literature values (Sprott, Julien Clinton, and Julien C. Sprott. Chaos and time-series
    analysis. Vol. 69. Oxford: Oxford university press, 2003.) for default parameters and
    starting_point:
    - Lyapunov exponents: (0.1034, 0, -0.1534)
    - Kaplan-Yorke dimension: 2.6741
    - Correlation dimension: 2.675 +- 0.132
    """
    def __init__(self, b: float = 0.05, A: float = 7.5, omega: float = 1, dt: float = 0.05) -> None:
        """Define the system parameters.

        Args:
            b: 'b' parameter of Ueda Oscillator.
            A: 'A' parameter of Ueda Oscillator.
            omega: 'omega' parameter of Ueda Oscillator.
            dt: Size of time steps.
        """
        self.b = b
        self.A = A
        self.omega = omega
        self.dt = dt

    def flow(self, x: np.ndarray) -> np.ndarray:
        """Calculates (dx/dt, dy/dt, dt/dt) with given (x,y,t) for RK4.

        Args:
            x: (x,y,z) coordinates. Needs to have shape (3,).
        Returns:
            : (dx/dt, dy/dt, dt/dt=1) corresponding to input x.

        """
        return np.array([x[1], -(x[0]**3) - self.b * x[1] + self.A * np.sin(self.omega * x[2]), 1])

    def iterate(self, x: np.ndarray) -> np.ndarray:
        """Calculates next timestep (x(i+1), y(i+1), t(i+1)) with given (x(i),y(i),t(i)) and dt
        with RK4. The third coordinate is time: t(i+1) = t(i) + dt.

        Args:
            x: (x,y,t) coordinates. Needs to have shape (3,).
        Returns:
            : (x(i+1), y(i+1), t(i+1)) corresponding to input x.

        """
        return _runge_kutta(self.flow, self.dt, x)

    def simulate(self, time_steps: int,
                 starting_point: np.ndarray = np.array([2.5, 0.0])) -> np.ndarray:
        """Simulate Ueda oscillator trajectory.

         Args:
             time_steps: Number of time steps t to simulate.
             starting_point: Starting point of the trajectory of shape (2,) (time dimension
             excluded).
         Returns:
             Trajectory of shape (t, 2).

         """
        starting_point = np.hstack((starting_point, 0.0))
        return _timestep_iterator(self.iterate, time_steps, starting_point)[:, :-1]


class KuramotoSivashinsky:
    """Simulate the n-dimensional Kuramoto-Sivashinsky PDE.

    PDE: y_t = -y*y_x - (1+eps)*y_xx - y_xxxx.

    Reference for the numerical integration:
    "fourth order time stepping for stiff pde-kassam trefethen 2005" at
    https://people.maths.ox.ac.uk/trefethen/publication/PDF/2005_111.pdf

    Python implementation at: https://github.com/E-Renshaw/kuramoto-sivashinsky

    """
    def __init__(self, dimensions: int = 50, system_size: float = 36, eps: float = 0,
                 dt: float = 0.1) -> None:
        """

        Args:
            dimensions: The dimensions of the KS system.
            system_size: The system size of the KS system.
            eps: A parameter in the KS system: y_t = -y*y_x - (1+eps)*y_xx - y_xxxx.
            dt: Size of time steps.
        """
        self.dimensions = dimensions
        self.system_size = system_size
        self.eps = eps
        self.dt = dt
        self._prepare()

    def _prepare(self):
        """function to calculate auxiliary variables."""
        k = (
            np.transpose(np.conj(np.concatenate((np.arange(0, self.dimensions / 2), np.array([0]),
                                                 np.arange(-self.dimensions / 2 + 1, 0)))))
            * 2
            * np.pi
            / self.system_size
            )

        L = (1 + self.eps) * k**2 - k**4

        self.E = np.exp(self.dt * L)
        self.E_2 = np.exp(self.dt * L / 2)
        M = 64
        r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
        LR = self.dt * np.transpose(np.repeat([L], M, axis=0)) + np.repeat([r], self.dimensions,
                                                                           axis=0)
        self.Q = self.dt * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=1))
        self.f1 = self.dt * np.real(
            np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR ** 2)) / LR ** 3, axis=1))
        self.f2 = self.dt * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR ** 3, axis=1))
        self.f3 = self.dt * np.real(
            np.mean((-4 - 3 * LR - LR ** 2 + np.exp(LR) * (4 - LR)) / LR ** 3, axis=1))

        self.g = -0.5j * k

    def iterate(self, x: np.ndarray) -> np.ndarray:
        """Calculates next timestep (x_0(i+1),x_1(i+1),..) with given (x_0(i),x_0(i),..) and dt.

        Args:
            x: (x_0(i),x_1(i),..) coordinates. Needs to have shape (self.dimensions,).
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

    def simulate(self, time_steps: int,
                 starting_point: np.ndarray | None = None) -> np.ndarray:
        """Simulate Kuramoto-Sivashinsky trajectory.

        Args:
            time_steps: Number of time steps t to simulate.
            starting_point: Starting point of the trajectory of shape (self.dimensions,).
        Returns:
            Trajectory of shape (t, self.dimensions).

        """
        if starting_point is None:
            # Use the starting point from the Kassam_2005 paper
            x = self.system_size * np.transpose(np.conj(np.arange(1, self.dimensions + 1))) \
                / self.dimensions
            starting_point = np.cos(2 * np.pi * x / self.system_size) * \
                             (1 + np.sin(2 * np.pi * x / self.system_size))
        else:
            if starting_point.shape[0] != self.dimensions:
                raise Exception(f"starting_point wrong dimension: Expected {self.dimensions} but "
                                f"got {starting_point.shape[0]}")

        return _timestep_iterator(self.iterate, time_steps, starting_point)


class KuramotoSivashinskyCustom:
    """Simulate the n-dimensional Kuramoto-Sivashinsky PDE with custom precision and fft backend.
    PDE: y_t = -y*y_x - y_xx - y_xxxx.

    Reference for the numerical integration:
    "fourth order time stepping for stiff pde-kassam trefethen 2005" at
    https://people.maths.ox.ac.uk/trefethen/publication/PDF/2005_111.pdf

    Python implementation at: https://github.com/E-Renshaw/kuramoto-sivashinsky

    """
    def __init__(self, dimensions: int = 50, system_size: float = 36, dt: float = 0.01,
                 precision: int | None = None, fft_type: str | None = None) -> None:
        """

        Args:
            dimensions: The dimensions of the KS system.
            system_size: The system size of the KS system.
            dt: Size of time steps.
            precision: The numerical precision for the simulation:
                    - None: no precision change
                    - 16, 32, 64, or 128 for the corresponding precision
            fft_type: Either "numpy" or "scipy".
        """
        self.dimensions = dimensions
        self.system_size = system_size
        self.dt = dt

        if precision is None:
            self.change_precision = False
        elif precision == 128:
            # NOTE: 128 precision is actually the same as longdouble precision on most (all?)
            # 64 bit machines, that is
            # 80 bits of precision, padded with zeros to 128 bits in memory.
            self.change_precision = True
            self.f_dtype = "float128"
            self.c_dtype = "complex256"
        elif precision == 64:
            self.change_precision = True
            self.f_dtype = "float64"
            self.c_dtype = "complex128"
        elif precision == 32:
            self.change_precision = True
            self.f_dtype = "float32"
            self.c_dtype = "complex64"
        elif precision == 16:
            self.change_precision = True
            self.f_dtype = "float16"
            self.c_dtype = "complex32"
        else:
            raise ValueError("specified precision not recognized")

        if fft_type is None or fft_type == "numpy":
            self.custom_fft = np.fft.fft
            self.custom_ifft = np.fft.ifft
        elif fft_type == "scipy":
            import scipy
            import scipy.fft

            self.custom_fft = scipy.fft.fft
            self.custom_ifft = scipy.fft.ifft
        else:
            raise ValueError("fft_type not recognized")

        self._prepare()

    def _prepare(self):
        """function to calculate auxiliary variables."""
        k = (
            np.transpose(np.conj(np.concatenate((np.arange(0, self.dimensions / 2), np.array([0]),
                                                 np.arange(-self.dimensions / 2 + 1, 0)))))
            * 2
            * np.pi
            / self.system_size
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
        LR = self.dt * np.transpose(np.repeat([L], M, axis=0)) + np.repeat([r], self.dimensions,
                                                                           axis=0)
        if self.change_precision:
            LR = LR.astype(self.c_dtype)
        self.Q = self.dt * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=1))
        if self.change_precision:
            self.Q = self.Q.astype(self.c_dtype)
        self.f1 = self.dt * np.real(
            np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR ** 2)) / LR ** 3, axis=1))
        if self.change_precision:
            self.f1 = self.f1.astype(self.c_dtype)
        self.f2 = self.dt * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR ** 3, axis=1))
        if self.change_precision:
            self.f2 = self.f2.astype(self.c_dtype)
        self.f3 = self.dt * np.real(
            np.mean((-4 - 3 * LR - LR ** 2 + np.exp(LR) * (4 - LR)) / LR ** 3, axis=1))
        if self.change_precision:
            self.f3 = self.f3.astype(self.c_dtype)

        self.g = -0.5j * k
        if self.change_precision:
            self.g = self.g.astype(self.c_dtype)

    def iterate(self, x: np.ndarray) -> np.ndarray:
        """Calculates next timestep (x_0(i+1),x_1(i+1),..) with given (x_0(i),x_0(i),..) and dt.

        Args:
            x: (x_0(i),x_1(i),..) coordinates. Needs to have shape (self.dimensions,).
        Returns:
            : (x_0(i+1),x_1(i+1),..) corresponding to input x.

        """

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

    def simulate(self, time_steps: int,
                 starting_point: np.ndarray | None = None) -> np.ndarray:
        """Simulate Kuramoto-Sivashinsky trajectory.

        Args:
            time_steps: Number of time steps t to simulate.
            starting_point: Starting point of the trajectory of shape (self.dimensions,).
        Returns:
            Trajectory of shape (t, self.dimensions).

        """
        if starting_point is None:
            # Use the starting point from the Kassam_2005 paper
            x = self.system_size * np.transpose(np.conj(np.arange(1, self.dimensions + 1))) \
                / self.dimensions
            starting_point = np.cos(2 * np.pi * x / self.system_size) * \
                             (1 + np.sin(2 * np.pi * x / self.system_size))
        else:
            if starting_point.shape[0] != self.dimensions:
                raise Exception(f"starting_point wrong dimension: Expected {self.dimensions} but "
                                f"got {starting_point.shape[0]}")
        if self.change_precision:
            starting_point = starting_point.astype(self.f_dtype)
        return _timestep_iterator(self.iterate, time_steps, starting_point)


class Lorenz96:
    """Simulate the n-dimensional dynamical system: Lorenz 96 model.

    """
    def __init__(self,  force: float = 8, dt: float = 0.05) -> None:
        """Define the system parameters.

        Args:
            force: 'force' parameter in the Lorenz 96 equations.
            dt: Size of time steps.
        """
        self.force = force
        self.dt = dt

    def flow(self, x: np.ndarray) -> np.ndarray:
        """Calculates (dx_0/dt, dx_1/dt, ..) with given (x_0,x_1,..) for RK4.

        Args:
            x: (x_0,x_1,..) coordinates. Adapts automatically to shape (dimensions, ).
        Returns:
            : (dx_0/dt, dx_1/dt, ..) corresponding to input x.

        """
        system_dimension = x.shape[0]
        derivative = np.zeros(system_dimension)
        # Periodic Boundary Conditions for the 3 edge cases i=1,2,system_dimension
        derivative[0] = (x[1] - x[system_dimension - 2]) * x[system_dimension - 1] - x[0]
        derivative[1] = (x[2] - x[system_dimension - 1]) * x[0] - x[1]
        derivative[system_dimension - 1] = (x[0] - x[system_dimension - 3]) * \
                                           x[system_dimension - 2] - x[system_dimension - 1]

        # TODO: Rewrite using numpy vectorization to make faster
        for i in range(2, system_dimension - 1):
            derivative[i] = (x[i + 1] - x[i - 2]) * x[i - 1] - x[i]

        derivative = derivative + self.force
        return derivative

    def iterate(self, x: np.ndarray) -> np.ndarray:
        """Calculates next timestep (x_0(i+1),x_1(i+1),..) with given (x_0(i),x_0(i),..) and dt
        with RK4.

        Args:
            x: (x_0(i),x_1(i),..) coordinates. Needs to have shape (self.dimensions,).
        Returns:
            : (x_0(i+1),x_1(i+1),..) corresponding to input x.

        """
        return _runge_kutta(self.flow, self.dt, x)

    def simulate(self, time_steps: int,
                 starting_point: np.ndarray | None = None) -> np.ndarray:
        """Simulate Lorenz 93 model.

        Args:
            time_steps: Number of time steps t to simulate.
            starting_point: Starting point of the trajectory. Automatically adapts to dimension of
            input.
        Returns:
            Trajectory of shape (t, input_dimension).

        """
        if starting_point is None:
            starting_point = np.sin(np.arange(30))

        return  _timestep_iterator(self.iterate, time_steps, starting_point)
