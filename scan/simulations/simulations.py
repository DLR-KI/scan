"""
Simulate timeseries of various dynamical systems.
Every dynamical system is represented as a class.
The syntax for simulating the trajectory is:
trajectory = SystemClass(parameters).simulate(time_steps, starting_point)
"""

from __future__ import annotations

from typing import Callable

import numpy as np

def _runge_kutta(
    f: Callable[[np.ndarray], np.ndarray], dt: float, x: np.ndarray) -> np.ndarray:
    """Simulate one step for ODEs of the form dx/dt = f(x(t)), returns x(t + dt)
    Args:
        f: function used to calculate the time derivate at point y
        dt: time step size
        x: d-dim position at time t

    Returns:
       d-dim position at time t+dt

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
    """Utility function to iterate a function f: x(i+1) = f(x(i)) multiple times to obtain a
    full trajectory
    Args:
        f: The iterator function x(i+1) = f(x(i))
        time_steps: The number of time_steps of the output trajectory
                    The starting_point is included as the 0th element in the trajectory
        starting_point: Starting point of the trajectory

    Returns:
        trajectory: system-state at every simulated timestep

    """
    starting_point = np.array(starting_point)
    traj_size = (time_steps, starting_point.shape[0])
    traj = np.zeros(traj_size)
    traj[0, :] = starting_point
    for t in range(1, traj_size[0]):
        traj[t] = f(traj[t-1])
    return traj


class Lorenz63:
    """Simulate the 3-dimensional autonomous flow: Lorenz-63 attractor

    Literature values (Sprott, Julien Clinton, and Julien C. Sprott. Chaos and time-series
    analysis. Vol. 69. Oxford: Oxford university press, 2003.) for default parameters and
    starting_point:
    - Lyapunov exponents: (0.9059, 0.0, -14.5723)
    - Kaplan-Yorke dimension: 2.06215
    - Correlation dimension: 2.068 +- 0.086
    """
    def __init__(self, sigma: float = 10, rho: float = 28, beta: float = 8 / 3,
                 dt: float = 0.05) -> None:
        """Define the system parameters

        Args:
            sigma: 'sigma' parameter in the Lorenz 63 equations
            rho: 'rho' parameter in the Lorenz 63 equations
            beta: 'beta' parameter in the Lorenz 63 equations
            dt: Size of time steps
        """
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.dt = dt

    def flow(self, x: np.ndarray) -> np.ndarray:
        """Calculates (dx/dt, dy/dt, dz/dt) with given (x,y,z) for RK4

        Args:
            x: (x,y,z) coordinates. Needs to have shape (3,)
        Returns:
            : (dx/dt, dy/dt, dz/dt) corresponding to input x

        """
        return np.array([self.sigma * (x[1] - x[0]),
                         x[0] * (self.rho - x[2]) - x[1],
                         x[0] * x[1] - self.beta * x[2]])

    def iterate(self, x: np.ndarray) -> np.ndarray:
        """Calculates next timestep (x(i+1), y(i+1), z(i+1)) with given (x(i),y(i),z(i)) and dt
        with RK4

        Args:
            x: (x,y,z) coordinates. Needs to have shape (3,)
        Returns:
            : (x(i+1), y(i+1), z(i+1)) corresponding to input x

        """
        return _runge_kutta(self.flow, self.dt, x)

    def simulate(self, time_steps: int,
                 starting_point: np.ndarray = np.array([0.0, -0.01, 9.0])) -> np.ndarray:
        """Simulate Lorenz63 trajectory

        Args:
            time_steps: Number of time steps t to simulate
            starting_point: Starting point of the trajectory of shape (3,)
        Returns:
            Trajectory of shape (t, 3)

        """
        return  _timestep_iterator(self.iterate, time_steps, starting_point)


class Henon:
    """Simulate the 2-dimensional dissipative map: Henon map

    Literature values (Sprott, Julien Clinton, and Julien C. Sprott. Chaos and time-series
    analysis. Vol. 69. Oxford: Oxford university press, 2003.) for default parameters and
    starting_point:
    - Lyapunov exponents: (0.41922, -1.62319)
    - Kaplan-Yorke dimension: 1.25827
    - Correlation dimension: 1.220 +- 0.036
    """
    def __init__(self, a: float = 1.4, b: float = 0.3) -> None:
        """Define the system parameters

        Args:
            a: 'a' parameter of Henon map
            b: 'b' parameter of Henon map
        """
        self.a = a
        self.b = b

    def iterate(self, x: np.ndarray) -> np.ndarray:
        """Calculates next timestep (x(i+1), y(i+1)) with given (x(i),y(i))

        Args:
            x: (x,y,z) coordinates. Needs to have shape (2,)
        Returns:
            : (x(i+1), y(i+1)) corresponding to input x

        """
        return np.array([1 - self.a*x[0]**2 + self.b * x[1], x[0]])

    def simulate(self, time_steps: int,
                 starting_point: np.ndarray = np.array([0.0, 0.9])) -> np.ndarray:
        """Simulate Henon trajectory

        Args:
            time_steps: Number of time steps t to simulate
            starting_point: Starting point of the trajectory of shape (2,)
        Returns:
            Trajectory of shape (t, 2)

        """
        return _timestep_iterator(self.iterate, time_steps, starting_point)


class Logistic:
    """Simulate the 1-dimensional noninvertable map: Logistic map

    Literature values (Sprott, Julien Clinton, and Julien C. Sprott. Chaos and time-series
    analysis. Vol. 69. Oxford: Oxford university press, 2003.) for default parameters and
    starting_point:
    - Lyapunov exponents: ln(2) = 0.6931147..
    - Kaplan-Yorke dimension: 1.0
    - Correlation dimension: 1.0
    """
    def __init__(self, r: float = 4) -> None:
        """Define the system parameters

        Args:
            r: 'r' parameter of Logistic map
        """
        self.r = r

    def iterate(self, x: np.ndarray) -> np.ndarray:
        """Calculates next timestep (x(i+1), ) with given (x(i), )

        Args:
            x: (x, ) coordinates. Needs to have shape (1,)
        Returns:
            : (x(i+1), ) corresponding to input x

        """
        return np.array([self.r * x[0] * (1 - x[0]), ])

    def simulate(self, time_steps: int,
                 starting_point: np.ndarray = np.array([0.1, ])) -> np.ndarray:
        """Simulate Lorenz63 trajectory

        Args:
            time_steps: Number of time steps t to simulate
            starting_point: Starting point of the trajectory of shape (1,)
        Returns:
            Trajectory of shape (t, 1)

        """
        return _timestep_iterator(self.iterate, time_steps, starting_point)


class SimplestDrivenChaotic:
    """Simulate the 2+1 (2 space, 1 time-dimension) conservative flow: Simplest Driven Chaotic flow

    Literature values (Sprott, Julien Clinton, and Julien C. Sprott. Chaos and time-series
    analysis. Vol. 69. Oxford: Oxford university press, 2003.) for default parameters and
    starting_point:
    - Lyapunov exponents: (0.0971, 0, -0.0971)
    - Kaplan-Yorke dimension: 3.0
    - Correlation dimension: 2.634 +- 0.160
    """
    def __init__(self, omega: float = 1.88, dt: float = 0.1) -> None:
        """Define the system parameters

        Args:
            omega: 'omega' parameter of Simplest Driven Chaotic flow
            dt: Size of time steps
        """
        self.omega = omega
        self.dt = dt

    def flow(self, x: np.ndarray) -> np.ndarray:
        """Calculates (dx/dt, dy/dt, dt/dt) with given (x,y,t) for RK4

        Args:
            x: (x,y,z) coordinates. Needs to have shape (3,)
        Returns:
            : (dx/dt, dy/dt, dt/dt=1) corresponding to input x

        """
        return np.array([x[1], -(x[0] ** 3) + np.sin(self.omega * x[2]), 1])

    def iterate(self, x: np.ndarray) -> np.ndarray:
        """Calculates next timestep (x(i+1), y(i+1), t(i+1)) with given (x(i),y(i),t(i)) and dt
        with RK4. The third coordinate is time: t(i+1) = t(i) + dt

        Args:
            x: (x,y,t) coordinates. Needs to have shape (3,)
        Returns:
            : (x(i+1), y(i+1), t(i+1)) corresponding to input x

        """
        return _runge_kutta(self.flow, self.dt, x)

    def simulate(self, time_steps: int,
                 starting_point: np.ndarray = np.array([0.0, 0.0])) -> np.ndarray:
        """Simulate Simplest Driven Chaotic flow trajectory

         Args:
             time_steps: Number of time steps t to simulate
             starting_point: Starting point of the trajectory of shape (2,) (time dimension
             excluded)
         Returns:
             Trajectory of shape (t, 2)

         """
        starting_point = np.hstack((starting_point, 0.0))
        return _timestep_iterator(self.iterate, time_steps, starting_point)[:, :-1]


class KuramotoSivashinsky:
    """Simulate the n-dimensional Kuramoto-Sivashinsky PDE
    PDE: y_t = -y*y_x - (1+eps)*y_xx - y_xxxx

    Reference for the numerical integration:
    "fourth order time stepping for stiff pde-kassam trefethen 2005" at
    https://people.maths.ox.ac.uk/trefethen/publication/PDF/2005_111.pdf

    Python implementation at: https://github.com/E-Renshaw/kuramoto-sivashinsky

    """
    def __init__(self, dimensions: int = 50, system_size: float = 36, eps: float = 0,
                 dt: float = 0.01) -> None:
        """

        Args:
            dimensions: The dimensions of the KS system
            system_size: The system size of the KS system
            eps: A parameter in the KS system: y_t = -y*y_x - (1+eps)*y_xx - y_xxxx
            dt: Size of time steps
        """
        self.dimensions = dimensions
        self.system_size = system_size
        self.eps = eps
        self.dt = dt
        self._prepare()

    def _prepare(self):
        """function to calculate auxiliary variables"""
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
        """Calculates next timestep (x_0(i+1),x_1(i+1),..) with given (x_0(i),x_0(i),..) and dt

        Args:
            x: (x_0(i),x_1(i),..) coordinates. Needs to have shape (self.dimensions,)
        Returns:
            : (x_0(i+1),x_1(i+1),..) corresponding to input x

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
        """Simulate Kuramoto-Sivashinsky trajectory

        Args:
            time_steps: Number of time steps t to simulate
            starting_point: Starting point of the trajectory of shape (self.dimensions,)
        Returns:
            Trajectory of shape (t, self.dimensions)

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


class Lorenz96:
    """Simulate the n-dimensional dynamical system: Lorenz 96 model

    """
    def __init__(self,  force: float = 8, dt: float = 0.05) -> None:
        """Define the system parameters

        Args:
            force: 'force' parameter in the Lorenz 96 equations
            dt: Size of time steps
        """
        self.force = force
        self.dt = dt

    def flow(self, x: np.ndarray) -> np.ndarray:
        """Calculates (dx_0/dt, dx_1/dt, ..) with given (x_0,x_1,..) for RK4

        Args:
            x: (x_0,x_1,..) coordinates. Adapts automatically to shape (dimensions, )
        Returns:
            : (dx_0/dt, dx_1/dt, ..) corresponding to input x

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
        with RK4

        Args:
            x: (x_0(i),x_1(i),..) coordinates. Needs to have shape (self.dimensions,)
        Returns:
            : (x_0(i+1),x_1(i+1),..) corresponding to input x

        """
        return _runge_kutta(self.flow, self.dt, x)

    def simulate(self, time_steps: int,
                 starting_point: np.ndarray = np.arange(10)) -> np.ndarray:
        """Simulate Lorenz 93 model

        Args:
            time_steps: Number of time steps t to simulate
            starting_point: Starting point of the trajectory. Automatically adapts to dimension of
            input
        Returns:
            Trajectory of shape (t, input_dimension)

        """
        # TODO: sensible starting_point
        return  _timestep_iterator(self.iterate, time_steps, starting_point)

