## Changelog
### Scan 0.6.2 - Added simple Lyapunov exponent calculation function
- Added a simple measure function to calculate the largest Lyapunov exponent of a dynamical system:
`scan.measures.largest_lyapunov_exponent`
- The positional arguments are:
  - `iterator_func`: A function to iterate the system from one to the next time-step: x(i+1) = F(x(i))
  - `starting_point`: The starting point for iterating the system. 

**Outline of Algorithm**: 
- The algorithm simulates two trajectories (the _base_ and the _perturbed_ trajectory) for some 
initially very close points, for some time steps (given by `part_time_steps`).
- The largest lyapunov exponent corresponding to this particular trajectory divergence is extracted and saved.
- The last point of the perturbed trajectory is "pulled" to the last point of the base trajectory by
keeping the direction between both points constant, but renormalizing the distance to a small value (given by `deviation_scale`). 
This new perturbed point serves as the initial point for the next trajectory simulation as explained in the first bulletpoint. 
- Thus, one repeats this scheme for `steps` renormalization steps and calculates the largest lyapunov exponent as the average of all renormalization steps.

See: _Sprott, Julien Clinton, and Julien C. Sprott. Chaos and time-series analysis. Vol. 69.
    Oxford: Oxford university press, 2003._ for an explanation of the algorithm. 

### Scan 0.6.1 - Simulating dynamical systems: Enhancements, bugfix and new system
* Fixed bug in `scan.simulations.LinearSystem` where every non-default matrix `A` would give an error. 
* Changed how the simulation classes are build up: 
  * Every simulation class is a subclass of the abstract base class `SimBase`. 
  * `SimBase` takes care of the basic structure of a simulation class: 
    * Every system must implement the `iterate` method, since `iterate` is now an `@abstractmethod`
    in `SimBase`.
    * `SimBase` defines the `simulate` function, which can be used in all child classes. 
    * `SimBase` assumes that every child simulation class defines a `default_starting_point`and the 
    `sys_dim` (system dimension) of the system. 
  * If the system is simulated with RungeKutta, it is based on the subclass `SimBaseRungeKutta`, 
  which is already a subclass of `SimBase`.
    * Every subclass of `SimBaseRungeKutta` is required to have the `flow` method (using `@abstractmethod`). 
    * The `iterate` method is defined in `SimBaseRungeKutta` by applying runge kutta on the flow. 
  * Every simulation class now has the default parameters saved in `self.default_parameters`.
  * Every simulation class now has the default starting point saved in `self.default_starting_point`.
    * Note: In KuramotoSivashinsky, KuramotoSivashinskyCustom, Lorenz96 and LinearSystem the default starting point is dependent
    on the parameters, since the dimension of the system is also a parameter. Thus, one can only access the `default_starting_point`
    after initializing the class-instance. E.g. `KuramotoSivashinsky.default_starting_point` does not exist, but `KuramotoSivashinsky().default_starting_point` does
  * **Examples**: 
    * `Lorenz63().default_parameters` gives: `{"sigma": 10.0, "rho": 28.0, "beta": 8 / 3, "dt": 0.05}`
    * `Lorenz63().default_starting_point` gives: `np.array([0.0, -0.01, 9.0])`
* Added new 2-dimensional autonomous flow `LotkaVolterra` system.
* Added `sys_dim` parameter to Lorenz96 (instead of indireclty specifying the dimension in `simulate`)

### Scan 0.6.0 - Simulating dynamical systems: Refactoring and new systems
* Removed the function `scan.simulations.simulate_trajectory`.
* Instead of using `simulate_trajectory`, there is now a class for every dynamical system implemented.
  * The dynamical-system classes have a common structure: 
    * The system parameters (including time step `dt`, if the system is a flow) are specified in 
    the class constructor (i.e. when calling `__init__`).
    * Every class has the method `simulate` which can be used to simulate the whole trajectory. 
    It takes the positional argument `time_steps` and the keyword argument `starting_point`.
    * If you simulate for _n_ `time_steps`, the returned trajectory has exactly _n_ elements, where 
    the first element is the starting point. So actually one only simulates _n-1_ new time steps.
    * All system classes have a method `iterate`, that iterates the system to the next time step. Some 
    systems also have the method `flow`, which calculates the time derivative at a given point. 
  * Example to simulate a trajectory of the _Lorenz63_ system with 1000 time steps:
    * `trajectory = Lorenz63(sigma=10, rho=28, beta=8/3, dt=0.05).simulate(1000, starting_point=np.array([0, 1, 2]))` 
    or just `Lorenz63().simulate(1000)` with default parameters and default starting_point.
* Added more dynamical Systems, with default parameters and starting_points from _Sprott, Julien 
Clinton, and Julien C. Sprott. Chaos and time-series
    analysis. Vol. 69. Oxford: Oxford university press, 2003._
* List of supported dynamical systems: 
  - Lorenz63 
  - Roessler
  - ComplexButterfly (see _Sprott_)
  - Chen
  - ChuaCircuit
  - Thomas
  - WindmiAttractor (see _Sprott_)
  - Rucklidge
  - SimplestQuadraticChaotic (see _Sprott_)
  - SimplestCubicChaotic (see _Sprott_)
  - SimplestPiecewiseLinearChaotic (see _Sprott_)
  - DoubleScroll
  - Henon
  - Logistic
  - SimplestDrivenChaotic (see _Sprott_)
  - UedaOscillator
  - KuramotoSivashinsky
  - KuramotoSivashinskyCustom
  - Lorenz96
  - LinearSystem (A general n_dimensional linear system: x_t = A*x with matrix A)
  
### Scan 0.5.0 - Compared to Rescomp 0.3.2

#### Non-Breaking Changes

* Achieved 100% test coverage 
  * Pytest as replacement for unittest
  * pytest-cov for coverage reporting
  * pytest-mpi for MPI code testing
  * We can now finally, actually, trust our code!
* Achieved 100% type hint coverage 
  * It's documentation that's also IDE readable, resulting in usage checks and warnings before the code is even run
* Mypy testing of those type hints and their flow throughout the program
* Tox testing framework to ensure installation and test verification in a variety of environments
  * In total 24 different environments are tested:
  * Conda and pip installations
  * Python versions 3.8 - 3.10
  * Current, oldest supported, newest supported and cutting edge package versions
* Adjusted style and formatting to follow the new InRe Style Guide
* Automated formatting tools black and isort to assist in achieving

* More expansive module structure
  * Added the modules data_processing, file_handling, logging and mpi with appropriate contents
* Added "slices" for train and predict input 
  * 3d input of shape (t, d, s), instead of only 2d input of shape (t, d), supported to train and/or predict at multiple 
positions in the time series at once 
  * Backwards compatible
* Significantly reduced peak memory usage for the ESN classes. For larger data and reservoir sizes, where memory 
constrains matter most, peak memory usage is down by about 50%.
  * Old peak memory usage:
    * "linear_r": (time_steps * slices * n_dim * **2**) * 64 bit + 150 MB
    * "linear_and_square_r": (time_steps * slices * n_dim * **4** ) * 64 bit + 150 MB
  * New:
    * "linear_r": (time_steps * slices * n_dim + 2 * n_dim**2) * 64 bit + 150 MB
    * "linear_and_square_r": (time_steps * slices * n_dim * **2** + 2 * (n_dim * 2)**2) * 64 bit + 150 MB
* Complete independence from the global numpy random seed via two new, also independent, seed parameters: n_seed and 
w_in_seed
  * This enables full software level reproducibility, even in parallel or not entirely user controlled contexts!
* The "reset_r" parameter, resetting the reservoir to a known state of all zeros before each slice. This makes training
and prediction depend only on the content of the time series, not the slice order
* Optional MPI implementation. For details see the mpi.py package
  * Most notably this includes the MPIScheduler()
  * Compatible with simple usage on laptops and cluster scheduling tools like slurm alike
  * Pytest-mpi as testing framework
* Loguru as replacement for the default logging backend
  * Pickleable
  * (Mpi-)Threadsafe 
  * All around just easier to work with as there is "only one single logger"
* Due to Loguru, all ESN classes are now pickleable
* SynonymDict initialization via constructor
* Hash operator for the ESN classes
* Equality operator for the ESN classes
* Downsampling function in data_processing.py
* Embedding function in data_processing.py
* Removed unused ArpackNoConvergence error handling during network creating
  * Should we, unexpectedly ever need it again, find a suitable example and then add it to a test


* Various changes to make installation more consistent
  * Check for a supported python version on install
  * Check for the required packages and their supported version on install
  * Started the transition from setup.py to pyproject.toml
* Added support for python 3.10
* Requirements_dev.txt for easy setup of the development environment
* Added conventions.md, describing the naming and other conventions of the package
* Update docstrings to reflect slightly updated conventions outlined in scan_conventions.md


#### Breaking Changes
* Dropped support for python 3.6 and 3.7. Advantages:
  * Python 3.6 has no good way to [type hint custom classes](https://stackoverflow.com/questions/33533148/how-do-i-type-hint-a-method-with-the-type-of-the-enclosing-class)
  * Python 3.6 doesn't support networkx>=2.6 See below
  * Python 3.8 added the f"{some_var=}" feature for f-strings, which is just nice to have
  * Python 3.9 added support for the "type" syntax and [added lots of type hint stuff into the main library](https://adamj.eu/tech/2021/05/16/python-type-hints-return-class-not-instance/). 
All of that is part of \_\_future\_\_.annotations though, so 3.8 effectively supports it too
  * Also update old patterns to conform to the new minimum python version 3.8 via pyupgrade
* Move to networkx >= 2.6. Networkx < 2.6 and >= 2.6 create the scale free and small world networks from a different 
seed, i.e. are incompatible
* Default of the "reset_r" parameter is "True". To get the old behavior, set it to "False"
* Removed lots of unused ESN and some unused utility and measure functions
  * Either no use case was apparent or the current contributors to the scan package didn't write them originally, hence 
they couldn't be included for license reasons
* Removed some unused parameters
  * Most notably, this includes the "save_input" parameter which is unecessary, as the user, by definition, already has 
the input
* Changed behavior of the "save_r" parameter to only save the r state itself, not it's generalized r_gen version
  * It's redundant because the r_to_generalized_r method exists and potentially, very memory intensive
* Removed most of the non-essential simulation functions
  * Either no use case was apparent or the current contributors to the scan package didn't write them originally, hence 
they couldn't be included for license reasons
* Refactored all ESN attributes and many methods from protected to public
  * All of them have legitimate use cases in a research context like ours, so making them less accessible doesn't make 
much sense. 
* Stubed out the ESNGenLoc class and locality_measures methods
  * They still exist, they were just moved to the GLS branch until they comply with the standard of the rest of the 
package. Mostly it's missing tests and some documentation
* Fixed lots of small errors for edge case inputs
