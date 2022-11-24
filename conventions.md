## Scan Conventions

List of style, naming, package and content conventions.


### Style Convention

The scan package follows the InRe code style.

### Naming Conventions

* (t, d, s) = (time_steps, x_dim/y_dim/sys_dim, slice_nr): np.array shape of the input/output/simulation data
* r = r(t): the reservoir state over time with shape (t, n_dim) or, for some internal functions, shape (n_dim,)
* "n_" prefix relates to network attributes, e.g. "n_dim" is the network dimension
* w_in: input matrix and prefix for input matrix related parameters
* "x_" prefix means input
* "y_" prefix means output/prediction
* "sys_" prefix means simulation data

### Package Conventions
* MPI is, and should stay, entirely optional

### Content Conventions
What should be part of scan:
* Everything related to the data and to the ML algorithms themselves: 
* Data pre- and postprocessing
* Generic data saving and loading where explicitly needed
* (ML) algorithms 
* Cost functions
* Hyperparameter optimization
* etc.

What should not be part of scan:
* Plotting
* Data "Interpretation" code
* Any wrapping code that you, and only you, will use
* Any data files
* Any non-text documentation files. This includes jupyter notebooks due to them 
