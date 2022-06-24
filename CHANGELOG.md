## Changelog

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
* Slight restructure of internal esn train/predict methods for a reduced memory overhead
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
