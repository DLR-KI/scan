## Scan 

This repository contains the python 3 package SCAN, implementing the machine 
learning technique Reservoir Computing (RC) and related methods.

SCAN: **S**oftware for **C**omplex system **A**nalysis via **N**etworks.

### License
The SCAN software is provided under the MIT License.

### Installation
#### Prerequisites
* Git
* Python 3.8+
* Pip or Conda (the latter typically via Miniforge)

#### The Basics

First, clone the scan repository to a folder of your choice. 

Once you have done this, enter the cloned folder, and tell git who you are:
```
git config user.name "Lastname, Firstname"
git config user.email "Firstname.Lastname@dlr.de"
```
From here, follow either the conda or pip installation instructions below. If you don't know which one to choose, 
[install miniforge](https://github.com/conda-forge/miniforge) as conda distribution and then follow the conda 
installation instructions.

#### Installation using Conda
Create a new python 3.9 environment named scan and install the scan package requirements as well as the packages used 
for further development:
```
conda create --name scan python=3.9 --file requirements.txt --file requirements_dev.txt
```
On some systems, mostly Windows, the above may fail due to no MPI being installed. In that case, use the no_mpi 
requirements file. You will miss out on some features and performance, but as MPI is optional, you'll still be able to 
use the package:
```
conda create --name scan python=3.9 --file requirements_no_mpi.txt --file requirements_dev.txt
```
Notably, if you are not intersted in changing the package or running its tests, you can also install the minimal 
requirements necessary to install scan via:
```
conda create --name scan python=3.9 --file requirements.txt
```
or 
```
conda create --name scan python=3.9 --file requirements_no_mpi.txt
```
respectively. This is not recommended as running the tests is the only reasonable way to know if the package actually 
does what it's supposed to.

Now activate the conda environment
```
conda activate scan
```
and install the package in editable mode:
```
pip install -e .
```

Finally, run the unit tests to see if everything is working as intended:
```
pytest
```
If no errors pop up, you have installed the scan package successfully. 

#### Installation using Pip
If you are using pip, it is still _highly_ recommended to not change your system python and instead create a virtual 
environment using a tool of your choice to install scan in. Pyenv with the pyenv-virtualenv plugin are recommended, but exchangeable.  

Once you have activated said virtual environment, install the requirements using:
```
pip install -r requirements.txt -r requirements_dev.txt
```
Similarly to the conda installation instruction above, the requirements files can be adjusted if your system doesn't have MPI installed:
```
pip install -r requirements_no_mpi.txt -r requirements_dev.txt
```
or you don't want to edit the package and run its tests:
```
pip install -r requirements.txt
```
or both:
```
pip install -r requirements_no_mpi.txt
```

Now, install the package in editable mode:
```
pip install -e .
```
and run the unit tests to see if everything is working as intended:
```
pytest
```
If no errors pop up, you have installed the scan package successfully. 


### Push and Merge Request Steps

Before any push or merge request do the following from inside the scan repository main folder within an active 
development environment.
following:  
1. Use black and isort the format the code:
    ```
    black . && isort .
    ```
2. Run mypy to check for type hint coverage and type related problems:
    ```
    mypy
    ```
3. Run the testing suite in your active environment to check for bugs and code coverage:
    ```
    pytest --cov-report term-missing --cov=scan tests/
    ```
4. Run the testing suite in all tox-conda environments:
    ```
    tox --develop -c tox-conda.ini
    ```