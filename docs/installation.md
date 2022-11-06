## Installation

The following is a guide detailing how to install the scan package the way it "should" be done, i.e. as safely as possible.  
If you already know what you are doing, you can just install this package like any other locally installed package.

These instructions are for unix systems, but do work with at most minor modifications on Windows too.

### Installation
#### Prerequisites
* Git
* Python 3.8+
* Pip or Conda (the latter typically via Miniforge)

#### The Basics

First, clone the scan repository to a folder of your choice. Once you have done this, enter the cloned folder, and 
tell git who you are:
```
git config user.name "Lastname, Firstname"
git config user.email "Firstname.Lastname@dlr.de"
```
From here, follow either the conda or pip installation instructions below. If you don't know which one to choose, 
[install miniforge](https://github.com/conda-forge/miniforge) as conda distribution and then follow the conda 
instructions.

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

After the installation, check out our [examples](examples.rst) to familiarize yourself with the package.  

### Common Installation Problems
####  The installation seems to have worked without error, but the package is not found when I try to import it. 

Make sure that pip is installed and active in the environment you want to install the package to:

    which pip

The output of the above should include the active environment name. For the environment _scan_ from above it should be something like

  > "/Users/<username>/anaconda3/envs/scan/bin/pip"

If it doesn't include the environment name, you installed the scan package in what ever environment the above specifies.   
To undo this, first uninstall the package,

    pip uninstall scan

then deactivate and activate your anaconda environment 

    conda deactivate
    conda activate scan

and then, should pip still not be located in the active environment by "which pip", install pip explicitly:

    conda install pip

#### Nothing works, please help me!  

If you can't install or use the package despite following the above instructions, [write us][maintainer mail adresses]. For a package as new as this one, such problems are to be expected and we will try to help as soon as possible.

### Updating

To keep the scan package up to date with the most current version in the repository, enter your local repository folder (the same folder we cloned during the installation) and run

    git pull
    
This updates your local copy of the package to be up to date with the one on the GitLab website.  
If you installed the package as an editable install during installation, you are done, the update is complete.
If you installed it as a regular package, you can finish the update by running

    pip install --upgrade .


### Uninstalling 

To uninstall the scan package, simply activate the respective environment and then type:

    pip uninstall scan

[maintainer mail adresses]: mailto:Sebastian.Baur@dlr.de?cc=Christoph.Raeth@dlr.de