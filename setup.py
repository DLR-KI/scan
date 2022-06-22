"""Setup.py for installation of the scan package.

Content to be moved to pyproject.toml eventually.

"""
import setuptools
from setuptools import setup

# TODO: Everything here needs to transition to pyproject.toml, to keep up with the times and to avoid code/version
#  dupliation
from scan._version import __version__

if __name__ == "__main__":
    setup(
        name="scan",
        version=__version__,
        # description='scan',
        # license_files=('LICENSE.txt',),
        author="Sebastian Baur",
        author_email="Sebastian.Baur@dlr.de",
        maintainer="Sebastian.Baur",
        maintainer_email="Sebastian.Baur@dlr.de",
        url="https://gitlab.dlr.de/baur_se/scan",
        download_url="git@gitlab.dlr.de:baur_se/scan.git",
        packages=setuptools.find_packages(),
        classifiers=[
            "Development Status :: 2 - Pre-Alpha",
            "Intended Audience :: Science/Research",
            "Topic :: Software Development :: Build Tools",
            # 'License :: OSI Approved :: MIT License',
            "Natural Language :: English",
            "Operating System :: OS IndependentProgramming Language :: Python",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            # "Programming Language :: Python :: 3.11",
        ],
        python_requires=">=3.8",
        # In the interest of ease of use, the highest package versions supported and tested are not specified here.
        # If an upper version limit is specified, that means there is a known bug with that version
        install_requires=[
            "numpy>=1.14.5",
            "networkx>=2.6.0",
            "pandas>=1.0.0",
            "scipy>=1.4.0",
            "scikit-learn>=0.20.0",
            "loguru>=0.4.0",
        ],
        extras_require={
            "mpi": ["mpi4py>=3.0.3"],
            "examples": ["matplotlib>=3.0.0,<4.0"],
        },
        provides=["scan"],
    )
