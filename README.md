# ![QuCumber](https://raw.githubusercontent.com/PIQuIL/QuCumber/master/docs/_static/img/QuCumber_readme.png)

[![PyPI version](https://badge.fury.io/py/qucumber.svg)](https://badge.fury.io/py/qucumber)
[![Documentation Status](https://readthedocs.org/projects/qucumber/badge/?version=latest)](https://qucumber.readthedocs.io/en/latest/?badge=latest)
[![Build Status (Travis)](https://travis-ci.com/PIQuIL/QuCumber.svg?branch=master)](https://travis-ci.com/PIQuIL/QuCumber)
[![Build Status (AppVeyor)](https://ci.appveyor.com/api/projects/status/lqdrc8qp94w4b9kf/branch/master?svg=true)](https://ci.appveyor.com/project/emerali/qucumber/branch/master)
[![codecov](https://codecov.io/gh/PIQuIL/QuCumber/branch/master/graph/badge.svg)](https://codecov.io/gh/PIQuIL/QuCumber)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

## A Quantum Calculator Used for Many-body Eigenstate Reconstruction

QuCumber is a program that reconstructs an unknown quantum wavefunction
from a set of measurements. The measurements should consist of binary counts;
for example, the occupation of an atomic orbital, or the Sz eigenvalue of
a qubit. These measurements form a training set, which is used to train a
stochastic neural network called a Restricted Boltzmann Machine. Once trained, the
neural network is a reconstructed representation of the unknown wavefunction
underlying the measurement data. It can be used for generative modelling, i.e.
producing new instances of measurements, and to calculate estimators not
contained in the original data set.

QuCumber is developed by the Perimeter Institute Quantum Intelligence Lab (PIQuIL).
The project is currently in beta. So, expect some rough edges, bugs,
and backward incompatible updates.

## Features

QuCumber implements unsupervised generative modelling with a two-layer RBM.
Each layer is a number of binary stochastic variables (with values 0 or 1). The
size of the visible layer corresponds to the input data, i.e. the number of
qubits. The size of the hidden layer is varied to systematically control
representation error.

Currently the reconstruction can be performed on pure states with either
positive-definite or complex wavefunctions. In the case of a positive-definite
wavefunction, data is only required in one basis. For complex wavefunctions,
tomographically complete basis sets will be required to train the wavefunction.

## Documentation

Documentation can be found [here](https://piquil.github.io/QuCumber/).

## Getting Started

These instructions will get you a copy of the project up and running on your
local machine for development and testing purposes.

### Installing

If you're on Windows, you will have to install PyTorch manually; instructions
can be found on their website: [pytorch.org](https://pytorch.org).

You can install the latest stable version of QuCumber, along with its dependencies,
using [`pip`](https://pip.pypa.io/en/stable/quickstart/):

```bash
pip install qucumber
```

If, for some reason, `pip` fails to install PyTorch, you can find installation
instructions on their website. Once that's done you should be able to install
QuCumber through `pip` as above.

QuCumber supports Python 3.5 and newer stable versions.

### Installing the bleeding-edge version

If you'd like to install the most upto date, but potentially unstable version,
you can clone the repository's develop branch and then build from source like so:

```bash
git clone git@github.com:PIQuIL/QuCumber.git
cd ./QuCumber
git checkout develop
python setup.py install
```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute
to the project, and the process for submitting pull requests to us.

## License

QuCumber is licensed under the Apache License Version 2.0.

## Acknowledgments

- Lauren Hayward Sierens for many helpful discussions.

- [Nick Mercer](https://www.behance.net/nickdmercec607) for the awesome logo.
