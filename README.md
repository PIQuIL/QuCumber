# QuCumber
[![Build Status](https://travis-ci.org/MelkoCollective/QuCumber.svg?branch=master)](https://travis-ci.org/MelkoCollective/QuCumber)

A Quantum Calculator Used for Many-body Eigenstate Reconstruction.

QuCumber is a program that reconstructs an unknown quantum wavefunction
from a set of measurements.  The measurements should consist of binary counts;
for example, the occupation of an atomic orbital, or the Sz eigenvalue of
a qubit.  These measurements form a training set, which is used to train a
stochastic neural network called a Restricted Boltzmann Machine.  Once trained, the
neural network is a reconstructed representation of the unknown wavefunction
underlying the measurement data. It can be used for generative modelling, i.e.
producing new instances of measurements, and to calculate estimators not
contained in the original data set.

QuCumber is developed by the Perimeter Institue Quantum Intelligence Lab (PIQuIL).

## License
QuCumber is licensed under the Apache License Version 2.0.

## Features
QuCumber implements unsupervised generative modelling with a two-layer RBM.
Each layer is a number of binary stochastic variables (with values 0 or 1).  The size of the visible
layer corresponds to the input data, i.e. the number of qubits.  The size of the hidden
layer is varied to systematically control representation error.

Currently the reconstruction can be performed on pure states with a positive-definite
wavefunction.  Data is thus only required in one basis.  Upcoming versions will
allow reconstruction of more general wavefunctions and density matrices; in this case
tomographyically-complete basis sets may be required in the training data.

## Requirements
Python 3. QuCumber is written in PyTorch, with CPU and GPU support.  See https://pytorch.org.

## Documentation

Documentation can be found [here](https://piquil.github.io/QuCumber/).
