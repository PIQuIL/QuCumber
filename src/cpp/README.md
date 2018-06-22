# Quantum State Tomography with Neural Networks

The code implement neural-network quantum state tomography (QST), i.e. the recontruction reconstruction of an unknown quantum state from a set of measurements. It relies on the parametrization of the quantum state with a restricted Boltzmann Machine (RBM). The reconstruction is performed using standard unsupervised machine learning.


### Requirements 

The code is writted in C++11, with the only requirement being [Eigen3][1], a header-only library for linear algebra.

### Compiling
   
`g++ main.cpp -O2 -I PATH_TO_EIGEN3 -std=c++11  -o run.x`
   
### Running
   
`./run.x -PARAMETER1 par1 -PARAMETER2 par2 ...`
  
The parametrs are:
  
  * `-nv`: Number of visible units
  * `-nh`: Number of hidden units
  * `-w `: Width of initial weights distribution
  * `-nc`: Numer of sampling chains
  * `-cd`: Number of Gibbs updates in Contrastive Divergence
  * `-lr`: Learning rate
  * `-l2`: L2 regularization constant
  * `-bs`: Batch size
  * `-ns`: Numer of training samples
  * `-ep`: Number of training iterations
  * `-basis`: Set of measurements bases

## Features

### Current features

* RBM for binary Hilbert spaces.
* RBM-state for positive wavefunctions.
* QST of positive wavefunctions through minimization of the KL divergence. 
* RBM-state for complex wavefunctions.
* QST of complex wavefunctions through minimization of a generalized KL divergence 

### Under testing

* RBM-state for density matrices.

### Upcoming features

* QST of mixed states.
* RBM for multinomial Hilbert spaces.
* RBM-state for real non-positive wavefunctions.
* MPI support.
* Code documentation.
 
## Tutorials (coming soon)

* 1d quantum Ising model.
* Entangled photonic pure states.

#

 [1]: https://eigen.tuxfamily.org
