# C++ RBM code
This is the c++ base-code for learning a probability distribution with a Restricted Botlzmann machine. The code is written in C++11, it is well commented and pretty self-explanatory. The only requirement is [Eigen3][1]. Eigen3 is a header-only linear algebra library, so you can either install it with your favourite package manager, or just donwload it somewhere. Just include the path and compile as

`g++ main.cpp -O2 -I PATH_TO_EIGEN3 -std=c++11  -o run.x`

You can run it using all default parameters as

`./run.x`

If you want to change the parameters, just add the following argument directly in the command line, after the run.x command:

* `-nv`: Number of visible units
* `-nh`: Number of hidden units
* `-w `: Width of initial weights distribution
* `-nc`: Numer of sampling chains
* `-cd`: Number of Gibbs updates in Contrastive Divergence
* `-lr`: Learning rate
* `-l2`: L2 regularization constant
* `-bs`: Batch size
* `-ns`: Numer of training samples (max 10000)
* `-ep`: Number of training iterations 


RIght now I included data for the transverse field Ising model in 1d at the critical point, for N=10 spins. Each sample in the dataset is obtained from the projection of the wavefunction on the sigma_z basis. I also included the true wavefunction, so you can compute the overlap to monitor the learning (note that this is only possible for small systems). As you run the code, the network will learn the probability distribution given by the square of the wavefunction, and it will print the negative log-likelihood evaluated on a validation, as well as the overlap (you should get ~0.999 overlap with the default parameters).
 
Finally, at the bottom of the `tomography.hpp` class, you will find a function that checks the derivatives numerically and compare with the ones computed in the `rbm.hpp` class (it's the first thing to check when you have get in trouble).



[1]: https://eigen.tuxfamily.org
