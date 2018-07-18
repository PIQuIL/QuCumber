from qucumber.rbm import BinomialRBM

import torch
import numpy as np

rbm_real   = BinomialRBM.autoload('saved_params_real.pkl')
data       = torch.tensor(np.loadtxt(
                          'examples/tfim1d_N10_train_samples.txt'), 
                          dtype = torch.double)
target_psi = torch.tensor(np.loadtxt('examples/tfim1d_N10_psi.txt'), 
                          dtype = torch.double)
vis        = rbm_real.rbm_module.generate_visible_space()
k          = 100
eps        = 1.e-8
alg_grads  = rbm_real.compute_batch_gradients(k, data, data)

def compute_numerical_kl(target_psi, vis, Z):
    KL = 0.0
    for i in range(len(vis)):
        KL += ((target_psi[i])**2)*((target_psi[i])**2).log()
        KL -= ((target_psi[i])**2)*(rbm_real.rbm_module.probability(vis[i], Z)).log().item()

    return KL

def compute_numerical_NLL(data, Z):
    NLL = 0
    batch_size = len(data)

    for i in range(batch_size):
        NLL -= (rbm_real.rbm_module.probability(data[i], Z)).log().item()/batch_size

    return NLL

def test_gradient(target_psi, param, alg_grad, vis, data, Z):
    print("Numerical NLL\t Numerical KL\t Alg")
    for i in range(len(param)):
        param[i] += eps
        
        Z     = rbm_real.rbm_module.partition(vis)
        KL_p  = compute_numerical_kl(target_psi, vis, Z)
        NLL_p = compute_numerical_NLL(data, Z)

        param[i] -= 2*eps

        Z     = rbm_real.rbm_module.partition(vis)
        KL_m  = compute_numerical_kl(target_psi, vis, Z)
        NLL_m = compute_numerical_NLL(data, Z)

        param[i] += eps

        num_gradKL  = (KL_p - KL_m) / (2*eps)
        num_gradNLL = (NLL_p - NLL_m) / (2*eps)

        print("{: 10.8f}\t{: 10.8f}\t{: 10.8f}"
              .format(num_gradNLL, num_gradKL, alg_grad[i]))

def test_gradients(data, vis, eps):
    w_grad   = alg_grads['rbm_module']['weights']
    v_b_grad = alg_grads['rbm_module']['visible_bias']
    h_b_grad = alg_grads['rbm_module']['hidden_bias']

    Z = rbm_real.rbm_module.partition(vis)

    flat_weights      = rbm_real.rbm_module.weights.data.view(-1)
    flat_weights_grad = w_grad.view(-1)

    print("Testing visible bias...")
    test_gradient(target_psi, rbm_real.rbm_module.visible_bias, v_b_grad, vis, data, Z)
    print("\nTesting hidden bias...")
    test_gradient(target_psi, rbm_real.rbm_module.hidden_bias, h_b_grad, vis, data, Z)
    print("\nTesting weights...")
    test_gradient(target_psi, flat_weights, flat_weights_grad, vis, data, Z)

test_gradients(data, vis, eps)
