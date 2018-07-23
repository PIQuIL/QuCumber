from qucumber.rbm import BinomialRBM

import torch
import numpy as np
from positive_wavefunction import PositiveWavefunction
from quantum_reconstruction import QuantumReconstruction

def generate_visible_space(num_visible):
    """Generates all possible visible states.

    :returns: A tensor of all possible spin configurations.
    :rtype: torch.Tensor
    """
    space = torch.zeros((1 << num_visible, num_visible),
                        device="cpu", dtype=torch.double)
    for i in range(1 << num_visible):
        d = i
        for j in range(num_visible):
            d, r = divmod(d, 2)
            space[i, nn_state.num_visible - j - 1] = int(r)

    return space

def partition(nn_state,visible_space):
    """The natural logarithm of the partition function of the RBM.

    :param visible_space: A rank 2 tensor of the entire visible space.
    :type visible_space: torch.Tensor

    :returns: The natural log of the partition function.
    :rtype: torch.Tensor
    """
    free_energies = -nn_state.rbm.effective_energy(visible_space)
    max_free_energy = free_energies.max()

    f_reduced = free_energies - max_free_energy
    logZ = max_free_energy + f_reduced.exp().sum().log()
    return logZ.exp()
    
    #return logZ

def probability(nn_state,v, Z):
    """Evaluates the probability of the given vector(s) of visible
    units; NOT RECOMMENDED FOR RBMS WITH A LARGE # OF VISIBLE UNITS

    :param v: The visible states.
    :type v: torch.Tensor
    :param Z: The partition function.
    :type Z: float

    :returns: The probability of the given vector(s) of visible units.
    :rtype: torch.Tensor
    """
    return (nn_state.psi(v))**2 / Z


def compute_numerical_kl(target_psi, vis, Z):
    KL = 0.0
    for i in range(len(vis)):
        KL += ((target_psi[i])**2)*((target_psi[i])**2).log()
        KL -= ((target_psi[i])**2)*(probability(nn_state,vis[i], Z)).log().item()

    return KL

def compute_numerical_NLL(data, Z):
    NLL = 0
    batch_size = len(data)

    for i in range(batch_size):
        NLL -= (probability(nn_state,data[i], Z)).log().item()/batch_size

    return NLL

def algorithmic_gradKL(nn_state,target_psi,vis):
    grad_KL={}
    for rbmType in nn_state.gradient(vis[0]):
        grad_KL[rbmType] = {}
        for pars in nn_state.gradient(vis[0])[rbmType]:
            grad_KL[rbmType][pars]=0
    Z = partition(nn_state,vis)
    for i in range(len(vis)):
        for rbmType in nn_state.gradient(vis[i]):
            for pars in nn_state.gradient(vis[i])[rbmType]:    
                grad_KL[rbmType][pars] += ((target_psi[i])**2)*nn_state.gradient(vis[i])[rbmType][pars]            
                grad_KL[rbmType][pars] -= probability(nn_state,vis[i], Z)*nn_state.gradient(vis[i])[rbmType][pars]
    return grad_KL            

def algorithmic_gradNLL(qr,data,k):
   
    grad_NLL = qr.compute_batch_gradients(k, data, data)
    #for rbmType in nn_state.gradient(vis[0]):
    #    grad_KL[rbmType] = {}
    #    for pars in nn_state.gradient(vis[0])[rbmType]:
    #        grad_KL[rbmType][pars]=0
    #Z = partition(nn_state,vis)
    #for i in range(len(vis)):
    #    for rbmType in nn_state.gradient(vis[i]):
    #        for pars in nn_state.gradient(vis[i])[rbmType]:    
    #            grad_KL[rbmType][pars] += ((target_psi[i])**2)*nn_state.gradient(vis[i])[rbmType][pars]            
    #            grad_KL[rbmType][pars] -= probability(nn_state,vis[i], Z)*nn_state.gradient(vis[i])[rbmType][pars]
    return grad_NLL            
    
    
def numeric_gradKL(nn_state,target_psi, param, vis):
    num_gradKL = []
    for i in range(len(param)):
        param[i] += eps
        
        Z     = partition(nn_state,vis)
        KL_p  = compute_numerical_kl(target_psi, vis, Z)
        #NLL_p = compute_numerical_NLL(data, Z)

        param[i] -= 2*eps

        Z     = partition(nn_state,vis)
        KL_m  = compute_numerical_kl(target_psi, vis, Z)
        #NLL_m = compute_numerical_NLL(data, Z)

        param[i] += eps

        num_gradKL.append( (KL_p - KL_m) / (2*eps) )
        #num_gradNLL = (NLL_p - NLL_m) / (2*eps)
    return num_gradKL

def numeric_gradNLL(nn_state, param,data):
    num_gradNLL = []
    for i in range(len(param)):
        param[i] += eps
        
        Z     = partition(nn_state,vis)
        NLL_p = compute_numerical_NLL(data, Z)

        param[i] -= 2*eps

        Z     = partition(nn_state,vis)
        NLL_m = compute_numerical_NLL(data, Z)

        param[i] += eps

        num_gradNLL.append( (NLL_p - NLL_m) / (2*eps) )
    return num_gradNLL

def test_gradients(qr,target_psi,data, vis, eps,k):
    nn_state = qr.nn_state
    alg_grad_KL = algorithmic_gradKL(nn_state,target_psi,vis)
    alg_grad_NLL = algorithmic_gradNLL(qr,data,k)
    
    flat_weights      = nn_state.rbm.weights.data.view(-1)
    flat_weights_grad_KL = alg_grad_KL["rbm_am"]["weights"].view(-1)
    flat_weights_grad_NLL = alg_grad_NLL["rbm_am"]["weights"].view(-1)
    num_grad_KL = numeric_gradKL(nn_state,target_psi,flat_weights,vis)
    num_grad_NLL = numeric_gradNLL(nn_state,flat_weights,data)
    print("\nTesting weights...")
    print("Numerical KL\tAlg KL\t\t\tNumerical NLL\tAlg NLL")
    for i in range(len(flat_weights)):
        print("{: 10.8f}\t{: 10.8f}\t\t".format(num_grad_KL[i],flat_weights_grad_KL[i]),end="", flush=True)
        print("{: 10.8f}\t{: 10.8f}\t\t".format(num_grad_NLL[i],flat_weights_grad_NLL[i]))
   
    num_grad_KL = numeric_gradKL(nn_state,target_psi,nn_state.rbm.visible_bias,vis)
    num_grad_NLL = numeric_gradNLL(nn_state,nn_state.rbm.visible_bias,data)
    print("\nTesting visible bias...")
    print("Numerical KL\tAlg KL\t\t\tNumerical NLL\tAlg NLL")
    for i in range(len(nn_state.rbm.visible_bias)):
        print("{: 10.8f}\t{: 10.8f}\t\t".format(num_grad_KL[i],alg_grad_KL["rbm_am"]["visible_bias"][i]),end="", flush=True)
        print("{: 10.8f}\t{: 10.8f}\t\t".format(num_grad_NLL[i],alg_grad_NLL["rbm_am"]["visible_bias"][i]))
 
    num_grad_KL = numeric_gradKL(nn_state,target_psi,nn_state.rbm.hidden_bias,vis)
    num_grad_NLL = numeric_gradNLL(nn_state,nn_state.rbm.hidden_bias,data)
    print("\nTesting hidden bias...")
    print("Numerical KL\tAlg KL\t\t\tNumerical NLL\tAlg NLL")
    for i in range(len(nn_state.rbm.hidden_bias)):
        print("{: 10.8f}\t{: 10.8f}\t\t".format(num_grad_KL[i],alg_grad_KL["rbm_am"]["hidden_bias"][i]),end="", flush=True)
        print("{: 10.8f}\t{: 10.8f}\t\t".format(num_grad_NLL[i],alg_grad_NLL["rbm_am"]["hidden_bias"][i]))

    print('')


# MAIN
data       = torch.tensor(np.loadtxt(
                          '../examples/tfim1d_N10_train_samples.txt'), 
                          dtype = torch.double)
target_psi = torch.tensor(np.loadtxt('../examples/tfim1d_N10_psi.txt'), 
                          dtype = torch.double)

nh = 2
seed=1234
nn_state = PositiveWavefunction(num_visible=data.shape[-1],
                                num_hidden=nh, seed=seed)

qr = QuantumReconstruction(nn_state)

vis        = generate_visible_space(data.shape[-1])
k          = 100
eps        = 1.e-8

#alg_grads  = nn_state.compute_batch_gradients(k, data, data)
algorithmic_gradKL(nn_state,target_psi,vis)
test_gradients(qr,target_psi,data[0:1000], vis, eps,k)#,alg_grads)
