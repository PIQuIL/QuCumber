from qucumber.rbm import ComplexRBM
from qucumber import unitaries
from qucumber import cplx

import torch
import numpy as np

data                    = torch.tensor(np.loadtxt(
                           'tools/benchmarks/data/2qubits_complex_RBMlike/2qubits_train_samples.txt'), 
                           dtype = torch.double)
basis_data              = np.loadtxt('tools/benchmarks/data/2qubits_complex_RBMlike/2qubits_train_bases.txt', dtype=str)

unitary_dict            = unitaries.create_dict()

full_unitary_file       = np.loadtxt('tools/benchmarks/data/2qubits_complex/2qubits_unitaries.txt')
full_unitary_dictionary = {}

# Dictionary for true wavefunctions
basis_list     = ['Z' 'Z', 'X' 'Z', 'Z' 'X', 'Y' 'Z', 'Z' 'Y']
psi_file       = np.loadtxt('tools/benchmarks/data/2qubits_complex_RBMlike/2qubits_psi.txt')
psi_dictionary = {}

for i in range(len(basis_list)):
    # 5 possible wavefunctions: ZZ, XZ, ZX, YZ, ZY
    psi      = torch.zeros(2, 2**data.shape[-1], dtype=torch.double)
    psi_real = torch.tensor(psi_file[i*4:(i*4+4),0], dtype=torch.double)
    psi_imag = torch.tensor(psi_file[i*4:(i*4+4),1], dtype=torch.double)
    psi[0]   = psi_real
    psi[1]   = psi_imag
    
    psi_dictionary[basis_list[i]] = psi

    full_unitary      = torch.zeros(2, 2**data.shape[-1],
                                    2**data.shape[-1],
                                    dtype=torch.double)
    full_unitary_real = torch.tensor(full_unitary_file[i*8:(i*8+4)],
                                     dtype=torch.double)
    full_unitary_imag = torch.tensor(full_unitary_file[(i*8+4):(i*8+8)],
                                     dtype=torch.double)
    full_unitary[0]   = full_unitary_real
    full_unitary[1]   = full_unitary_imag
    full_unitary_dictionary[basis_list[i]] = full_unitary

num_visible      = data.shape[-1]
num_hidden_amp   = num_visible
num_hidden_phase = num_visible


rbm_complex = ComplexRBM(full_unitaries=full_unitary_dictionary,
                         psi_dictionary=psi_dictionary,
                         num_visible=num_visible,
                         num_hidden_amp=num_hidden_amp,
                         num_hidden_phase=num_hidden_phase)

vis         = rbm_complex.rbm_amp.generate_visible_space()
k           = 100
eps         = 1.e-8
alg_grads   = rbm_complex.compute_batch_gradients(unitary_dict, k, data, data, basis_data, basis_data)

def compute_numerical_KL(visible_space, Z):
    '''Computes the total KL divergence.
    '''
    KL = 0.0
    basis_list = ['Z' 'Z', 'X' 'Z', 'Z' 'X', 'Y' 'Z', 'Z' 'Y']

    # Wavefunctions (RBM and true) in the computational basis.
    # psi_ZZ      = self.normalized_wavefunction(visible_space)
    # true_psi_ZZ = self.get_true_psi('ZZ')

    #Compute the KL divergence for the non computational bases.
    for i in range(len(basis_list)):
        rotated_RBM_psi = cplx.MV_mult(
            full_unitary_dictionary[basis_list[i]],
            rbm_complex.normalized_wavefunction(visible_space, Z)).view(2,-1)
        rotated_true_psi = rbm_complex.get_true_psi(basis_list[i]).view(2,-1)

        #print ("RBM >>> ", rotated_RBM_psi,"\n norm >>> ",cplx.norm(cplx.inner_prod(rotated_RBM_psi, rotated_RBM_psi)))
        #print ("True >> ", rotated_true_psi)

        for j in range(len(visible_space)):
            elementof_rotated_RBM_psi = torch.tensor(
                                        [rotated_RBM_psi[0][j],
                                         rotated_RBM_psi[1][j]]
                                        ).view(2, 1)

            elementof_rotated_true_psi = torch.tensor(
                                          [rotated_true_psi[0][j],
                                           rotated_true_psi[1][j]] 
                                          ).view(2, 1)

            norm_true_psi = cplx.norm(cplx.inner_prod(
                                      elementof_rotated_true_psi,
                                      elementof_rotated_true_psi))

            norm_RBM_psi = cplx.norm(cplx.inner_prod(
                                     elementof_rotated_RBM_psi,
                                     elementof_rotated_RBM_psi))
            '''
            if norm_true_psi < 0.01 or norm_RBM_psi < 0.01:
                print ('True >>> ',norm_true_psi)
                print ('RBM >>> ', norm_RBM_psi)
            '''
            # TODO: numerical grads are NAN here if I don't do this if statement (july 16)
            #if norm_true_psi>0.0 and norm_RBM_psi>0.0:
            #print ('Basis      : ',basis_list[i])
            #print ("Plus term  : ",norm_true_psi*torch.log(norm_true_psi))
            #print ("Minus term : ",norm_true_psi*torch.log(norm_RBM_psi),'\n')

            KL += norm_true_psi*torch.log(norm_true_psi)
            KL -= norm_true_psi*torch.log(norm_RBM_psi)

    #print ('KL >>> ',KL)

    return KL

def compute_numerical_NLL(batch, Z):
    NLL = 0.0
    batch_size = len(batch)

    for i in range(len(batch)):
        NLL -= (rbm_complex.rbm_amp.probability(batch[i], Z)).log().item()/batch_size       

    return NLL 

def compute_numerical_gradient(batch, visible_space, param, alg_grad, Z):
    eps = 1.e-8
    print("Numerical NLL\t Numerical KL\t Alg.")

    for i in range(len(param)):

        param[i].data += eps
        Z = rbm_complex.rbm_amp.partition(visible_space)
        NLL_pos = compute_numerical_NLL(batch, Z)
        KL_pos  = compute_numerical_KL(visible_space, Z)

        param[i].data -= 2*eps
        Z = rbm_complex.rbm_amp.partition(visible_space)
        NLL_neg = compute_numerical_NLL(batch, Z)
        KL_neg  = compute_numerical_KL(visible_space, Z)

        param[i].data += eps

        num_gradKL  = (KL_pos - KL_neg) / (2*eps)
        num_gradNLL = (NLL_pos - NLL_neg) / (2*eps)

        print("{: 10.8f}\t{: 10.8f}\t{: 10.8f}\t"
              .format(num_gradNLL, num_gradKL, alg_grad[i]))

def test_gradients(batch, visible_space, k, alg_grads):
    # Must have negative sign because the compute_batch_grads returns the neg of the grads.
    # key_list = ["weights_amp", "visible_bias_amp", "hidden_bias_amp", "weights_phase", "visible_bias_phase", "hidden_bias_phase"]

    flat_weights_amp   = rbm_complex.rbm_amp.weights.data.view(-1)
    flat_weights_phase = rbm_complex.rbm_phase.weights.data.view(-1)

    flat_grad_weights_amp   = alg_grads["rbm_amp"]["weights"].view(-1)
    flat_grad_weights_phase = alg_grads["rbm_phase"]["weights"].view(-1)

    Z = rbm_complex.rbm_amp.partition(visible_space)

    print('-------------------------------------------------------------------------------')

    print('Weights amp gradient')
    compute_numerical_gradient(
        batch, visible_space, flat_weights_amp, -flat_grad_weights_amp, Z)
    print ('\n')

    print('Visible bias amp gradient')
    compute_numerical_gradient(
        batch, visible_space, rbm_complex.rbm_amp.visible_bias, -alg_grads["rbm_amp"]["visible_bias"], Z)
    print ('\n')

    print('Hidden bias amp gradient')
    compute_numerical_gradient(
        batch, visible_space, rbm_complex.rbm_amp.hidden_bias, -alg_grads["rbm_amp"]["hidden_bias"], Z)
    print ('\n')

    print('Weights phase gradient')
    compute_numerical_gradient(
        batch, visible_space, flat_weights_phase, -flat_grad_weights_phase, Z)
    print ('\n')

    print('Visible bias phase gradient')
    compute_numerical_gradient(
        batch, visible_space, rbm_complex.rbm_phase.visible_bias, -alg_grads["rbm_phase"]["visible_bias"], Z)
    print ('\n')

    print('Hidden bias phase gradient')
    compute_numerical_gradient(
        batch, visible_space, rbm_complex.rbm_phase.hidden_bias, -alg_grads["rbm_phase"]["hidden_bias"], Z)

test_gradients(data, vis, k, alg_grads)
