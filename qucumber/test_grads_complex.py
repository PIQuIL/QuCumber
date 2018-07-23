#from qucumber.rbm import ComplexRBM
from qucumber import unitaries
from qucumber import cplx

import torch
import numpy as np
from complex_wavefunction import ComplexWavefunction
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
            space[i, num_visible - j - 1] = int(r)
    return space

def partition(nn_state,visible_space):
    """The natural logarithm of the partition function of the RBM.

    :param visible_space: A rank 2 tensor of the entire visible space.
    :type visible_space: torch.Tensor

    :returns: The natural log of the partition function.
    :rtype: torch.Tensor
    """
    free_energies = -nn_state.rbm_am.effective_energy(visible_space)
    max_free_energy = free_energies.max()

    f_reduced = free_energies - max_free_energy
    logZ = max_free_energy + f_reduced.exp().sum().log()
    return logZ.exp()
    
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
    return (nn_state.amplitude(v))**2 / Z

#def algorithmic_gradKL(nn_state,target_psi,vis):
#    grad_KL={}
#    for rbmType in nn_state.gradient(vis[0]):
#        grad_KL[rbmType] = {}
#        for pars in nn_state.gradient(vis[0])[rbmType]:
#            grad_KL[rbmType][pars]=0
#    Z = partition(nn_state,vis)
#    for i in range(len(vis)):
#        for rbmType in nn_state.gradient(vis[i]):
#            for pars in nn_state.gradient(vis[i])[rbmType]:    
#                grad_KL[rbmType][pars] += ((target_psi[i])**2)*nn_state.gradient(vis[i])[rbmType][pars]            
#                grad_KL[rbmType][pars] -= probability(nn_state,vis[i], Z)*nn_state.gradient(vis[i])[rbmType][pars]
#    return grad_KL            
#

def load_target_psi(bases,path_to_target_psi):
    psi_data = np.loadtxt(path_to_target_psi)
    psi_dict={}
    D = int(len(psi_data)/float(len(bases)))
    for b in range(len(bases)):
        # 5 possible wavefunctions: ZZ, XZ, ZX, YZ, ZY
        psi      = torch.zeros(2,D, dtype=torch.double)
        psi_real = torch.tensor(psi_data[b*D:D*(b+1),0], dtype=torch.double)
        psi_imag = torch.tensor(psi_data[b*D:D*(b+1),1], dtype=torch.double)
        psi[0]   = psi_real
        psi[1]   = psi_imag
        
        psi_dict[bases[b]] = psi
    
    return psi_dict

def load_full_unitaries(bases,path_to_full_unitaries):
    unitaries_data = np.loadtxt(path_to_full_unitaries)
    full_unitaries_dict = {}
    #D = 1 << len(bases[0])
    #print(D)
    for b in range(len(bases)):
        full_unitary      = torch.zeros(2,D,D,dtype=torch.double)
        full_unitary_real = torch.tensor(unitaries_data[2*D*b:D*(2*b+1)],
                                         dtype=torch.double)
        full_unitary_imag = torch.tensor(unitaries_data[D*(2*b+1):2*D*(b+1)],
                                         dtype=torch.double)
        full_unitary[0]   = full_unitary_real
        full_unitary[1]   = full_unitary_imag
        full_unitaries_dict[bases[b]] = full_unitary
    return full_unitaries_dict


def rotate_psi_full(basis,full_unitary_dict,psi):

    U = full_unitary_dict[basis]
    Upsi = cplx.MV_mult(U,psi)
    return Upsi

def rotate_psi(nn_state,basis,unitary_dict):
    N=nn_state.num_visible
    v = torch.zeros(N, dtype=torch.double)
    psi_r = torch.zeros(2,1<<N,dtype=torch.double)
    
    for x in range(1<<N):
        Upsi = torch.zeros(2, dtype=torch.double)
        num_nontrivial_U = 0
        nontrivial_sites = []
        for j in range(N):
            if (basis[j] is not 'Z'):
                num_nontrivial_U += 1
                nontrivial_sites.append(j)
        sub_state = generate_visible_space(num_nontrivial_U)
         
        for xp in range(1<<num_nontrivial_U):
            cnt = 0
            for j in range(N):
                if (basis[j] is not 'Z'):
                    v[j]=sub_state[xp][cnt] 
                    cnt += 1
                else:
                    v[j]=vis[x,j]
            U = torch.tensor([1., 0.], dtype=torch.double)
            for ii in range(num_nontrivial_U):
                tmp = unitary_dict[basis[nontrivial_sites[ii]]][:,int(vis[x][nontrivial_sites[ii]]),int(v[nontrivial_sites[ii]])]
                U = cplx.scalar_mult(U,tmp)
            Upsi += cplx.scalar_mult(U,nn_state.psi(v))
        psi_r[:,x] = Upsi
    return psi_r


def compute_numerical_kl(nn_state,psi_dict,vis,Z,unitary_dict,bases):
    N=nn_state.num_visible 
    psi_r = torch.zeros(2,1<<N,dtype=torch.double)
    KL = 0.0
    for i in range(len(vis)):
        KL += cplx.norm(psi_dict[bases[0]][:,i])*cplx.norm(psi_dict[bases[0]][:,i]).log()
        KL -= cplx.norm(psi_dict[bases[0]][:,i])*(probability(nn_state,vis[i],Z)).log().item()
    for b in range(1,len(bases)):
        psi_r = rotate_psi(nn_state,bases[b],unitary_dict)
        print(psi_r)
        for ii in range(len(vis)):
            if(cplx.norm(psi_dict[bases[b]][:,ii])>0.0):
                KL += cplx.norm(psi_dict[bases[b]][:,ii])*cplx.norm(psi_dict[bases[b]][:,ii]).log()
            KL -= cplx.norm(psi_dict[bases[b]][:,ii])*cplx.norm(psi_r[:,ii]).log()
            KL += cplx.norm(psi_dict[bases[b]][:,ii])*Z.log()

    print(KL)
    return KL


#def test_psi_rotations(bases,unitary_dict,fullunitary_dict,psi_dict,vis):
#    for b in range(1,len(bases)):
#        psi_r = rotate_psi_full(bases[b],fullunitary_dict,psi_dict['ZZ'])
#        print("\n\nBases: %s\n" % bases[b])
#        D = 1 << len(bases[b])
#        psi_alg = rotate_psi(bases[b],unitary_dict,psi_dict['ZZ'])
#        print('\t   Exact \t\t\t\tFullRotation \t\t\t\tAlgorithmic')
#        for j in range(D):
#            print("{: 10.8f}  +  {: 10.8f}\t\t".format(psi_dict[bases[b]][0][j].item(),psi_dict[bases[b]][1][j].item()),end="", flush=True)
#            print("{: 10.8f}  +  {: 10.8f}\t\t".format(psi_r[0][j].item(),psi_r[1][j].item()),end="", flush=True)
#            print("{: 10.8f}  +  {: 10.8f}\t\t".format(psi_alg[0][j].item(),psi_alg[1][j].item()))
#        
#
#

path_to_train_data = '../tools/benchmarks/data/2qubits_complex/2qubits_train_samples.txt'
path_to_train_bases= '../tools/benchmarks/data/2qubits_complex/2qubits_train_bases.txt'
path_to_full_unitaries = '../tools/benchmarks/data/2qubits_complex/2qubits_unitaries.txt'
path_to_bases = '../tools/benchmarks/data/2qubits_complex/2qubits_bases.txt'
path_to_target_psi = '../tools/benchmarks/data/2qubits_complex/2qubits_psi.txt'

train_data = torch.tensor(np.loadtxt(path_to_train_data),dtype = torch.double)
train_bases = np.loadtxt(path_to_train_bases, dtype=str)
unitary_dict = unitaries.create_dict()

num_visible      = train_data.shape[-1]
num_hidden   = num_visible
D = 1<<num_visible
vis = generate_visible_space(num_visible)
bases = []#np.loadtxt(path_to_bases,dtype=str)
with open(path_to_bases) as fin:
    for line in fin:
        bases.append(line.strip())
psi_dict = load_target_psi(bases,path_to_target_psi)
fullunitary_dict = load_full_unitaries(bases,path_to_full_unitaries)

nn_state = ComplexWavefunction(full_unitaries=fullunitary_dict,
                         psi_dictionary=psi_dict,
                         num_visible=num_visible,
                         num_hidden=num_hidden)


Z = partition(nn_state,vis)
compute_numerical_kl(nn_state,psi_dict,vis,Z,unitary_dict,bases)


#print (unitary_dict['X'][:,0,0])

#test_psi_rotations(bases,unitary_dict,fullunitary_dict,psi_dict,vis)
#k           = 100
#eps         = 1.e-8
##alg_grads   = rbm_complex.compute_batch_gradients(unitary_dict, k, data, data, basis_data, basis_data)




#def compute_numerical_KL(visible_space, Z):
#    '''Computes the total KL divergence.
#    '''
#    KL = 0.0
#    basis_list = ['Z' 'Z', 'X' 'Z', 'Z' 'X', 'Y' 'Z', 'Z' 'Y']
#
#    # Wavefunctions (RBM and true) in the computational basis.
#    # psi_ZZ      = self.normalized_wavefunction(visible_space)
#    # true_psi_ZZ = self.get_true_psi('ZZ')
#
#    #Compute the KL divergence for the non computational bases.
#    for i in range(len(basis_list)):
#        rotated_RBM_psi = cplx.MV_mult(
#            full_unitary_dictionary[basis_list[i]],
#            rbm_complex.normalized_wavefunction(visible_space, Z)).view(2,-1)
#        rotated_true_psi = rbm_complex.get_true_psi(basis_list[i]).view(2,-1)
#
#        #print ("RBM >>> ", rotated_RBM_psi,"\n norm >>> ",cplx.norm(cplx.inner_prod(rotated_RBM_psi, rotated_RBM_psi)))
#        #print ("True >> ", rotated_true_psi)
#
#        for j in range(len(visible_space)):
#            elementof_rotated_RBM_psi = torch.tensor(
#                                        [rotated_RBM_psi[0][j],
#                                         rotated_RBM_psi[1][j]]
#                                        ).view(2, 1)
#
#            elementof_rotated_true_psi = torch.tensor(
#                                          [rotated_true_psi[0][j],
#                                           rotated_true_psi[1][j]] 
#                                          ).view(2, 1)
#
#            norm_true_psi = cplx.norm(cplx.inner_prod(
#                                      elementof_rotated_true_psi,
#                                      elementof_rotated_true_psi))
#
#            norm_RBM_psi = cplx.norm(cplx.inner_prod(
#                                     elementof_rotated_RBM_psi,
#                                     elementof_rotated_RBM_psi))
#            '''
#            if norm_true_psi < 0.01 or norm_RBM_psi < 0.01:
#                print ('True >>> ',norm_true_psi)
#                print ('RBM >>> ', norm_RBM_psi)
#            '''
#            # TODO: numerical grads are NAN here if I don't do this if statement (july 16)
#            #if norm_true_psi>0.0 and norm_RBM_psi>0.0:
#            #print ('Basis      : ',basis_list[i])
#            #print ("Plus term  : ",norm_true_psi*torch.log(norm_true_psi))
#            #print ("Minus term : ",norm_true_psi*torch.log(norm_RBM_psi),'\n')
#
#            KL += norm_true_psi*torch.log(norm_true_psi)
#            KL -= norm_true_psi*torch.log(norm_RBM_psi)
#
#    #print ('KL >>> ',KL)
#
#    return KL
#
#def compute_numerical_NLL(batch, Z):
#    NLL = 0.0
#    batch_size = len(batch)
#
#    for i in range(len(batch)):
#        NLL -= (rbm_complex.rbm_amp.probability(batch[i], Z)).log().item()/batch_size       
#
#    return NLL 
#
#def compute_numerical_gradient(batch, visible_space, param, alg_grad, Z):
#    eps = 1.e-8
#    print("Numerical NLL\t Numerical KL\t Alg.")
#
#    for i in range(len(param)):
#
#        param[i].data += eps
#        Z = rbm_complex.rbm_amp.partition(visible_space)
#        NLL_pos = compute_numerical_NLL(batch, Z)
#        KL_pos  = compute_numerical_KL(visible_space, Z)
#
#        param[i].data -= 2*eps
#        Z = rbm_complex.rbm_amp.partition(visible_space)
#        NLL_neg = compute_numerical_NLL(batch, Z)
#        KL_neg  = compute_numerical_KL(visible_space, Z)
#
#        param[i].data += eps
#
#        num_gradKL  = (KL_pos - KL_neg) / (2*eps)
#        num_gradNLL = (NLL_pos - NLL_neg) / (2*eps)
#
#        print("{: 10.8f}\t{: 10.8f}\t{: 10.8f}\t"
#              .format(num_gradNLL, num_gradKL, alg_grad[i]))
#
#def test_gradients(batch, visible_space, k, alg_grads):
#    # Must have negative sign because the compute_batch_grads returns the neg of the grads.
#    # key_list = ["weights_amp", "visible_bias_amp", "hidden_bias_amp", "weights_phase", "visible_bias_phase", "hidden_bias_phase"]
#
#    flat_weights_amp   = rbm_complex.rbm_amp.weights.data.view(-1)
#    flat_weights_phase = rbm_complex.rbm_phase.weights.data.view(-1)
#
#    flat_grad_weights_amp   = alg_grads["rbm_amp"]["weights"].view(-1)
#    flat_grad_weights_phase = alg_grads["rbm_phase"]["weights"].view(-1)
#
#    Z = rbm_complex.rbm_amp.partition(visible_space)
#
#    print('-------------------------------------------------------------------------------')
#
#    print('Weights amp gradient')
#    compute_numerical_gradient(
#        batch, visible_space, flat_weights_amp, -flat_grad_weights_amp, Z)
#    print ('\n')
#
#    print('Visible bias amp gradient')
#    compute_numerical_gradient(
#        batch, visible_space, rbm_complex.rbm_amp.visible_bias, -alg_grads["rbm_amp"]["visible_bias"], Z)
#    print ('\n')
#
#    print('Hidden bias amp gradient')
#    compute_numerical_gradient(
#        batch, visible_space, rbm_complex.rbm_amp.hidden_bias, -alg_grads["rbm_amp"]["hidden_bias"], Z)
#    print ('\n')
#
#    print('Weights phase gradient')
#    compute_numerical_gradient(
#        batch, visible_space, flat_weights_phase, -flat_grad_weights_phase, Z)
#    print ('\n')
#
#    print('Visible bias phase gradient')
#    compute_numerical_gradient(
#        batch, visible_space, rbm_complex.rbm_phase.visible_bias, -alg_grads["rbm_phase"]["visible_bias"], Z)
#    print ('\n')
#
#    print('Hidden bias phase gradient')
#    compute_numerical_gradient(
#        batch, visible_space, rbm_complex.rbm_phase.hidden_bias, -alg_grads["rbm_phase"]["hidden_bias"], Z)
#
#test_gradients(data, vis, k, alg_grads)
