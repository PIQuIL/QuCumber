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
def compute_numerical_kl(nn_state,psi_dict,vis,Z,unitary_dict,bases):
    N=nn_state.num_visible 
    psi_r = torch.zeros(2,1<<N,dtype=torch.double)
    KL = 0.0
    for i in range(len(vis)):
        KL += cplx.norm(psi_dict[bases[0]][:,i])*cplx.norm(psi_dict[bases[0]][:,i]).log()
        KL -= cplx.norm(psi_dict[bases[0]][:,i])*(probability(nn_state,vis[i],Z)).log().item()
    for b in range(1,len(bases)):
        psi_r = rotate_psi(nn_state,bases[b],unitary_dict)
        for ii in range(len(vis)):
            if(cplx.norm(psi_dict[bases[b]][:,ii])>0.0):
                KL += cplx.norm(psi_dict[bases[b]][:,ii])*cplx.norm(psi_dict[bases[b]][:,ii]).log()
            KL -= cplx.norm(psi_dict[bases[b]][:,ii])*cplx.norm(psi_r[:,ii]).log()
            KL += cplx.norm(psi_dict[bases[b]][:,ii])*Z.log()

    return KL


def numeric_gradKL(param,nn_state,psi_dict,vis,unitary_dict,bases):
    num_gradKL = []
    for i in range(len(param)):
        param[i] += eps
        
        Z     = partition(nn_state,vis)
        KL_p  = compute_numerical_kl(nn_state,psi_dict, vis, Z,unitary_dict,bases)

        param[i] -= 2*eps

        Z     = partition(nn_state,vis)
        KL_m  = compute_numerical_kl(nn_state,psi_dict, vis, Z,unitary_dict,bases)

        param[i] += eps

        num_gradKL.append( (KL_p - KL_m) / (2*eps) )
    return num_gradKL

def algorithmic_gradKL(nn_state,psi_dict,vis,unitary_dict,bases):
    grad_KL={}
    for net in nn_state.networks:
        tmp = {}
        rbm = getattr(nn_state, net)
        for par in rbm.state_dict():
            tmp[par]=0.0    
        grad_KL[net] = tmp
    Z = partition(nn_state,vis)
    
    for i in range(len(vis)):
        for par in rbm.state_dict():
            grad_KL['rbm_am'][par] += cplx.norm(psi_dict[bases[0]][:,i])*nn_state.gradient(vis[i])['rbm_am'][par]
            grad_KL['rbm_am'][par] -= probability(nn_state,vis[i], Z)*nn_state.gradient(vis[i])['rbm_am'][par]

    for b in range(1,len(bases)):
        psi_r = rotate_psi(nn_state,bases[b],unitary_dict)
        for i in range(len(vis)):
            rotated_grad = nn_state.rotate_grad(bases[b],vis[i],unitary_dict)
            for net in nn_state.networks:
                for par in rbm.state_dict():
                    grad_KL[net][par] += cplx.norm(psi_dict[bases[b]][:,i])*rotated_grad[net][par]
            for par in rbm.state_dict():
                grad_KL['rbm_am'][par] -= probability(nn_state,vis[i], Z)*nn_state.gradient(vis[i])['rbm_am'][par]
    return grad_KL            


#def compute_numerical_NLL(batch, Z):
#    NLL = 0.0
#    batch_size = len(batch)
#
#    for i in range(len(batch)):
#        NLL -= (rbm_complex.rbm_amp.probability(batch[i], Z)).log().item()/batch_size       
#
#    return NLL 
#


#def test_gradients(qr,target_psi,data, vis, eps,k):
def test_gradients(nn_state,psi_dict,unitary_dict,bases,vis,eps):
    #alg_grad_NLL = algorithmic_gradNLL(qr,data,k)
    alg_grad_KL = algorithmic_gradKL(nn_state,psi_dict,vis,unitary_dict,bases)
    for net in nn_state.networks:
        print('\n\nRBM: %s' %net) 
        #alg_grad_KL = algorithmic_gradKL(nn_state,net,psi_dict,vis,unitary_dict,bases)
        rbm = getattr(nn_state, net)
        flat_weights = rbm.weights.data.view(-1)
        #print(alg_grad_KL[net])
        flat_weights_grad_KL = alg_grad_KL[net]["weights"].view(-1)
        #flat_weights_grad_NLL = alg_grad_NLL["rbm_am"]["weights"].view(-1)
        num_grad_KL=numeric_gradKL(flat_weights,nn_state,psi_dict,vis,unitary_dict,bases)
        #num_grad_NLL = numeric_gradNLL(nn_state,flat_weights,data)
        print("\nTesting weights...")
        print("Numerical KL\tAlg KL\t\t\tNumerical NLL\tAlg NLL")
        for i in range(len(flat_weights)):
            print("{: 10.8f}\t{: 10.8f}\t\t".format(num_grad_KL[i],flat_weights_grad_KL[i]))#,end="", flush=True)
        ##    print("{: 10.8f}\t{: 10.8f}\t\t".format(num_grad_NLL[i],flat_weights_grad_NLL[i]))
        #
        num_grad_KL=numeric_gradKL(rbm.visible_bias,nn_state,psi_dict,vis,unitary_dict,bases)
        ##num_grad_NLL = numeric_gradNLL(nn_state,nn_state.rbm_am.visible_bias,data)
        print("\nTesting visible bias...")
        print("Numerical KL\tAlg KL\t\t\tNumerical NLL\tAlg NLL")
        ##for i in range(len(rbm.visible_bias)):
        ##    print("{: 10.8f}".format(num_grad_KL[i]))
        for i in range(len(rbm.visible_bias)):
            print("{: 10.8f}\t{: 10.8f}\t\t".format(num_grad_KL[i],alg_grad_KL[net]["visible_bias"][i]))#,end="", flush=True)
        ##    print("{: 10.8f}\t{: 10.8f}\t\t".format(num_grad_NLL[i],alg_grad_NLL["rbm_am"]["visible_bias"][i]))
 
        
        num_grad_KL=numeric_gradKL(rbm.hidden_bias,nn_state,psi_dict,vis,unitary_dict,bases)
        ##num_grad_NLL = numeric_gradNLL(nn_state,nn_state.rbm_am.hidden_bias,data)
        print("\nTesting hidden bias...")
        print("Numerical KL\tAlg KL\t\t\tNumerical NLL\tAlg NLL")
        #for i in range(len(rbm.hidden_bias)):
        ##    print("{: 10.8f}".format(num_grad_KL[i]))
        for i in range(len(rbm.hidden_bias)):
            print("{: 10.8f}\t{: 10.8f}\t\t".format(num_grad_KL[i],alg_grad_KL[net]["hidden_bias"][i]))#,end="", flush=True)
        ##    print("{: 10.8f}\t{: 10.8f}\t\t".format(num_grad_NLL[i],alg_grad_NLL["rbm_am"]["hidden_bias"][i]))

    print('')






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

nn_state = ComplexWavefunction(num_visible=num_visible,
                               num_hidden=num_hidden)
k           = 100
eps         = 1.e-8

test_gradients(nn_state,psi_dict,unitary_dict,bases,vis,eps)
