import sys
sys.path.append('../')
from utils import unitaries
from utils import cplx
import torch
import numpy as np
from complex_wavefunction import ComplexWavefunction
from quantum_reconstruction import QuantumReconstruction
import pickle
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

def load_target_psi(bases,psi_data):
    psi_dict={}
    D = int(len(psi_data)/float(len(bases)))
    for b in range(len(bases)):
        
        psi      = torch.zeros(2,D, dtype=torch.double)
        psi_real = torch.tensor(psi_data[b*D:D*(b+1),0], dtype=torch.double)
        psi_imag = torch.tensor(psi_data[b*D:D*(b+1),1], dtype=torch.double)
        psi[0]   = psi_real
        psi[1]   = psi_imag
        psi_dict[bases[b]] = psi
    
    return psi_dict

#def load_target_psi(psi_data):
#    D = int(len(psi_data))
#    psi      = torch.zeros(2,D, dtype=torch.double)
#    psi_real = torch.tensor(psi_data[:,0], dtype=torch.double)
#    psi_imag = torch.tensor(psi_data[:,1], dtype=torch.double)
#    psi[0]   = psi_real
#    psi[1]   = psi_imag
#    return psi

def transform_bases(bases_data):
    bases = []
    for i in range(len(bases_data)):
        tmp = ""
        for j in range(len(bases_data[i])):
            if bases_data[i][j] is not " ":
                tmp += bases_data[i][j]
        bases.append(tmp)
    return bases

#def BuildPsiDict(N,bases,target_psi,unitary_dict,vis):
#    
#    psi_dict = {}
#    D = int(len(target_psi))
#    psi_dict[bases[0]] = target_psi
#    for b in range(1,len(bases)):
#        print(bases)
#        psi      = torch.zeros(2,D, dtype=torch.double)
#        psi_r = rotate_target_psi(N,target_psi,bases[b],unitary_dict,vis)
#        #psi[0] = torch.tensor(psi_r[0,:], dtype=torch.double)
#        #psi[1] = torch.tensor(psi_r[1,:], dtype=torch.double)
#        print(psi_r[0,:])
#        print()
#        psi_dict[bases[b]] = psi
#    return psi_dict

def rotate_psi_full(basis,full_unitary_dict,psi):

    U = full_unitary_dict[basis]
    Upsi = cplx.matmul(U,psi)
    return Upsi

#def rotate_target_psi(N,psi,basis,unitary_dict,vis):
#    v = torch.zeros(N, dtype=torch.double)
#    psi_r = torch.zeros(2,1<<N,dtype=torch.double)
#    for x in range(1<<N):
#        Upsi = torch.zeros(2, dtype=torch.double)
#        psi_torch = torch.zeros(2, dtype=torch.double)
#        num_nontrivial_U = 0
#        nontrivial_sites = []
#        for j in range(N):
#            if (basis[j] is not 'Z'):
#                num_nontrivial_U += 1
#                nontrivial_sites.append(j)
#        sub_state = generate_visible_space(num_nontrivial_U)
#         
#        for xp in range(1<<num_nontrivial_U):
#            cnt = 0
#            for j in range(N):
#                if (basis[j] is not 'Z'):
#                    v[j]=sub_state[xp][cnt] 
#                    cnt += 1
#                else:
#                    v[j]=vis[x,j]
#            U = torch.tensor([1., 0.], dtype=torch.double)
#            for ii in range(num_nontrivial_U):
#                tmp = unitary_dict[basis[nontrivial_sites[ii]]][:,int(vis[x][nontrivial_sites[ii]]),int(v[nontrivial_sites[ii]])]
#                U = cplx.scalar_mult(U,tmp)
#            Upsi += cplx.scalar_mult(U,psi[:,x])
#        psi_r[:,x] = Upsi
#    return psi_r

def rotate_psi(nn_state,basis,unitary_dict,vis):
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

def compute_numerical_NLL(nn_state,data_samples,data_bases,Z,unitary_dict,vis):
    NLL = 0
    batch_size = len(data_samples)
    b_flag = 0
    for i in range(batch_size):
        bitstate = []
        for j in range(nn_state.num_visible):
            ind = 0
            if (data_bases[i][j] != 'Z'):
                b_flag = 1
            bitstate.append(int(data_samples[i,j].item()))
        ind = int("".join(str(i) for i in bitstate), 2)
        if (b_flag == 0): 
            NLL -= (probability(nn_state,data_samples[i], Z)).log().item()/batch_size
        else:
            psi_r = rotate_psi(nn_state,data_bases[i],unitary_dict,vis)
            NLL -= (cplx.norm(psi_r[:,ind]).log()-Z.log()).item()/batch_size
    return NLL

def compute_numerical_kl(nn_state,psi_dict,vis,Z,unitary_dict,bases):
    N=nn_state.num_visible 
    psi_r = torch.zeros(2,1<<N,dtype=torch.double)
    KL = 0.0
    for i in range(len(vis)):
        KL += cplx.norm(psi_dict[bases[0]][:,i])*cplx.norm(psi_dict[bases[0]][:,i]).log()/float(len(bases))
        KL -= cplx.norm(psi_dict[bases[0]][:,i])*(probability(nn_state,vis[i],Z)).log().item()/float(len(bases))
    for b in range(1,len(bases)):
        psi_r = rotate_psi(nn_state,bases[b],unitary_dict,vis)
        for ii in range(len(vis)):
            if(cplx.norm(psi_dict[bases[b]][:,ii])>0.0):
                KL += cplx.norm(psi_dict[bases[b]][:,ii])*cplx.norm(psi_dict[bases[b]][:,ii]).log()/float(len(bases))
            KL -= cplx.norm(psi_dict[bases[b]][:,ii])*cplx.norm(psi_r[:,ii]).log()/float(len(bases))
            KL += cplx.norm(psi_dict[bases[b]][:,ii])*Z.log()/float(len(bases))

    return KL

def algorithmic_gradNLL(qr,data_samples,data_bases,k):
    qr.nn_state.set_visible_layer(data_samples)
    return qr.compute_batch_gradients(k, data_samples,data_bases)

def numeric_gradNLL(nn_state,data_samples,data_bases,unitary_dict,param,vis,eps):
    
    num_gradNLL = []
    for i in range(len(param)):
        param[i] += eps
        
        Z     = partition(nn_state,vis)
        NLL_p = compute_numerical_NLL(nn_state,data_samples,data_bases,Z,unitary_dict,vis)

        param[i] -= 2*eps

        Z     = partition(nn_state,vis)
        NLL_m = compute_numerical_NLL(nn_state,data_samples,data_bases,Z,unitary_dict,vis)

        param[i] += eps

        num_gradNLL.append( (NLL_p - NLL_m) / (2*eps) )
    return num_gradNLL

def numeric_gradKL(param,nn_state,psi_dict,vis,unitary_dict,bases,eps):
    
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

    grad_KL = [torch.zeros(nn_state.rbm_am.num_pars,dtype=torch.double),torch.zeros(nn_state.rbm_ph.num_pars,dtype=torch.double)]
    Z = partition(nn_state,vis)
    
    for i in range(len(vis)):
        grad_KL[0] += cplx.norm(psi_dict[bases[0]][:,i])*nn_state.rbm_am.effective_energy_gradient(vis[i])/float(len(bases))
        grad_KL[0] -= probability(nn_state,vis[i], Z)*nn_state.rbm_am.effective_energy_gradient(vis[i])/float(len(bases))

    for b in range(1,len(bases)):
        psi_r = rotate_psi(nn_state,bases[b],unitary_dict,vis)
        for i in range(len(vis)):
            rotated_grad = nn_state.gradient(bases[b],vis[i])
            grad_KL[0] += cplx.norm(psi_dict[bases[b]][:,i])*rotated_grad[0]/float(len(bases))
            grad_KL[1] += cplx.norm(psi_dict[bases[b]][:,i])*rotated_grad[1]/float(len(bases))
            grad_KL[0] -=probability(nn_state,vis[i], Z)*nn_state.rbm_am.effective_energy_gradient(vis[i])/float(len(bases))
    return grad_KL            



def run(qr,psi_dict,data_samples,data_bases,unitary_dict,bases,vis,eps,k):
    alg_grad_NLL = algorithmic_gradNLL(qr,data_samples,data_bases,k)
    alg_grad_KL = algorithmic_gradKL(qr.nn_state,psi_dict,vis,unitary_dict,bases)
    for n,net in enumerate(qr.nn_state.networks):
        counter = 0
        print('\n\nRBM: %s' %net) 
        rbm = getattr(qr.nn_state, net)
        
        num_grad_KL = numeric_gradKL(rbm.weights.view(-1),qr.nn_state,psi_dict,vis,unitary_dict,bases,eps)
        num_grad_NLL = numeric_gradNLL(qr.nn_state,data_samples,data_bases,unitary_dict,rbm.weights.view(-1),vis,eps)
        
        print("\nTesting weights...")
        print("Numerical KL\tAlg KL\t\t\tNumerical NLL\tAlg NLL")
        for i in range(len(rbm.weights.view(-1))):
            print("{: 10.8f}\t{: 10.8f}\t\t".format(num_grad_KL[i],alg_grad_KL[n][counter].item()),end="",flush=True)
            print("{: 10.8f}\t{: 10.8f}\t\t".format(num_grad_NLL[i],alg_grad_NLL[n][i].item()))
            counter += 1
 
        num_grad_KL = numeric_gradKL(rbm.visible_bias,qr.nn_state,psi_dict,vis,unitary_dict,bases,eps)
        num_grad_NLL = numeric_gradNLL(qr.nn_state,data_samples,data_bases,unitary_dict,rbm.visible_bias,vis,eps)
        print("\nTesting visible bias...")
        print("Numerical KL\tAlg KL\t\t\tNumerical NLL\tAlg NLL")
        for i in range(len(rbm.visible_bias)):
            print("{: 10.8f}\t{: 10.8f}\t\t".format(num_grad_KL[i],alg_grad_KL[n][counter].item()),end="", flush=True)
            print("{: 10.8f}\t{: 10.8f}\t\t".format(num_grad_NLL[i],alg_grad_NLL[n][counter].item()))
            counter += 1
 
        num_grad_KL = numeric_gradKL(rbm.hidden_bias,qr.nn_state,psi_dict,vis,unitary_dict,bases,eps)
        num_grad_NLL = numeric_gradNLL(qr.nn_state,data_samples,data_bases,unitary_dict,rbm.hidden_bias,vis,eps)
        print("\nTesting visible bias...")
        print("Numerical KL\tAlg KL\t\t\tNumerical NLL\tAlg NLL")
        for i in range(len(rbm.hidden_bias)):
            print("{: 10.8f}\t{: 10.8f}\t\t".format(num_grad_KL[i],alg_grad_KL[n][counter].item()),end="", flush=True)
            print("{: 10.8f}\t{: 10.8f}\t\t".format(num_grad_NLL[i],alg_grad_NLL[n][counter].item()))
            counter += 1
 

    print('')

if __name__ == '__main__':
   
    k=2
    num_chains=10
    seed=1234 
    with open('data_test.pkl', 'rb') as fin:
        test_data = pickle.load(fin)
    train_bases = test_data['2qubits']['train_bases']
    train_samples = torch.tensor(test_data['2qubits']['train_samples'],dtype = torch.double)
    bases_data = test_data['2qubits']['bases'] 
    target_psi_tmp=torch.tensor(test_data['2qubits']['target_psi'],dtype = torch.double)
    nh = train_samples.shape[-1]
    bases = transform_bases(bases_data)
    unitary_dict = unitaries.create_dict()
    psi_dict = load_target_psi(bases,target_psi_tmp) 
    vis = generate_visible_space(train_samples.shape[-1]) 
    nn_state = ComplexWavefunction(num_visible=train_samples.shape[-1],
                                   num_hidden=nh)
    qr = QuantumReconstruction(nn_state)
    eps= 1.e-6
    run(qr,psi_dict,train_samples,train_bases,unitary_dict,bases,vis,eps,k) 
 
