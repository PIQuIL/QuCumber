import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import tqdm, tqdm_notebook
import warnings
from cplx import *
from matplotlib import pyplot as plt

class RBM(nn.Module):
    def __init__(self, unitaries, num_visible, num_hidden_amp, num_hidden_phase, gpu=True, seed=1234):
        super(RBM, self).__init__()
        self.num_visible      = int(num_visible)
        self.num_hidden_amp   = int(num_hidden_amp)
        self.num_hidden_phase = int(num_hidden_phase)
        self.unitaries        = unitaries

        if gpu and not torch.cuda.is_available():
            warnings.warn("Could not find GPU: will continue with CPU.",
                          ResourceWarning)
            
        self.gpu = gpu and torch.cuda.is_available()
        if self.gpu:
            torch.cuda.manual_seed(seed)
            self.device = torch.device('cuda')
        else:
            torch.manual_seed(seed)
            self.device = torch.device('cpu')

        self.weights_amp = nn.Parameter(
            (torch.randn(self.num_hidden_amp, self.num_visible,
                         device=self.device, dtype=torch.double)
             / np.sqrt(self.num_visible)),
            requires_grad=True)
        
        #self.weights_phase = nn.Parameter(torch.zeros_like(self.weights_amp, device=self.device, dtype = torch.double, requires_grad=True))
        
        self.weights_phase = nn.Parameter(
            (torch.randn(self.num_hidden_phase, self.num_visible,
                         device=self.device, dtype=torch.double)
             / np.sqrt(self.num_visible)),
            requires_grad=True)
        
        self.visible_bias_amp   = nn.Parameter(torch.zeros(self.num_visible,
                                                     device=self.device,
                                                     dtype=torch.double),
                                         requires_grad=True)

        self.visible_bias_phase = nn.Parameter(torch.zeros(self.num_visible,
                                                     device=self.device,
                                                     dtype=torch.double),
                                         requires_grad=True)

        self.hidden_bias_amp    = nn.Parameter(torch.zeros(self.num_hidden_amp,
                                                    device=self.device,
                                                    dtype=torch.double),
                                        requires_grad=True)

        self.hidden_bias_phase  = nn.Parameter(torch.zeros(self.num_hidden_phase,
                                                    device=self.device,
                                                    dtype=torch.double),
                                        requires_grad=True)

    def compute_batch_gradients(self, k, batch, chars_batch, l1_reg, l2_reg, stddev=0):
        '''This function will compute the gradients of a batch of the training data (data_file) given the basis measurements (chars_file).'''
        
        if len(batch) == 0:
            return (torch.zeros_like(self.weights_amp,
                                     device=self.device,
                                     dtype=torch.double),
                    torch.zeros_like(self.visible_bias_amp,
                                     device=self.device,
                                     dtype=torch.double),
                    torch.zeros_like(self.hidden_bias_amp,
                                     device=self.device,
                                     dtype=torch.double),
                    torch.zeros_like(self.weights_phase,
                                     device=self.device,
                                     dtype=torch.double),
                    torch.zeros_like(self.visible_bias_phase,
                                     device=self.device,
                                     dtype=torch.double),
                    torch.zeros_like(self.hidden_bias_phase,
                                     device=self.device,
                                     dtype=torch.double))
    
        w_grad_amp  = torch.zeros_like(self.weights_amp)
        vb_grad_amp = torch.zeros_like(self.visible_bias_amp)
        hb_grad_amp = torch.zeros_like(self.hidden_bias_amp)
    
        w_grad_phase  = torch.zeros_like(self.weights_phase)
        vb_grad_phase = torch.zeros_like(self.visible_bias_phase)
        hb_grad_phase = torch.zeros_like(self.hidden_bias_phase)

        g_weights_amp = torch.zeros_like(self.weights_amp)
        g_vb_amp      = torch.zeros_like(self.visible_bias_amp)
        g_hb_amp      = torch.zeros_like(self.hidden_bias_amp)

        g_weights_phase = torch.zeros_like(self.weights_phase)
        g_vb_phase      = torch.zeros_like(self.visible_bias_phase)
        g_hb_phase      = torch.zeros_like(self.hidden_bias_phase)

        zeros_for_w  = torch.zeros_like(w_grad_amp)
        zeros_for_vb = torch.zeros_like(vb_grad_amp)
        zeros_for_hb = torch.zeros_like(hb_grad_amp) #NOTE! THIS WILL CURRENTLY ONLY WORK IF AND ONLY IF THE NUMBER OF HIDDEN UNITS FOR PHASE AND AMP ARE THE SAME!!!

        for row_count, v0 in enumerate(batch):
            num_non_trivial_unitaries = 0
            
            '''tau_indices will contain the index numbers of spins not in the computational basis (Z). z_indices will contain the index numbers of spins in the computational basis.'''
            tau_indices = []
            z_indices   = []
            for j in range(self.num_visible):
                if chars_batch[row_count][j] != 'Z':
                    num_non_trivial_unitaries += 1
                    tau_indices.append(j)

                else:
                    z_indices.append(j) 

            '''Condition if data point is in the comptational basis.'''
            if num_non_trivial_unitaries == 0:
                '''Do regular grad updates, like you would if there was no phase.'''
                v0, h0, vk_amp, hk_amp, phk_amp = self.gibbs_sampling_amp(k, v0)
    
                norm_factor = len(batch)*self.unnormalized_probability_amp(vk_amp)

                '''Positive phase of gradient.'''
                g_weights_amp -= torch.einsum("j,k->jk", (h0, v0)) / (norm_factor)  #Outer product.
                g_vb_amp      -= v0 / (norm_factor)
                g_hb_amp      -= h0 / (norm_factor)

                '''Negative phase of amp gradient.'''
                g_weights_amp += torch.einsum("j,k->jk", (phk_amp, vk_amp)) / (norm_factor)
                g_vb_amp      += vk_amp / (norm_factor)
                g_hb_amp      += phk_amp / (norm_factor)
                '''Divide by unnormalized prob because we took gradient of log(unnormalized prob).'''

                continue

            if num_non_trivial_unitaries > 0:
                A_weights_amp = torch.zeros(2, self.weights_amp.size()[0], self.weights_amp.size()[1], 
                                            device=self.device, dtype = torch.double)
                A_vb_amp      = torch.zeros(2, self.visible_bias_amp.size()[0], 
                                            device=self.device, dtype = torch.double)
                A_hb_amp      = torch.zeros(2, self.hidden_bias_amp.size()[0], 
                                            device=self.device, dtype = torch.double)
                
                A_weights_phase = torch.zeros(2, self.weights_phase.size()[0], self.weights_phase.size()[1], 
                                              device=self.device, dtype = torch.double)
                A_vb_phase      = torch.zeros(2, self.visible_bias_phase.size()[0], 
                                              device=self.device, dtype = torch.double)
                A_hb_phase      = torch.zeros(2, self.hidden_bias_phase.size()[0], 
                                              device=self.device, dtype = torch.double)

                B = 0.

                _, _, vk_amp, hk_amp, phk_amp = self.gibbs_sampling_amp(k, v0)

                norm_factor1 = len(batch)*self.unnormalized_probability_amp(vk_amp)

                '''Loop over Hilbert space of the non trivial unitaries to build the state |sigma> in Giacomo's pseudo code (alg 4.2).'''
                for j in range(2**num_non_trivial_unitaries):
                    s = self.state_generator(num_non_trivial_unitaries)[j]
                    '''Creates a matrix where the jth row is the desired state, |S>'''

                    '''This is the |sigma> state in Giacomo's pseudo code.'''
                    constructed_state = torch.zeros(self.num_visible, dtype = torch.double)
                    U = 1.

                    for index in range(len(z_indices)):
                        constructed_state[z_indices[index]] = batch[row_count][z_indices[index]]

                    for index in range(len(tau_indices)):

                        constructed_state[tau_indices[index]] = s[index]

                        U *= cplx_dot( cplx_MV_mult(self.unitaries[chars_batch[row_count][tau_indices[index]]], 
                                               self.basis_state_generator(batch[row_count][tau_indices[index]])), 
                                       self.basis_state_generator(s[index]) )

                    norm_factor2 = self.unnormalized_probability_amp(constructed_state)
                    norm_factor3 = self.unnormalized_probability_phase(constructed_state)

                    '''Gradients for phase and amp.'''
                    w_grad_amp  = torch.matmul(F.sigmoid(F.linear(constructed_state, self.weights_amp.t(), self.hidden_bias_amp)), 
                                               constructed_state) 
                    vb_grad_amp = constructed_state / norm_factor2
                    hb_grad_amp = F.sigmoid(F.linear(constructed_state, self.weights_amp.t(), self.hidden_bias_amp)) / norm_factor2

                    
                    w_grad_phase  = torch.matmul(F.sigmoid(F.linear(constructed_state, self.weights_phase.t(), self.hidden_bias_phase)), 
                                               constructed_state) 
                    vb_grad_phase = constructed_state / norm_factor3
                    hb_grad_phase = F.sigmoid(F.linear(constructed_state, self.weights_phase.t(), self.hidden_bias_phase)) / norm_factor3


                    w_grad_amp  = self.regularize_weight_gradients_amp(w_grad_amp, l1_reg, l2_reg)
                    w_grad_amp += (stddev * torch.randn_like(w_grad_amp, device=self.device))

                    w_grad_phase  = self.regularize_weight_gradients_phase(w_grad_phase, l1_reg, l2_reg)
                    w_grad_phase += (stddev * torch.randn_like(w_grad_amp, device=self.device))

                    w_grad_amp   /= norm_factor2
                    w_grad_phase /= norm_factor3
                    
                    '''
                    In order to calculate the 'A' parameters below with my current complex library, I need to make the weights and biases complex.
                    '''
                    temp_w_grad_amp  = cplx_make_complex_matrix(w_grad_amp, zeros_for_w)
                    temp_vb_grad_amp = cplx_make_complex_vector(vb_grad_amp, zeros_for_vb)
                    temp_hb_grad_amp = cplx_make_complex_vector(hb_grad_amp, zeros_for_hb)

                    temp_w_grad_phase  = cplx_make_complex_matrix(w_grad_phase, zeros_for_w)
                    temp_vb_grad_phase = cplx_make_complex_vector(vb_grad_phase, zeros_for_vb)
                    temp_hb_grad_phase = cplx_make_complex_vector(hb_grad_phase, zeros_for_hb)

                    A_weights_amp += cplx_MS_mult(cplx_scalar_mult(U, self.unnormalized_wavefunction(constructed_state)), 
                                                  temp_w_grad_amp)
                    A_vb_amp      += cplx_VS_mult(cplx_scalar_mult(U, self.unnormalized_wavefunction(constructed_state)), 
                                                  temp_vb_grad_amp)
                    A_hb_amp      += cplx_VS_mult(cplx_scalar_mult(U, self.unnormalized_wavefunction(constructed_state)), 
                                                  temp_hb_grad_amp)

                    A_weights_phase += cplx_MS_mult(cplx_scalar_mult(U, self.unnormalized_wavefunction(constructed_state)), 
                                                  temp_w_grad_phase)
                    A_vb_phase      += cplx_VS_mult(cplx_scalar_mult(U, self.unnormalized_wavefunction(constructed_state)), 
                                                  temp_vb_grad_phase)
                    A_hb_phase      += cplx_VS_mult(cplx_scalar_mult(U, self.unnormalized_wavefunction(constructed_state)), 
                                                  temp_hb_grad_phase)
                    
                    B += cplx_scalar_mult(U, self.unnormalized_wavefunction(constructed_state))

                L_weights_amp = cplx_MS_divide(A_weights_amp, B)
                L_vb_amp      = cplx_VS_divide(A_vb_amp, B)
                L_hb_amp      = cplx_VS_divide(A_hb_amp, B)

                L_weights_phase = cplx_MS_divide(A_weights_phase, B)
                L_vb_phase      = cplx_VS_divide(A_vb_phase, B)
                L_hb_phase      = cplx_VS_divide(A_hb_phase, B)

                '''Gradents of amplitude parameters take the real part of the L gradients.'''
                g_weights_amp -= L_weights_amp[0] / len(batch)
                g_vb_amp      -= L_vb_amp[0] / len(batch)
                g_hb_amp      -= L_hb_amp[0] / len(batch)

                '''Negative phase of amp gradient.'''
                g_weights_amp += torch.einsum("j,k->jk", (phk_amp, vk_amp)) / (norm_factor1)
                g_vb_amp      += vk_amp / (norm_factor1)
                g_hb_amp      += phk_amp / (norm_factor1)
                '''Divide by unnormalized prob because we took gradient of log(unnormalized prob).'''

                '''Gradents of phase parameters take the real part of the L gradients. Phase gradients have no neg phase.'''
                g_weights_phase += L_weights_phase[1]/len(batch)
                g_vb_phase      += L_vb_phase[1]/len(batch)
                g_hb_phase      += L_hb_phase[1]/len(batch)

                continue
        
            '''Return negative gradients to match up nicely with the usual
            parameter update rules, which *subtract* the gradient from
            the parameters. This is in contrast with the RBM update
            rules which ADD the gradients (scaled by the learning rate)
            to the parameters.'''

        return {"weights_amp": g_weights_amp,
                "visible_bias_amp": g_vb_amp,
                "hidden_bias_amp": g_hb_amp,
                "weights_phase": g_weights_phase,
                "visible_bias_phase": g_vb_phase,
                "hidden_bias_phase": g_hb_phase
                }   

    def train(self, data, character_data, epochs, batch_size,
              k=10, lr=1e-3, momentum=0.0,
              method='sgd', l1_reg=0.0, l2_reg=0.0,
              initial_gaussian_noise=0.01, gamma=0.55,
              callbacks=[], progbar=False,
              log_every=50,
              **kwargs):
        # callback_outputs = []
        disable_progbar = (progbar is False)
        progress_bar = tqdm_notebook if progbar == "notebook" else tqdm

        data = torch.tensor(data).to(device=self.device)
        optimizer = torch.optim.Adam([self.weights_amp,
                                      self.visible_bias_amp,
                                      self.hidden_bias_amp,
                                      self.weights_phase,
                                      self.visible_bias_phase,
                                      self.hidden_bias_phase],
                                      lr=lr)

        vis = self.generate_visible_space()
        print ('Generated visible space. Ready to begin training.')
        fidelity_list = []
        epoch_list = []

        for ep in range(epochs+1):
            
            random_permutation = torch.randperm(data.shape[0])

            shuffled_data           = data[random_permutation]   
            shuffled_character_data = character_data[random_permutation]

            batches = [shuffled_data[batch_start:(batch_start + batch_size)] 
                       for batch_start in range(0, len(data), batch_size)]

            char_batches = [shuffled_character_data[batch_start:(batch_start + batch_size)] 
                            for batch_start in range(0, len(data), batch_size)]

            if ep % log_every == 0:
                #logZ = self.log_partition(vis)
                #nll = self.nll(data, logZ)
                fidelity_ = self.fidelity(vis)
                print ('Epoch = ',ep,'\nFidelity = ',fidelity_)
                fidelity_list.append(fidelity_)
                epoch_list.append(ep)

            if ep == epochs:
                fidelity_file = open('fidelity_file.txt', 'w')
                print ('Finished training. Saving results...' )               
                for i in range(len(fidelity_list)):
                    fidelity_file.write('%.5f' % fidelity_list[i] + ' %d\n' % epoch_list[i])
                break

            stddev = torch.tensor(
                [initial_gaussian_noise / ((1 + ep) ** gamma)],
                dtype=torch.double, device=self.device).sqrt()

            for batch_index in range(len(batches)):

                grads = self.compute_batch_gradients(k, batches[batch_index], char_batches[batch_index],
                                                     l1_reg, l2_reg,
                                                     stddev=stddev)

                optimizer.zero_grad()  # clear any cached gradients

                # assign all available gradients to the corresponding parameter
                for name in grads.keys():
                    getattr(self, name).grad = grads[name]

                optimizer.step()  # tell the optimizer to apply the gradients

            # TODO: run callbacks

    def prob_v_given_h_amp(self, h):
        p = F.sigmoid(F.linear(h, self.weights_amp.t(), self.visible_bias_amp))
        return p

    def prob_v_given_h_phase(self, h):
        p = F.sigmoid(F.linear(h, self.weights_phase.t(), self.visible_bias_phase))
        return 

    def prob_h_given_v_amp(self, v):
        p = F.sigmoid(F.linear(v, self.weights_amp, self.hidden_bias_amp))
        return p

    def prob_h_given_v_phase(self, v):
        p = F.sigmoid(F.linear(v, self.weights_phase, self.hidden_bias_phase))
        return p

    def sample_v_given_h_amp(self, h):
        p = self.prob_v_given_h_amp(h)
        v = p.bernoulli()
        return p, v

    def sample_h_given_v_amp(self, v):
        p = self.prob_h_given_v_amp(v)
        h = p.bernoulli()
        return p, h

    def sample_v_given_h_phase(self, h):
        p = self.prob_v_given_h_phase(h)
        v = p.bernoulli()
        return p, v

    def sample_h_given_v_phase(self, v):
        p = self.prob_h_given_v_phase(v)
        h = p.bernoulli()
        return p, h

    def gibbs_sampling_amp(self, k, v0):
        _, h0 = self.sample_h_given_v_amp(v0)
        h = h0
        for _ in range(k):
            pv, v = self.sample_v_given_h_amp(h)
            ph, h = self.sample_h_given_v_amp(v)
        return v0, h0, v, h, ph

    def gibbs_sampling_phase(self, k, v0):
        _, h0 = self.sample_h_given_v_phase(v0)
        h = h0
        for _ in range(k):
            pv, v = self.sample_v_given_h_phase(h)
            ph, h = self.sample_h_given_v_phase(v)
        return v0, h0, v, h, ph

    def sample_amp(self, k, num_samples):
        dist = torch.distributions.bernoulli.Bernoulli(probs=0.5)
        v0 = (dist.sample(torch.Size([num_samples, self.num_visible]))
                  .to(device=self.device, dtype=torch.double))
        _, _, v, _, _ = self.gibbs_sampling_amp(k, v0)
        return v

    def sample_phase(self, k, num_samples):
        dist = torch.distributions.bernoulli.Bernoulli(probs=0.5)
        v0 = (dist.sample(torch.Size([num_samples, self.num_visible]))
                  .to(device=self.device, dtype=torch.double))
        _, _, v, _, _ = self.gibbs_sampling_phase(k, v0)
        return v

    def regularize_weight_gradients_amp(self, w_grad, l1_reg, l2_reg):
        return (w_grad
                + (l2_reg * self.weights_amp)
                + (l1_reg * self.weights_amp.sign()))

    def regularize_weight_gradients_phase(self, w_grad, l1_reg, l2_reg):
        return (w_grad
                + (l2_reg * self.weights_phase)
                + (l1_reg * self.weights_phase.sign()))

    def free_energy_amp(self, v):
        if len(v.shape) < 2:
            v = v.view(1, -1)

        visible_bias_term = torch.mv(v, self.visible_bias_amp)
        hidden_bias_term = F.softplus(
            F.linear(v, self.weights_amp, self.hidden_bias_amp)).sum(1)

        return visible_bias_term + hidden_bias_term

    def free_energy_phase(self, v):
        if len(v.shape) < 2:
            v = v.view(1, -1)

        visible_bias_term = torch.mv(v, self.visible_bias_phase)
        hidden_bias_term = F.softplus(
            F.linear(v, self.weights_phase, self.hidden_bias_phase)).sum(1)

        return visible_bias_term + hidden_bias_term

    def unnormalized_probability_amp(self, v):
        return self.free_energy_amp(v).exp()

    def unnormalized_probability_phase(self, v):
        return self.free_energy_phase(v).exp()

    def normalized_wavefunction(self, v):
        #v_prime   = v.view(-1,2)
        v_prime = v
        #v_prime = v
        temp1     = (self.unnormalized_probability_amp(v_prime)).sqrt()
        temp2     = ((self.unnormalized_probability_phase(v_prime)).log())*0.5

        cos_angle = temp2.cos()
        sin_angle = temp2.sin()
        
        psi       = torch.zeros(2, v_prime.size()[0], dtype = torch.double)
        psi[0]    = temp1*cos_angle
        psi[1]    = temp1*sin_angle

        sqrt_Z    = (self.partition(v_prime)).sqrt()

        return psi / sqrt_Z

    def unnormalized_wavefunction(self, v):
        v_prime   = v.view(-1,self.num_visible)
        #v_prime = v
        temp1     = (self.unnormalized_probability_amp(v_prime)).sqrt()
        temp2     = ((self.unnormalized_probability_phase(v_prime)).log())*0.5
        cos_angle = temp2.cos()
        sin_angle = temp2.sin()
        
        psi       = torch.zeros(2, v_prime.size()[0], dtype = torch.double)
        
        psi[0]    = temp1*cos_angle
        psi[1]    = temp1*sin_angle

        return psi

    def spin_list(self, n_vis):
        '''returns a list of all possible spin configurations for n_vis spins'''
        spins = [spin_config(number, n_vis) for number in range(2**n_vis)]
        spins = Variable(torch.FloatTensor(spins))
        return spins

    def true_psi(self):
        #psi = torch.tensor(np.loadtxt('../benchmarks/c++/complex_target_psi.txt'), dtype = torch.double)
        '''Picks out the correct psi in the correct basis.'''
        
        psi_file = np.loadtxt('../benchmarks/data/2qubits_complex/2qubits_psi.txt')
        psi = torch.zeros(2, 2**self.num_visible, dtype = torch.double)

        psi_real = torch.tensor(psi_file[:4,0], dtype = torch.double)
        psi_imag = torch.tensor(psi_file[:4,1], dtype = torch.double)

        psi[0] = psi_real
        psi[1] = psi_imag
        
        return psi

    def overlap(self, visible_space):
        '''
        print ('RBM psi norm >>> \n',cplx_norm( cplx_inner(self.normalized_wavefunction(visible_space),
                           self.normalized_wavefunction(visible_space)) ))

        print ('True psi norm >>> \n',cplx_norm( cplx_inner(self.true_psi(),
                           self.true_psi()) ))
        '''
        overlap_ = cplx_inner(self.true_psi(),
                           self.normalized_wavefunction(visible_space))
        '''
        overlap_ = cplx_inner(self.true_psi(),
                           self.normalized_wavefunction(visible_space).t())
        '''
        return overlap_

    def fidelity(self, visible_space):
        return cplx_norm(self.overlap(visible_space))

    def generate_visible_space(self):
        space = torch.zeros((1 << self.num_visible, self.num_visible),
                            device=self.device, dtype=torch.double)
        for i in range(1 << self.num_visible):
            d = i
            for j in range(self.num_visible):
                d, r = divmod(d, 2)
                space[i, self.num_visible - j - 1] = int(r)

        return space

    def log_partition(self, visible_space):
        free_energies = self.free_energy_amp(visible_space)
        max_free_energy = free_energies.max()

        f_reduced = free_energies - max_free_energy
        logZ = max_free_energy + f_reduced.exp().sum().log()

        return logZ

    def partition(self, visible_space):
        return self.log_partition(visible_space).exp()

    def nll(self, data, logZ):
        total_free_energy = self.free_energy_amp(data).sum()

        return (len(data)*logZ) - total_free_energy

    def state_generator(self, num_non_trivial_unitaries):
        '''A function that returns all possible configurations of 'num_non_trivial_unitaries' spins.'''
        states = torch.zeros(2**num_non_trivial_unitaries, num_non_trivial_unitaries)
        for i in range(2**num_non_trivial_unitaries):
            temp = i
            for j in range(num_non_trivial_unitaries): 
                temp, remainder = divmod(temp, 2)
                states[i][j] = remainder
        return states

    def basis_state_generator(self, s):
        '''Only works for binary at the moment. If s = 0, this is the (1,0) state in the basis of the measurement. If s = 1, this is the (0,1) state in the basis of the measurement.'''
        if s == 0.:
            return torch.tensor([[1., 0.],[0., 0.]], dtype = torch.double)
        if s == 1.:
            return torch.tensor([[0., 1.],[0., 0.]], dtype = torch.double) 