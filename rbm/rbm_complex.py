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
    """Class to build the Restricted Boltzmann Machine

    Parameters
    ----------
    unitaries : array_like
        list of unitaries
    num_visible : int
        Number of visible units
    num_hidden_amp : int
        Number of hidden units to learn the amplitude
    num_hidden_phase : int
        Number of hidden units to learn the phase
    gpu : bool
        Should the GPU be used for the training
    seed : int
        Fix the random number seed to make results reproducable
        
    
    """

    def __init__(self, full_unitaries, unitaries, psi_dictionary, num_visible, num_hidden_amp, num_hidden_phase, gpu=True, seed=1234):
        super(RBM, self).__init__()
        self.num_visible      = int(num_visible)
        self.num_hidden_amp   = int(num_hidden_amp)
        self.num_hidden_phase = int(num_hidden_phase)
        self.full_unitaries   = full_unitaries
        self.unitaries        = unitaries
        self.psi_dictionary   = psi_dictionary

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
                      
        self.weights_phase = nn.Parameter(
            (torch.randn(self.num_hidden_phase, self.num_visible,
                         device=self.device, dtype=torch.double)
             / np.sqrt(self.num_visible)),
            requires_grad=True)
        '''
        self.weights_phase = nn.Parameter(torch.zeros(self.num_hidden_phase, self.num_visible, 
                         device=self.device, dtype=torch.double), requires_grad=True)
        '''
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
        '''This function will compute the gradients of a batch of the training 
        data (data_file) given the basis measurements (chars_file).

        Parameters
        ----------
        k : int
            number of Gibbs steps in amplitude training
        batch : array_like
            batch of the input data
        chars_batch : array_like
            batch of the unitaries. Indicates in which basis the input data was
            measured
        l1_reg : float
            L1 regularization hyperparameter
        l2_reg : float
            L2 regularization hyperparameter
        stddev : float
            standard deviation of random noise that can be added to the weights.
            This is also a hyperparamter.

        Returns 
        ----------
        Gradients : dictionary
            Dictionary containing all the gradients in the following order:
            Gradient of weights, visible bias and hidden bias for the amplitude 
            Gradients of weights, visible bias and hidden bias for the phase.

        '''
        
        '''This function will compute the gradients of a batch of the training data (data_file) given the basis measurements (chars_file).'''
        vis = self.generate_visible_space()
        if len(batch) == 0:
            print ('Batch length is zero...')
        #If batch has length 0, return zero matrices as grad update
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
    
        #unrotated_RBM_psi_initial = self.normalized_wavefunction(vis)
        #constructed_RBM_psi_initial = torch.zeros_like(unrotated_RBM_psi_initial)   
        # GRADIENT FOR HIDDEN BIAS PHASE VANISHES...

        batch_size = len(batch)

        w_grad_amp      = torch.zeros_like(self.weights_amp)
        vb_grad_amp     = torch.zeros_like(self.visible_bias_amp)
        hb_grad_amp     = torch.zeros_like(self.hidden_bias_amp)
    
        w_grad_phase    = torch.zeros_like(self.weights_phase)
        vb_grad_phase   = torch.zeros_like(self.visible_bias_phase)
        hb_grad_phase   = torch.zeros_like(self.hidden_bias_phase)

        g_weights_amp   = torch.zeros_like(self.weights_amp)
        g_vb_amp        = torch.zeros_like(self.visible_bias_amp)
        g_hb_amp        = torch.zeros_like(self.hidden_bias_amp)

        g_weights_phase = torch.zeros_like(self.weights_phase)
        g_vb_phase      = torch.zeros_like(self.visible_bias_phase)
        g_hb_phase      = torch.zeros_like(self.hidden_bias_phase)

        zeros_for_w  = torch.zeros_like(w_grad_amp)
        zeros_for_vb = torch.zeros_like(vb_grad_amp)
        zeros_for_hb = torch.zeros_like(hb_grad_amp) 
        #NOTE! THIS WILL CURRENTLY ONLY WORK IF AND ONLY IF THE NUMBER OF HIDDEN UNITS FOR PHASE AND AMP ARE THE SAME!!!

        for row_count, v0 in enumerate(batch):
            num_non_trivial_unitaries = 0

            '''tau_indices will contain the index numbers of spins not in the   
            computational basis (Z). 
            z_indices will contain the index numbers of spins in the computational 
            basis.'''
            tau_indices = []
            z_indices   = []

            for j in range(self.num_visible):
                """Go through list of unitaries and save inidices of non-trivial"""
                if chars_batch[row_count][j] != 'Z':
                    num_non_trivial_unitaries += 1
                    tau_indices.append(j)
                else:
                    z_indices.append(j)
                    """"Create list of indices of trivial unitaries """
                    z_indices.append(j) 

            v0, h0_amp, vk_amp, hk_amp, phk_amp = self.gibbs_sampling_amp(k, v0)
            _, h0_phase = self.sample_h_given_v_phase(v0)

            '''Condition if data point is in the comptational basis.'''
            if num_non_trivial_unitaries == 0:
                '''Do regular grad updates, like you would if there was no phase.'''

                '''Positive phase of gradient.'''
                g_weights_amp -= torch.einsum("j,k->jk", (h0_amp, v0)) / batch_size #Outer product.
                g_vb_amp      -= v0 / batch_size
                g_hb_amp      -= h0_amp / batch_size

            if num_non_trivial_unitaries > 0:
                '''Initialize the 'A' parameters (see alg 4.2).'''
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

                B = torch.zeros(2, device = self.device, dtype = torch.double)

                '''Loop over Hilbert space of the non trivial unitaries to build the state |sigma> in Giacomo's pseudo code (alg 4.2).'''
                for j in range(2**num_non_trivial_unitaries):
                    s = self.state_generator(num_non_trivial_unitaries)[j]
                    '''Creates a matrix where the jth row is the desired state, |S>, a vector.'''

                    '''This is the |sigma> state in Giacomo's pseudo code.'''
                    constructed_state = torch.zeros(self.num_visible, dtype = torch.double)
                    
                    U = torch.tensor([1., 0.], dtype = torch.double, device = self.device)

                    for index in range(len(z_indices)):
                        constructed_state[z_indices[index]] = batch[row_count][z_indices[index]]

                    for index in range(len(tau_indices)):
                        constructed_state[tau_indices[index]] = s[index]

                        temp = cplx_inner( cplx_MV_mult(self.unitaries[chars_batch[row_count][tau_indices[index]]], 
                                                        self.basis_state_generator(batch[row_count][tau_indices[index]])), 
                                           self.basis_state_generator(s[index]) )

                        U = cplx_scalar_mult(U, temp)

                    '''Positive phase gradients for phase and amp. Will be added into the 'A' parameters.'''
                    w_grad_amp  = torch.einsum("j,k->jk", (h0_amp, constructed_state)) #Outer product.
                    vb_grad_amp = constructed_state
                    hb_grad_amp = h0_amp

                    w_grad_phase  = torch.einsum("j,k->jk", (h0_phase, constructed_state)) #Outer product.
                    vb_grad_phase = constructed_state
                    hb_grad_phase = h0_phase
                    '''
                    In order to calculate the 'A' parameters below with my current complex library, I need to make the weights and biases complex.
                    '''
                    temp_w_grad_amp  = cplx_make_complex_matrix(w_grad_amp, zeros_for_w)
                    temp_vb_grad_amp = cplx_make_complex_vector(vb_grad_amp, zeros_for_vb)
                    temp_hb_grad_amp = cplx_make_complex_vector(hb_grad_amp, zeros_for_hb)

                    temp_w_grad_phase  = cplx_make_complex_matrix(w_grad_phase, zeros_for_w)
                    temp_vb_grad_phase = cplx_make_complex_vector(vb_grad_phase, zeros_for_vb)
                    temp_hb_grad_phase = cplx_make_complex_vector(hb_grad_phase, zeros_for_hb)

                    temp = cplx_scalar_mult(U, self.unnormalized_wavefunction(constructed_state))

                    A_weights_amp += cplx_MS_mult(temp, temp_w_grad_amp)
                    A_vb_amp      += cplx_VS_mult(temp, temp_vb_grad_amp)
                    A_hb_amp      += cplx_VS_mult(temp, temp_hb_grad_amp)

                    A_weights_phase += cplx_MS_mult(temp, temp_w_grad_phase)
                    A_vb_phase      += cplx_VS_mult(temp, temp_vb_grad_phase)
                    A_hb_phase      += cplx_VS_mult(temp, temp_hb_grad_phase)
                    
                    '''Rotated wavefunction.'''
                    B += temp
                    #B_norm += cplx_scalar_mult(U, self.normalized_wavefunction(constructed_state)) # for debugging 

                #constructed_RBM_psi_initial[0][self.state_to_index(batch[row_count])] = B_norm[0]
                #constructed_RBM_psi_initial[1][self.state_to_index(batch[row_count])] = B_norm[1]

                #U_matrix = self.get_full_unitary(chars_batch[row_count])

                L_weights_amp = cplx_MS_divide(A_weights_amp, B)
                L_vb_amp      = cplx_VS_divide(A_vb_amp, B)
                L_hb_amp      = cplx_VS_divide(A_hb_amp, B)

                L_weights_phase = cplx_MS_divide(A_weights_phase, B)
                L_vb_phase      = cplx_VS_divide(A_vb_phase, B)
                L_hb_phase      = cplx_VS_divide(A_hb_phase, B)

                '''Gradents of amplitude parameters take the real part of the L gradients.'''
                g_weights_amp -= L_weights_amp[0] / batch_size
                g_vb_amp      -= L_vb_amp[0] / batch_size
                g_hb_amp      -= L_hb_amp[0] / batch_size

                '''Gradents of phase parameters take the real part of the L gradients. Phase gradients have no neg phase.'''
                g_weights_phase += L_weights_phase[1]/batch_size
                g_vb_phase      += L_vb_phase[1]/batch_size
                g_hb_phase      += L_hb_phase[1]/batch_size

        '''Negative phase of amp gradient. Phase parameters do not have a negative phase.'''
        g_weights_amp += torch.einsum("j,k->jk", (phk_amp, vk_amp)) / batch_size
        g_vb_amp      += vk_amp / batch_size
        g_hb_amp      += phk_amp / batch_size

        '''Return negative gradients to match up nicely with the usual
        parameter update rules, which *subtract* the gradient from
        the parameters. This is in contrast with the RBM update
        rules which ADD the gradients (scaled by the learning rate)
            to the parameters.'''

        '''
        print ('Initial RBM psi: \n',unrotated_RBM_psi_initial)
        print ('Constructed RBM psi with rotation: \n', constructed_RBM_psi_initial)
        print ('Undo the rotation: \n', cplx_MV_mult(U_matrix, constructed_RBM_psi_initial)) # U^T = U for XZ
        '''    
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
                fidelity_ = self.fidelity(vis, 'Z' 'Z')
                print ('Epoch = ',ep,'\nFidelity = ',fidelity_)
                fidelity_list.append(fidelity_)
                #print('Not calculating anything right now, just checking grads.')

            if ep == epochs:
                print ('Finished training. Saving results...' )               
                fidelity_file = open('fidelity_file.txt', 'w')

                for i in range(len(fidelity_list)):
                    fidelity_file.write('%.5f' % fidelity_list[i] + ' %d\n' % epoch_list[i])

                fidelity_file.close()
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
                
                self.test_gradients(vis, k, batches[batch_index], char_batches[batch_index],
                                                     l1_reg, l2_reg,
                                                     stddev=stddev)                
                
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

    def eff_energy_amp(self, v):
        if len(v.shape) < 2:
            v = v.view(1, -1)

        visible_bias_term = torch.mv(v, self.visible_bias_amp)
        hidden_bias_term = F.softplus(F.linear(v, self.weights_amp, self.hidden_bias_amp)).sum(1)

        return visible_bias_term + hidden_bias_term

    def eff_energy_phase(self, v):
        if len(v.shape) < 2:
            v = v.view(1, -1)

        visible_bias_term = torch.mv(v, self.visible_bias_phase)
        hidden_bias_term = F.softplus(F.linear(v, self.weights_phase, self.hidden_bias_phase)).sum(1)

        return visible_bias_term + hidden_bias_term

    def unnormalized_probability_amp(self, v):
        return self.eff_energy_amp(v).exp()

    def unnormalized_probability_phase(self, v):
        return self.eff_energy_phase(v).exp()

    def normalized_wavefunction(self, v):
        v_prime   = v.view(-1,self.num_visible)
        temp1     = (self.unnormalized_probability_amp(v_prime)).sqrt()
        temp2     = ((self.unnormalized_probability_phase(v_prime)).log())*0.5

        cos_angle = temp2.cos()
        sin_angle = temp2.sin()
        
        psi       = torch.zeros(2, v_prime.size()[0], dtype = torch.double)
        psi[0]    = temp1*cos_angle
        psi[1]    = temp1*sin_angle

        sqrt_Z    = (self.partition(self.generate_visible_space())).sqrt()

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

    def get_true_psi(self, basis):
        #psi = torch.tensor(np.loadtxt('../benchmarks/c++/complex_target_psi.txt'), dtype = torch.double)
        '''Picks out the correct psi in the correct basis.'''
        key = ''
        for i in range(len(basis)):
            key += basis[i]
        return self.psi_dictionary[key]

    def get_full_unitary(self, basis):
        key = ''
        for i in range(len(basis)):
            key += basis[i]
        return self.full_unitaries[key]

    def overlap(self, visible_space, basis):
        '''
        print ('RBM psi norm >>> \n',cplx_norm( cplx_inner(self.normalized_wavefunction(visible_space),
                           self.normalized_wavefunction(visible_space)) ))

        print ('True psi norm >>> \n',cplx_norm( cplx_inner(self.true_psi(),
                           self.true_psi()) ))
                           
        '''
        overlap_ = cplx_inner(self.get_true_psi(basis),
                           self.normalized_wavefunction(visible_space))
        '''
        overlap_ = cplx_inner(self.true_psi(),
                           self.normalized_wavefunction(visible_space).t())
                           
        '''
        return overlap_

    def fidelity(self, visible_space, basis):
        return cplx_norm(self.overlap(visible_space, basis))

    def generate_visible_space(self):
        space = torch.zeros((2**self.num_visible, self.num_visible),
                            device=self.device, dtype=torch.double)
        for i in range(2**self.num_visible):
            d = i
            for j in range(self.num_visible):
                d, r = divmod(d, 2)
                space[i, self.num_visible - j - 1] = int(r)

        return space

    def log_partition(self, visible_space):
        eff_energies = self.eff_energy_amp(visible_space)
        max_eff_energy = eff_energies.max()

        reduced = eff_energies - max_eff_energy
        logZ = max_eff_energy + reduced.exp().sum().log()

        return logZ

    def partition(self, visible_space):
        return self.log_partition(visible_space).exp()

    def nll(self, data, logZ):
        total_eff_energy = self.eff_energy_amp(data).sum()

        return (len(data)*logZ) - total_eff_energy

    def state_generator(self, num_non_trivial_unitaries):
        '''A function that returns all possible configurations of 'num_non_trivial_unitaries' spins.'''
        states = torch.zeros((2**num_non_trivial_unitaries, num_non_trivial_unitaries), device = self.device, dtype=torch.double)
        for i in range(2**num_non_trivial_unitaries):
            temp = i
            for j in range(num_non_trivial_unitaries): 
                temp, remainder = divmod(temp, 2)
                states[i][num_non_trivial_unitaries - j - 1] = remainder
        return states

    def basis_state_generator(self, s):
        '''Only works for binary at the moment. If s = 0, this is the (1,0) state in the basis of the measurement. If s = 1, this is the (0,1) state in the basis of the measurement.'''
        if s == 0.:
            return torch.tensor([[1., 0.],[0., 0.]], dtype = torch.double)
        if s == 1.:
            return torch.tensor([[0., 1.],[0., 0.]], dtype = torch.double) 

    def KL_divergence(self, visible_space):
        KL = 0
        basis_list = ['Z' 'Z', 'X' 'Z', 'Z' 'X', 'Y' 'Z', 'Z' 'Y']

        for i in range(len(basis_list)):
            rotated_RBM_psi  = cplx_MV_mult(self.full_unitaries[basis_list[i]], self.normalized_wavefunction(visible_space))
            rotated_true_psi = self.get_true_psi(basis_list[i])

            for j in range(len(visible_space)):
                elementof_rotated_RBM_psi  = torch.tensor([rotated_RBM_psi[0][j], rotated_RBM_psi[1][j]]).view(2,1)
                elementof_rotated_true_psi = torch.tensor([rotated_true_psi[0][j], rotated_true_psi[1][j]]).view(2,1)

                norm_true_psi = cplx_norm( cplx_inner(elementof_rotated_true_psi, elementof_rotated_true_psi) )
                norm_RBM_psi  = cplx_norm( cplx_inner(elementof_rotated_RBM_psi, elementof_rotated_RBM_psi) )

                KL += norm_true_psi*torch.log(norm_RBM_psi)

        return KL

    def compute_numerical_gradient(self, visible_space, hyper_param, alg_grad):
        eps = 1.e-6

        for i in range(len(hyper_param)):
            hyper_param[i].data += eps
            KL_pos = self.KL_divergence(visible_space)

            hyper_param[i].data -= 2*eps
            KL_neg = self.KL_divergence(visible_space)

            hyper_param[i].data += eps

            num_grad = (KL_pos - KL_neg) / (2*eps)

            print ('numerical:', num_grad)
            print ('algebraic: ', alg_grad[i],'\n')

    def test_gradients(self, visible_space, k, batch, chars_batch, l1_reg, l2_reg, stddev):

        '''Must have negative sign because the compute_batch_grads returns the neg of the grads.'''
        alg_grads = self.compute_batch_gradients(k, batch, chars_batch, l1_reg, l2_reg, stddev)
        # key_list = ["weights_amp", "visible_bias_amp", "hidden_bias_amp", "weights_phase", "visible_bias_phase", "hidden_bias_phase"]

        flat_weights_amp   = self.weights_amp.view(-1)
        flat_weights_phase = self.weights_phase.view(-1)
        
        flat_grad_weights_amp   = alg_grads["weights_amp"].view(-1)
        flat_grad_weights_phase = alg_grads["weights_phase"].view(-1)

        print ('-------------------------------------------------------------------------------')

        print ('Weights amp gradient')
        self.compute_numerical_gradient(visible_space, flat_weights_amp, -flat_grad_weights_amp)
        
        print ('Visible bias amp gradient')
        self.compute_numerical_gradient(visible_space, self.visible_bias_amp, -alg_grads["visible_bias_amp"])

        print ('Hidden bias amp gradient')
        self.compute_numerical_gradient(visible_space, self.hidden_bias_amp, -alg_grads["hidden_bias_amp"])

        print ('Weights phase gradient')
        self.compute_numerical_gradient(visible_space, flat_weights_phase, -flat_grad_weights_phase)

        print ('Visible bias phase gradient')
        self.compute_numerical_gradient(visible_space, self.visible_bias_phase, -alg_grads["visible_bias_phase"])

        print ('Hidden bias phase gradient')
        self.compute_numerical_gradient(visible_space, self.hidden_bias_phase, -alg_grads["hidden_bias_phase"])

    def state_to_index(self, state):
        ''' Only for debugging how the unitary is applied to the unnormalized wavefunction - the 'B' term in alg 4.2.'''
        states = torch.zeros(2**self.num_visible, self.num_visible)
        npstates = states.numpy()
        npstate  = state.numpy()
        for i in range(2**self.num_visible):
            temp = i
            
            for j in range(self.num_visible): 
                temp, remainder = divmod(temp, 2)
                npstates[i][self.num_visible - j - 1] = remainder
            
            if np.array_equal(npstates[i], npstate):
                return i
