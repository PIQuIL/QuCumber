import cplx
import numpy as np
from matplotlib import pyplot as plt
plt.style.use('classic')
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
import warnings
from tqdm import tqdm, tqdm_notebook
import csv
import unitary_library
from observables_tutorial import TFIMChainEnergy, TFIMChainMagnetization

class RBM_Module(nn.Module):
    def __init__(self, num_visible, num_hidden, seed=1234, zero_weights=False):
        super(RBM_Module, self).__init__()
        self.num_visible = int(num_visible)
        self.num_hidden  = int(num_hidden)
        
        if seed != None:
            torch.manual_seed(seed)
        
        self.device = torch.device('cpu')

        if zero_weights:
            self.weights = nn.Parameter((torch.zeros(self.num_hidden, self.num_visible,
                             device=self.device, dtype=torch.double)), requires_grad=True)
        else:
            self.weights = nn.Parameter(
                (torch.randn(self.num_hidden, self.num_visible,
                             device=self.device, dtype=torch.double)
                 / np.sqrt(self.num_visible)),
                requires_grad=True)

        self.visible_bias = nn.Parameter(torch.zeros(self.num_visible,
                                                     device=self.device,
                                                     dtype=torch.double),
                                         requires_grad=True)
        self.hidden_bias = nn.Parameter(torch.zeros(self.num_hidden,
                                                    device=self.device,
                                                    dtype=torch.double),
                                        requires_grad=True)

    def effective_energy(self, v):
        if len(v.shape) < 2:
            v = v.view(1, -1)
        visible_bias_term = torch.mv(v, self.visible_bias)
        hidden_bias_term = F.softplus(
            F.linear(v, self.weights, self.hidden_bias)
        ).sum(1)

        return visible_bias_term + hidden_bias_term

    def prob_v_given_h(self, h):
        p = F.sigmoid(F.linear(h, self.weights.t(), self.visible_bias))
        return p

    def prob_h_given_v(self, v):
        p = F.sigmoid(F.linear(v, self.weights, self.hidden_bias))
        return p

    def sample_v_given_h(self, h):
        p = self.prob_v_given_h(h)
        v = p.bernoulli()
        return p, v

    def sample_h_given_v(self, v):
        p = self.prob_h_given_v(v)
        h = p.bernoulli()
        return p, h

    def gibbs_sampling(self, k, v0, sampler=None, observable=None):
        ph, h0 = self.sample_h_given_v(v0)
        v, h = v0, h0

        if sampler != None:
            step_list   = []

            if observable == 'Energy':
                energy_list = [] 
                energy_calc = TFIMChainEnergy(h=0.0)
                energy_list.append(energy_calc.apply(v0, sampler).mean())
                step_list.append(0)
                for i in range(k):
                    pv, v = self.sample_v_given_h(h)
                    ph, h = self.sample_h_given_v(v)
            
                    energy_list.append(energy_calc.apply(v, sampler).mean())
                    step_list.append(i+1)
        
                ax1 = plt.axes()
                ax1.plot(step_list, energy_list)
                ax1.grid()
                ax1.set_xlabel('k')
                ax1.set_ylabel('Energy')
                return v0, h0, v, h, ph, step_list, energy_list
    
            if observable == 'Magnetization':
                mag_list = []
                mag_calc = TFIMChainMagnetization()
                mag_list.append(mag_calc.apply(v0).mean())
                step_list.append(0)
                for i in range(k):
                    pv, v = self.sample_v_given_h(h)
                    ph, h = self.sample_h_given_v(v)
                    
                    mag_list.append(mag_calc.apply(v).mean())
                    step_list.append(i+1)   
                
                ax2 = plt.axes()
                ax2.plot(step_list, mag_list)
                ax2.grid()
                ax2.set_xlabel('k')
                ax2.set_ylabel('Magnetization')
                return v0, h0, v, h, ph, step_list, mag_list

        else:
            for i in range(k):
                pv, v = self.sample_v_given_h(h)
                ph, h = self.sample_h_given_v(v)

            return v0, h0, v, h, ph

    def unnormalized_probability(self, v):
        return self.effective_energy(v).exp()

    def log_probability_ratio(self, a, b):
        log_prob_a = self.effective_energy(a)
        log_prob_b = self.effective_energy(b)

        return log_prob_a.sub(log_prob_b)

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
        free_energies = self.effective_energy(visible_space)
        max_free_energy = free_energies.max()

        f_reduced = free_energies - max_free_energy
        logZ = max_free_energy + f_reduced.exp().sum().log()

        return logZ

    def partition(self, visible_space):
        return self.log_partition(visible_space).exp()

    def nll(self, data, visible_space):
        total_free_energy = self.effective_energy(data).sum()
        logZ = self.log_partition(visible_space)
        return (len(data)*logZ) - total_free_energy
    
class BinomialRBM:

    def __init__(self, num_visible, num_hidden, seed=1234):
        self.num_visible = int(num_visible)
        self.num_hidden  = int(num_hidden)
        self.rbm_module  = RBM_Module(num_visible, num_hidden, seed=seed)

    def overlap(self, visible_space, psi):
        sqrt_Z   = (self.rbm_module.partition(visible_space)).sqrt()
        RBM_psi  = ((self.rbm_module.unnormalized_probability(visible_space)).sqrt()) / sqrt_Z

        overlap_ = torch.dot(RBM_psi, psi)
        return overlap_

    def fidelity(self, visible_space, psi):
        return (self.overlap(visible_space, psi))**2.

    def compute_batch_gradients(self, k, batch):
        if len(batch) == 0:
            return (torch.zeros_like(self.rbm_module.weights,
                                     device=self.self.rbm_module.device,
                                     dtype=torch.double),
                    torch.zeros_like(self.rbm_module.visible_bias,
                                     device=self.rbm_module.device,
                                     dtype=torch.double),
                    torch.zeros_like(self.rbm_module.hidden_bias,
                                     device=self.rbm_module.device,
                                     dtype=torch.double))

        v0, h0, vk, hk, phk = self.rbm_module.gibbs_sampling(k, batch)

        prob = F.sigmoid(F.linear(v0, self.rbm_module.weights, 
                         self.rbm_module.hidden_bias))

        w_grad  = torch.einsum("ij,ik->jk", (h0, v0))
        vb_grad = torch.einsum("ij->j", (v0,))
        hb_grad = torch.einsum("ij->j", (h0,))

        w_grad  -= torch.einsum("ij,ik->jk", (phk, vk))
        vb_grad -= torch.einsum("ij->j", (vk,))
        hb_grad -= torch.einsum("ij->j", (phk,))

        w_grad  /= float(len(batch))
        vb_grad /= float(len(batch))
        hb_grad /= float(len(batch))

        # Return negative gradients to match up nicely with the usual
        # parameter update rules, which *subtract* the gradient from
        # the parameters. This is in contrast with the RBM update
        # rules which ADD the gradients (scaled by the learning rate)
        # to the parameters.

        return {"rbm_module": {"weights": -w_grad,
             "visible_bias": -vb_grad,
             "hidden_bias": -hb_grad}}

    def fit(self, data, psi, epochs, batch_size, k, lr, log_every):

        progress_bar = tqdm_notebook

        data = torch.tensor(data, device=self.rbm_module.device, dtype=torch.double)
        optimizer = torch.optim.SGD([self.rbm_module.weights,
                                     self.rbm_module.visible_bias,
                                     self.rbm_module.hidden_bias],
                                     lr=lr)

        vis = self.rbm_module.generate_visible_space()

        fidelity_list = []
        epoch_list = []

        for ep in progress_bar(range(0, epochs + 1), desc="Training: ", total=epochs):
            
            batches = DataLoader(data, batch_size=batch_size, shuffle=True)

            if ep % log_every == 0:

                fidelity_ = self.fidelity(vis, psi)
                print ('Epoch = ',ep,'\tFidelity = %.5f' %fidelity_.item())
                fidelity_list.append(fidelity_)
                epoch_list.append(ep)
           
            # Save fidelities at the end of training.
            if ep == epochs:
                print ('Finished training.' )               

                ax = plt.axes()
                ax.plot(epoch_list, fidelity_list)
                ax.grid()
                ax.set_xlabel('Epochs')
                ax.set_ylabel('Fidelity')

                self.save_params()
                print ('Saved weights and biases.')
                break

            for batch in batches:
                all_grads = self.compute_batch_gradients(k, batch)

                optimizer.zero_grad()  # clear any cached gradients

                # assign all available gradients to the corresponding parameter
                for name, grads  in all_grads.items():
                    selected_RBM = getattr(self, name)
                    for param in grads.keys():
                        getattr(selected_RBM, param).grad = grads[param]
                
                optimizer.step()  # tell the optimizer to apply the gradients

    def save_params(self):
        """A function that saves weights and biases individually."""
        trained_params = [self.rbm_module.weights.data.numpy(), 
                          self.rbm_module.visible_bias.data.numpy(), 
                          self.rbm_module.hidden_bias.data.numpy()] 
    
        names = ["Weights", "Visible bias", "Hidden bias"]
        
        with open('trained_weights.csv', 'w') as csvfile1:
            np.savetxt(csvfile1, trained_params[0], fmt='%.5f', delimiter=',')
        
        with open('trained_visible_bias.csv', 'w') as csvfile2:
            np.savetxt(csvfile2, trained_params[1], fmt='%.5f', delimiter=',')

        with open('trained_hidden_bias.csv', 'w') as csvfile3:
            np.savetxt(csvfile3, trained_params[2], fmt='%.5f', delimiter=',')
    
    def sample(self, num_samples, k, sampler=None, observable=None):
        dist = torch.distributions.bernoulli.Bernoulli(probs=0.5)
        v0 = (dist.sample(torch.Size([num_samples, self.num_visible]))
                  .to(dtype=torch.double))

        if sampler != None:
            _, _, v, _, _, step_list, energy_list = self.rbm_module.gibbs_sampling(k, v0, sampler=sampler, observable=observable)
            return v

        else:
            _, _, v, _, _ = self.rbm_module.gibbs_sampling(k, v0)
            return v

class ComplexRBM:
    def __init__(self, num_visible, num_hidden_amp, num_hidden_phase, seed=1234):

        self.num_visible      = int(num_visible)
        self.num_hidden_amp   = int(num_hidden_amp)
        self.num_hidden_phase = int(num_hidden_phase)
        self.rbm_amp          = RBM_Module(num_visible, num_hidden_amp, seed=seed)
        self.rbm_phase        = RBM_Module(num_visible, num_hidden_phase, zero_weights=True, seed=None)
        self.device           = self.rbm_amp.device
            
    def basis_state_generator(self, s):
        if s == 0.:
            return torch.tensor([[1., 0.],[0., 0.]], dtype = torch.double)
        if s == 1.:
            return torch.tensor([[0., 1.],[0., 0.]], dtype = torch.double) 

    def state_generator(self, num_non_trivial_unitaries):
        states = torch.zeros((2**num_non_trivial_unitaries, num_non_trivial_unitaries), device = self.device, dtype=torch.double)
        for i in range(2**num_non_trivial_unitaries):
            temp = i
            for j in range(num_non_trivial_unitaries): 
                temp, remainder = divmod(temp, 2)
                states[i][num_non_trivial_unitaries - j - 1] = remainder
        return states

    def unnormalized_probability_amp(self, v):
        return self.rbm_amp.effective_energy(v).exp()

    def unnormalized_probability_phase(self, v):
        return self.rbm_phase.effective_energy(v).exp()

    def normalized_wavefunction(self, v):
        v_prime   = v.view(-1,self.num_visible)
        temp1     = (self.unnormalized_probability_amp(v_prime)).sqrt()
        temp2     = ((self.unnormalized_probability_phase(v_prime)).log())*0.5

        cos_angle = temp2.cos()
        sin_angle = temp2.sin()
        
        psi       = torch.zeros(2, v_prime.size()[0], dtype = torch.double)
        psi[0]    = temp1*cos_angle
        psi[1]    = temp1*sin_angle

        sqrt_Z    = (self.rbm_amp.partition(self.rbm_amp.generate_visible_space())).sqrt()

        return psi / sqrt_Z

    def unnormalized_wavefunction(self, v):
        v_prime   = v.view(-1,self.num_visible)
        temp1     = (self.unnormalized_probability_amp(v_prime)).sqrt()
        temp2     = ((self.unnormalized_probability_phase(v_prime)).log())*0.5
        cos_angle = temp2.cos()
        sin_angle = temp2.sin()
        
        psi       = torch.zeros(2, v_prime.size()[0], dtype = torch.double)
        
        psi[0]    = temp1*cos_angle
        psi[1]    = temp1*sin_angle

        return psi

    def overlap(self, visible_space, true_psi):
        overlap_ = cplx.inner_prod(true_psi, self.normalized_wavefunction(visible_space))
        return overlap_

    def fidelity(self, visible_space, true_psi):
        return cplx.norm(self.overlap(visible_space, true_psi))

    def compute_batch_gradients(self, unitaries, k, batch, chars_batch):
        vis = self.rbm_amp.generate_visible_space()
        batch_size = len(batch)

        g_weights_amp   = torch.zeros_like(self.rbm_amp.weights)
        g_vb_amp        = torch.zeros_like(self.rbm_amp.visible_bias)
        g_hb_amp        = torch.zeros_like(self.rbm_amp.hidden_bias)

        g_weights_phase = torch.zeros_like(self.rbm_phase.weights)
        g_vb_phase      = torch.zeros_like(self.rbm_phase.visible_bias)
        g_hb_phase      = torch.zeros_like(self.rbm_phase.hidden_bias)

        [batch, h0_amp_batch, vk_amp_batch, hk_amp_batch, phk_amp_batch] = self.rbm_amp.gibbs_sampling(k, batch) 
        # Iterate through every data point in the batch.
        for row_count, v0 in enumerate(batch):

            # A counter for the number of non-trivial unitaries 
            # (non-computational basis) in the data point.
            num_non_trivial_unitaries = 0

            # tau_indices will contain the index numbers of spins not in the 
            # computational basis (Z). z_indices will contain the index numbers 
            # of spins in the computational basis.
            tau_indices = []
            z_indices   = []

            for j in range(self.num_visible):
                # Go through the unitaries (chars_batch[row_count]) of each site
                # in the data point, v0, and save inidices of non-trivial.
                if chars_batch[row_count][j] != 'Z':
                    num_non_trivial_unitaries += 1
                    tau_indices.append(j)
                else:
                    z_indices.append(j)

            if num_non_trivial_unitaries == 0:
                # If there are no non-trivial unitaries for the data point v0, 
                # calculate the positive phase of regular (i.e. non-complex RBM)
                # gradient. Use the actual data point, v0.
                prob_amp = F.sigmoid(F.linear(v0, self.rbm_amp.weights, self.rbm_amp.hidden_bias))
                g_weights_amp -= torch.einsum("i,j->ij", (prob_amp, v0)) / batch_size
                g_vb_amp      -= v0 / batch_size
                g_hb_amp      -= prob_amp / batch_size

            else:
                # Compute the rotated gradients.
                [L_weights_amp, L_vb_amp, L_hb_amp, L_weights_phase, L_vb_phase, L_hb_phase] = self.compute_rotated_grads(unitaries, k, v0, chars_batch[row_count], num_non_trivial_unitaries, z_indices, tau_indices) 


                # Gradents of amplitude parameters take the real part of the 
                # rotated gradients.
                g_weights_amp -= L_weights_amp[0] / batch_size
                g_vb_amp      -= L_vb_amp[0] / batch_size
                g_hb_amp      -= L_hb_amp[0] / batch_size
                
                # Gradents of phase parameters take the imaginary part of the 
                # rotated gradients.
                g_weights_phase += L_weights_phase[1] / batch_size
                g_vb_phase      += L_vb_phase[1] / batch_size
                g_hb_phase      += L_hb_phase[1] / batch_size
                
        # Block gibbs sampling for negative phase.
        g_weights_amp += torch.einsum("ij,ik->jk", (phk_amp_batch, vk_amp_batch)) / batch_size
        g_vb_amp      += torch.einsum("ij->j", (vk_amp_batch,)) / batch_size
        g_hb_amp      += torch.einsum("ij->j", (phk_amp_batch,)) / batch_size
       
        """Return negative gradients to match up nicely with the usual
        parameter update rules, which *subtract* the gradient from
        the parameters. This is in contrast with the RBM update
        rules which ADD the gradients (scaled by the learning rate)
        to the parameters."""
 
        return { "rbm_amp":{"weights": g_weights_amp,
                "visible_bias": g_vb_amp,
                "hidden_bias": g_hb_amp},
                 "rbm_phase":{
                "weights": g_weights_phase,
                "visible_bias": g_vb_phase,
                "hidden_bias": g_hb_phase}
                }   

    def compute_rotated_grads(self, unitaries, k, v0, characters, num_non_trivial_unitaries, 
                              z_indices, tau_indices): 
        A_weights_amp = torch.zeros(2, self.rbm_amp.weights.size()[0], 
                                    self.rbm_amp.weights.size()[1], 
                                    device=self.device, dtype = torch.double)
        A_vb_amp      = torch.zeros(2, self.rbm_amp.visible_bias.size()[0], 
                                    device=self.device, dtype = torch.double)
        A_hb_amp      = torch.zeros(2, self.rbm_amp.hidden_bias.size()[0], 
                                    device=self.device, dtype = torch.double)
        
        A_weights_phase = torch.zeros(2, self.rbm_phase.weights.size()[0], 
                                      self.rbm_phase.weights.size()[1], 
                                      device=self.device, dtype = torch.double)
        A_vb_phase      = torch.zeros(2, self.rbm_phase.visible_bias.size()[0], 
                                      device=self.device, dtype = torch.double)
        A_hb_phase      = torch.zeros(2, self.rbm_phase.hidden_bias.size()[0], 
                                      device=self.device, dtype = torch.double)
        # 'B' will contain the coefficients of the rotated unnormalized wavefunction.
        B = torch.zeros(2, device = self.device, dtype = torch.double)

        w_grad_amp      = torch.zeros_like(self.rbm_amp.weights)
        vb_grad_amp     = torch.zeros_like(self.rbm_amp.visible_bias)
        hb_grad_amp     = torch.zeros_like(self.rbm_amp.hidden_bias)
    
        w_grad_phase    = torch.zeros_like(self.rbm_phase.weights)
        vb_grad_phase   = torch.zeros_like(self.rbm_phase.visible_bias)
        hb_grad_phase   = torch.zeros_like(self.rbm_phase.hidden_bias)

        zeros_for_w_amp    = torch.zeros_like(w_grad_amp)
        zeros_for_w_phase  = torch.zeros_like(w_grad_phase)
        zeros_for_vb       = torch.zeros_like(vb_grad_amp)
        zeros_for_hb_amp   = torch.zeros_like(hb_grad_amp) 
        zeros_for_hb_phase = torch.zeros_like(hb_grad_phase)

        # Loop over Hilbert space of the non trivial unitaries to build the state.
        for j in range(2**num_non_trivial_unitaries):
            s = self.state_generator(num_non_trivial_unitaries)[j]
            # Creates a matrix where the jth row is the desired state, |S>, a vector.
    
            # This is the sigma state.
            constructed_state = torch.zeros(self.num_visible, dtype = torch.double)
            
            U = torch.tensor([1., 0.], dtype = torch.double, device = self.device)
        
            # Populate the |sigma> state (aka constructed_state) accirdingly.
            for index in range(len(z_indices)):
                # These are the sites in the computational basis.
                constructed_state[z_indices[index]] = v0[z_indices[index]]
        
            for index in range(len(tau_indices)):
                # These are the sites that are NOT in the computational basis.
                constructed_state[tau_indices[index]] = s[index]
        
                aa = unitaries[characters[tau_indices[index]]]
                bb = self.basis_state_generator(v0[tau_indices[index]])
                cc = self.basis_state_generator(s[index])
            
                temp = cplx.inner_prod(cplx.MV_mult(cplx.compT_matrix(aa), bb), cc)
        
                U = cplx.scalar_mult(U, temp)
            
            # Positive phase gradients for phase and amp. Will be added into the
            # 'A' parameters.

            prob_amp   = F.sigmoid(F.linear(constructed_state, self.rbm_amp.weights, self.rbm_amp.hidden_bias))
            prob_phase = F.sigmoid(F.linear(constructed_state, self.rbm_phase.weights, self.rbm_phase.hidden_bias))
 
            w_grad_amp  = torch.einsum("i,j->ij", (prob_amp, constructed_state))
            vb_grad_amp = constructed_state
            hb_grad_amp = prob_amp
            
            w_grad_phase  = torch.einsum("i,j->ij", (prob_phase, constructed_state))
            vb_grad_phase = constructed_state
            hb_grad_phase = prob_phase

            temp_w_grad_amp  = cplx.make_complex_matrix(w_grad_amp, 
                                                        zeros_for_w_amp)
            temp_vb_grad_amp = cplx.make_complex_vector(vb_grad_amp, 
                                                        zeros_for_vb)
            temp_hb_grad_amp = cplx.make_complex_vector(hb_grad_amp, 
                                                        zeros_for_hb_amp)
 
            temp_w_grad_phase  = cplx.make_complex_matrix(w_grad_phase, 
                                                          zeros_for_w_phase)
            temp_vb_grad_phase = cplx.make_complex_vector(vb_grad_phase,
                                                          zeros_for_vb)
            temp_hb_grad_phase = cplx.make_complex_vector(hb_grad_phase,
                                                          zeros_for_hb_phase)
        
            # Temp = U*psi(sigma)
            temp = cplx.scalar_mult(U, self.unnormalized_wavefunction(constructed_state))
            
            A_weights_amp += cplx.MS_mult(temp, temp_w_grad_amp)
            A_vb_amp      += cplx.VS_mult(temp, temp_vb_grad_amp)
            A_hb_amp      += cplx.VS_mult(temp, temp_hb_grad_amp)
        
            A_weights_phase += cplx.MS_mult(temp, temp_w_grad_phase)
            A_vb_phase      += cplx.VS_mult(temp, temp_vb_grad_phase)
            A_hb_phase      += cplx.VS_mult(temp, temp_hb_grad_phase)
           
            # Rotated wavefunction.
            B += temp
        
        L_weights_amp = cplx.MS_divide(A_weights_amp, B)
        L_vb_amp      = cplx.VS_divide(A_vb_amp, B)
        L_hb_amp      = cplx.VS_divide(A_hb_amp, B)
        
        L_weights_phase = cplx.MS_divide(A_weights_phase, B)
        L_vb_phase      = cplx.VS_divide(A_vb_phase, B)
        L_hb_phase      = cplx.VS_divide(A_hb_phase, B)
       
        return [L_weights_amp, L_vb_amp, L_hb_amp, L_weights_phase, L_vb_phase,
                L_hb_phase ]

    def fit(self, data, character_data, true_psi, unitaries, epochs, batch_size, k, lr, log_every):
        # Make data file into a torch tensor.
        data = torch.tensor(data, dtype = torch.double, device=self.device)

        # Use the Adam optmizer to update the weights and biases.
        optimizer = torch.optim.Adam([self.rbm_amp.weights,
                                      self.rbm_amp.visible_bias,
                                      self.rbm_amp.hidden_bias,
                                      self.rbm_phase.weights,
                                      self.rbm_phase.visible_bias,
                                      self.rbm_phase.hidden_bias],
                                      lr=lr)

        progress_bar = tqdm_notebook
        vis = self.rbm_amp.generate_visible_space()
    
        # Empty lists to put calculated convergence quantities in.
        fidelity_list = []
        epoch_list = []

        for ep in progress_bar(range(0, epochs + 1), desc="Training: ", total=epochs):
            # Shuffle the data to ensure that the batches taken from the data 
            # are random data points.
            random_permutation = torch.randperm(data.shape[0])

            shuffled_data           = data[random_permutation]   
            shuffled_character_data = character_data[random_permutation]

            # List of all the batches.
            batches = [shuffled_data[batch_start:(batch_start + batch_size)] 
                       for batch_start in range(0, len(data), batch_size)]

            # List of all the bases.
            char_batches = [shuffled_character_data[batch_start:(batch_start + batch_size)] 
                            for batch_start in range(0, len(data), batch_size)]

            # Calculate convergence quantities every "log-every" steps.
            if ep % log_every == 0:
                fidelity_ = self.fidelity(vis, true_psi)
                print ('Epoch = ',ep,'\nFidelity = ',fidelity_)
                fidelity_list.append(fidelity_)
                epoch_list.append(ep)
           
            # Save fidelities at the end of training.
            if ep == epochs:
                print ('Finished training. Saving results...' )               
                ax = plt.axes()
                ax.plot(epoch_list, fidelity_list)
                ax.grid()
                ax.set_xlabel('Epochs')
                ax.set_ylabel('Fidelity')
                self.save_params()
                print ('Done.')
                break

            # Loop through all of the batches and calculate the batch gradients.

            for index in range(len(batches)):           
                all_grads = self.compute_batch_gradients(unitaries, k, batches[index],
                                                     char_batches[index])
                
                # Clear any cached gradients.
                optimizer.zero_grad()  

                # Assign all available gradients to the corresponding parameter.
                for name, grads  in all_grads.items():
                    selected_RBM = getattr(self, name)
                    for param in grads.keys():
                        getattr(selected_RBM, param).grad = grads[param]
               
                # Tell the optimizer to apply the gradients and update the parameters.
                optimizer.step()  

    def save_params(self):
        trained_params = [self.rbm_amp.weights.data.numpy(), 
                          self.rbm_amp.visible_bias.data.numpy(), 
                          self.rbm_amp.hidden_bias.data.numpy(), 
                          self.rbm_phase.weights.data.numpy(), 
                          self.rbm_phase.visible_bias.data.numpy(), 
                          self.rbm_phase.hidden_bias.data.numpy()]
        
        with open('trained_weights_amp.csv', 'w') as csvfile:
            np.savetxt(csvfile, trained_params[0], fmt='%.5f', delimiter=',')

        with open('trained_visible_bias_amp.csv', 'w') as csvfile:      
            np.savetxt(csvfile, trained_params[1], fmt='%.5f', delimiter=',')
            
        with open('trained_hidden_bias_amp.csv', 'w') as csvfile:
            np.savetxt(csvfile, trained_params[2], fmt='%.5f', delimiter=',')
            
        with open('trained_weights_phase.csv', 'w') as csvfile:
            np.savetxt(csvfile, trained_params[3], fmt='%.5f', delimiter=',')
            
        with open('trained_visible_bias_phase.csv', 'w') as csvfile:
            np.savetxt(csvfile, trained_params[4], fmt='%.5f', delimiter=',')
            
        with open('trained_hidden_bias_phase.csv', 'w') as csvfile:
            np.savetxt(csvfile, trained_params[5], fmt='%.5f', delimiter=',')

    def sample(self, num_samples, k):
        dist = torch.distributions.bernoulli.Bernoulli(probs=0.5)
        v0 = (dist.sample(torch.Size([num_samples, self.num_visible]))
                  .to(dtype=torch.double))
        _, _, v, _, _ = self.rbm_amp.gibbs_sampling(k, v0)

        return v
