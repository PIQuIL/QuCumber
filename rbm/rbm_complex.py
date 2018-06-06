import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import tqdm, tqdm_notebook
import warnings

class RBM(nn.Module):
    def __init__(self, num_visible, num_hidden_amp, num_hidden_phase, gpu=True, unitary_name, seed=1234):
        super(RBM, self).__init__()
        self.num_visible      = int(num_visible)
        self.num_hidden_amp   = int(num_hidden_amp)
        self.num_hidden_phase = int(num_hidden_phase)
        self.unitary_name     = unitary_name

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

    def regularize_weight_gradients(self, w_grad, l1_reg, l2_reg):
        return (w_grad
                + (l2_reg * self.weights)
                + (l1_reg * self.weights.sign()))

    def compute_batch_gradients(self, k, batch, chars_batch, l1_reg, l2_reg, stddev=0):
        '''This function will compute the gradients of a batch of the training data (data_file) given the basis measurements (chars_file).'''
        '''
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
        '''
        '''For the gradient update.'''
        # TODO: must make these complex...
        A_weights_amp = torch.zeros(2, self.weights_amp.size()[0], self.weights_amp.size()[1], device=self.device)
        A_vb_amp      = torch.zeros(2, self.visible_biases_amp.size()[0], device=self.device)
        A_hb_amp      = torch.zeros(2, self.hidden_biases_amp.size()[0], device=self.device)
        
        A_weights_phase = torch.zeros(2, self.weights_phase.size()[0], self.weights_phase.size()[1], device=self.device)
        A_vb_phase      = torch.zeros(2, self.visible_biases_phase.size()[0], device=self.device)
        A_hb_phase      = torch.zeros(2, self.hidden_biases_phase.size()[0], device=self.device)

        B = 0.
    
        grad_weights_amp = torch.zeros_like(A_weights_amp)
        grad_vb_amp      = torch.zeros_like(A_vb_amp)
        grad_hb_amp      = torch.zeros_like(A_hb_amp)
    
        grad_weights_phase = torch.zeros_like(A_weights_phase)
        grad_vb_phase      = torch.zeros_like(A_vb_phase)
        grad_hb_phase      = torch.zeros_like(A_hb_phase)

        for v0 in batch:

            '''Giacomo stuff: (alg 4.2 and 4.3 in pseudo code)'''
            num_non_trivial_unitaries = 0
            
            '''tau_indices will contain the index numbers of spins not in the computational basis (Z). z_indices will contain the index numbers of spins in the computational basis.'''
            tau_indices = []
            z_indices   = []

            for j in range(chars_batch.shape[1]):
                if chars_batch[i][j] != 'z':
                    num_non_trivial_unitaries += 1
                    tau_indices.append(j)

                else:
                    z_indices.append(j)

            constructed_state = torch.zeros(self.num_visible)

            for j in range(2**num_non_trivial_unitaries):
                s                 = self.state_generator(num_non_trivial_unitaries)[j]
                U = 1.

                for index in range(len(z_indices)):
                    constructed_state[z_indices[index]] = data[i][z_indices[index]]

                for index in range(len(tau_indices)):
                    constructed_state[tau_indices[index]] = s[index]
                    U *= cplx_DOT( cplx_MV(self.unitary(self.unitary_name), self.basis_state_generator(data[i][tau_indices[index]])), 
                                   self.basis_state_generator(s[index]) ) 

            '''Gradients for phase and amp.'''
            w_grad_amp  = torch.matmul(F.sigmoid(F.linear(constructed_state, self.weights_amp.t(), self.hidden_bias_amp)), 
                                       constructed_state) 
            vb_grad_amp = constructed_state
            hb_grad_amp = F.sigmoid(F.linear(constructed_state, self.weights_amp.t(), self.hidden_bias_amp))

            
            w_grad_phase  = torch.matmul(F.sigmoid(F.linear(constructed_state, self.weights_phase.t(), self.hidden_bias_phase)), 
                                       constructed_state) 
            vb_grad_phase = constructed_state
            hb_grad_phase = F.sigmoid(F.linear(constructed_state, self.weights_phase.t(), self.hidden_bias_phase))

            
            w_grad_amp  = self.regularize_weight_gradients(w_grad, l1_reg, l2_reg)
            w_grad_amp += (stddev * torch.randn_like(w_grad, device=self.device))

            w_grad_phase  = self.regularize_weight_gradients(w_grad, l1_reg, l2_reg)
            w_grad_phase += (stddev * torch.randn_like(w_grad, device=self.device))

            w_grad_amp  /= self.unnormalized_probability_amp(constructed_state)
            vb_grad_amp /= self.unnormalized_probability_amp(constructed_state)
            hb_grad_amp /= self.unnormalized_probability_amp(constructed_state)

            w_grad_phase  /= self.unnormalized_probability_phase(constructed_state)
            vb_grad_phase /= self.unnormalized_probability_phase(constructed_state)
            hb_grad_phase /= self.unnormalized_probability_phase(constructed_state)
            
            A_weights_amp += cplx_VS(cplx_SS(U, self.unnormalized_wavefunction(constructed_state)), weight_grad_amp)
            A_vb_amp      += cplx_VS(cplx_SS(U, self.unnormalized_wavefunction(constructed_state)), vb_grad_amp)
            A_hb_amp      += cplx_VS(cplx_SS(U, self.unnormalized_wavefunction(constructed_state)), hb_grad_amp)

            A_weights_phase += cplx_VS(cplx_SS(U, self.unnormalized_wavefunction(constructed_state)), weight_grad_phase)
            A_vb_phase      += cplx_VS(cplx_SS(U, self.unnormalized_wavefunction(constructed_state)), vb_grad_phase)
            A_hb_phase      += cplx_VS(cplx_SS(U, self.unnormalized_wavefunction(constructed_state)), hb_grad_phase)
            
            B += cplx_SS(U, self.unnormalized_wavefunction(constructed_state))

            L_weights_amp = cplx_divideVS(B, A_weights_amp)
            L_vb_amp      = cplx_divideVS(B, A_vb_amp)
            L_hb_amp      = cplx_divideVS(B, A_hb_amp)

            L_weights_phase = cplx_divideVS(B, A_weights_phase)
            L_vb_phase      = cplx_divideVS(B, A_vb_phase)
            L_hb_phase      = cplx_divideVS(B, A_hb_phase)

            grad_weights_amp -= L_weights_amp[0]/batch_size
            grad_vb_amp      -= L_vb_amp[0]/batch_size
            grad_hb_amp      -= L_hb_amp[0]/batch_size
    
            grad_weights_phase += L_weights_phase[1]/batch_size
            grad_vb_phase      += L_vb_phase[1]/batch_size
            grad_hb_phase      += L_hb_phase[1]/batch_size
        
        ''' Mc = batch size for now... should fix this '''
        for v0 in batch:
            v0_amp, h0_amp, vk_amp, hk_amp, phk_amp = self.gibbs_sampling_amp(k, v0)

            w_grad_amp  = torch.einsum("j,k->jk", (h0_amp, v0_amp))
            vb_grad_amp = v0_phase
            hb_grad_amp = h0_phase
    
            w_grad_amp  -= torch.einsum("j,k->jk", (phk_amp, vk_amp))
            vb_grad_amp -= vk_amp
            hb_grad_amp -= phk_amp

            w_grad_amp  /= self.unnormalized_probability_amp(vk_amp)
            vb_grad_amp /= self.unnormalized_probability_amp(vk_amp)
            hb_grad_amp /= self.unnormalized_probability_amp(vk_amp)

            grad_weights_amp += w_grad_amp/batch_size
            grad_vb_amp      += vb_grad_amp/batch_size
            grad_hb_amp      += hb_grad_amp/batch_size

        # Return negative gradients to match up nicely with the usual
        # parameter update rules, which *subtract* the gradient from
        # the parameters. This is in contrast with the RBM update
        # rules which ADD the gradients (scaled by the learning rate)
        # to the parameters.
        return {"weight_amp": -grad_weights_amp,
                "visible_bias_amp": -grad_vb_amp,
                "hidden_bias_amp": -grad_hb_amp,
                "weights_phase": -grad_weights_phase,
                "visible_bias_phase": -grad_vb_phase,
                "hidden_bias_phase": -grad_hb_phase
                }

    def train(self, data, character_data epochs, batch_size,
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
        for ep in progress_bar(range(epochs + 1), desc="Epochs ",
                               total=epochs, disable=disable_progbar):
            
            random_permutation = torch.randperm(data.shape[0])

            shuffled_data           = data[random_permutation]   
            shuffled_character_data = character_data[random_permutation]

            batches = [shuffled_data[batch_start:(batch_start + batch_size)] 
                       for batch_start in range(0, len(data), batch_size)]

            char_batches = [shuffled_character_data[batch_start:(batch_start + batch_size)] 
                            for batch_start in range(0, len(data), batch_size)]

            if ep % log_every == 0:
                logZ = self.log_partition(vis)
                nll = self.nll(data, logZ)
                tqdm.write("{}: {}".format(ep, nll.item() / len(data)))

            if ep == epochs:
                break

            stddev = torch.tensor(
                [initial_gaussian_noise / ((1 + ep) ** gamma)],
                dtype=torch.double, device=self.device).sqrt()

            for batch in progress_bar(batches, desc="Batches",
                                      leave=False, disable=True):

                grads = self.compute_batch_gradients(k, batch, char_batches,
                                                     l1_reg, l2_reg,
                                                     stddev=stddev)

                optimizer.zero_grad()  # clear any cached gradients

                # assign all available gradients to the corresponding parameter
                for name in grads.keys():
                    getattr(self, name).grad = grads[name]

                optimizer.step()  # tell the optimizer to apply the gradients
            # TODO: run callbacks

    def free_energy_amp(self, v):
        if len(v.shape) < 2:
            v = v.view(1, -1)
        visible_bias_term = torch.mv(v, self.visible_bias_amp)
        hidden_bias_term = F.softplus(
            F.linear(v, self.weights_amp, self.hidden_bias_amp)
        ).sum(1)

        return visible_bias_term + hidden_bias_term

    def free_energy_phase(self, v):
        if len(v.shape) < 2:
            v = v.view(1, -1)
        visible_bias_term = torch.mv(v, self.visible_bias_phase)
        hidden_bias_term = F.softplus(
            F.linear(v, self.weights_phase, self.hidden_bias_phase)
        ).sum(1)

        return visible_bias_term + hidden_bias_term

    def unnormalized_probability_amp(self, v):
        return self.free_energy_amp(v).exp()

    def unnormalized_probability_phase(self, v):
        return self.free_energy_phase(v).exp()

    def unnormalized_wavefunction(self, v):
        temp1     = self.unnormalized_probability_amp(v).sqrt()
        temp2     = (self.unnormalized_probability_phase(v).log())*0.5
        cos_angle = temp2.cos()
        sin_angle = temp2.sin()
        psi[0]    = temp1*cos_angle
        psi[1]    = temp2*sin_angle
        return psi

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
        free_energies = self.free_energy(visible_space)
        max_free_energy = free_energies.max()

        f_reduced = free_energies - max_free_energy
        logZ = max_free_energy + f_reduced.exp().sum().log()

        return logZ

    def partition(self, visible_space):
        return self.log_partition(visible_space).exp()

    def nll(self, data, logZ):
        total_free_energy = self.free_energy(data).sum()

        return (len(data)*logZ) - total_free_energy

    def f_length(self, path):i
        '''A function that returns the number of rows in a text file.'''
        f = open('{!s}'.format(path))
        num_rows = len(f.readlines())
        f.close()
        return num_rows

    def unitary(self):
        '''A function that pytrochifies the unitary matrix given its name. It must be in the unitary_library!'''
        num_rows = self.f_length('unitary_library.txt')
        with open('unitary_library.txt') as f:
            for i, line in enumerate(f):
                if self.unitary_name in line:
                    a = torch.from_numpy(np.genfromtxt('unitary_library.txt', delimiter='\t', skip_header = i+1, skip_footer = num_rows - i - 3))
                    b = torch.from_numpy(np.genfromtxt('unitary_library.txt', delimiter='\t', skip_header = i+3, skip_footer = num_rows - i - 5))
            f.close()
        return make_cplx(a,b)

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
        '''If s = 0, this is the (1,0) state in the basis of the measurement. If s = 1, this is the (0,1) state in the basis of the measurement.'''
        if s == 0.:
            return torch.tensor([[1., 0.],[0., 0.]])
        if s == 1.:
            return torch.tensor([[0., 1.],[0., 0.]]) 
