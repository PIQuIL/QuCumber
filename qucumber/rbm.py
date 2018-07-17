# Copyright 2018 PIQuIL - All Rights Reserved

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import warnings
from itertools import chain

import numpy as np
from math import sqrt
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm, tqdm_notebook

import qucumber.cplx as cplx
from qucumber.samplers import Sampler
from qucumber.callbacks import CallbackList

__all__ = [
    "BinomialRBMModule",
    "BinomialRBM"
]


def _warn_on_missing_gpu(gpu):
    if gpu and not torch.cuda.is_available():
        warnings.warn("Could not find GPU: will continue with CPU.",
                      ResourceWarning)


class BinomialRBMModule(nn.Module, Sampler):
    def __init__(self, num_visible, num_hidden, zero_weights=False,
                 gpu=True, seed=None):
        super(BinomialRBMModule, self).__init__()
        self.num_visible = int(num_visible)
        self.num_hidden = int(num_hidden)

        _warn_on_missing_gpu(gpu)
        self.gpu = gpu and torch.cuda.is_available()

        if seed:
            if self.gpu:
                torch.cuda.manual_seed(seed)
            else:
                torch.manual_seed(seed)

        self.device = torch.device('cuda') if self.gpu else torch.device('cpu')

        if zero_weights:
            self.weights = nn.Parameter((torch.zeros(self.num_hidden,
                                                     self.num_visible,
                                                     device=self.device,
                                                     dtype=torch.double)),
                                        requires_grad=True)
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

    def __repr__(self):
        return ("BinomialRBMModule(num_visible={}, num_hidden={}, gpu={})"
                .format(self.num_visible, self.num_hidden, self.gpu))

    def effective_energy(self, v):
        r"""The effective energies of the given visible states.

        .. math::

            \mathcal{E}(\bm{v}) &= \sum_{j}b_j v_j
                        + \sum_{i}\log
                            \left\lbrack 1 +
                                  \exp\left(c_{i} + \sum_{j} W_{ij} v_j\right)
                            \right\rbrack

        :param v: The visible states.
        :type v: torch.Tensor

        :returns: The effective energies of the given visible states.
        :rtype: torch.Tensor
        """
        if len(v.shape) < 2:
            v = v.view(1, -1)
        visible_bias_term = torch.mv(v, self.visible_bias)
        hidden_bias_term = F.softplus(
            F.linear(v, self.weights, self.hidden_bias)
        ).sum(1)

        return visible_bias_term + hidden_bias_term

    def prob_v_given_h(self, h):
        """Given a hidden unit configuration, compute the probability
        vector of the visible units being on.

        :param h: The hidden unit
        :type h: torch.Tensor

        :returns: The probability of visible units being active given the
                  hidden state.
        :rtype: torch.Tensor
        """
        p = F.sigmoid(F.linear(h, self.weights.t(), self.visible_bias))
        return p

    def prob_h_given_v(self, v):
        """Given a visible unit configuration, compute the probability
        vector of the hidden units being on.

        :param h: The hidden unit.
        :type h: torch.Tensor

        :returns: The probability of hidden units being active given the
                  visible state.
        :rtype: torch.Tensor
        """
        p = F.sigmoid(F.linear(v, self.weights, self.hidden_bias))
        return p

    def sample_v_given_h(self, h):
        """Sample/generate a visible state given a hidden state.

        :param h: The hidden state.
        :type h: torch.Tensor

        :returns: Tuple containing prob_v_given_h(h) and the sampled visible
                  state.
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        p = self.prob_v_given_h(h)
        v = p.bernoulli()
        return p, v

    def sample_h_given_v(self, v):
        """Sample/generate a hidden state given a visible state.

        :param h: The visible state.
        :type h: torch.Tensor

        :returns: Tuple containing prob_h_given_v(v) and the sampled hidden
                  state.
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        p = self.prob_h_given_v(v)
        h = p.bernoulli()
        return p, h

    def gibbs_sampling(self, k, v0):
        """Performs k steps of Block Gibbs sampling given an initial visible
        state v0.

        :param k: Number of Block Gibbs steps.
        :type k: int
        :param v0: The initial visible state.
        :type v0: torch.Tensor

        :returns: Tuple containing the initial visible state, v0,
                  the hidden state sampled from v0,
                  the visible state sampled after k steps,
                  the hidden state sampled after k steps and its corresponding

                  probability vector.
        :rtype: tuple(torch.Tensor, torch.Tensor,
                      torch.Tensor, torch.Tensor,
                      torch.Tensor)
        """
        ph, h0 = self.sample_h_given_v(v0)
        v, h = v0, h0
        for _ in range(k):
            pv, v = self.sample_v_given_h(h)
            ph, h = self.sample_h_given_v(v)
        return v0, h0, v, h, ph

    def sample(self, num_samples, k=10):
        """Samples from the RBM using k steps of Block Gibbs sampling.
        :param num_samples: The number of samples to be generated
        :type num_samples: int
        :param k: Number of Block Gibbs steps.
        :type k: int
        :returns: Samples drawn from the RBM
        :rtype: torch.Tensor
        """
        dist = torch.distributions.bernoulli.Bernoulli(probs=0.5)
        v0 = (dist.sample(torch.Size([num_samples, self.num_visible]))
                  .to(device=self.device, dtype=torch.double))
        _, _, v, _, _ = self.gibbs_sampling(k, v0)
        return v

    def unnormalized_probability(self, v):
        r"""The unnormalized probabilities of the given visible states.

        .. math:: p(\bm{v}) = e^{\mathcal{E}(\bm{v})}

        :param v: The visible states.
        :type v: torch.Tensor

        :returns: The unnormalized probability of the given visible state(s).
        :rtype: torch.Tensor
        """
        return self.effective_energy(v).exp()

    def probability_ratio(self, a, b):
        return self.log_probability_ratio(a, b).exp()

    def log_probability_ratio(self, a, b):
        return self.effective_energy(a).sub(self.effective_energy(b))

    def generate_visible_space(self):
        """Generates all possible visible states.

        :returns: A tensor of all possible spin configurations.
        :rtype: torch.Tensor
        """
        space = torch.zeros((1 << self.num_visible, self.num_visible),
                            device=self.device, dtype=torch.double)
        for i in range(1 << self.num_visible):
            d = i
            for j in range(self.num_visible):
                d, r = divmod(d, 2)
                space[i, self.num_visible - j - 1] = int(r)

        return space

    def log_partition(self, visible_space):
        """The natural logarithm of the partition function of the RBM.

        :param visible_space: A rank 2 tensor of the entire visible space.
        :type visible_space: torch.Tensor

        :returns: The natural log of the partition function.
        :rtype: torch.Tensor
        """
        free_energies = self.effective_energy(visible_space)
        max_free_energy = free_energies.max()

        f_reduced = free_energies - max_free_energy
        logZ = max_free_energy + f_reduced.exp().sum().log()

        return logZ

    def partition(self, visible_space):
        """The partition function of the RBM.

        :param visible_space: A rank 2 tensor of the entire visible space.
        :type visible_space: torch.Tensor

        :returns: The partition function.
        :rtype: torch.Tensor
        """
        return self.log_partition(visible_space).exp()

    def probability(self, v, Z):
        """Evaluates the probability of the given vector(s) of visible
        units; NOT RECOMMENDED FOR RBMS WITH A LARGE # OF VISIBLE UNITS

        :param v: The visible states.
        :type v: torch.Tensor
        :param Z: The partition function.
        :type Z: float

        :returns: The probability of the given vector(s) of visible units.
        :rtype: torch.Tensor
        """
        return self.unnormalized_probability(v) / Z


class BinomialRBM(Sampler):
    def __init__(self, num_visible, num_hidden=None, gpu=True, seed=None):
        super(BinomialRBM, self).__init__()
        self.num_visible = int(num_visible)
        self.num_hidden = (int(num_hidden)
                           if num_hidden is not None
                           else self.num_visible)
        self.rbm_module = BinomialRBMModule(self.num_visible, self.num_hidden,
                                            gpu=gpu, seed=seed)
        self.stop_training = False

    def save(self, location, metadata={}):
        """Saves the RBM parameters to the given location along with
        any given metadata.

        :param location: The location to save the RBM parameters + metadata
        :type location: str or file
        :param metadata: Any extra metadata to store alongside the RBM
                         parameters
        :type metadata: dict
        """
        # add extra metadata to dictionary before saving it to disk
        data = {**self.rbm_module.state_dict(), **metadata}
        torch.save(data, location)

    def load(self, location):
        """Loads the RBM parameters from the given location ignoring any
        metadata stored in the file. Overwrites the RBM's parameters.

        .. note::
            The RBM object on which this function is called must
            have the same shape as the one who's parameters are being
            loaded.

        :param location: The location to load the RBM parameters from
        :type location: str or file
        """

        try:
            state_dict = torch.load(location)
        except AssertionError as e:
            state_dict = torch.load(location, lambda storage, loc: 'cpu')

        self.rbm_module.load_state_dict(state_dict, strict=False)

    @staticmethod
    def autoload(location, gpu=True):
        """Initializes an RBM from the parameters in the given location,
        ignoring any metadata stored in the file.

        :param location: The location to load the RBM parameters from
        :type location: str or file

        :returns: A new RBM initialized from the given parameters
        :rtype: BinomialRBM
        """
        _warn_on_missing_gpu(gpu)
        gpu = gpu and torch.cuda.is_available()

        if gpu:
            state_dict = torch.load(location, lambda storage, loc: 'cuda')
        else:
            state_dict = torch.load(location, lambda storage, loc: 'cpu')

        rbm = BinomialRBM(num_visible=len(state_dict['visible_bias']),
                          num_hidden=len(state_dict['hidden_bias']),
                          gpu=gpu,
                          seed=None)
        rbm.rbm_module.load_state_dict(state_dict, strict=False)

        return rbm

    def compute_batch_gradients(self, k, pos_batch, neg_batch):
        """This function will compute the gradients of a batch of the training
        data (data_file) given the basis measurements (chars_file).

        :param k: Number of contrastive divergence steps in training.
        :type k: int
        :param pos_batch: Batch of the input data for the positive phase.
        :type pos_batch: torch.Tensor
        :param neg_batch: Batch of the input data for the negative phase.
        :type neg_batch: torch.Tensor

        :returns: Dictionary containing all the gradients of the parameters.
        :rtype: dict
        """

        v0, _, _, _, _ = self.rbm_module.gibbs_sampling(k, pos_batch)
        _, _, vk, hk, phk = self.rbm_module.gibbs_sampling(k, neg_batch)

        prob = F.sigmoid(F.linear(v0, self.rbm_module.weights,
                                  self.rbm_module.hidden_bias))

        pos_batch_size = float(len(pos_batch))
        neg_batch_size = float(len(neg_batch))


        w_grad = torch.einsum("ij,ik->jk", (prob, v0))/pos_batch_size
        vb_grad = torch.einsum("ij->j", (v0,))/pos_batch_size
        hb_grad = torch.einsum("ij->j", (prob,))/pos_batch_size

        w_grad -= torch.einsum("ij,ik->jk", (phk, vk))/neg_batch_size
        vb_grad -= torch.einsum("ij->j", (vk,))/neg_batch_size
        hb_grad -= torch.einsum("ij->j", (phk,))/neg_batch_size

        # Return negative gradients to match up nicely with the usual
        # parameter update rules, which *subtract* the gradient from
        # the parameters. This is in contrast with the RBM update
        # rules which ADD the gradients (scaled by the learning rate)
        # to the parameters.

        return {"rbm_module": {"weights": -w_grad,
                               "visible_bias": -vb_grad,
                               "hidden_bias": -hb_grad}}

    def fit(self, data, epochs=100, pos_batch_size=100, neg_batch_size=200,
            k=1, lr=1e-2, progbar=False, callbacks=[]):
        """Execute the training of the RBM.

        :param data: The actual training data
        :type data: list(float)
        :param epochs: The number of parameter (i.e. weights and biases)
                       updates
        :type epochs: int
        :param pos_batch_size: The size of batches for the positive phase
                               taken from the data.
        :type pos_batch_size: int
        :param neg_batch_size: The size of batches for the negative phase
                               taken from the data
        :type neg_batch_size: int
        :param k: The number of contrastive divergence steps
        :type k: int
        :param lr: Learning rate
        :type lr: float
        :param progbar: Whether or not to display a progress bar. If "notebook"
                        is passed, will use a Jupyter notebook compatible
                        progress bar.
        :type progbar: bool or str
        :param callbacks: Callbacks to run while training.
        :type callbacks: list(qucumber.callbacks.Callback)
        """

        disable_progbar = (progbar is False)
        progress_bar = tqdm_notebook if progbar == "notebook" else tqdm
        callbacks = CallbackList(callbacks)

        data = torch.tensor(data, device=self.rbm_module.device,
                            dtype=torch.double)
        optimizer = torch.optim.SGD([self.rbm_module.weights,
                                     self.rbm_module.visible_bias,
                                     self.rbm_module.hidden_bias],
                                    lr=lr)

        callbacks.on_train_start(self)

        for ep in progress_bar(range(epochs), desc="Epochs ",
                               disable=disable_progbar):
            pos_batches = DataLoader(data, batch_size=pos_batch_size,
                                     shuffle=True)

            multiplier = int((neg_batch_size / pos_batch_size) + 0.5)
            neg_batches = [DataLoader(data, batch_size=neg_batch_size,
                                      shuffle=True)
                           for i in range(multiplier)]
            neg_batches = chain(*neg_batches)

            callbacks.on_epoch_start(self, ep)

            if self.stop_training:  # check for stop_training signal
                break

            for batch_num, (pos_batch, neg_batch) in enumerate(zip(pos_batches,
                                                               neg_batches)):
                callbacks.on_batch_start(self, ep, batch_num)

                all_grads = self.compute_batch_gradients(k, pos_batch,
                                                         neg_batch)
                optimizer.zero_grad()  # clear any cached gradients

                # assign all available gradients to the corresponding parameter
                for name, grads in all_grads.items():
                    selected_RBM = getattr(self, name)
                    for param in grads.keys():
                        getattr(selected_RBM, param).grad = grads[param]

                optimizer.step()  # tell the optimizer to apply the gradients

                callbacks.on_batch_end(self, ep, batch_num)

            callbacks.on_epoch_end(self, ep)

        callbacks.on_train_end(self)

    def sample(self, num_samples, k):
        """Samples from the RBM using k steps of Block Gibbs sampling.
        :param num_samples: The number of samples to be generated
        :type num_samples: int
        :param k: Number of Block Gibbs steps.
        :type k: int
        :returns: Samples drawn from the RBM.
        :rtype: torch.Tensor
        """
        return self.rbm_module.sample(num_samples, k)

    def probability_ratio(self, a, b):
        return self.rbm_module.log_probability_ratio(a, b).exp()

    def log_probability_ratio(self, a, b):
        return self.rbm_module.effective_energy(a) \
                              .sub(self.effective_energy(b))


class ComplexRBM:
    # NOTE: In development. Might be unstable.
    # NOTE: The 'full_unitaries' argument is not needed for training/sampling.
    # This is only here for debugging the gradients. Delete this argument when
    # gradients have been debugged.
    def __init__(self, full_unitaries, psi_dictionary, num_visible,
                 num_hidden_amp, num_hidden_phase, test_grads, gpu=True,
                 seed=1234):

        self.num_visible = int(num_visible)
        self.num_hidden_amp = int(num_hidden_amp)
        self.num_hidden_phase = int(num_hidden_phase)
        self.full_unitaries = full_unitaries
        self.psi_dictionary = psi_dictionary
        self.rbm_amp = RBM_Module(num_visible, num_hidden_amp, gpu=gpu,
                                  seed=seed)
        self.rbm_phase = RBM_Module(num_visible, num_hidden_phase,
                                    zero_weights=True, gpu=gpu, seed=None)
        self.device = self.rbm_amp.device
        self.test_grads = test_grads

        self.rbm_amp.weights        = nn.Parameter((torch.tensor(
                                             [[-0.8,0.3],[0.5,-0.2]], 
                                             device=self.rbm_amp.device, 
                                             dtype = torch.double)), 
                                            requires_grad = True)

        self.rbm_phase.weights      = nn.Parameter((torch.tensor(
                                             [[-1.0, 0.1],[-0.5,0.48]], 
                                             device = self.rbm_amp.device, 
                                             dtype = torch.double)), 
                                            requires_grad = True)   

        self.rbm_amp.visible_bias   = nn.Parameter((torch.tensor([0.1, -0.74], 
                                                  device=self.rbm_amp.device,
                                                  dtype = torch.double)), 
                                                 requires_grad = True)

        self.rbm_phase.visible_bias = nn.Parameter((torch.tensor([0.31, -0.4], 
                                                  device=self.rbm_amp.device,
                                                  dtype = torch.double)), 
                                                 requires_grad = True)     

        self.rbm_amp.hidden_bias    = nn.Parameter((torch.tensor([-0.4, -1.2], 
                                                  device=self.rbm_amp.device,
                                                  dtype = torch.double)), 
                                                 requires_grad = True)

        self.rbm_phase.hidden_bias  = nn.Parameter((torch.tensor([-0.45, -0.2], 
                                                  device=self.rbm_amp.device,
                                                  dtype = torch.double)), 
                                                 requires_grad = True)   


    def basis_state_generator(self, s):
        """Only works for binary visible units at the moment. Generates a vector
        given a spin value (0 or 1).

        :param s: A spin's value (either 0 or 1).
        :type s: float

        :returns: If s = 0, this is the (1,0) state in the basis of the
                  measurement. If s = 1, this is the (0,1) state in the basis
                  of the measurement.
        :rtype: torch.doubleTensor
        """
        if s == 0.:
            return torch.tensor([[1., 0.], [0., 0.]], dtype=torch.double)
        if s == 1.:
            return torch.tensor([[0., 1.], [0., 0.]], dtype=torch.double)

    def state_generator(self, num_non_trivial_unitaries):
        """A function that returns all possible configurations of
        'num_non_trivial_unitaries' spins. Similar to generate_visible_space.

        :param num_non_trivial_unitaries: The number of sites measured in the
                                          non-computational basis.
        :type num_non_trivial_unitaries: int

        :returns: An array of all possible spin configurations of
                  'num_non_trivial_unitaries' spins.
        :rtype: torch.doubleTensor
        """
        states = torch.zeros((2**num_non_trivial_unitaries,
                              num_non_trivial_unitaries),
                             device=self.device,
                             dtype=torch.double)
        for i in range(2**num_non_trivial_unitaries):
            temp = i
            for j in range(num_non_trivial_unitaries):
                temp, remainder = divmod(temp, 2)
                states[i][num_non_trivial_unitaries - j - 1] = remainder
        return states

    def unnormalized_probability_amp(self, v):
        r"""The effective energy of the phase RBM.

        :param v: Visible unit(s).
        :type v: torch.doubleTensor

        :returns:
            :math:`p_{\lambda}(\bm{v}) = e^{\mathcal{E}_{\lambda}(\bm{v})}`
        :rtype: torch.doubleTensor
        """
        return self.rbm_amp.unnormalized_probability(v)

    def unnormalized_probability_phase(self, v):
        r"""The effective energy of the phase RBM.

        :param v: Visible unit(s).
        :type v: torch.doubleTensor

        :returns: :math:`p_{\mu}(\bm{v}) = e^{\mathcal{E}_{\mu}(\bm{v})}`
        :rtype: torch.doubleTensor
        """
        return self.rbm_phase.unnormalized_probability(v)

    def normalized_wavefunction(self, v, Z):
        r"""The RBM wavefunction.

        :param v: Visible unit(s).
        :type v: torch.doubleTensor

        :returns:

        .. math:: \psi_{\lambda\mu} =
                    \sqrt{\frac{p_{\lambda}}{Z_{\lambda}}}
                    \exp\left(\frac{i\log(p_{\mu})}{2}\right)

        :rtype: torch.doubleTensor
        """
        return (self.unnormalized_wavefunction(v) / Z.sqrt()) 

    def unnormalized_wavefunction(self, v):
        r"""The unnormalized RBM wavefunction.

        :param v: Visible unit(s).
        :type v: torch.doubleTensor

        :returns:

        .. math:: \tilde{\psi}_{\lambda\mu} =
                        \sqrt{p_{\lambda}}
                        \exp\left(\frac{i\log(p_{\mu})}{2}\right)

        :rtype: torch.doubleTensor
        """
        v_prime = v.view(-1, self.num_visible)
        temp1 = (self.unnormalized_probability_amp(v_prime)).sqrt()
        temp2 = ((self.unnormalized_probability_phase(v_prime)).log())*0.5
        cos_angle = temp2.cos()
        sin_angle = temp2.sin()

        psi = torch.zeros(2, v_prime.size()[0], dtype=torch.double)

        psi[0] = temp1*cos_angle
        psi[1] = temp1*sin_angle

        return psi

    def compute_batch_gradients(self, unitary_dict, k, pos_batch, neg_batch,
                                pos_chars_batch, neg_chars_batch):
        """This function will compute the gradients of a batch of the training
        data (data_file) given the basis measurements (chars_file).

        :param k: Number of contrastive divergence steps in amplitude training.
        :type k: int
        :param batch: Batch of the input data.
        :type batch: torch.doubleTensor
        :param chars_batch: Batch of bases that correspondingly indicates the
                            basis each site in the batch was measured in.
        :type chars_batch: list(str)

        :returns: Dictionary containing all the gradients (negative): Gradient
                  of weights, visible bias and hidden bias for the amplitude,
                  Gradients of weights, visible bias and hidden bias for the
                  phase.
        :rtype: dict
        """

        pos_batch_size = len(pos_batch)
        neg_batch_size = len(neg_batch)

        g_weights_amp = torch.zeros_like(self.rbm_amp.weights)
        g_vb_amp = torch.zeros_like(self.rbm_amp.visible_bias)
        g_hb_amp = torch.zeros_like(self.rbm_amp.hidden_bias)

        g_weights_phase = torch.zeros_like(self.rbm_phase.weights)
        g_vb_phase = torch.zeros_like(self.rbm_phase.visible_bias)
        g_hb_phase = torch.zeros_like(self.rbm_phase.hidden_bias)

        pos_batch, _, _, _, _                       = self.rbm_amp.gibbs_sampling(k, pos_batch)
        _, _, neg_vkbatch, neg_hkbatch, neg_phkbatch = self.rbm_amp.gibbs_sampling(k, neg_batch)

        # Iterate through every data point in the batch.
        for row_count, v0 in enumerate(pos_batch):

            # A counter for the number of non-trivial unitaries
            # (non-computational basis) in the data point.
            num_non_trivial_unitaries = 0

            # tau_indices will contain the index numbers of spins not in the
            # computational basis (Z). z_indices will contain the index numbers
            # of spins in the computational basis.
            tau_indices = []
            z_indices = []

            for j in range(self.num_visible):
                # Go through the unitaries (chars_batch[row_count]) of each
                # site in the data point, v0, and save inidices of non-trivial.
                if pos_chars_batch[row_count][j] != 'Z':
                    num_non_trivial_unitaries += 1
                    tau_indices.append(j)
                else:
                    z_indices.append(j)

            if num_non_trivial_unitaries == 0:
                # If there are no non-trivial unitaries for the data point v0,
                # calculate the positive phase of regular (i.e. non-complex
                # RBM) gradient. Use the actual data point, v0.
                prob_amp = F.sigmoid(F.linear(v0, self.rbm_amp.weights,
                                              self.rbm_amp.hidden_bias))
                g_weights_amp -= (torch.einsum("i,j->ij", (prob_amp, v0))
                                  / pos_batch_size)
                g_vb_amp -= v0 / pos_batch_size
                g_hb_amp -= prob_amp / pos_batch_size

            else:
                # Compute the rotated gradients.
                [L_weights_amp, L_vb_amp, L_hb_amp,
                 L_weights_phase, L_vb_phase, L_hb_phase] = \
                    self.compute_rotated_grads(unitary_dict, k, v0,
                                               pos_chars_batch[row_count],
                                               num_non_trivial_unitaries,
                                               z_indices, tau_indices)

                # Gradents of amplitude parameters take the real part of the
                # rotated gradients.
                g_weights_amp -= L_weights_amp[0] / pos_batch_size
                g_vb_amp -= L_vb_amp[0] / pos_batch_size
                g_hb_amp -= L_hb_amp[0] / pos_batch_size

                # Gradents of phase parameters take the imaginary part of the
                # rotated gradients.
                g_weights_phase += L_weights_phase[1] / pos_batch_size
                g_vb_phase += L_vb_phase[1] / pos_batch_size
                g_hb_phase += L_hb_phase[1] / pos_batch_size

        # Block gibbs sampling for negative phase.
        g_weights_amp += (torch.einsum("ij,ik->jk",
                                       (neg_phkbatch, neg_vkbatch))
                          / neg_batch_size)
        g_vb_amp += torch.einsum("ij->j", (neg_vkbatch,)) / neg_batch_size
        g_hb_amp += torch.einsum("ij->j", (neg_phkbatch,)) / neg_batch_size

        """Return negative gradients to match up nicely with the usual
        parameter update rules, which *subtract* the gradient from
        the parameters. This is in contrast with the RBM update
        rules which ADD the gradients (scaled by the learning rate)
        to the parameters."""

        return {
            "rbm_amp": {
                "weights": g_weights_amp,
                "visible_bias": g_vb_amp,
                "hidden_bias": g_hb_amp
            },
            "rbm_phase": {
                "weights": g_weights_phase,
                "visible_bias": g_vb_phase,
                "hidden_bias": g_hb_phase
            }
        }

    def compute_rotated_grads(self, unitary_dict, k, v0, pos_characters,
                              num_non_trivial_unitaries,
                              z_indices, tau_indices):
        """Computes the rotated gradients.

        :param v0: A visible unit.
        :type v0: torch.doubleTensor
        :param characters: A string of characters corresponding to the basis
                           that each site in v0 was measured in.
        :type characters: str
        :param num_non_trivial_unitaries: The number of sites in v0 that aren't
                                          measured in the computational basis.
        :type num_non_trivial_unitaries: int
        :param z_indices: A list of indices that correspond to sites of v0 that
                          are measured in the computational basis.
        :type z_indices: list(int)
        :param tau_indices: A list of indices that correspond to sites of v0
                            that are not measured in the computational basis.
        :type tau_indices: list(int)

        :returns: Dictionary of the rotated gradients: L_weights_amp, L_vb_amp,
                  L_hb_amp, L_weights_phase, L_vb_phase, L_hb_phase
        :rtype: dict
        """
        """Initialize the 'A' parameters (see alg 4.2)."""
        A_weights_amp = torch.zeros(2, self.rbm_amp.weights.size()[0],
                                    self.rbm_amp.weights.size()[1],
                                    device=self.device, dtype=torch.double)
        A_vb_amp = torch.zeros(2, self.rbm_amp.visible_bias.size()[0],
                               device=self.device, dtype=torch.double)
        A_hb_amp = torch.zeros(2, self.rbm_amp.hidden_bias.size()[0],
                               device=self.device, dtype=torch.double)

        A_weights_phase = torch.zeros(2, self.rbm_phase.weights.size()[0],
                                      self.rbm_phase.weights.size()[1],
                                      device=self.device, dtype=torch.double)
        A_vb_phase = torch.zeros(2, self.rbm_phase.visible_bias.size()[0],
                                 device=self.device, dtype=torch.double)
        A_hb_phase = torch.zeros(2, self.rbm_phase.hidden_bias.size()[0],
                                 device=self.device, dtype=torch.double)
        # 'B' will contain the coefficients of the rotated unnormalized
        # wavefunction.
        B = torch.zeros(2, device=self.device, dtype=torch.double)

        w_grad_amp = torch.zeros_like(self.rbm_amp.weights)
        vb_grad_amp = torch.zeros_like(self.rbm_amp.visible_bias)
        hb_grad_amp = torch.zeros_like(self.rbm_amp.hidden_bias)

        w_grad_phase = torch.zeros_like(self.rbm_phase.weights)
        vb_grad_phase = torch.zeros_like(self.rbm_phase.visible_bias)
        hb_grad_phase = torch.zeros_like(self.rbm_phase.hidden_bias)

        zeros_for_w_amp = torch.zeros_like(w_grad_amp)
        zeros_for_w_phase = torch.zeros_like(w_grad_phase)
        zeros_for_vb = torch.zeros_like(vb_grad_amp)
        zeros_for_hb_amp = torch.zeros_like(hb_grad_amp)
        zeros_for_hb_phase = torch.zeros_like(hb_grad_phase)

        # Loop over Hilbert space of the non trivial unitaries to build
        # the state.
        for j in range(2**num_non_trivial_unitaries):
            s = self.state_generator(num_non_trivial_unitaries)[j]
            # Creates a matrix where the jth row is the desired state, |S>,
            # a vector.

            # This is the sigma state.
            constructed_state = torch.zeros(self.num_visible, dtype=torch.double)

            U = torch.tensor([1., 0.], dtype=torch.double, device=self.device)

            # Populate the |sigma> state (aka constructed_state) accirdingly.
            for index in range(len(z_indices)):
                # These are the sites in the computational basis.
                constructed_state[z_indices[index]] = v0[z_indices[index]]

            for index in range(len(tau_indices)):
                # These are the sites that are NOT in the computational basis.
                constructed_state[tau_indices[index]] = s[index]

                aa = unitary_dict[pos_characters[tau_indices[index]]]
                bb = self.basis_state_generator(v0[tau_indices[index]])
                cc = self.basis_state_generator(s[index])

                temp = cplx.inner_prod(cplx.MV_mult(
                    cplx.compT_matrix(aa), bb), cc)

                U = cplx.scalar_mult(U, temp)

            # Positive phase gradients for phase and amp. Will be added into
            # the 'A' parameters.

            prob_amp = F.sigmoid(F.linear(constructed_state,
                                          self.rbm_amp.weights,
                                          self.rbm_amp.hidden_bias))
            prob_phase = F.sigmoid(F.linear(constructed_state,
                                            self.rbm_phase.weights,
                                            self.rbm_phase.hidden_bias))

            w_grad_amp = torch.einsum("i,j->ij", (prob_amp, constructed_state))
            vb_grad_amp = constructed_state
            hb_grad_amp = prob_amp

            w_grad_phase = torch.einsum("i,j->ij",
                                        (prob_phase, constructed_state))
            vb_grad_phase = constructed_state
            hb_grad_phase = prob_phase

            """
            In order to calculate the 'A' parameters below with the current
            complex library, I need to make the weights and biases complex.
            I fill the complex parts of the parameters with a tensor of zeros.
            """
            temp_w_grad_amp = cplx.make_complex_matrix(w_grad_amp,
                                                       zeros_for_w_amp)
            temp_vb_grad_amp = cplx.make_complex_vector(vb_grad_amp,
                                                        zeros_for_vb)
            temp_hb_grad_amp = cplx.make_complex_vector(hb_grad_amp,
                                                        zeros_for_hb_amp)

            temp_w_grad_phase = cplx.make_complex_matrix(w_grad_phase,
                                                         zeros_for_w_phase)
            temp_vb_grad_phase = cplx.make_complex_vector(vb_grad_phase,
                                                          zeros_for_vb)
            temp_hb_grad_phase = cplx.make_complex_vector(hb_grad_phase,
                                                          zeros_for_hb_phase)

            # Temp = U*psi(sigma)
            temp = cplx.scalar_mult(
                U, self.unnormalized_wavefunction(constructed_state))

            A_weights_amp += cplx.MS_mult(temp, temp_w_grad_amp)
            A_vb_amp += cplx.VS_mult(temp, temp_vb_grad_amp)
            A_hb_amp += cplx.VS_mult(temp, temp_hb_grad_amp)

            A_weights_phase += cplx.MS_mult(temp, temp_w_grad_phase)
            A_vb_phase += cplx.VS_mult(temp, temp_vb_grad_phase)
            A_hb_phase += cplx.VS_mult(temp, temp_hb_grad_phase)

            # Rotated wavefunction.
            B += temp

        L_weights_amp = cplx.MS_divide(A_weights_amp, B)
        L_vb_amp = cplx.VS_divide(A_vb_amp, B)
        L_hb_amp = cplx.VS_divide(A_hb_amp, B)

        L_weights_phase = cplx.MS_divide(A_weights_phase, B)
        L_vb_phase = cplx.VS_divide(A_vb_phase, B)
        L_hb_phase = cplx.VS_divide(A_hb_phase, B)

        return [L_weights_amp, L_vb_amp, L_hb_amp, L_weights_phase, L_vb_phase,
                L_hb_phase]

################################################################################
############################# EXACT GRADIENTS ##################################
################################################################################

    def compute_numerical_NLL(self, batch, Z):
        NLL = 0.0
        batch_size = len(batch)

        for i in range(len(batch)):
            NLL -= (self.rbm_amp.probability(batch[i], Z)).log().item()/batch_size       

        return NLL 

    def compute_exact_gradients_NLL(self, unitary_dict, k, pos_batch, neg_batch,
                                pos_chars_batch, neg_chars_batch, visible_space, Z):
        """This function will compute the gradients of a batch of the training
        data (data_file) given the basis measurements (chars_file).

        :param k: Number of contrastive divergence steps in amplitude training.
        :type k: int
        :param batch: Batch of the input data.
        :type batch: torch.doubleTensor
        :param chars_batch: Batch of bases that correspondingly indicates the
                            basis each site in the batch was measured in.
        :type chars_batch: list(str)

        :returns: Dictionary containing all the gradients (negative): Gradient
                  of weights, visible bias and hidden bias for the amplitude,compute_exact_gradients_NLL(self, unitary_dict, k, pos_batch, neg_batch,
                                pos_chars_batch, neg_chars_batch, visible_space, Z):

                  Gradients of weights, visible bias and hidden bias for the
                  phase.
        :rtype: dict
        """

        pos_batch_size = len(pos_batch)
        neg_batch_size = len(neg_batch)

        g_weights_amp = torch.zeros_like(self.rbm_amp.weights)
        g_vb_amp = torch.zeros_like(self.rbm_amp.visible_bias)
        g_hb_amp = torch.zeros_like(self.rbm_amp.hidden_bias)

        g_weights_phase = torch.zeros_like(self.rbm_phase.weights)
        g_vb_phase = torch.zeros_like(self.rbm_phase.visible_bias)
        g_hb_phase = torch.zeros_like(self.rbm_phase.hidden_bias)

        # Iterate through every data point in the positive phase batch.
        for row_count, v0 in enumerate(pos_batch):

            # A counter for the number of non-trivial unitaries
            # (non-computational basis) in the data point.
            num_non_trivial_unitaries = 0

            # tau_indices will contain the index numbers of spins not in the
            # computational basis (Z). z_indices will contain the index numbers
            # of spins in the computational basis.
            tau_indices = []
            z_indices   = []

            for j in range(self.num_visible):
                # Go through the unitaries (chars_batch[row_count]) of each
                # site in the data point, v0, and save inidices of non-trivial.
                if pos_chars_batch[row_count][j] != 'Z':
                    num_non_trivial_unitaries += 1
                    tau_indices.append(j)
                else:
                    z_indices.append(j)

            if num_non_trivial_unitaries == 0:
                # If there are no non-trivial unitaries for the data point v0,
                # calculate the positive phase of regular (i.e. non-complex
                # RBM) gradient. Use the actual data point, v0.
                prob1_amp = F.sigmoid(F.linear(v0, self.rbm_amp.weights,
                                              self.rbm_amp.hidden_bias))
                g_weights_amp -= (torch.einsum("i,j->ij", (prob1_amp, v0))
                                  / pos_batch_size)
                g_vb_amp -= v0 / pos_batch_size
                g_hb_amp -= prob1_amp / pos_batch_size

            else:
                # Compute the rotated gradients.
                [L_weights_amp, L_vb_amp, L_hb_amp,
                 L_weights_phase, L_vb_phase, L_hb_phase] = \
                    self.compute_exactrotated_grads_NLL(unitary_dict, k, v0,
                                               pos_chars_batch[row_count],
                                               num_non_trivial_unitaries,
                                               z_indices, tau_indices)

                # Gradents of amplitude parameters take the real part of the
                # rotated gradients.
                g_weights_amp -= L_weights_amp[0] / pos_batch_size
                g_vb_amp -= L_vb_amp[0] / pos_batch_size
                g_hb_amp -= L_hb_amp[0] / pos_batch_size

                # Gradents of phase parameters take the imaginary part of the
                # rotated gradients.
                g_weights_phase += L_weights_phase[1] / pos_batch_size
                g_vb_phase += L_vb_phase[1] / pos_batch_size
                g_hb_phase += L_hb_phase[1] / pos_batch_size

        for j in range(len(visible_space)):
            prob2_amp = F.sigmoid(F.linear(visible_space[j], self.rbm_amp.weights,
                                              self.rbm_amp.hidden_bias))

            temp_weights_amp = (torch.einsum('i,j->ij',(prob2_amp, visible_space[j])))/neg_batch_size
            temp_vb_amp      = (visible_space[j])/neg_batch_size
            temp_hb_amp      = (prob2_amp)/neg_batch_size
  
            g_weights_amp   += (self.rbm_amp.probability(visible_space[j], Z))*temp_weights_amp
            g_vb_amp        += (self.rbm_amp.probability(visible_space[j], Z))*temp_vb_amp
            g_hb_amp        += (self.rbm_amp.probability(visible_space[j], Z))*temp_hb_amp
 
        """Return negative gradients to match up nicely with the usual
        parameter update rules, which *subtract* the gradient from
        the parameters. This is in contrast with the RBM update
        rules which ADD the gradients (scaled by the learning rate)
        to the parameters."""

        return {
            "rbm_amp": {
                "weights": g_weights_amp,
                "visible_bias": g_vb_amp,
                "hidden_bias": g_hb_amp
            },
            "rbm_phase": {
                "weights": g_weights_phase,
                "visible_bias": g_vb_phase,
                "hidden_bias": g_hb_phase
            }
        }

    def compute_exactrotated_grads_NLL(self, unitary_dict, k, v0, pos_characters,
                              num_non_trivial_unitaries,
                              z_indices, tau_indices):
        """Computes the rotated gradients.

        :param v0: A visible unit.
        :type v0: torch.doubleTensor
        :param characters: A string of characters corresponding to the basis
                           that each site in v0 was measured in.
        :type characters: str
        :param num_non_trivial_unitaries: The number of sites in v0 that aren't
                                          measured in the computational basis.
        :type num_non_trivial_unitaries: int
        :param z_indices: A list of indices that correspond to sites of v0 that
                          are measured in the computational basis.
        :type z_indices: list(int)
        :param tau_indices: A list of indices that correspond to sites of v0
                            that are not measured in the computational basis.
        :type tau_indices: list(int)

        :returns: Dictionary of the rotated gradients: L_weights_amp, L_vb_amp,
                  L_hb_amp, L_weights_phase, L_vb_phase, L_hb_phase
        :rtype: dict
        """
        """Initialize the 'A' parameters (see alg 4.2)."""
        A_weights_amp = torch.zeros(2, self.rbm_amp.weights.size()[0],
                                    self.rbm_amp.weights.size()[1],
                                    device=self.device, dtype=torch.double)
        A_vb_amp = torch.zeros(2, self.rbm_amp.visible_bias.size()[0],
                               device=self.device, dtype=torch.double)
        A_hb_amp = torch.zeros(2, self.rbm_amp.hidden_bias.size()[0],
                               device=self.device, dtype=torch.double)

        A_weights_phase = torch.zeros(2, self.rbm_phase.weights.size()[0],
                                      self.rbm_phase.weights.size()[1],
                                      device=self.device, dtype=torch.double)
        A_vb_phase = torch.zeros(2, self.rbm_phase.visible_bias.size()[0],
                                 device=self.device, dtype=torch.double)
        A_hb_phase = torch.zeros(2, self.rbm_phase.hidden_bias.size()[0],
                                 device=self.device, dtype=torch.double)
        # 'B' will contain the coefficients of the rotated unnormalized
        # wavefunction.
        B = torch.zeros(2, device=self.device, dtype=torch.double)

        w_grad_amp = torch.zeros_like(self.rbm_amp.weights)
        vb_grad_amp = torch.zeros_like(self.rbm_amp.visible_bias)
        hb_grad_amp = torch.zeros_like(self.rbm_amp.hidden_bias)

        w_grad_phase = torch.zeros_like(self.rbm_phase.weights)
        vb_grad_phase = torch.zeros_like(self.rbm_phase.visible_bias)
        hb_grad_phase = torch.zeros_like(self.rbm_phase.hidden_bias)

        zeros_for_w_amp = torch.zeros_like(w_grad_amp)
        zeros_for_w_phase = torch.zeros_like(w_grad_phase)
        zeros_for_vb = torch.zeros_like(vb_grad_amp)
        zeros_for_hb_amp = torch.zeros_like(hb_grad_amp)
        zeros_for_hb_phase = torch.zeros_like(hb_grad_phase)

        # Loop over Hilbert space of the non trivial unitaries to build
        # the state.
        for j in range(2**num_non_trivial_unitaries):
            s = self.state_generator(num_non_trivial_unitaries)[j]
            # Creates a matrix where the jth row is the desired state, |S>,
            # a vector.

            # This is the sigma state.
            constructed_state = torch.zeros(self.num_visible, dtype=torch.double)

            U = torch.tensor([1., 0.], dtype=torch.double, device=self.device)

            # Populate the |sigma> state (aka constructed_state) accirdingly.
            for index in range(len(z_indices)):
                # These are the sites in the computational basis.
                constructed_state[z_indices[index]] = v0[z_indices[index]]

            for index in range(len(tau_indices)):
                # These are the sites that are NOT in the computational basis.
                constructed_state[tau_indices[index]] = s[index]

                aa = unitary_dict[pos_characters[tau_indices[index]]]
                bb = self.basis_state_generator(v0[tau_indices[index]])
                cc = self.basis_state_generator(s[index])

                temp = cplx.inner_prod(cplx.MV_mult(
                    cplx.compT_matrix(aa), bb), cc)

                U = cplx.scalar_mult(U, temp)

            # Positive phase gradients for phase and amp. Will be added into
            # the 'A' parameters.

            prob_amp = F.sigmoid(F.linear(constructed_state,
                                          self.rbm_amp.weights,
                                          self.rbm_amp.hidden_bias))
            prob_phase = F.sigmoid(F.linear(constructed_state,
                                            self.rbm_phase.weights,
                                            self.rbm_phase.hidden_bias))

            w_grad_amp = torch.einsum("i,j->ij", (prob_amp, constructed_state))
            vb_grad_amp = constructed_state
            hb_grad_amp = prob_amp

            w_grad_phase = torch.einsum("i,j->ij",
                                        (prob_phase, constructed_state))
            vb_grad_phase = constructed_state
            hb_grad_phase = prob_phase

            """
            In order to calculate the 'A' parameters below with the current
            complex library, I need to make the weights and biases complex.
            I fill the complex parts of the parameters with a tensor of zeros.
            """
            temp_w_grad_amp = cplx.make_complex_matrix(w_grad_amp,
                                                       zeros_for_w_amp)
            temp_vb_grad_amp = cplx.make_complex_vector(vb_grad_amp,
                                                        zeros_for_vb)
            temp_hb_grad_amp = cplx.make_complex_vector(hb_grad_amp,
                                                        zeros_for_hb_amp)

            temp_w_grad_phase = cplx.make_complex_matrix(w_grad_phase,
                                                         zeros_for_w_phase)
            temp_vb_grad_phase = cplx.make_complex_vector(vb_grad_phase,
                                                          zeros_for_vb)
            temp_hb_grad_phase = cplx.make_complex_vector(hb_grad_phase,
                                                          zeros_for_hb_phase)

            # Temp = U*psi(sigma)
            temp = cplx.scalar_mult(
                U, self.unnormalized_wavefunction(constructed_state))

            A_weights_amp += cplx.MS_mult(temp, temp_w_grad_amp)
            A_vb_amp += cplx.VS_mult(temp, temp_vb_grad_amp)
            A_hb_amp += cplx.VS_mult(temp, temp_hb_grad_amp)

            A_weights_phase += cplx.MS_mult(temp, temp_w_grad_phase)
            A_vb_phase += cplx.VS_mult(temp, temp_vb_grad_phase)
            A_hb_phase += cplx.VS_mult(temp, temp_hb_grad_phase)

            # Rotated wavefunction.
            B += temp

        L_weights_amp = cplx.MS_divide(A_weights_amp, B)
        L_vb_amp = cplx.VS_divide(A_vb_amp, B)
        L_hb_amp = cplx.VS_divide(A_hb_amp, B)

        L_weights_phase = cplx.MS_divide(A_weights_phase, B)
        L_vb_phase = cplx.VS_divide(A_vb_phase, B)
        L_hb_phase = cplx.VS_divide(A_hb_phase, B)

        return [L_weights_amp, L_vb_amp, L_hb_amp, L_weights_phase, L_vb_phase,
                L_hb_phase]

################################################################################
################################################################################
################################################################################

    def fit(self, data, character_data, unitary_dict, epochs, pos_batch_size,
            neg_batch_size, k=1, lr=1e-2, log_every=0, progbar=False):
        """Execute the training of the RBM.

        :param data: The actual training data
        :type data: list(float)
        :param character_data: The corresponding bases that each site in the
                               data has been measured in.
        :type character_data: list(str)
        :param epochs: The number of parameter (i.e. weights and biases)
                       updates
        :type epochs: int
        :param batch_size: The size of batches taken from the data
        :type batch_size: int
        :param k: The number of contrastive divergence steps
        :type k: int
        :param lr: Learning rate
        :type lr: float
        :param progbar: Whether or not to display a progress bar. If "notebook"
                        is passed, will use a Jupyter notebook compatible
                        progress bar.
        :type progbar: bool or str
        """
        # Make data file into a torch tensor.
        data = torch.tensor(data, dtype=torch.double).to(device=self.device)

        # Use the Adam optmizer to update the weights and biases.
        optimizer = torch.optim.Adam([self.rbm_amp.weights,
                                      self.rbm_amp.visible_bias,
                                      self.rbm_amp.hidden_bias,
                                      self.rbm_phase.weights,
                                      self.rbm_phase.visible_bias,
                                      self.rbm_phase.hidden_bias],
                                     lr=lr)

        disable_progbar = (progbar is False)
        progress_bar = tqdm_notebook if progbar == "notebook" else tqdm

        vis = self.rbm_amp.generate_visible_space()
        Z   = self.rbm_amp.partition(vis)

        for ep in progress_bar(range(0, epochs + 1),
                               desc="Epochs ", total=epochs,
                               disable=disable_progbar):
            # Shuffle the data to ensure that the batches taken from the data
            # are random data points.
            pos_random_permutation      = torch.randperm(data.shape[0])
            pos_shuffled_data           = data[pos_random_permutation]
            pos_shuffled_character_data = character_data[pos_random_permutation]

            neg_random_permutation      = torch.randperm(data.shape[0])
            neg_shuffled_data           = data[neg_random_permutation]
            neg_shuffled_character_data = character_data[neg_random_permutation]

            # List of all the batches for positive phase.
            pos_batches = [pos_shuffled_data[batch_start:(batch_start + pos_batch_size)]
                           for batch_start in range(0, len(data), pos_batch_size)]

            # List of all the bases for positive phase.
            pos_char_batches = [pos_shuffled_character_data[batch_start:(batch_start+pos_batch_size)]
                            for batch_start in range(0, len(data), pos_batch_size)]

            # List of all the batches for negative phase.
            neg_batches = [neg_shuffled_data[batch_start:(batch_start + neg_batch_size)]
                           for batch_start in range(0, len(data), neg_batch_size)]

            # List of all the bases for negative phase.
            neg_char_batches = [neg_shuffled_character_data[batch_start:(batch_start+neg_batch_size)]
                                for batch_start in range(0, len(data), neg_batch_size)]

            # Calculate convergence quantities every "log-every" steps.
            '''
            if ep % log_every == 0:
                fidelity_ = self.fidelity(vis, 'Z' 'Z')
                print ('Epoch = ',ep,'\nFidelity = ',fidelity_)
            '''

            # Save parameters at the end of training.
            if ep == epochs:
                print('Finished training. Saving results...')
                self.save_params()
                print('Done.')
                break

            # Loop through all of the batches and calculate the batch
            # gradients.
            for batch_num, (pos_batch, neg_batch) in enumerate(zip(pos_batches, neg_batches)): 

                #alg_grads = self.compute_batch_gradients(unitary_dict, k, pos_batch, neg_batch, pos_char_batches[batch_num], neg_char_batches[batch_num])

                alg_grads = self.compute_exact_gradients_NLL(unitary_dict, k, pos_batch, neg_batch,
                            pos_char_batches[batch_num], neg_char_batches[batch_num], vis, Z)

                if self.test_grads:
                    self.test_gradients(pos_batch, vis, k, alg_grads, Z)

                # Clear any cached gradients.
                optimizer.zero_grad()

                # Assign all available gradients to the
                # corresponding parameter.
                for name, grads in alg_grads.items():
                    selected_RBM = getattr(self, name)
                    for param in grads.keys():
                        getattr(selected_RBM, param).grad = grads[param]

                # Tell the optimizer to apply the gradients and update
                # the parameters.
                optimizer.step()

    def save_params(self):
        """A function that saves the weights and biases for the amplitude and
        phase individually."""
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

    def get_true_psi(self, basis):
        """Picks out the true psi in the correct basis.

        :param basis: E.g. XZZZX.
        :type basis: str

        :returns: The true wavefunction in the basis.
        :rtype: torch.doubleTensor
        """
        key = ''
        for i in range(len(basis)):
            key += basis[i]
        return self.psi_dictionary[key]

    def overlap(self, visible_space, basis):
        """Computes the overlap between the RBM and true wavefunctions.

        :param visible_space: An array of all possible spin configurations.
        :type visible_space: torch.doubleTensor
        :param basis: E.g. XZZZX.
        :type basis: str

        :returns: :math:`O = \\langle{\\psi_{true}}\\vert\\psi_{\\lambda\\mu}\\rangle`.
        :rtype: float
        """
        overlap_ = cplx.inner_prod(self.get_true_psi(basis),
                           self.normalized_wavefunction(visible_space))
        return overlap_

    def fidelity(self, visible_space, basis):
        """Computed the fidelity of the RBM and true wavefunctions.

        :param visible_space: An array of all possible spin configurations.
        :type visible_space: torch.doubleTensor
        :param basis: E.g. XZZZX.
        :type basis: str

        :returns: :math:`F = |O|^2`.
        :rtype: float
        """
        return cplx.norm(self.overlap(visible_space, basis))

    def KL_divergence(self, visible_space, Z):
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
                self.full_unitaries[basis_list[i]],
                self.normalized_wavefunction(visible_space, Z)).view(2,-1)
            rotated_true_psi = self.get_true_psi(basis_list[i]).view(2,-1)

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

    def compute_numerical_gradient(self, batch, visible_space, param, alg_grad, Z):
        eps = 1.e-8
        print("Numerical\t Alg.\t Abs. Diff.")

        for i in range(len(param)):

            param[i].data += eps
            Z = self.rbm_amp.partition(visible_space)
            NLL_pos = self.compute_numerical_NLL(batch, Z)
            #KL_pos  = self.KL_divergence(visible_space, Z)

            param[i].data -= 2*eps
            Z = self.rbm_amp.partition(visible_space)
            NLL_neg = self.compute_numerical_NLL(batch, Z)
            #KL_neg  = self.KL_divergence(visible_space, Z)

            param[i].data += eps

            #num_grad = (KL_pos - KL_neg) / (2*eps)
            num_grad = (NLL_pos - NLL_neg) / (2*eps)

            print("{: 10.8f}\t{: 10.8f}\t{: 10.8f}\t"
                  .format(num_grad, alg_grad[i],
                          abs(num_grad - alg_grad[i])))

    def test_gradients(self, batch, visible_space, k, alg_grads, Z):
        # Must have negative sign because the compute_batch_grads returns the neg of the grads.
        # key_list = ["weights_amp", "visible_bias_amp", "hidden_bias_amp", "weights_phase", "visible_bias_phase", "hidden_bias_phase"]

        flat_weights_amp   = self.rbm_amp.weights.data.view(-1)
        flat_weights_phase = self.rbm_phase.weights.data.view(-1)

        flat_grad_weights_amp   = alg_grads["rbm_amp"]["weights"].view(-1)
        flat_grad_weights_phase = alg_grads["rbm_phase"]["weights"].view(-1)

        print('-------------------------------------------------------------------------------')

        print('Weights amp gradient')
        self.compute_numerical_gradient(
            batch, visible_space, flat_weights_amp, -flat_grad_weights_amp, Z)
        print ('\n')

        print('Visible bias amp gradient')
        self.compute_numerical_gradient(
            batch, visible_space, self.rbm_amp.visible_bias, -alg_grads["rbm_amp"]["visible_bias"], Z)
        print ('\n')

        print('Hidden bias amp gradient')
        self.compute_numerical_gradient(
            batch, visible_space, self.rbm_amp.hidden_bias, -alg_grads["rbm_amp"]["hidden_bias"], Z)
        print ('\n')

        print('Weights phase gradient')
        self.compute_numerical_gradient(
            batch, visible_space, flat_weights_phase, -flat_grad_weights_phase, Z)
        print ('\n')

        print('Visible bias phase gradient')
        self.compute_numerical_gradient(
            batch, visible_space, self.rbm_phase.visible_bias, -alg_grads["rbm_phase"]["visible_bias"], Z)
        print ('\n')

        print('Hidden bias phase gradient')
        self.compute_numerical_gradient(
            batch, visible_space, self.rbm_phase.hidden_bias, -alg_grads["rbm_phase"]["hidden_bias"], Z)

    def state_to_index(self, state):
        ''' Only for debugging how the unitary is applied to the unnormalized wavefunction - the 'B' term in alg 4.2.'''
        states = torch.zeros(2**self.num_visible, self.num_visible)
        npstates = states.numpy()
        npstate = state.numpy()
        for i in range(2**self.num_visible):
            temp = i

            for j in range(self.num_visible):
                temp, remainder = divmod(temp, 2)
                npstates[i][self.num_visible - j - 1] = remainder

            if np.array_equal(npstates[i], npstate):
                return i