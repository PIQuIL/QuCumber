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
    #"BinomialRBMModule",
    #"BinomialRBM"
    "BinaryRBM"
]


def _warn_on_missing_gpu(gpu):
    if gpu and not torch.cuda.is_available():
        warnings.warn("Could not find GPU: will continue with CPU.",
                      ResourceWarning)


class BinaryRBM(nn.Module, Sampler):
    def __init__(self, num_visible, num_hidden, zero_weights=False,
                 gpu=True, seed=None, num_chains = 200):
        super(BinaryRBM, self).__init__()
        self.num_visible = int(num_visible)
        self.num_hidden = int(num_hidden)
        self.num_chains = int(num_chains)
        self.num_pars = self.num_visible*self.num_hidden+self.num_visible+self.num_hidden
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
                 / np.sqrt(self.num_visible)),requires_grad=True)

        self.visible_bias = nn.Parameter(
                (torch.randn(self.num_visible,device=self.device, 
                 dtype=torch.double)/np.sqrt(self.num_visible)),requires_grad=True)
        self.hidden_bias = nn.Parameter(
                (torch.randn(self.num_hidden,device=self.device, 
                 dtype=torch.double)/np.sqrt(self.num_hidden)),requires_grad=True)
        #self.visible_bias = nn.Parameter(torch.zeros(self.num_visible,
        #                                             device=self.device,
        #                                             dtype=torch.double),
        #                                 requires_grad=True)
        #self.hidden_bias = nn.Parameter(torch.zeros(self.num_hidden,
        #                                            device=self.device,
        #                                            dtype=torch.double),
        #                                requires_grad=True)
        
        self.visible_state = torch.zeros(self.num_chains,self.num_visible,
                                         device=self.device,
                                         dtype=torch.double)
        self.hidden_state = torch.zeros(self.num_chains,self.num_hidden,
                                         device=self.device,
                                         dtype=torch.double)
        self.visible_prob = torch.zeros(self.num_chains,self.num_visible,
                                         device=self.device,
                                         dtype=torch.double)
        self.hidden_prob = torch.zeros(self.num_chains,self.num_hidden,
                                         device=self.device,
                                         dtype=torch.double)
    
    def __repr__(self):
        return ("BinaryRBM(num_visible={}, num_hidden={}, gpu={})"
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

    def effective_energy_gradient(self,v):
        """The gradients of the effective energies for the given visible states.

        :param v: The visible states.
        :type v: torch.Tensor

        :returns: The gradients of the effective energies for the given 
                  visible states.
        :rtype: torch.Tensor
        """
        prob = F.sigmoid(F.linear(v, self.weights,self.hidden_bias))     
        
        
        #W_grad = torch.einsum("ij,ik->jk", (prob, v))
        #b_grad = torch.einsum("ij->j", (v,))
        #c_grad = torch.einsum("ij->j", (prob,))
        W_grad = -torch.einsum("j,k->jk", (prob, v))
        b_grad = -v
        c_grad = -prob
        
        return {'weights':W_grad,'visible_bias':b_grad,'hidden_bias':c_grad}
    
    
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

    def sample_layer(self,prob_hv):
        hv = prob_hv.bernoulli()
        return hv

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
        #self.visible_state = v0
        #for _ in range(k):
        #    self.hidden_prob = self.prob_h_given_v(self.visible_state)
        #    self.sample_layer(self.hidden_prob)
        #    self.visible_prob = self.prob_v_given_h(self.hidden_state)
        #    self.sample_layer(self.visible_prob)
        
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

    #def unnormalized_probability(self, v):
    #    r"""The unnormalized probabilities of the given visible states.

    #    .. math:: p(\bm{v}) = e^{\mathcal{E}(\bm{v})}

    #    :param v: The visible states.
    #    :type v: torch.Tensor

    #    :returns: The unnormalized probability of the given visible state(s).
    #    :rtype: torch.Tensor
    #    """
    #    return self.effective_energy(v).exp()

    #def probability_ratio(self, a, b):
    #    return self.log_probability_ratio(a, b).exp()

    #def log_probability_ratio(self, a, b):
    #    return self.effective_energy(a).sub(self.effective_energy(b))

    #def generate_visible_space(self):
    #    """Generates all possible visible states.

    #    :returns: A tensor of all possible spin configurations.
    #    :rtype: torch.Tensor
    #    """
    #    space = torch.zeros((1 << self.num_visible, self.num_visible),
    #                        device=self.device, dtype=torch.double)
    #    for i in range(1 << self.num_visible):
    #        d = i
    #        for j in range(self.num_visible):
    #            d, r = divmod(d, 2)
    #            space[i, self.num_visible - j - 1] = int(r)

    #    return space

    #def log_partition(self, visible_space):
    #    """The natural logarithm of the partition function of the RBM.

    #    :param visible_space: A rank 2 tensor of the entire visible space.
    #    :type visible_space: torch.Tensor

    #    :returns: The natural log of the partition function.
    #    :rtype: torch.Tensor
    #    """
    #    free_energies = self.effective_energy(visible_space)
    #    max_free_energy = free_energies.max()

    #    f_reduced = free_energies - max_free_energy
    #    logZ = max_free_energy + f_reduced.exp().sum().log()

    #    return logZ

    #def partition(self, visible_space):
    #    """The partition function of the RBM.

    #    :param visible_space: A rank 2 tensor of the entire visible space.
    #    :type visible_space: torch.Tensor

    #    :returns: The partition function.
    #    :rtype: torch.Tensor
    #    """
    #    return self.log_partition(visible_space).exp()

    #def probability(self, v, Z):
    #    """Evaluates the probability of the given vector(s) of visible
    #    units; NOT RECOMMENDED FOR RBMS WITH A LARGE # OF VISIBLE UNITS

    #    :param v: The visible states.
    #    :type v: torch.Tensor
    #    :param Z: The partition function.
    #    :type Z: float

    #    :returns: The probability of the given vector(s) of visible units.
    #    :rtype: torch.Tensor
    #    """
    #    return self.unnormalized_probability(v) / Z
