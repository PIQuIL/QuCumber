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
from torch.nn.utils import parameters_to_vector
from tqdm import tqdm, tqdm_notebook

#import qucumber.cplx as cplx
import utils.cplx as cplx
from qucumber.samplers import Sampler
from qucumber.callbacks import CallbackList

__all__ = [
    "BinaryRBM"
]


def _warn_on_missing_gpu(gpu):
    if gpu and not torch.cuda.is_available():
        warnings.warn("Could not find GPU: will continue with CPU.",
                      ResourceWarning)


class BinaryRBM(nn.Module, Sampler):
    def __init__(self, num_visible, num_hidden, zero_weights=False,
                 gpu=True, seed=None, num_chains = 100):
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
            self.visible_bias = nn.Parameter(torch.zeros(self.num_visible,
                                                         device=self.device,
                                                         dtype=torch.double),
                                             requires_grad=True)
            self.hidden_bias = nn.Parameter(torch.zeros(self.num_hidden,
                                                        device=self.device,
                                                        dtype=torch.double),
                                            requires_grad=True)
        else:
            self.randomize()
        
    def __repr__(self):
        return ("BinaryRBM(num_visible={}, num_hidden={}, gpu={})"
                .format(self.num_visible, self.num_hidden, self.gpu))

    def randomize(self):
        """Randomize the parameters of the RBM"""
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
 

    def effective_energy(self, v):
        r"""The effective energies of the given visible states.

        .. math::

            \mathcal{E}(\bm{v}) &= -\sum_{j}b_j v_j
                        - \sum_{i}\log
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

        return -(visible_bias_term + hidden_bias_term)

    def effective_energy_gradient(self,v):
        """The gradients of the effective energies for the given visible states.

        :param v: The visible states.
        :type v: torch.Tensor

        :returns: 1d vector containing the gradients for all parameters (computed on the given 
                  visible states v). 
        :rtype: torch.Tensor
        """
        prob = F.sigmoid(F.linear(v, self.weights,self.hidden_bias))     
        
        if len(v.shape) < 2:
            W_grad = -torch.einsum("j,k->jk", (prob, v))
            b_grad = -v
            c_grad = -prob
        else:        
            W_grad = -torch.einsum("ij,ik->jk", (prob, v))
            b_grad = -torch.einsum("ij->j", (v,))
            c_grad = -torch.einsum("ij->j", (prob,))
        
        return parameters_to_vector([W_grad,b_grad,c_grad])
    
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
        return v

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
        return h
