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

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parameters_to_vector

from qucumber import _warn_on_missing_gpu

__all__ = ["BinaryRBM"]


class BinaryRBM(nn.Module):
    def __init__(
        self, num_visible, num_hidden, zero_weights=False, gpu=True, num_chains=100
    ):
        super(BinaryRBM, self).__init__()
        self.num_visible = int(num_visible)
        self.num_hidden = int(num_hidden)
        self.num_chains = int(num_chains)
        self.num_pars = (
            (self.num_visible * self.num_hidden) + self.num_visible + self.num_hidden
        )

        _warn_on_missing_gpu(gpu)
        self.gpu = gpu and torch.cuda.is_available()

        # Maximum number of visible units for exact enumeration
        self.size_cut = 16

        self.device = torch.device("cuda") if self.gpu else torch.device("cpu")

        if zero_weights:
            self.weights = nn.Parameter(
                (
                    torch.zeros(
                        self.num_hidden,
                        self.num_visible,
                        device=self.device,
                        dtype=torch.double,
                    )
                ),
                requires_grad=True,
            )
            self.visible_bias = nn.Parameter(
                torch.zeros(self.num_visible, device=self.device, dtype=torch.double),
                requires_grad=True,
            )
            self.hidden_bias = nn.Parameter(
                torch.zeros(self.num_hidden, device=self.device, dtype=torch.double),
                requires_grad=True,
            )
        else:
            self.initialize_parameters()

    def __repr__(self):
        return "BinaryRBM(num_visible={}, num_hidden={}, gpu={})".format(
            self.num_visible, self.num_hidden, self.gpu
        )

    def initialize_parameters(self):
        """Randomize the parameters of the RBM"""
        self.weights = nn.Parameter(
            (
                torch.randn(
                    self.num_hidden,
                    self.num_visible,
                    device=self.device,
                    dtype=torch.double,
                )
                / np.sqrt(self.num_visible)
            ),
            requires_grad=True,
        )

        self.visible_bias = nn.Parameter(
            (
                torch.randn(self.num_visible, device=self.device, dtype=torch.double)
                / np.sqrt(self.num_visible)
            ),
            requires_grad=True,
        )
        self.hidden_bias = nn.Parameter(
            (
                torch.randn(self.num_hidden, device=self.device, dtype=torch.double)
                / np.sqrt(self.num_hidden)
            ),
            requires_grad=True,
        )

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
        hid_bias_term = F.softplus(F.linear(v, self.weights, self.hidden_bias)).sum(1)

        return -(visible_bias_term + hid_bias_term)

    def effective_energy_gradient(self, v):
        """The gradients of the effective energies for the given visible states.

        :param v: The visible states.
        :type v: torch.Tensor

        :returns: 1d vector containing the gradients for all parameters
                  (computed on the given visible states v).
        :rtype: torch.Tensor
        """
        prob = self.prob_h_given_v(v)

        if v.dim() < 2:
            W_grad = -torch.einsum("j,k->jk", (prob, v))
            vb_grad = -v
            hb_grad = -prob
        else:
            W_grad = -torch.einsum("ij,ik->jk", (prob, v))
            vb_grad = -torch.einsum("ij->j", (v,))
            hb_grad = -torch.einsum("ij->j", (prob,))

        return parameters_to_vector([W_grad, vb_grad, hb_grad])

    def prob_v_given_h(self, h, out=None):
        """Given a hidden unit configuration, compute the probability
        vector of the visible units being on.

        :param h: The hidden unit
        :type h: torch.Tensor
        :param out: The output tensor to write to.
        :type out: torch.Tensor

        :returns: The probability of visible units being active given the
                  hidden state.
        :rtype: torch.Tensor
        """
        if h.dim() < 2:  # create extra axis, if needed
            h = h.unsqueeze(0)
        p = (
            torch.addmm(self.visible_bias.data, h, self.weights.data, out=out)
            .sigmoid_()
            .squeeze_(0)
        )  # remove superfluous axis, if it exists
        return p

    def prob_h_given_v(self, v, out=None):
        """Given a visible unit configuration, compute the probability
        vector of the hidden units being on.

        :param h: The hidden unit.
        :type h: torch.Tensor
        :param out: The output tensor to write to.
        :type out: torch.Tensor

        :returns: The probability of hidden units being active given the
                  visible state.
        :rtype: torch.Tensor
        """
        if v.dim() < 2:  # create extra axis, if needed
            v = v.unsqueeze(0)
        p = (
            torch.addmm(self.hidden_bias.data, v, self.weights.data.t(), out=out)
            .sigmoid_()
            .squeeze_(0)
        )  # remove superfluous axis, if it exists
        return p

    def sample_v_given_h(self, h, out=None):
        """Sample/generate a visible state given a hidden state.

        :param h: The hidden state.
        :type h: torch.Tensor
        :param out: The output tensor to write to.
        :type out: torch.Tensor

        :returns: Tuple containing prob_v_given_h(h) and the sampled visible
                  state.
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        v = self.prob_v_given_h(h, out=out)
        v = v.copy_(v.bernoulli())  # overwrite v with its sample
        return v

    def sample_h_given_v(self, v, out=None):
        """Sample/generate a hidden state given a visible state.

        :param h: The visible state.
        :type h: torch.Tensor
        :param out: The output tensor to write to.
        :type out: torch.Tensor

        :returns: Tuple containing prob_h_given_v(v) and the sampled hidden
                  state.
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        h = self.prob_h_given_v(v, out=out)
        h = h.copy_(h.bernoulli())  # overwrite h with its sample
        return h

    def compute_partition_function(self, space):
        """The natural logarithm of the partition function of the RBM.

        :param space: A rank 2 tensor of the visible space.
        :type space: torch.Tensor

        :returns: The natural log of the partition function.
        :rtype: torch.Tensor
        """
        neg_free_energies = -self.effective_energy(space)
        logZ = neg_free_energies.logsumexp(0)
        Z = logZ.exp().item()
        return torch.tensor(Z, dtype=torch.double, device=self.device)
