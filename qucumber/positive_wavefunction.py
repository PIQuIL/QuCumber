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

import torch
from qucumber.binary_rbm import BinaryRBM

__all__ = [
    "PositiveWavefunction"
]


class PositiveWavefunction(object):
    def __init__(self, num_visible, num_hidden=None, gpu=True):
        super(PositiveWavefunction, self).__init__()
        self.num_visible = int(num_visible)
        self.num_hidden = (int(num_hidden)
                           if num_hidden is not None
                           else self.num_visible)

        self.rbm_am = BinaryRBM(self.num_visible, self.num_hidden,
                                gpu=gpu)

        # Maximum size of the Hilbert space for full enumeration
        self.size_cut = 20

        self.space = None
        self.Z = 0.0
        self.num_pars = self.rbm_am.num_pars
        self.networks = ["rbm_am"]
        self.device = self.rbm_am.device

    def initialize_parameters(self):
        r"""Randomize the parameters :math:`\bm{\lambda}=\{\bm{W},\bm{b},\bm{c}\}` of
        the RBM parametrizing the wavefunction."""
        self.rbm_am.initialize_parameters()

    def psi(self, v):
        r""" Compute the wavefunction of a given vector/matrix of visible states:

        .. math::

            \psi_{\bm{\lambda}}(\bm{\sigma})
                = e^{-\mathcal{E}_{\bm{\lambda}}(\bm{\sigma})/2}

        :param v: visible states, :math:`\bm{\sigma}`
        :type v: torch.tensor

        :returns: Complex object containing the value of the wavefunction for
                  each visible state
        :rtype: torch.tensor
        """
        psi = torch.zeros(2, dtype=torch.double, device=self.device)
        psi[0] = (-self.rbm_am.effective_energy(v)).exp().sqrt()
        psi[1] = 0.0
        return psi

    def gradient(self, v):
        r"""Compute the gradient
        :math:`\nabla_{\bm{\lambda}}\mathcal{E}_{\bm{\lambda}}(\bm{\sigma})`
        of the effective visible energy for a batch of visible states v.

        :param v: visible states, :math:`\bm{\sigma}`
        :type v: torch.tensor

        :returns dictionary with one key (rbm_am)
        :rtype  dictionary(dictionary(torch.tensor,torch.tensor,torch.tensor)
        """
        return self.rbm_am.effective_energy_gradient(v)

    def gibbs_steps(self, k, initial_state, overwrite=False):
        v = initial_state.to(device=self.device, dtype=torch.double)

        if overwrite is False:
            v = v.clone()

        h = torch.zeros(v.shape[0], self.num_hidden,
                        device=self.device, dtype=torch.double)

        for _ in range(k):
            self.rbm_am.sample_h_given_v(v, out=h)
            self.rbm_am.sample_v_given_h(h, out=v)

        return v

    def sample(self, num_samples, k, initial_state=None, overwrite=False):
        r"""Performs k steps of Block Gibbs sampling. One step consists of sampling
        the hidden state :math:`\bm{h}` from the conditional distribution
        :math:`p_{\bm{\lambda}}(\bm{h}\:|\:\bm{v})`, and sampling the visible
        state :math:`\bm{v}` from the conditional distribution
        :math:`p_{\bm{\lambda}}(\bm{v}\:|\:\bm{h})`.

        :param k: Number of Block Gibbs steps.
        :type k: int
        """
        if initial_state is None:
            dist = torch.distributions.Bernoulli(probs=0.5)
            sample_size = torch.Size((num_samples, self.num_visible))
            initial_state = dist.sample(sample_size) \
                                .to(device=self.device, dtype=torch.double)

        return self.gibbs_steps(k, initial_state, overwrite=overwrite)

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
        data = {"rbm_am": self.rbm_am.state_dict(), **metadata}
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

        self.rbm_am.load_state_dict(state_dict, strict=False)

    def generate_hilbert_space(self, size):
        r"""Generates Hilbert space of dimension :math:`2^{\text{size}}`.

        :returns: A tensor with all the basis states of the Hilbert space.
        :rtype: torch.Tensor
        """
        if (size > self.size_cut):
            raise ValueError('Size of the Hilbert space too large!')
        else:
            space = torch.zeros((1 << size, size),
                                device=self.device, dtype=torch.double)
            for i in range(1 << size):
                d = i
                for j in range(size):
                    d, r = divmod(d, 2)
                    space[i, size - j - 1] = int(r)
            return space

    def compute_normalization(self):
        r"""Compute the normalization constant of the wavefunction.

        .. math::

            Z_{\bm{\lambda}}=\sqrt{\sum_{\bm{\sigma}}|\psi_{\bm{\lambda}}|^2}=
            \sqrt{\sum_{\bm{\sigma}} p_{\bm{\lambda}}(\bm{\sigma})}

        :param space: A rank 2 tensor of the entire visible space.
        :type space: torch.Tensor

        """
        if (self.space is None):
            raise ValueError('Missing Hilbert space')
        else:
            self.Z = self.rbm_am.compute_partition_function(self.space)

    @staticmethod
    def autoload(location, gpu=False):
        """Initializes an RBM from the parameters in the given location,
        ignoring any metadata stored in the file.

        :param location: The location to load the RBM parameters from
        :type location: str or file

        :returns: A new RBM initialized from the given parameters
        :rtype: BinomialRBM
        """
        state_dict = torch.load(location)

        rbm = BinaryRBM(num_visible=len(state_dict["rbm_am"]['visible_bias']),
                        num_hidden=len(state_dict["rbm_am"]['hidden_bias']),
                        gpu=gpu,
                        seed=None)

        rbm.load_state_dict(state_dict, strict=False)

        return rbm
