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

import qucumber.utils.cplx as cplx
from qucumber.binary_rbm import BinaryRBM

__all__ = ["ComplexWavefunction"]


class ComplexWavefunction(object):
    def __init__(self, unitary_dict, num_visible, num_hidden, gpu=True):
        super(ComplexWavefunction, self).__init__()

        self.num_visible = int(num_visible)
        self.num_hidden = int(num_hidden)
        self.rbm_am = BinaryRBM(num_visible, num_hidden, gpu=gpu)
        self.rbm_ph = BinaryRBM(num_visible, num_hidden, gpu=gpu)

        # Maximum size of the Hilbert space for full enumeration
        self.size_cut = 20
        self.space = None
        self.Z = 0.0

        self.networks = ["rbm_am", "rbm_ph"]
        self.device = self.rbm_am.device
        self.unitary_dict = {
            k: v.to(device=self.device) for k, v in unitary_dict.items()
        }

    def initialize_parameters(self):
        r"""Randomize the parameters
        :math:`\bm{\lambda}=\{\bm{W}^{\bm{\lambda}},
        \bm{b}^{\bm{\lambda}},\bm{c}^{\bm{\lambda}}\}` and
        :math:`\bm{\mu}=\{\bm{W}^{\bm{\mu}},
        \bm{b}^{\bm{\mu}},\bm{c}^{\bm{\mu}}\}` of the amplitude and phase
        RBMs respectively."""
        self.rbm_am.initialize_parameters()
        self.rbm_ph.initialize_parameters()

    def amplitude(self, v):
        r""" Compute the amplitude of a given vector/matrix of visible states:

        .. math::

            \text{amplitude}(\bm{\sigma})=|\psi_{\bm{\lambda\mu}}(\bm{\sigma})|=
            e^{-\mathcal{E}_{\bm{\lambda}}(\bm{\sigma})/2}

        :param v: visible states :math:`\bm{\sigma}`
        :type v: torch.tensor

        :returns Matrix/vector containing the amplitudes of v
        :rtype torch.tensor
        """
        return (-self.rbm_am.effective_energy(v)).exp().sqrt()

    def phase(self, v):
        r""" Compute the phase of a given vector/matrix of visible states

        .. math::

            \text{phase}(\bm{\sigma})=-\mathcal{E}_{\bm{\mu}}(\bm{\sigma})/2

        :param v: visible states :math:`\bm{\sigma}`
        :type v: torch.tensor

        :returns Matrix/vector containing the phases of v
        :rtype torch.tensor
        """
        return -0.5 * self.rbm_ph.effective_energy(v)

    def psi(self, v):
        r""" Compute the wavefunction of a given vector/matrix of visible states:

        .. math::

            \psi_{\bm{\lambda\mu}}(\bm{\sigma})
                = e^{-[\mathcal{E}_{\bm{\lambda}}(\bm{\sigma})
                        + i\mathcal{E}_{\bm{\mu}}(\bm{\sigma}]/2}

        :param v: visible states :math:`\bm{\sigma}`
        :type v: torch.tensor

        :returns: Complex object containing the value of the wavefunction for
                  each visible state
        :rtype: torch.tensor
        """
        amplitude = self.amplitude(v)
        phase = self.phase(v)

        cos_phase = phase.cos()
        sin_phase = phase.sin()

        psi = torch.zeros(2, dtype=torch.double, device=self.device)
        psi[0] = amplitude * cos_phase
        psi[1] = amplitude * sin_phase
        return psi

    def init_gradient(self, basis, sites):
        Upsi = torch.zeros(2, dtype=torch.double, device=self.device)
        vp = torch.zeros(self.num_visible, dtype=torch.double, device=self.device)
        Us = np.array(torch.stack([self.unitary_dict[b] for b in basis[sites]]))
        rotated_grad = [
            torch.zeros(
                2, getattr(self, net).num_pars, dtype=torch.double, device=self.device
            )
            for net in self.networks
        ]
        return Upsi, vp, Us, rotated_grad

    def rotated_gradient(self, grad, basis, sites, sample):
        Upsi, vp, Us, rotated_grad = self.init_gradient(basis, sites)
        int_sample = np.array(sample[sites].round().int())
        vp = sample.round().clone()

        for x in range(2 ** sites.size):
            vp = sample.round().clone()
            vp[sites] = self.subspace_vector(
                sites.size, x
            )  # overwrite rotated elements
            # Gradient on the current configuration
            grad_vp = [
                self.rbm_am.effective_energy_gradient(vp),
                self.rbm_ph.effective_energy_gradient(vp),
            ]
            # Gradient from the rotation
            int_vp = np.array(vp[sites].round().int())
            all_Us = Us[np.arange(sites.size), :, int_sample, int_vp]
            U = np.prod(all_Us[:, 0] + 1j * all_Us[:, 1])
            U = torch.tensor([U.real, U.imag], dtype=torch.double, device=self.device)
            Upsi_v = cplx.scalar_mult(U, self.psi(vp))
            Upsi += Upsi_v
            rotated_grad[0] += cplx.scalar_mult(
                Upsi_v, cplx.make_complex(grad_vp[0], torch.zeros_like(grad_vp[0]))
            )
            rotated_grad[1] += cplx.scalar_mult(
                Upsi_v, cplx.make_complex(grad_vp[1], torch.zeros_like(grad_vp[1]))
            )

        grad.append(cplx.scalar_divide(rotated_grad[0], Upsi)[0, :])
        grad.append(-cplx.scalar_divide(rotated_grad[1], Upsi)[1, :])

        return grad

    def gradient(self, basis, sample):
        r"""Compute the gradient of a set (v_state) of samples, measured
        in different bases

        :param basis: A set of basis, (i.e.vector of strings)
        :type basis: np.array
        """
        grad = []
        basis = np.array(list(basis))  # list is silly, but works for now
        rot_sites = np.where(basis != "Z")[0]
        if rot_sites.size == 0:
            grad.append(self.rbm_am.effective_energy_gradient(sample))  # Real
            grad.append(0.0)  # Imaginary
        else:
            grad = self.rotated_gradient(grad, basis, rot_sites, sample)
        return grad

    def gibbs_steps(self, k, initial_state, overwrite=False):
        v = initial_state.to(device=self.device, dtype=torch.double)

        if overwrite is False:
            v = v.clone()

        h = torch.zeros(
            v.shape[0], self.num_hidden, device=self.device, dtype=torch.double
        )

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

        :param k_cd: Number of Block Gibbs steps.
        :type k_cd: int
        """
        if initial_state is None:
            dist = torch.distributions.Bernoulli(probs=0.5)
            sample_size = torch.Size((num_samples, self.num_visible))
            initial_state = dist.sample(sample_size).to(
                device=self.device, dtype=torch.double
            )

        return self.gibbs_steps(k, initial_state, overwrite=overwrite)

    def subspace_vector(self, size, num):
        r"""Generates Hilbert space of dimension :math:`2^{\text{size}}`.

        :returns: A tensor with all the basis states of the Hilbert space.
        :rtype: torch.Tensor
        """
        if size > self.size_cut:
            raise ValueError("Size of the Hilbert space too large!")
        else:
            space = (((num & (1 << np.arange(size)))) > 0)[::-1]
            space = space.astype(int)
            return torch.tensor(space, dtype=torch.double, device=self.device)

    def generate_hilbert_space(self, size):
        r"""Generates Hilbert space of dimension :math:`2^{\text{size}}`.

        :returns: A tensor with all the basis states of the Hilbert space.
        :rtype: torch.Tensor
        """
        if size > self.size_cut:
            raise ValueError("Size of the Hilbert space too large!")
        else:
            dim = np.arange(2 ** size)
            space = (((dim[:, None] & (1 << np.arange(size)))) > 0)[:, ::-1]
            space = space.astype(int)
            return torch.tensor(space, dtype=torch.double, device=self.device)

    def compute_normalization(self):
        r"""Compute the normalization constant of the wavefunction.

        .. math::

            Z_{\bm{\lambda}}=
            \sqrt{\sum_{\bm{\sigma}}|\psi_{\bm{\lambda\mu}}|^2}=
            \sqrt{\sum_{\bm{\sigma}} p_{\bm{\lambda}}(\bm{\sigma})}

        :param space: A rank 2 tensor of the entire visible space.
        :type space: torch.Tensor

        """
        if self.space is None:
            raise ValueError("Missing Hilbert space")
        else:
            self.Z = self.rbm_am.compute_partition_function(self.space)

    def save(self, location, metadata=None):
        """Saves the RBM parameters to the given location along with
        any given metadata.

        :param location: The location to save the RBM parameters + metadata
        :type location: str or file
        :param metadata: Any extra metadata to store alongside the RBM
                         parameters
        :type metadata: dict
        """
        # add extra metadata to dictionary before saving it to disk
        metadata = metadata if metadata else {}
        data = {
            "rbm_am": self.rbm_am.state_dict(),
            "rbm_ph": self.rbm_ph.state_dict(),
            **metadata,
        }
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
            state_dict = torch.load(location, lambda storage, loc: "cpu")

        self.rbm_am.load_state_dict(state_dict["rbm_am"], strict=False)
        self.rbm_ph.load_state_dict(state_dict["rbm_ph"], strict=False)
