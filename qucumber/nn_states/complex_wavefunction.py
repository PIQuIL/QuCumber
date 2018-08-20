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

from qucumber.utils import cplx
from qucumber.utils import unitaries
from qucumber.rbm import BinaryRBM
from .wavefunction import Wavefunction


class ComplexWavefunction(Wavefunction):
    """Class capable of learning Wavefunctions with a non-zero phase.

    :param num_visible: The number of visible units, ie. the size of the system being learned.
    :type num_visible: int
    :param num_hidden: The number of hidden units in both internal RBMs. Defaults to
                    the number of visible units.
    :type num_hidden: int
    :param unitary_dict: A dictionary mapping unitary names to their matrix representations.
    :type unitary_dict: dict[str, torch.Tensor]
    :param gpu: Whether to perform computations on the default gpu.
    :type gpu: bool
    """

    _rbm_am = None
    _rbm_ph = None
    _device = None

    def __init__(self, num_visible, num_hidden=None, unitary_dict=None, gpu=True):
        self.num_visible = int(num_visible)
        self.num_hidden = int(num_hidden) if num_hidden else self.num_visible
        self.rbm_am = BinaryRBM(self.num_visible, self.num_hidden, gpu=gpu)
        self.rbm_ph = BinaryRBM(self.num_visible, self.num_hidden, gpu=gpu)

        self.device = self.rbm_am.device

        self.unitary_dict = unitary_dict if unitary_dict else unitaries.create_dict()
        self.unitary_dict = {
            k: v.to(device=self.device) for k, v in self.unitary_dict.items()
        }

    @property
    def networks(self):
        return ["rbm_am", "rbm_ph"]

    @property
    def rbm_am(self):
        return self._rbm_am

    @rbm_am.setter
    def rbm_am(self, new_val):
        self._rbm_am = new_val

    @property
    def rbm_ph(self):
        """RBM used to learn the wavefunction phase."""
        return self._rbm_ph

    @rbm_ph.setter
    def rbm_ph(self, new_val):
        self._rbm_ph = new_val

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, new_val):
        self._device = new_val

    def amplitude(self, v):
        r"""Compute the (unnormalized) amplitude of a given vector/matrix of visible states.

        .. math::

            \text{amplitude}(\bm{\sigma})=|\psi_{\bm{\lambda\mu}}(\bm{\sigma})|=
            e^{-\mathcal{E}_{\bm{\lambda}}(\bm{\sigma})/2}

        :param v: visible states :math:`\bm{\sigma}`.
        :type v: torch.Tensor

        :returns: Vector containing the amplitudes of the given states.
        :rtype: torch.Tensor
        """
        return super().amplitude(v)

    def phase(self, v):
        r"""Compute the phase of a given vector/matrix of visible states.

        .. math::

            \text{phase}(\bm{\sigma})=-\mathcal{E}_{\bm{\mu}}(\bm{\sigma})/2

        :param v: visible states :math:`\bm{\sigma}`.
        :type v: torch.Tensor

        :returns: Vector containing the phases of the given states.
        :rtype: torch.Tensor
        """
        return -0.5 * self.rbm_ph.effective_energy(v)

    def psi(self, v):
        r"""Compute the (unnormalized) wavefunction of a given vector/matrix of visible states.

        .. math::

            \psi_{\bm{\lambda\mu}}(\bm{\sigma})
                = e^{-[\mathcal{E}_{\bm{\lambda}}(\bm{\sigma})
                        + i\mathcal{E}_{\bm{\mu}}(\bm{\sigma})]/2}

        :param v: visible states :math:`\bm{\sigma}`
        :type v: torch.Tensor

        :returns: Complex object containing the value of the wavefunction for
                  each visible state
        :rtype: torch.Tensor
        """
        # vectors/tensors of shape (len(v),)
        amplitude, phase = self.amplitude(v), self.phase(v)

        # complex vector; shape: (2, len(v))
        psi = torch.zeros(
            (2,) + amplitude.shape, dtype=torch.double, device=self.device
        )

        # elementwise products
        psi[0] = amplitude * phase.cos()  # real part
        psi[1] = amplitude * phase.sin()  # imaginary part

        # squeeze down to complex scalar if there was only one visible state
        return psi.squeeze()

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

    def rotated_gradient(self, basis, sites, sample):
        Upsi, vp, Us, rotated_grad = self.init_gradient(basis, sites)
        int_sample = np.array(sample[sites].round().int())
        vp = sample.round().clone()

        for x in range(2 ** sites.size):
            vp = sample.round().clone()

            # overwrite rotated elements
            vp[sites] = self.subspace_vector(x, size=sites.size)

            # Gradient on the current configuration
            grad_vp = [
                self.rbm_am.effective_energy_gradient(vp),
                self.rbm_ph.effective_energy_gradient(vp),
            ]

            # Gradient from the rotation
            int_vp = np.array(vp[sites].round().int())
            all_Us = Us[np.arange(sites.size), :, int_sample, int_vp]
            U = np.prod(all_Us[:, 0] + (1j * all_Us[:, 1]))
            U = torch.tensor([U.real, U.imag], dtype=torch.double, device=self.device)
            Upsi_v = cplx.scalar_mult(U, self.psi(vp))
            Upsi += Upsi_v
            rotated_grad[0] += cplx.scalar_mult(
                Upsi_v, cplx.make_complex(grad_vp[0], torch.zeros_like(grad_vp[0]))
            )
            rotated_grad[1] += cplx.scalar_mult(
                Upsi_v, cplx.make_complex(grad_vp[1], torch.zeros_like(grad_vp[1]))
            )

        grad = [
            cplx.scalar_divide(rotated_grad[0], Upsi)[0, :],  # Real
            -cplx.scalar_divide(rotated_grad[1], Upsi)[1, :],  # Imaginary
        ]

        return grad

    def gradient(self, basis, sample):
        r"""Compute the gradient of a sample, measured in different bases.

        :param basis: A set of bases.
        :type basis: np.array
        :param sample: A sample to compute the gradient of.
        :type sample: np.array

        :returns: A list of 2 tensors containing the parameters of each of the
                  internal RBMs.
        :rtype: list[torch.Tensor]
        """
        basis = np.array(list(basis))  # list is silly, but works for now
        rot_sites = np.where(basis != "Z")[0]
        if rot_sites.size == 0:
            grad = [
                self.rbm_am.effective_energy_gradient(sample),  # Real
                0.0,  # Imaginary
            ]
        else:
            grad = self.rotated_gradient(basis, rot_sites, sample)
        return grad

    def compute_normalization(self, space):
        r"""Compute the normalization constant of the wavefunction.

        .. math::

            Z_{\bm{\lambda}}=
            \sqrt{\sum_{\bm{\sigma}}|\psi_{\bm{\lambda\mu}}|^2}=
            \sqrt{\sum_{\bm{\sigma}} p_{\bm{\lambda}}(\bm{\sigma})}

        :param space: A rank 2 tensor of the entire visible space.
        :type space: torch.Tensor

        """
        return super().compute_normalization(space)

    def save(self, location, metadata=None):
        metadata = metadata if metadata else {}
        metadata["unitary_dict"] = self.unitary_dict
        super().save(location, metadata=metadata)

    @staticmethod
    def autoload(location, gpu=False):
        state_dict = torch.load(location)
        wvfn = ComplexWavefunction(
            unitary_dict=state_dict["unitary_dict"],
            num_visible=len(state_dict["rbm_am"]["visible_bias"]),
            num_hidden=len(state_dict["rbm_am"]["hidden_bias"]),
            gpu=gpu,
        )
        wvfn.load(location)
        return wvfn
