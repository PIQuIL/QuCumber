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

from qucumber.rbm import BinaryRBM
from .wavefunction import AbstractWavefunction


class PositiveWavefunction(AbstractWavefunction):
    _rbm_am = None

    def __init__(self, num_visible, num_hidden=None, gpu=True):
        super(PositiveWavefunction, self).__init__()
        self.num_visible = int(num_visible)
        self.num_hidden = (
            int(num_hidden) if num_hidden is not None else self.num_visible
        )

        self.rbm_am = BinaryRBM(self.num_visible, self.num_hidden, gpu=gpu)

        self.space = None
        self.Z = 0.0
        self.device = self.rbm_am.device

    @property
    def networks(self):
        return ["rbm_am"]

    @property
    def rbm_am(self):
        return self._rbm_am

    @rbm_am.setter
    def rbm_am(self, new_val):
        self._rbm_am = new_val

    def amplitude(self, v):
        r"""Compute the (unnormalized) amplitude of a given vector/matrix of visible states:

        .. math::

            \text{amplitude}(\bm{\sigma})=|\psi_{\bm{\lambda}}(\bm{\sigma})|=
            e^{-\mathcal{E}_{\bm{\lambda}}(\bm{\sigma})/2}

        :param v: visible states :math:`\bm{\sigma}`
        :type v: torch.Tensor

        :returns: Matrix/vector containing the amplitudes of v
        :rtype: torch.Tensor
        """
        return super().amplitude(v)

    def phase(self, v):
        r"""Compute the phase of a given vector/matrix of visible states, which,
        in the case of a Positive Wavefunction, is just zero.

        :param v: visible states :math:`\bm{\sigma}`
        :type v: torch.Tensor

        :returns: Matrix/vector containing the phases of v
        :rtype: torch.Tensor
        """
        return torch.zeros(v.shape[0]).to(v)

    def psi(self, v):
        r"""Compute the wavefunction of a given vector/matrix of visible states:

        .. math::

            \psi_{\bm{\lambda}}(\bm{\sigma})
                = e^{-\mathcal{E}_{\bm{\lambda}}(\bm{\sigma})/2}

        :param v: visible states :math:`\bm{\sigma}`
        :type v: torch.Tensor

        :returns: Complex object containing the value of the wavefunction for
                  each visible state
        :rtype: torch.Tensor
        """
        psi = torch.zeros(2, dtype=torch.double, device=self.device)
        psi[0] = self.amplitude(v)
        psi[1] = 0.0
        return psi

    def gradient(self, v):
        r"""Compute the gradient
        :math:`\nabla_{\bm{\lambda}}\mathcal{E}_{\bm{\lambda}}(\bm{\sigma})`
        of the effective visible energy for a batch of visible states v.

        :param v: visible states :math:`\bm{\sigma}`
        :type v: torch.Tensor

        :returns: A single tensor containing all of the parameter gradients.
        :rtype: torch.Tensor
        """
        return self.rbm_am.effective_energy_gradient(v)

    def compute_normalization(self):
        r"""Compute the normalization constant of the wavefunction.

        .. math::

            Z_{\bm{\lambda}}=\sqrt{\sum_{\bm{\sigma}}|\psi_{\bm{\lambda}}|^2}=
            \sqrt{\sum_{\bm{\sigma}} p_{\bm{\lambda}}(\bm{\sigma})}

        :param space: A rank 2 tensor of the entire visible space.
        :type space: torch.Tensor
        """
        return super().compute_normalization()

    @staticmethod
    def autoload(location, gpu=False):
        state_dict = torch.load(location)
        wvfn = PositiveWavefunction(
            num_visible=len(state_dict["rbm_am"]["visible_bias"]),
            num_hidden=len(state_dict["rbm_am"]["hidden_bias"]),
            gpu=gpu,
        )
        wvfn.load(location)
        return wvfn
