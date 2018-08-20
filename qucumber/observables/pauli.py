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

from qucumber.utils import cplx
from .observable import Observable
from .utils import to_pm1


class SigmaZ(Observable):
    """The :math:`sigma_z` observable"""

    def __init__(self):
        self.name = "SigmaZ"
        self.symbol = "Z"

    def apply(self, nn_state, samples):
        """Computes the magnetization of each sample given a batch of samples.

        :param nn_state: The Wavefunction that drew the samples.
        :type nn_state: qucumber.nn_states.Wavefunction
        :param samples: A batch of samples to calculate the observable on.
                        Must be using the :math:`\sigma_i = 0, 1` convention.
        :type samples: torch.Tensor
        """
        # convert to +/- 1 convention, after computing the
        # mean, to reduce total computations; this works
        # because expectation is linear.
        return to_pm1(samples.mean(1)).abs()


class SigmaX(Observable):
    """The :math:`sigma_x` observable"""

    def __init__(self):
        self.name = "SigmaX"
        self.symbol = "X"

    @staticmethod
    def _flip_spin(i, s):
        torch.fmod(s[:, i] + 1., 2, out=s[:, i])

    def apply(self, nn_state, samples):
        """Computes the magnetization along X of each sample in the given batch of samples.

        :param nn_state: The Wavefunction that drew the samples.
        :type nn_state: qucumber.nn_states.Wavefunction
        :param samples: A batch of samples to calculate the observable on.
                        Must be using the :math:`\sigma_i = 0, 1` convention.
        :type samples: torch.Tensor
        """
        samples = samples.to(device=nn_state.device)

        # vectors of shape: (2, num_samples,)
        psis = nn_state.psi(samples)
        psi_ratio_sum = torch.zeros_like(psis)

        for i in range(samples.shape[-1]):  # sum over spin sites
            self._flip_spin(i, samples)  # flip the spin at site i

            # compute ratio of psi_(-i) / psi and add it to the running sum
            psi_ratio = nn_state.psi(samples)
            psi_ratio = cplx.elementwise_division(psi_ratio, psis)
            psi_ratio_sum.add_(psi_ratio)

            self._flip_spin(i, samples)  # flip it back

        return cplx.elementwise_norm(psi_ratio_sum).div_(samples.shape[-1])
