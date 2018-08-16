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


class SigmaZ(Observable):
    """The :math:`sigma_z` observable"""

    def apply(self, nn_state, samples):
        """Computes the magnetization of each sample given a batch of samples.

        :param nn_state: The Wavefunction that drew the samples.
        :type nn_state: qucumber.nn_states.Wavefunction
        :param samples: A batch of samples to calculate the observable on.
                        Must be using the :math:`\sigma_i = 0, 1` convention.
        :type samples: torch.Tensor
        """
        return (
            samples.mean(1)
            .mul(2.)  # convert to +/- 1 convention, after computing the
            .sub(1.)  # mean, to reduce total computations; this works
            .abs()  # because expectation is linear.
        )


class SigmaX(Observable):
    """The :math:`sigma_x` observable"""

    @staticmethod
    def _flip_spin(i, s):
        torch.fmod(s[:, i] + 1., 2, out=s[:, i])

    def apply(self, nn_states, samples):
        """Computes the magnetization along X of each sample in the given batch of samples.

        :param nn_state: The Wavefunction that drew the samples.
        :type nn_state: qucumber.nn_states.Wavefunction
        :param samples: A batch of samples to calculate the observable on.
                        Must be using the :math:`\sigma_i = 0, 1` convention.
        :type samples: torch.Tensor
        """
        psis = nn_states.psi(samples)  # vector (2, num_samples,)

        abs_psi = psis[0].pow(2) + psis[1].pow(2)  # abs of each psi val

        # vector (2, num_samples,)
        psi_ratios = torch.zeros(
            psis.shape, dtype=torch.double, device=nn_states.device
        )

        for i in range(samples.shape[-1]):  # sum over spin sites
            self._flip_spin(i, samples)  # flip the spin at site i
            psi_ratios[:, :] += nn_states.psi(samples)  # TODO: do division here
            self._flip_spin(i, samples)  # flip it back

        # TODO: divide psi_-i / psi_i using complex math
        #       -> make sure to broadcast properly

        return psi_ratios
