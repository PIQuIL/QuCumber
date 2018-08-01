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
from torch.distributions.utils import log_sum_exp

from .observable import Observable

__all__ = [
    "TFIMChainEnergy",
    "TFIMChainMagnetization"
]


def to_pm1(samples):
    """Converts a tensor of spins from the :math:`\sigma_i = 0, 1` convention
    to the :math:`\sigma_i = -1, +1` convention.

    :param samples: A tensor of spins to convert.
                    Must be using the :math:`\sigma_i = 0, 1` convention.
    :type samples: torch.Tensor
    """
    return samples.mul(2.).sub(1.)


def to_01(samples):
    """Converts a tensor of spins from the :math:`\sigma_i = -1, +1` convention
    to the :math:`\sigma_i = 0, 1` convention.

    :param samples: A tensor of spins to convert.
                    Must be using the :math:`\sigma_i = -1, +1` convention.
    :type samples: torch.Tensor
    """
    return samples.add(1.).div(2.)


class TFIMChainEnergy(Observable):
    """Observable defining the energy of a Transverse Field Ising Model (TFIM)
    spin chain with nearest neighbour interactions, and :math:`J=1`.

    :param h: The strength of the tranverse field
    :type h: float
    :param density: Whether to compute the energy per spin site.
    :type density: bool
    :param periodic_bcs: If `True` use periodic boundary conditions,
                         otherwise use open boundary conditions.
    :type periodic_bcs: bool
    """

    def __init__(self, h, density=True, periodic_bcs=False):
        super(TFIMChainEnergy, self).__init__()
        self.h = h
        self.density = density
        self.periodic_bcs = periodic_bcs

    def __repr__(self):
        return (f"TFIMChainEnergy(h={self.h}, density={self.density},"
                f"periodic_bcs={self.periodic_bcs})")

    @staticmethod
    def _flip_spin(i, s):
        s[:, i] *= -1.0

    def apply(self, samples, sampler):
        """Computes the energy of each sample given a batch of
        samples.

        :param samples: A batch of samples to calculate the observable on.
                        Must be using the :math:`\sigma_i = 0, 1` convention.
        :type samples: torch.Tensor
        :param sampler: The sampler that drew the samples. Must implement
                        the function :func:`effective_energy`, giving the
                        log probability of its inputs (up to an additive
                        constant).
        :type sampler: qucumber.rbm.BinomialRBM
        """
        samples = to_pm1(samples)
        log_psis = sampler.effective_energy(to_01(samples)).div(2.)

        shape = log_psis.shape + (samples.shape[-1],)
        log_flipped_psis = torch.zeros(*shape,
                                       dtype=torch.double,
                                       device=sampler.device)

        for i in range(samples.shape[-1]):  # sum over spin sites
            self._flip_spin(i, samples)  # flip the spin at site i
            log_flipped_psis[:, i] = sampler.effective_energy(
                to_01(samples)
            ).div(2.)
            self._flip_spin(i, samples)  # flip it back

        log_flipped_psis = log_sum_exp(
            log_flipped_psis, keepdim=True).squeeze()

        if self.periodic_bcs:
            perm_indices = list(range(sampler.shape[-1]))
            perm_indices = perm_indices[1:] + [0]
            interaction_terms = ((samples * samples[:, perm_indices])
                                 .sum(1))
        else:
            interaction_terms = ((samples[:, :-1] * samples[:, 1:])
                                 .sum(1))      # sum over spin sites

        transverse_field_terms = (log_flipped_psis
                                  .sub(log_psis)
                                  .exp())  # convert to ratio of probabilities

        energy = (transverse_field_terms
                  .mul(self.h)
                  .add(interaction_terms)
                  .mul(-1.))

        if self.density:
            return energy.div(samples.shape[-1])
        else:
            return energy


class TFIMChainMagnetization(Observable):
    """Observable defining the magnetization of a Transverse Field Ising Model
    (TFIM) spin chain.
    """

    def __init__(self):
        super(TFIMChainMagnetization, self).__init__()

    def __repr__(self):
        return "TFIMChainMagnetization()"

    def apply(self, samples, sampler=None):
        """Computes the magnetization of each sample given a batch of samples.

        :param samples: A batch of samples to calculate the observable on.
                        Must be using the :math:`\sigma_i = 0, 1` convention.
        :type samples: torch.Tensor
        :param sampler: The sampler that drew the samples. Will be ignored.
        :type sampler: qucumber.rbm.BinomialRBM
        """
        return (to_pm1(samples)
                .mean(1)
                .abs())
