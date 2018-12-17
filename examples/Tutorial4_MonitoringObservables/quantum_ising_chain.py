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
import numpy as np

from qucumber.observables import ObservableBase, to_01, to_pm1

__all__ = ["TFIMChainEnergy"]


class TFIMChainEnergy(ObservableBase):
    r"""Observable defining the energy of a Transverse Field Ising Model (TFIM)
    spin chain with nearest neighbour interactions, and :math:`J=1`.

    :param h: The strength of the tranverse field
    :type h: float
    :param density: Whether to compute the energy per spin site.
    :type density: bool
    :param periodic_bcs: If `True` use periodic boundary conditions,
                         otherwise use open boundary conditions.
    :type periodic_bcs: bool
    """

    def __init__(self, h):
        super(TFIMChainEnergy, self).__init__()
        self.h = h

    @staticmethod
    def _flip_spin(i, s):
        s[:, i] *= -1.0

    def apply(self, nn_state, samples):
        r"""Computes the energy of each sample given a batch of
        samples.

        :param samples: A batch of samples to calculate the observable on.
                        Must be using the :math:`\sigma_i = 0, 1` convention.
        :type samples: torch.Tensor
        """
        samples = to_pm1(samples)
        log_psis = -nn_state.rbm_am.effective_energy(to_01(samples)).div(2.0)

        shape = log_psis.shape + (samples.shape[-1],)
        log_flipped_psis = torch.zeros(
            *shape, dtype=torch.double, device=nn_state.rbm_am.device
        )

        for i in range(samples.shape[-1]):  # sum over spin sites
            self._flip_spin(i, samples)  # flip the spin at site i
            log_flipped_psis[:, i] = -nn_state.rbm_am.effective_energy(
                to_01(samples)
            ).div(2.0)
            self._flip_spin(i, samples)  # flip it back

        log_flipped_psis = torch.logsumexp(log_flipped_psis, 1, keepdim=True).squeeze()

        # sum over spin sites
        interaction_terms = (samples[:, :-1] * samples[:, 1:]).sum(1)

        # convert to ratio of probabilities
        transverse_field_terms = log_flipped_psis.sub(log_psis).exp()

        energy = transverse_field_terms.mul(self.h).add(interaction_terms).mul(-1.0)

        return energy.div(samples.shape[-1])


def Convergence(nn_state, tfim_energy, n_measurements, steps):
    energy_list = []
    err_energy = []

    v = torch.bernoulli(
        torch.ones(
            n_measurements,
            nn_state.num_visible,
            dtype=torch.double,
            device=nn_state.device,
        )
        * 0.5
    )

    energy_stats = tfim_energy.statistics_from_samples(nn_state, v)
    energy_list.append(energy_stats["mean"])
    err_energy.append(energy_stats["std_error"])

    for _steps in range(steps):
        v = nn_state.sample(1, n_measurements, initial_state=v, overwrite=True)

        energy_stats = tfim_energy.statistics_from_samples(nn_state, v)
        energy_list.append(energy_stats["mean"])
        err_energy.append(energy_stats["std_error"])

    out = {"energies": np.array(energy_list), "error": np.array(err_energy)}

    return out
