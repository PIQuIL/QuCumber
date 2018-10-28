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

from .observable import Observable
from .utils import to_pm1
import torch


class NeighbourInteraction(Observable):
    r"""The :math:`\sigma^z_i \sigma^z_{i+c}` observable

    Computes the `c`-th nearest neighbour interaction for a spin chain with
    either open or periodic boundary conditions.

    :param periodic_bcs: Specifies whether the system has periodic boundary
                         conditions.
    :type periodic_bcs: bool
    :param c: Interaction distance.
    :type c: int
    """

    def __init__(self, periodic_bcs=False, c=1):
        self.periodic_bcs = periodic_bcs
        self.c = c

        self.name = "NeighbourInteraction(periodic_bcs={}, c={})".format(
            self.periodic_bcs, self.c
        )
        self.symbol = "(Z_i * Z_(i+{}))".format(self.c)

    def apply(self, nn_state, samples):
        r"""Computes the energy of this neighbour interaction for each sample
        given a batch of samples.

        :param nn_state: The Wavefunction that drew the samples.
        :type nn_state: qucumber.nn_states.Wavefunction
        :param samples: A batch of samples to calculate the observable on.
                        Must be using the :math:`\sigma_i = 0, 1` convention.
        :type samples: torch.Tensor
        """

        samples = to_pm1(samples)  # convert to +/- 1 format
        L = samples.shape[-1]  # length of the spin chain
        if self.periodic_bcs:
            perm_indices = [(i + self.c) % L for i in range(L)]
            interaction_terms = samples * samples[:, perm_indices]
        else:
            interaction_terms = samples[:, : -self.c] * samples[:, self.c :]

        # average over spin sites.
        # not using mean bc interaction_terms.shape[-1] < num_spins = L
        return interaction_terms.sum(1).div_(L)

def transform(sample,j,operator):
    '''
    Transforms a given sample based on specified set of raising and
    lowering operators. Transformed state can be used to obtain
    corresponding amplitude coefficient in local estimator.

    :param sample: Sample for which value of observable is being determined.
    :type sample: str
    :param j: Subscript of first spin observable.
    :type j: int
    :param operator: The observable to be measured.
                     One of "S+S-" "S-S+".
    :type operator: str

    :returns: Transformed sample after applying two spin observable.
    :rtype: str
    '''
    newSample = torch.tensor(sample)
    if operator == "S+S-":
        newSample[j] = 1
        newSample[j+1] = 0
    elif operator == "S-S+":
        newSample[j] = 0
        newSample[j+1] = 1
    return newSample

class Heisenberg1DEnergy(Observable):
    r"""Observable defining the energy of a 1D Heisenberg model.

    :param h: The strength of the tranverse field
    :type h: float
    :param density: Whether to compute the energy per spin site.
    :type density: bool
    :param periodic_bcs: If `True` use periodic boundary conditions,
                         otherwise use open boundary conditions.
    :type periodic_bcs: bool
    """

    def __init__(self):
        super(Heisenberg1DEnergy, self).__init__()

    @staticmethod
    def _convert(operator,sample,nn_state):
        '''
        Calculates the value of an observable corresponding to a given
        spin configuration.

        :param operator: The observable to be measured.
                         One of "SzSz" "S+S-" "S-S+".
        :type operator: str
        :param sample: Sample for which value of observable is being determined.
        :type sample: str

        :returns: Value of the observable corresponding to the given sample.
        :rtype: float
        '''
        total = torch.tensor([0,0], dtype=torch.float64)
        org = nn_state.psi(sample)
        if operator == "SzSz":
            for i in range(len(sample)-1):
                if sample[i] == 0 and sample[i+1] == 0 or sample[i] == 1 and sample[i+1] == 1:
                    total += 0.25
                else:
                    total += -0.25
        elif operator == "S+S-":
            for i in range(len(sample)-1):
                if sample[i] == 0 and sample[i+1] == 1:
                    total += nn_state.psi(transform(sample,i,"S+S-"))/org
        elif operator == "S-S+":
            for i in range(len(sample)-1):
                if sample[i] == 1 and sample[i+1] == 0:
                    total += nn_state.psi(transform(sample,i,"S-S+"))/org

        return total

    def apply(self, nn_state, samples):
        r"""Computes the energy of each sample given a batch of
        samples.

        :param samples: A batch of samples to calculate the observable on.
                        Must be using the :math:`\sigma_i = 0, 1` convention.
        :type samples: torch.Tensor
        """

        total = torch.tensor([0,0], dtype=torch.float64)
        for i in range(len(samples)):
            total += -self._convert("SzSz",samples[i],nn_state)
            total += -0.5 * self._convert("S+S-",samples[i],nn_state)
            total += -0.5 * self._convert("S-S+",samples[i],nn_state)

        total[1] = 0
        print(total/len(samples))
        return total/len(samples)
