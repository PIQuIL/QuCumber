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

from qucumber.observables import Observable

__all__ = [
    "TransverseFieldIsingChain"
]


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
    def __init__(self, h, nc):
        super(TFIMChainEnergy, self).__init__()
        self.h                = h
        self.nc               = nc


    def apply(self, samples, nn_state):
        return self.Energy(nn_state, samples)


    @staticmethod
    def _flip_spin(i, s):
        s[:, i] *= -1.0


    def Randomize(self, N):
        p = torch.ones(self.nc, N) * 0.5
        return torch.bernoulli(p)


    def Energy(self, nn_state, samples):
        """Computes the eneself.Energy(nn_state, v)rgy of each sample given a batch of
        samples.

        :param samples: A batch of samples to calculate the observable on.
                        Must be using the :math:`\sigma_i = 0, 1` convention.
        :type samples: torch.Tensor
        """
        samples = to_pm1(samples)
        log_psis = -nn_state.rbm_am.effective_energy(to_01(samples)).div(2.)

        shape = log_psis.shape + (samples.shape[-1],)
        log_flipped_psis = torch.zeros(*shape,
                                       dtype=torch.double,
                                       device=nn_state.rbm_am.device)

        for i in range(samples.shape[-1]):  # sum over spin sites
            self._flip_spin(i, samples)  # flip the spin at site i
            log_flipped_psis[:, i] = -nn_state.rbm_am.effective_energy(
                to_01(samples)
            ).div(2.)
            self._flip_spin(i, samples)  # flip it back

        log_flipped_psis = torch.logsumexp(log_flipped_psis, 1,
                                           keepdim=True).squeeze()

        interaction_terms = ((samples[:, :-1] * samples[:, 1:])
                             .sum(1))      # sum over spin sites

        transverse_field_terms = (log_flipped_psis
                                  .sub(log_psis)
                                  .exp())  # convert to ratio of probabilities

        energy = (transverse_field_terms.mul(self.h).add(interaction_terms)
                  .mul(-1.))

        return energy.div(samples.shape[-1])


class TFIMChainMagnetization(Observable):
    """Observable defining the magnetization of a Transverse Field Ising Model
    (TFIM) spin chain.
    """
    def __init__(self, nc):
        super(TFIMChainMagnetization, self).__init__()
        self.nc = nc


    def __repr__(self):
        return "TFIMChainMagnetization()"


    def apply(self, samples, nn_state):
        """Computes the magnetization of each sample given a batch of samples.

        :param samples: A batch of samples to calculate the observable on.
                        Must be using the :math:`\sigma_i = 0, 1` convention.
        :type samples: torch.Tensor
        """
        return self.SigmaZ(samples)


    def Randomize(self, N):
        p = torch.ones(self.nc, N) * 0.5
        return torch.bernoulli(p)


    def Run(self, nn_state, n_eq):
        v = self.Randomize(nn_state.num_visible).to(dtype=torch.double,
                                                    device=nn_state.device)

        num_samples = self.nc
        if self.show_convergence:

            sZ_list = []
            err_sZ = []

            sZ = self.SigmaZ(v)
            sZ_list.append(sZ.apply(v, nn_state).mean())
            err_sZ.append((torch.std(sZ)/np.sqrt(sZ.size()[0])).item())

            for steps in range(n_eq):
                v = nn_state.gibbs_steps(1, v, overwrite=True)
                sZ_list.append(self.SigmaZ(v, show_convergence).mean().item())

            out = {'sZ':    np.array(sZ_list),
                   'error': np.array(err_sZ)
                  }

        else:
            
            out = {'error':  self.std_error(nn_state, num_samples), 
                   'sZ': self.expected_value(nn_state, num_samples)
                  }

    def SigmaZ(self, samples):
        """Computes the magnetization of each sample given a batch of samples.

        :param samples: A batch of samples to calculate the observable on.
                        Must be using the :math:`\sigma_i = 0, 1` convention.
        :type samples: torch.Tensor
        """
        return (to_pm1(samples)
                .mean(1)
                .abs())


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


def Convergence(nn_state, tfim_energy, tfim_sZ, n_measurements, n_eq):
    energy_list = []
    err_energy  = []

    sZ_list = []
    err_sZ  = []

    v = torch.bernoulli(torch.ones(n_measurements, nn_state.num_visible, dtype = torch.double, device = nn_state.device)*0.5)

    energy = tfim_energy.Energy(nn_state, v)
    energy_list.append(energy.mean().item())
    err_energy.append(torch.std(energy).div(np.sqrt(energy.size()[0])).item())

    sZ = tfim_sZ.SigmaZ(v)
    sZ_list.append(sZ.mean().item())
    err_sZ.append((torch.std(sZ)/np.sqrt(sZ.size()[0])).item())

    for steps in range(n_eq):
        v = nn_state.gibbs_steps(1, v, overwrite=True)
        
        energy = tfim_energy.Energy(nn_state, v)
        energy_list.append(energy.mean().item())
        err_energy.append(torch.std(energy).div(np.sqrt(energy.size()[0])).item())
        
        sZ = tfim_sZ.SigmaZ(v)
        sZ_list.append(sZ.mean().item())
        err_sZ.append((torch.std(sZ)/np.sqrt(sZ.size()[0])).item())

    out = {'energy': {
                      'energies': np.array(energy_list),
                      'error':    np.array(err_energy)
                     },
           'sigmaZ': {
                      'sZ':    np.array(sZ_list),
                      'error': np.array(err_sZ)
                     }
          }
    
    return out
