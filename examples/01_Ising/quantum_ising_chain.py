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
import random
import torch
import numpy as np

__all__ = [
    "TransverseFieldIsingChain"
]


class TransverseFieldIsingChain():

    def __init__(self,h,nc,pbc=False):
        super(TransverseFieldIsingChain, self).__init__()
        self.h = h
        self.pbc = pbc
        self.nc = nc 


    def __repr__(self):
        return (f"TransverseFieldIsingChain(h={self.h},"
                f"pbc={self.pbs})")


    @staticmethod
    def _flip_spin(i, s):
        s[:, i] *= -1.0
    
    
    def Randomize(self,N):
        v=np.zeros((self.nc,N))
        for i in range(v.shape[0]):
            for j in range(N):
                if (random.random()>0.5):
                    v[i,j]=1.0
                else:
                    v[i,j]=0.0
        return v
        

    def Run(self,nn_state,n_eq=100, show_convergence=False):
        v0 = torch.tensor(self.Randomize(nn_state.num_visible),dtype=torch.double, device=nn_state.device)
        nn_state.set_visible_layer(v0)
       
        if show_convergence:
            energy_list = []
            err_energy  = []

            sZ_list = []
            err_sZ  = []

            energy = self.Energy(nn_state, v0, show_convergence)
            energy_list.append(energy.mean().item())
            err_energy.append((torch.std(energy)/np.sqrt(energy.size()[0])).item())

            sZ = self.SigmaZ(v0, show_convergence)
            sZ_list.append(sZ.mean().item())
            err_sZ.append((torch.std(sZ)/np.sqrt(sZ.size()[0])).item())

            for steps in range(n_eq):
                nn_state.sample(1)
                samples = nn_state.visible_state
                energy_list.append(self.Energy(nn_state, samples,
                                               show_convergence)
                                   .mean().item())
                sZ_list.append(self.SigmaZ(samples,
                                           show_convergence)
                               .mean() .item())
                nn_state.set_visible_layer(samples)
                
            out = {'energy': {'energies': np.array(energy_list), 'error': np.array(err_energy)},
                   'sigmaZ': {'sZ': np.array(sZ_list), 'error': np.array(err_sZ)}
                  }
 
        else:
            nn_state.sample(n_eq)
            samples = nn_state.visible_state 
        
            energy = self.Energy(nn_state,samples)
            sZ = self.SigmaZ(samples)

            out = {'energy':energy,
                'sigmaZ': sZ
                }

        return out


    def Energy(self,nn_state,samples, show_convergence=False):
        """Computes the energy of each sample given a batch of
        samples.

        :param samples: A batch of samples to calculate the observable on.
                        Must be using the :math:`\sigma_i = 0, 1` convention.
        :type samples: torch.Tensor
        :param sampler: The sampler that drew the samples. Must implement
                        the function :func:`effective_energy`, giving the
                        log probability of its inputs (up to an additive
                        constant).
        :type sampler: qucumber.samplers.Sampler
        """
        samples = to_pm1(samples)
        log_psis = -nn_state.rbm_am.effective_energy(to_01(samples)).div(2.)

        shape = log_psis.shape + (samples.shape[-1],)
        log_flipped_psis = torch.zeros(*shape,
                                       dtype=torch.double,
                                       device=nn_state.rbm_am.device)

        for i in range(samples.shape[-1]):  # sum over spin sites
            self._flip_spin(i, samples)  # flip the spin at site i
            log_flipped_psis[:, i] =-nn_state.rbm_am.effective_energy(
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

        if show_convergence:
            energy = (transverse_field_terms.mul(self.h).add(interaction_terms)
                      .mul(-1.))
        
        else:
            energy = (transverse_field_terms.mul(self.h).add(interaction_terms)
                      .mul(-1.)).mean()          

        return energy.div(samples.shape[-1])

    def SigmaZ(self, samples, show_convergence=False):
        """Computes the magnetization of each sample given a batch of samples.

        :param samples: A batch of samples to calculate the observable on.
                        Must be using the :math:`\sigma_i = 0, 1` convention.
        :type samples: torch.Tensor
        """
        if show_convergence:
            return (to_pm1(samples)
                    .mean(1)
                    .abs())

        else:
            return (to_pm1(samples)
                    .mean(1)
                    .abs()).mean()           


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

