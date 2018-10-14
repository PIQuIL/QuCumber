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
from .observable import Observable
from qucumber.utils import cplx


def swap(s1, s2, A):
    for a in A:
        _s = s1[:, a].clone()
        s1[:, a] = s2[:, a]
        s2[:, a] = _s

    return s1, s2


class RenyiEntropy(Observable):
    r"""The :math:`\sigma_y` observable

    Computes the 2nd Renyi entropy of the region A based on the SWAP operator.
    Ref: PhysRevLett.104.157201
    """

    def __init__(self):
        self.name = "SWAP"
        self.symbol = "S"

    def apply(self, nn_state, samples, A):
        r"""Computes the entanglement entropy via a swap operator which an esimtaor for the 2nd Renyi entropy.
        The swap operator requires an access to two identical copies of a wavefunction. In practice, this translates
        to the requirement of having two independent sets of samples from the wavefunction replicas. For this 
        purpose, the batch of samples stored in the param samples is split into two subsets. Although this 
        procedure is designed to break the autocorrelation between the samples, it must be used with caution.   
        For a fully unbiased estimate of the entanglement entropy, the batch of samples needs to be built from two 
        independent initializations of the wavefucntion each having a different random number generator. 

        :param nn_state: The Wavefunction that drew the samples.
        :type nn_state: qucumber.nn_states.Wavefunction
        :param samples: A batch of samples to calculate the observable on.
                        Must be using the :math:`\sigma_i = 0, 1` convention.
        :type samples: torch.Tensor
        """
        samples = samples.to(device=nn_state.device).clone()

        # split the batch of samples into two equal batches
        # if their total number is odd, the last sample is ignored
        _ns = samples.shape[0] // 2
        samples1 = samples[:_ns, :]
        samples2 = samples[_ns : _ns * 2, :]

        # print('Wavefunction:')
        # print(nn_state.psi(samples))
        # vectors of shape: (2, num_samples,)
        psi_ket1 = nn_state.psi(samples1)
        psi_ket2 = nn_state.psi(samples2)

        psi_ket = cplx.elementwise_mult(psi_ket1, psi_ket2)
        psi_ket_star = cplx.conjugate(psi_ket)

        # print('Replicated wavefunction ket')
        # print(psi_ket)

        # sample_norm = cplx.elementwise_mult(psi_ket_star, psi_ket).mean(1)[0]
        # print()
        # print('Sample norm')
        # print(sample_norm)

        samples1_, samples2_ = swap(samples1, samples2, A)
        psi_bra1 = nn_state.psi(samples1_)
        psi_bra2 = nn_state.psi(samples2_)

        psi_bra = cplx.elementwise_mult(psi_bra1, psi_bra2)
        psi_bra_star = cplx.conjugate(psi_bra)
        # print('Replicated wavefunction bra')
        # print(psi_bra)

        # print("Weight ratios")
        # print(cplx.elementwise_division(psi_bra_star, psi_ket_star))

        # print('Entanglement')
        EE = -torch.log(
            cplx.elementwise_division(psi_bra_star, psi_ket_star).mean(1)
        )  # /sample_norm)
        # print(EE)
        return EE
