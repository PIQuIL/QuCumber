# Copyright 2019 PIQuIL - All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from .observable import ObservableBase
from qucumber.utils import cplx


def swap(s1, s2, A):
    r"""Applies the Swap operator onto two sets of samples, `s1` and `s2`.

    The sites that will be swapped are defined by `A`.

    :param s1: A rank 2 tensor of spin samples representing the first replica.
    :type s1: torch.Tensor
    :param s2: A rank 2 tensor of spin samples representing the second replica.
    :type s2: torch.Tensor
    :param A: The sites that will be swapped between the two replicas.
    :type A: int or list or np.array or torch.Tensor
    """
    _s = s1[:, A].clone()
    s1[:, A] = s2[:, A]
    s2[:, A] = _s
    return s1, s2


class SWAP(ObservableBase):
    r"""The :math:`\text{Swap}_A` observable.

    Can be used to compute the 2nd Renyi entropy of the region A through:

    :math:`S_2 = -\ln\langle \text{Swap}_A \rangle`

    Ref: PhysRevLett.104.157201

    :param A: The sites contained in the region A.
    :type A: int or list or np.array or torch.Tensor
    """

    def __init__(self, A):
        self.name = "SWAP"
        self.symbol = "S"
        self.A = A

    def apply(self, nn_state, samples):
        r"""Computes the swap operator which an estimator for the 2nd Renyi
        entropy.

        The swap operator requires access to two identical copies of a
        wavefunction. In practice, this translates to the requirement of
        having two independent sets of samples from the wavefunction replicas.
        For this purpose, the batch of samples stored in the param samples is
        split into two subsets. Although this procedure is designed to break
        the autocorrelation between the samples, it must be used with caution.
        For a fully unbiased estimate of the entanglement entropy, the batch
        of samples needs to be built from two independent initializations of
        the wavefunction each having a different random number generator.

        :param nn_state: The WaveFunction that drew the samples.
        :type nn_state: qucumber.nn_states.WaveFunctionBase
        :param samples: A batch of samples to calculate the observable on.
                        Must be using the :math:`\sigma_i = 0, 1` convention.
        :type samples: torch.Tensor
        """
        samples = samples.to(device=nn_state.device, copy=True)

        # split the batch of samples into two equal batches
        # if their total number is odd, the last sample is ignored
        _ns = samples.shape[0] // 2
        samples1 = samples[:_ns, :]
        samples2 = samples[_ns : _ns * 2, :]

        psi_ket1 = nn_state.psi(samples1)
        psi_ket2 = nn_state.psi(samples2)

        psi_ket = cplx.elementwise_mult(psi_ket1, psi_ket2)
        psi_ket_star = cplx.conjugate(psi_ket)

        samples1_, samples2_ = swap(samples1, samples2, self.A)
        psi_bra1 = nn_state.psi(samples1_)
        psi_bra2 = nn_state.psi(samples2_)

        psi_bra = cplx.elementwise_mult(psi_bra1, psi_bra2)
        psi_bra_star = cplx.conjugate(psi_bra)
        return cplx.real(cplx.elementwise_division(psi_bra_star, psi_ket_star))
