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


import torch

from qucumber.utils import cplx
from .observable import ObservableBase
from .utils import to_pm1


def flip_spin(i, samples):
    r"""Flip the i-th spin configuration in `samples`.

    :param i: The i-th spin.
    :type i: int
    :param samples: A batch of samples.
                    Must be using the :math:`\sigma_i = 0, 1` convention.
    :type samples: torch.Tensor
    """
    samples[:, i].sub_(1).abs_()


class SigmaX(ObservableBase):
    r"""The :math:`\sigma_x` observable

    Computes the magnetization in the X direction of a spin chain.

    :param absolute: Specifies whether to estimate the absolute magnetization.
    :type absolute: bool
    """

    def __init__(self, absolute=False):
        self.name = "SigmaX"
        self.symbol = "X"
        self.absolute = absolute

    def apply(self, nn_state, samples):
        r"""Computes the magnetization along X of each sample in the given batch of samples.

        Assumes that the computational basis that the WaveFunction was trained
        on was the Z basis.

        :param nn_state: The WaveFunction that drew the samples.
        :type nn_state: qucumber.nn_states.WaveFunctionBase
        :param samples: A batch of samples to calculate the observable on.
                        Must be using the :math:`\sigma_i = 0, 1` convention.
        :type samples: torch.Tensor
        """
        samples = samples.to(device=nn_state.device)

        # vectors of shape: (2, num_samples,)
        psis = nn_state.psi(samples)
        psi_ratio_sum = torch.zeros_like(psis)

        for i in range(samples.shape[-1]):  # sum over spin sites
            flip_spin(i, samples)  # flip the spin at site i

            # compute ratio of psi_(-i) / psi and add it to the running sum
            psi_ratio = nn_state.psi(samples)
            psi_ratio = cplx.elementwise_division(psi_ratio, psis)
            psi_ratio_sum.add_(psi_ratio)

            flip_spin(i, samples)  # flip it back

        # take real part (imaginary part should be approximately zero)
        # and divide by number of spins
        res = cplx.real(psi_ratio_sum).div_(samples.shape[-1])
        if self.absolute:
            return res.abs_()
        else:
            return res


class SigmaY(ObservableBase):
    r"""The :math:`\sigma_y` observable

    Computes the magnetization in the Y direction of a spin chain.

    :param absolute: Specifies whether to estimate the absolute magnetization.
    :type absolute: bool
    """

    def __init__(self, absolute=False):
        self.name = "SigmaY"
        self.symbol = "Y"
        self.absolute = absolute

    def apply(self, nn_state, samples):
        r"""Computes the magnetization along Y of each sample in the given batch of samples.

        Assumes that the computational basis that the WaveFunction was trained
        on was the Z basis.

        :param nn_state: The WaveFunction that drew the samples.
        :type nn_state: qucumber.nn_states.WaveFunctionBase
        :param samples: A batch of samples to calculate the observable on.
                        Must be using the :math:`\sigma_i = 0, 1` convention.
        :type samples: torch.Tensor
        """
        samples = samples.to(device=nn_state.device)

        # vectors of shape: (2, num_samples,)
        psis = nn_state.psi(samples)
        psi_ratio_sum = torch.zeros_like(psis)

        for i in range(samples.shape[-1]):  # sum over spin sites

            coeff = -to_pm1(samples[:, i])
            coeff = cplx.make_complex(torch.zeros_like(coeff), coeff)

            flip_spin(i, samples)  # flip the spin at site i

            # compute ratio of psi_(-i) / psi, multiply it by the appropriate
            # eigenvalue, and add it to the running sum
            psi_ratio = nn_state.psi(samples)
            psi_ratio = cplx.elementwise_division(psi_ratio, psis)
            psi_ratio = cplx.elementwise_mult(psi_ratio, coeff)
            psi_ratio_sum.add_(psi_ratio)

            flip_spin(i, samples)  # flip it back

        # take real part (imaginary part should be approximately zero)
        # and divide by number of spins
        res = cplx.real(psi_ratio_sum).div_(samples.shape[-1])
        if self.absolute:
            return res.abs_()
        else:
            return res


class SigmaZ(ObservableBase):
    r"""The :math:`\sigma_z` observable.

    Computes the magnetization in the Z direction of a spin chain.

    :param absolute: Specifies whether to estimate the absolute magnetization.
    :type absolute: bool
    """

    def __init__(self, absolute=False):
        self.name = "SigmaZ"
        self.symbol = "Z"
        self.absolute = absolute

    def apply(self, nn_state, samples):
        r"""Computes the magnetization along Z of each sample given a batch of samples.

        Assumes that the computational basis that the WaveFunction was trained
        on was the Z basis.

        :param nn_state: The WaveFunction that drew the samples.
        :type nn_state: qucumber.nn_states.WaveFunctionBase
        :param samples: A batch of samples to calculate the observable on.
                        Must be using the :math:`\sigma_i = 0, 1` convention.
        :type samples: torch.Tensor
        """
        # convert to +/- 1 convention, *after* computing the
        # mean to reduce total computations
        res = to_pm1(samples.mean(1))
        if self.absolute:
            return res.abs_()
        else:
            return res
