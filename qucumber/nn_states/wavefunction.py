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


import abc

from qucumber.utils import cplx
from .neural_state import NeuralStateBase


class WaveFunctionBase(NeuralStateBase):
    """Abstract Base Class for WaveFunctions."""

    def reinitialize_parameters(self):
        """Randomize the parameters of the internal RBMs."""
        for net in self.networks:
            getattr(self, net).initialize_parameters()

    def __getattr__(self, attr):
        return getattr(self.rbm_am, attr)

    def amplitude(self, v):
        r"""Compute the (unnormalized) amplitude of a given vector/matrix of visible states.

        .. math::

            \text{amplitude}(\bm{\sigma})=|\psi(\bm{\sigma})|

        :param v: visible states :math:`\bm{\sigma}`
        :type v: torch.Tensor

        :returns: Matrix/vector containing the amplitudes of v
        :rtype: torch.Tensor
        """
        return (-self.rbm_am.effective_energy(v)).exp().sqrt()

    @abc.abstractmethod
    def phase(self, v):
        r"""Compute the phase of a given vector/matrix of visible states.

        .. math::
            \text{phase}(\bm{\sigma})

        :param v: visible states :math:`\bm{\sigma}`
        :type v: torch.Tensor

        :returns: Matrix/vector containing the phases of v
        :rtype: torch.Tensor
        """

    def psi(self, v):
        r"""Compute the (unnormalized) wavefunction of a given vector/matrix of
        visible states.

        .. math::
            \psi(\bm{\sigma})

        :param v: visible states :math:`\bm{\sigma}`
        :type v: torch.Tensor

        :returns: Complex object containing the value of the wavefunction for
                  each visible state
        :rtype: torch.Tensor
        """
        # vectors/tensors of shape (len(v),)
        amplitude, phase = self.amplitude(v), self.phase(v)
        return cplx.make_complex(
            amplitude * phase.cos(),  # real part
            amplitude * phase.sin(),  # imaginary part
        )

    def importance_sampling_numerator(self, vp, v):
        return self.psi(vp)

    def importance_sampling_denominator(self, v):
        return self.psi(v)


# make module path show up properly in sphinx docs
WaveFunctionBase.__module__ = "qucumber.nn_states"
