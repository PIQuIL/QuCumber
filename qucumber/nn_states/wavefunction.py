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

import abc

import numpy as np
import torch


class Wavefunction(abc.ABC):
    """Abstract Base Class for Wavefunctions."""

    @property
    def max_size(self):
        """Maximum size of the Hilbert space for full enumeration"""
        return 20

    @property
    @abc.abstractmethod
    def networks(self):
        """A list of the names of the internal RBMs."""

    @property
    @abc.abstractmethod
    def rbm_am(self):
        """The RBM to be used to learn the wavefunction amplitude."""

    @rbm_am.setter
    @abc.abstractmethod
    def rbm_am(self, new_val):
        return

    @property
    @abc.abstractmethod
    def device(self):
        """The device that the model is on."""

    @device.setter
    @abc.abstractmethod
    def device(self, new_val):
        return

    def reinitialize_parameters(self):
        """Randomize the parameters of the internal RBMs."""
        for net in self.networks:
            getattr(self, net).initialize_parameters()

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

    @abc.abstractmethod
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

    def probability(self, v, Z):
        """Evaluates the probability of the given vector(s) of visible
        states.

        :param v: The visible states.
        :type v: torch.Tensor
        :param Z: The partition function.
        :type Z: float

        :returns: The probability of the given vector(s) of visible units.
        :rtype: torch.Tensor
        """
        v = v.to(device=self.device, dtype=torch.double)
        return (self.amplitude(v)[0]) ** 2 / Z

    @abc.abstractmethod
    def gradient(self):
        """Compute the gradient of a set of samples."""

    def sample(self, k, num_samples=1, initial_state=None, overwrite=False):
        r"""Performs k steps of Block Gibbs sampling. One step consists of sampling
        the hidden state :math:`\bm{h}` from the conditional distribution
        :math:`p_{\bm{\lambda}}(\bm{h}\:|\:\bm{v})`, and sampling the visible
        state :math:`\bm{v}` from the conditional distribution
        :math:`p_{\bm{\lambda}}(\bm{v}\:|\:\bm{h})`.

        :param k: Number of Block Gibbs steps.
        :type k: int
        :param num_samples: The number of samples to generate.
        :type num_samples: int
        :param initial_state: The initial state of the Markov Chain. If given,
                              `num_samples` will be ignored.
        :type initial_state: torch.Tensor
        :param overwrite: Whether to overwrite the initial_state tensor, if it is provided.
        :type overwrite: bool
        """
        if initial_state is None:
            dist = torch.distributions.Bernoulli(probs=0.5)
            sample_size = torch.Size((num_samples, self.num_visible))
            initial_state = dist.sample(sample_size).to(
                device=self.device, dtype=torch.double
            )

        return self.rbm_am.gibbs_steps(k, initial_state, overwrite=overwrite)

    def subspace_vector(self, num, size=None):
        r"""Generates a single vector from the Hilbert space of dimension
        :math:`2^{\text{size}}`.

        :param size: The size of each element of the Hilbert space.
        :type size: int
        :param num: The specific vector to return from the Hilbert space. Since
                    the Hilbert space can be represented by the set of binary strings
                    of length `size`, `num` is equivalent to the decimal representation
                    of the returned vector.
        :type num: int

        :returns: A state from the Hilbert space.
        :rtype: torch.Tensor
        """
        size = size if size else self.num_visible
        space = (((num & (1 << np.arange(size)))) > 0)[::-1]
        space = space.astype(int)
        return torch.tensor(space, dtype=torch.double, device=self.device)

    def generate_hilbert_space(self, size=None):
        r"""Generates Hilbert space of dimension :math:`2^{\text{size}}`.

        :param size: The size of each element of the Hilbert space. Defaults to
                     the number of visible units.
        :type size: int

        :returns: A tensor with all the basis states of the Hilbert space.
        :rtype: torch.Tensor
        """
        size = size if size else self.rbm_am.num_visible
        if size > self.max_size:
            raise ValueError("Size of the Hilbert space is too large!")
        else:
            dim = np.arange(2 ** size)
            space = (((dim[:, None] & (1 << np.arange(size)))) > 0)[:, ::-1]
            space = space.astype(int)
            return torch.tensor(space, dtype=torch.double, device=self.device)

    def compute_normalization(self, space):
        r"""Compute the normalization constant of the wavefunction.

        .. math::

            Z_{\bm{\lambda}}=
            \sqrt{\sum_{\bm{\sigma}}|\psi_{\bm{\lambda\mu}}|^2}=
            \sqrt{\sum_{\bm{\sigma}} p_{\bm{\lambda}}(\bm{\sigma})}

        :param space: A rank 2 tensor of the entire visible space.
        :type space: torch.Tensor

        """
        return self.rbm_am.partition(space)

    def save(self, location, metadata=None):
        """Saves the Wavefunction parameters to the given location along with
        any given metadata.

        :param location: The location to save the data.
        :type location: str or file
        :param metadata: Any extra metadata to store alongside the Wavefunction
                         parameters.
        :type metadata: dict
        """
        # add extra metadata to dictionary before saving it to disk
        metadata = metadata if metadata else {}

        # validate metadata
        for net in self.networks:
            if net in metadata.keys():
                raise ValueError(
                    "Invalid key in metadata; '{}' cannot be a key!".format(net)
                )

        data = {net: getattr(self, net).state_dict() for net in self.networks}
        data.update(**metadata)
        torch.save(data, location)

    def load(self, location):
        """Loads the Wavefunction parameters from the given location ignoring any
        metadata stored in the file. Overwrites the Wavefunction's parameters.

        .. note::
            The Wavefunction object on which this function is called must
            have the same parameter shapes as the one who's parameters are being
            loaded.

        :param location: The location to load the Wavefunction parameters from.
        :type location: str or file
        """
        try:
            state_dict = torch.load(location)
        except AssertionError as e:
            state_dict = torch.load(location, lambda storage, loc: "cpu")

        for net in self.networks:
            getattr(self, net).load_state_dict(state_dict[net])

    @staticmethod
    @abc.abstractmethod
    def autoload(location, gpu=False):
        """Initializes a Wavefunction from the parameters in the given
        location.

        :param location: The location to load the model parameters from.
        :type location: str or file
        :param gpu: Whether the returned model should be on the GPU.
        :type gpu: bool

        :returns: A new Wavefunction initialized from the given parameters.
                  The returned Wavefunction will be of whichever type this function
                  was called on.
        """


# make module path show up properly in sphinx docs
Wavefunction.__module__ = "qucumber.nn_states"
