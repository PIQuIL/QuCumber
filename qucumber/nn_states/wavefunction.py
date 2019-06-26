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
from itertools import chain
from math import ceil

import numpy as np
import torch
from tqdm import tqdm, tqdm_notebook

from qucumber.callbacks import CallbackList, Timer
from qucumber.utils.data import extract_refbasis_samples
from qucumber.utils.gradients_utils import vector_to_grads


class WaveFunctionBase(abc.ABC):
    """Abstract Base Class for WaveFunctions."""

    _stop_training = False

    @property
    def stop_training(self):
        """If `True`, will not train.

        If this property is set to `True` during the training cycle, training
        will terminate once the current batch or epoch ends (depending on when
        `stop_training` was set).
        """
        return self._stop_training

    @stop_training.setter
    def stop_training(self, new_val):
        if isinstance(new_val, bool):
            self._stop_training = new_val
        else:
            raise ValueError("stop_training must be a boolean value!")

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
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def device(self):
        """The device that the model is on."""

    @device.setter
    @abc.abstractmethod
    def device(self, new_val):
        raise NotImplementedError

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
        return (self.amplitude(v)) ** 2 / Z

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
        :param initial_state: The initial state of the Markov Chains. If given,
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

    def subspace_vector(self, num, size=None, device=None):
        r"""Generates a single vector from the Hilbert space of dimension
        :math:`2^{\text{size}}`.

        :param size: The size of each element of the Hilbert space.
        :type size: int
        :param num: The specific vector to return from the Hilbert space. Since
                    the Hilbert space can be represented by the set of binary strings
                    of length `size`, `num` is equivalent to the decimal representation
                    of the returned vector.
        :type num: int
        :param device: The device to create the vector on. Defaults to the
                       device this model is on.

        :returns: A state from the Hilbert space.
        :rtype: torch.Tensor
        """
        device = device if device is not None else self.device
        size = size if size else self.num_visible
        space = ((num & (1 << np.arange(size))) > 0)[::-1]
        space = space.astype(int)
        return torch.tensor(space, dtype=torch.double, device=device)

    def generate_hilbert_space(self, size=None, device=None):
        r"""Generates Hilbert space of dimension :math:`2^{\text{size}}`.

        :param size: The size of each element of the Hilbert space. Defaults to
                     the number of visible units.
        :type size: int
        :param device: The device to create the Hilbert space matrix on.
                       Defaults to the device this model is on.

        :returns: A tensor with all the basis states of the Hilbert space.
        :rtype: torch.Tensor
        """
        device = device if device is not None else self.device
        size = size if size else self.rbm_am.num_visible
        if size > self.max_size:
            raise ValueError("Size of the Hilbert space is too large!")
        else:
            dim = np.arange(2 ** size)
            space = ((dim[:, None] & (1 << np.arange(size))) > 0)[:, ::-1]
            space = space.astype(int)
            return torch.tensor(space, dtype=torch.double, device=device)

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
        """Saves the WaveFunction parameters to the given location along with
        any given metadata.

        :param location: The location to save the data.
        :type location: str or file
        :param metadata: Any extra metadata to store alongside the WaveFunction
                         parameters.
        :type metadata: dict
        """
        # add extra metadata to dictionary before saving it to disk
        metadata = metadata if metadata else {}

        # validate metadata
        for net in self.networks:
            if net in metadata.keys():
                raise ValueError(f"Invalid key in metadata; '{net}' cannot be a key!")

        data = {net: getattr(self, net).state_dict() for net in self.networks}
        data.update(**metadata)
        torch.save(data, location)

    def load(self, location):
        """Loads the WaveFunction parameters from the given location ignoring any
        metadata stored in the file. Overwrites the WaveFunction's parameters.

        .. note::
            The WaveFunction object on which this function is called must
            have the same parameter shapes as the one who's parameters are being
            loaded.

        :param location: The location to load the WaveFunction parameters from.
        :type location: str or file
        """
        state_dict = torch.load(location, map_location=self.device)

        for net in self.networks:
            getattr(self, net).load_state_dict(state_dict[net])

    @staticmethod
    @abc.abstractmethod
    def autoload(location, gpu=False):
        """Initializes a WaveFunction from the parameters in the given
        location.

        :param location: The location to load the model parameters from.
        :type location: str or file
        :param gpu: Whether the returned model should be on the GPU.
        :type gpu: bool

        :returns: A new WaveFunction initialized from the given parameters.
                  The returned WaveFunction will be of whichever type this function
                  was called on.
        """

    @abc.abstractmethod
    def gradient(self):
        """Compute the gradient of a set of samples."""

    def compute_batch_gradients(self, k, samples_batch, neg_batch, bases_batch=None):
        """Compute the gradients of a batch of the training data (`samples_batch`).

        If measurements are taken in bases other than the reference basis,
        a list of bases (`bases_batch`) must also be provided.

        :param k: Number of contrastive divergence steps in training.
        :type k: int
        :param samples_batch: Batch of the input samples.
        :type samples_batch: torch.Tensor
        :param neg_batch: Batch of the input samples for computing the
                          negative phase.
        :type neg_batch: torch.Tensor
        :param bases_batch: Batch of the input bases corresponding to the samples
                            in `samples_batch`.
        :type bases_batch: np.array

        :returns: List containing the gradients of the parameters.
        :rtype: list
        """
        # Negative phase: learning signal driven by the amplitude RBM of
        # the NN state
        vk = self.rbm_am.gibbs_steps(k, neg_batch)
        grad_model = self.rbm_am.effective_energy_gradient(vk)

        # If measurements are taken in the reference bases only
        if bases_batch is None:
            grad = [0.0]
            # Positive phase: learning signal driven by the data (and bases)
            grad_data = self.gradient(samples_batch)
            # Gradient = Positive Phase - Negative Phase
            grad[0] = grad_data / float(samples_batch.shape[0])
            grad[0] -= grad_model / float(neg_batch.shape[0])
        else:
            grad = [0.0, 0.0]

            # Initialize
            grad_data = [
                torch.zeros(
                    getattr(self, net).num_pars, dtype=torch.double, device=self.device
                )
                for net in self.networks
            ]

            # Loop over each sample in the batch
            for i in range(samples_batch.shape[0]):
                # Positive phase: learning signal driven by the data
                #                 (and bases)
                data_gradient = self.gradient(bases_batch[i], samples_batch[i])
                # Accumulate amplitude RBM gradient
                grad_data[0] += data_gradient[0]

                # Accumulate phase RBM gradient
                grad_data[1] += data_gradient[1]

            # Gradient = Positive Phase - Negative Phase
            grad[0] = grad_data[0] / float(samples_batch.shape[0])
            grad[0] -= grad_model / float(neg_batch.shape[0])

            # No negative signal for the phase parameters
            grad[1] = grad_data[1] / float(samples_batch.shape[0])

        return grad

    def _shuffle_data(
        self,
        pos_batch_size,
        neg_batch_size,
        num_batches,
        train_samples,
        input_bases,
        z_samples,
    ):
        pos_batch_perm = torch.randperm(train_samples.shape[0])

        shuffled_pos_samples = train_samples[pos_batch_perm]
        if input_bases is None:
            if neg_batch_size == pos_batch_size:
                neg_batch_perm = pos_batch_perm
            else:
                neg_batch_perm = torch.randint(
                    train_samples.shape[0],
                    size=(num_batches * neg_batch_size,),
                    dtype=torch.long,
                )
            shuffled_neg_samples = train_samples[neg_batch_perm]
        else:
            neg_batch_perm = torch.randint(
                z_samples.shape[0],
                size=(num_batches * neg_batch_size,),
                dtype=torch.long,
            )
            shuffled_neg_samples = z_samples[neg_batch_perm]

        # List of all the batches for positive phase.
        pos_batches = [
            shuffled_pos_samples[batch_start : (batch_start + pos_batch_size)]
            for batch_start in range(0, len(shuffled_pos_samples), pos_batch_size)
        ]

        neg_batches = [
            shuffled_neg_samples[batch_start : (batch_start + neg_batch_size)]
            for batch_start in range(0, len(shuffled_neg_samples), neg_batch_size)
        ]

        if input_bases is not None:
            shuffled_pos_bases = input_bases[pos_batch_perm]
            pos_batches_bases = [
                shuffled_pos_bases[batch_start : (batch_start + pos_batch_size)]
                for batch_start in range(0, len(train_samples), pos_batch_size)
            ]
            return zip(pos_batches, neg_batches, pos_batches_bases)
        else:
            return zip(pos_batches, neg_batches)

    def fit(
        self,
        data,
        epochs=100,
        pos_batch_size=100,
        neg_batch_size=None,
        k=1,
        lr=1e-3,
        input_bases=None,
        progbar=False,
        starting_epoch=1,
        time=False,
        callbacks=None,
        optimizer=torch.optim.SGD,
        **kwargs,
    ):
        """Train the WaveFunction.

        :param data: The training samples
        :type data: np.array
        :param epochs: The number of full training passes through the dataset.
                       Technically, this specifies the index of the *last* training
                       epoch, which is relevant if `starting_epoch` is being set.
        :type epochs: int
        :param pos_batch_size: The size of batches for the positive phase
                               taken from the data.
        :type pos_batch_size: int
        :param neg_batch_size: The size of batches for the negative phase
                               taken from the data. Defaults to `pos_batch_size`.
        :type neg_batch_size: int
        :param k: The number of contrastive divergence steps.
        :type k: int
        :param lr: Learning rate
        :type lr: float
        :param input_bases: The measurement bases for each sample. Must be provided
                            if training a ComplexWaveFunction.
        :type input_bases: np.array
        :param progbar: Whether or not to display a progress bar. If "notebook"
                        is passed, will use a Jupyter notebook compatible
                        progress bar.
        :type progbar: bool or str
        :param starting_epoch: The epoch to start from. Useful if continuing training
                               from a previous state.
        :type starting_epoch: int
        :param callbacks: Callbacks to run while training.
        :type callbacks: list[qucumber.callbacks.CallbackBase]
        :param optimizer: The constructor of a torch optimizer.
        :type optimizer: torch.optim.Optimizer
        :param kwargs: Keyword arguments to pass to the optimizer
        """
        if self.stop_training:  # terminate immediately if stop_training is true
            return

        disable_progbar = progbar is False
        progress_bar = tqdm_notebook if progbar == "notebook" else tqdm

        callbacks = CallbackList(callbacks if callbacks else [])
        if time:
            callbacks.append(Timer())

        neg_batch_size = neg_batch_size if neg_batch_size else pos_batch_size

        if isinstance(data, torch.Tensor):
            train_samples = (
                data.clone().detach().to(device=self.device, dtype=torch.double)
            )
        else:
            train_samples = torch.tensor(data, device=self.device, dtype=torch.double)

        if len(self.networks) > 1:
            all_params = [getattr(self, net).parameters() for net in self.networks]
            all_params = list(chain(*all_params))
            optimizer = optimizer(all_params, lr=lr, **kwargs)
        else:
            optimizer = optimizer(self.rbm_am.parameters(), lr=lr, **kwargs)

        if input_bases is not None:
            z_samples = extract_refbasis_samples(train_samples, input_bases).to(
                device=self.device
            )
        else:
            z_samples = None

        callbacks.on_train_start(self)

        num_batches = ceil(train_samples.shape[0] / pos_batch_size)
        for ep in progress_bar(
            range(starting_epoch, epochs + 1), desc="Epochs ", disable=disable_progbar
        ):
            data_iterator = self._shuffle_data(
                pos_batch_size,
                neg_batch_size,
                num_batches,
                train_samples,
                input_bases,
                z_samples,
            )
            callbacks.on_epoch_start(self, ep)

            for b, batch in enumerate(data_iterator):
                callbacks.on_batch_start(self, ep, b)

                all_grads = self.compute_batch_gradients(k, *batch)

                optimizer.zero_grad()  # clear any cached gradients

                # assign gradients to corresponding parameters
                for i, net in enumerate(self.networks):
                    rbm = getattr(self, net)
                    vector_to_grads(all_grads[i], rbm.parameters())

                optimizer.step()  # tell the optimizer to apply the gradients

                callbacks.on_batch_end(self, ep, b)
                if self.stop_training:  # check for stop_training signal
                    break

            callbacks.on_epoch_end(self, ep)
            if self.stop_training:  # check for stop_training signal
                break

        callbacks.on_train_end(self)


# make module path show up properly in sphinx docs
WaveFunctionBase.__module__ = "qucumber.nn_states"
