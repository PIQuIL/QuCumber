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
from qucumber.utils import cplx
from qucumber.utils.data import extract_refbasis_samples
from qucumber.utils.gradients_utils import vector_to_grads
from .neural_state import NeuralState


class WaveFunctionBase(NeuralState):
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

    def positive_phase_gradients(self, samples_batch, bases_batch=None):

        # If measurements are taken in the reference bases only
        if bases_batch is None:
            grad = [0.0]
            # Positive phase: learning signal driven by the data (and bases)
            grad_data = self.gradient(samples_batch)
            grad[0] = grad_data[0] / float(samples_batch.shape[0])
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
                data_gradient = self.gradient(samples_batch[i], bases_batch[i])
                # Accumulate amplitude RBM gradient
                grad_data[0] += data_gradient[0]

                # Accumulate phase RBM gradient
                grad_data[1] += data_gradient[1]

            grad[0] = grad_data[0] / float(samples_batch.shape[0])

            grad[1] = grad_data[1] / float(samples_batch.shape[0])

        return grad

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
        :type data: numpy.ndarray
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
        :type input_bases: numpy.ndarray
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
