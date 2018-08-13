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

from itertools import chain
from math import ceil
import time

import torch
from tqdm import tqdm, tqdm_notebook

from qucumber.callbacks import CallbackList
from qucumber.utils.gradients_utils import vector_to_grads


class QuantumReconstruction(object):
    def __init__(self, nn_state):
        super(QuantumReconstruction, self).__init__()
        self.nn_state = nn_state
        self.num_visible = nn_state.num_visible
        self.stop_training = False

    def compute_batch_gradients(self, k_cd, samples_batch, neg_batch, bases_batch=None):
        """This function will compute the gradients of a batch of the training
        data (samples_batch). If measurements are taken in bases other than the
        reference basis, a list of bases (bases_batch) must also be provided.

        :param k_cd: Number of contrastive divergence steps in training.
        :type k_cd: int
        :param samples_batch: Batch of the input samples.
        :type samples_batch: torch.Tensor
        :param neg_batch: Batch of the input samples for computing the
                          negative phase.
        :type neg_batch: torch.Tensor
        :param bases_batch: Batch of the input bases
        :type bases_batch: np.array

        :returns: List containing the gradients of the parameters.
        :rtype: list
        """
        # Negative phase: learning signal driven by the amplitude RBM of
        # the NN state
        vk = self.nn_state.rbm_am.gibbs_steps(k_cd, neg_batch)
        grad_model = self.nn_state.rbm_am.effective_energy_gradient(vk)

        # If measurements are taken in the reference bases only
        if bases_batch is None:
            grad = [0.0]
            # Positive phase: learning signal driven by the data (and bases)
            grad_data = self.nn_state.gradient(samples_batch)
            # Gradient = Positive Phase - Negative Phase
            grad[0] = grad_data / float(samples_batch.shape[0])
            grad[0] -= grad_model / float(neg_batch.shape[0])
        else:
            grad = [0.0, 0.0]

            # Initialize
            grad_data = [
                torch.zeros(
                    getattr(self.nn_state, net).num_pars,
                    dtype=torch.double,
                    device=self.nn_state.device,
                )
                for net in self.nn_state.networks
            ]

            # Loop over each sample in the batch
            for i in range(samples_batch.shape[0]):
                # Positive phase: learning signal driven by the data
                #                 (and bases)
                data_gradient = self.nn_state.gradient(bases_batch[i], samples_batch[i])
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

    def fit(
        self,
        input_samples,
        epochs,
        pos_batch_size,
        neg_batch_size,
        k_cd,
        lr,
        input_bases=None,
        z_samples=None,
        progbar=False,
        callbacks=None,
    ):
        """Execute the training of the RBM.

        :param input_samples: The training samples
        :type input_samples: np.array
        :param epochs:
        :type epochs: int
        :param pos_batch_size: The size of batches for the positive phase
                               taken from the data.
        :type pos_batch_size: int
        :param neg_batch_size: The size of batches for the negative phase
                               taken from the data
        :type neg_batch_size: int
        :param k: The number of contrastive divergence steps
        :type k: int
        :param lr: Learning rate
        :type lr: float
        :param progbar: Whether or not to display a progress bar. If "notebook"
                        is passed, will use a Jupyter notebook compatible
                        progress bar.
        :type progbar: bool or str
        :param callbacks: Callbacks to run while training.
        :type callbacks: list(qucumber.callbacks.Callback)
        """
        disable_progbar = progbar is False
        progress_bar = tqdm_notebook if progbar == "notebook" else tqdm
        callbacks = CallbackList(callbacks if callbacks else [])

        train_samples = torch.tensor(
            input_samples, device=self.nn_state.device, dtype=torch.double
        )

        if len(self.nn_state.networks) > 1:
            optimizer = torch.optim.SGD(
                list(
                    chain(
                        *[
                            getattr(self.nn_state, net).parameters()
                            for net in self.nn_state.networks
                        ]
                    )
                ),
                lr=lr,
            )
        else:
            optimizer = torch.optim.SGD(self.nn_state.rbm_am.parameters(), lr=lr)
            batch_bases = None

        callbacks.on_train_start(self.nn_state)

        t0 = time.time()
        batch_num = ceil(train_samples.shape[0] / pos_batch_size)
        for ep in progress_bar(range(epochs), desc="Epochs ", disable=disable_progbar):
            if self.stop_training:  # check for stop_training signal
                break

            random_permutation = torch.randperm(train_samples.shape[0])
            shuffled_samples = train_samples[random_permutation]

            # List of all the batches for positive phase.
            pos_batches = [
                shuffled_samples[batch_start : (batch_start + pos_batch_size)]
                for batch_start in range(0, len(train_samples), pos_batch_size)
            ]

            if input_bases is not None:
                shuffled_bases = input_bases[random_permutation]
                pos_batches_bases = [
                    shuffled_bases[batch_start : (batch_start + pos_batch_size)]
                    for batch_start in range(0, len(train_samples), pos_batch_size)
                ]

            callbacks.on_epoch_start(self.nn_state, ep)

            for b in range(batch_num):
                callbacks.on_batch_start(self.nn_state, ep, b)

                if input_bases is None:
                    random_permutation = torch.randperm(train_samples.shape[0])
                    neg_batch = train_samples[random_permutation]
                    neg_batch = neg_batch[0:neg_batch_size]
                else:
                    z_samples = z_samples.to(self.nn_state.device)
                    random_permutation = torch.randperm(z_samples.shape[0])
                    neg_batch = z_samples[random_permutation][0:neg_batch_size]
                    batch_bases = pos_batches_bases[b]

                all_grads = self.compute_batch_gradients(
                    k_cd, pos_batches[b], neg_batch, batch_bases
                )

                optimizer.zero_grad()  # clear any cached gradients

                for p, net in enumerate(self.nn_state.networks):
                    rbm = getattr(self.nn_state, net)
                    vector_to_grads(all_grads[p], rbm.parameters())

                optimizer.step()  # tell the optimizer to apply the gradients

                callbacks.on_batch_end(self.nn_state, ep, b)

            callbacks.on_epoch_end(self.nn_state, ep)

        callbacks.on_train_end(self.nn_state)
        t1 = time.time()
        print("\nElapsed time = %.2f" % (t1 - t0))
