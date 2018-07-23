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

#import warnings
from itertools import chain
#
#import numpy as np
#from math import sqrt
import torch
#from torch import nn
#from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm, tqdm_notebook

import qucumber.cplx as cplx
from qucumber.samplers import Sampler
from qucumber.callbacks import CallbackList
#from binary_rbm import BinaryRBM
#from positive_wavefunction import PositiveWavefunction
__all__ = [
    "QuantumReconstruction"
]

class QuantumReconstruction(Sampler):
    def __init__(self, nn_state):
        super(QuantumReconstruction, self).__init__()
        self.nn_state = nn_state 
        self.num_visible = nn_state.num_visible
        self.stop_training = False

    def compute_batch_gradients(self, k, pos_batch, neg_batch):
        """This function will compute the gradients of a batch of the training
        data (data_file) given the basis measurements (chars_file).

        :param k: Number of contrastive divergence steps in training.
        :type k: int
        :param pos_batch: Batch of the input data for the positive phase.
        :type pos_batch: torch.Tensor
        :param neg_batch: Batch of the input data for the negative phase.
        :type neg_batch: torch.Tensor

        :returns: Dictionary containing all the gradients of the parameters.
        :rtype: dict
        """
        grad = {}
        pos_batch_size = float(len(pos_batch))
        neg_batch_size = float(len(neg_batch))
        
        # Positive Phase
        grad_data = self.nn_state.gradient(pos_batch)
        
        # Negative Phase
        self.nn_state.set_visible_layer(neg_batch)
        self.nn_state.sample(k)
        grad_model =self.nn_state.gradient(self.nn_state.visible_state)
       
        for net in self.nn_state.networks:
            tmp = {}
            for par in grad_data[net].keys():
                tmp[par] = grad_data[net][par]/pos_batch_size - grad_model[net][par]/neg_batch_size
            grad[net] = tmp
        return grad

    def fit(self, data, epochs=100, pos_batch_size=100, neg_batch_size=200,
            k=1, lr=1e-2, progbar=False, callbacks=[]):
        """Execute the training of the RBM.

        :param data: The actual training data
        :type data: list(float)
        :param epochs: The number of parameter (i.e. weights and biases)
                       updates
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

        disable_progbar = (progbar is False)
        progress_bar = tqdm_notebook if progbar == "notebook" else tqdm
        callbacks = CallbackList(callbacks)

        data = torch.tensor(data, device=self.nn_state.device,
                            dtype=torch.double)
        optimizer = torch.optim.SGD([self.nn_state.rbm_am.weights,
                                     self.nn_state.rbm_am.visible_bias,
                                     self.nn_state.rbm_am.hidden_bias],
                                    lr=lr)

        callbacks.on_train_start(self)

        for ep in progress_bar(range(epochs), desc="Epochs ",
                               disable=disable_progbar):
            pos_batches = DataLoader(data, batch_size=pos_batch_size,
                                     shuffle=True)

            multiplier = int((neg_batch_size / pos_batch_size) + 0.5)
            neg_batches = [DataLoader(data, batch_size=neg_batch_size,
                                      shuffle=True)
                           for i in range(multiplier)]
            neg_batches = chain(*neg_batches)

            callbacks.on_epoch_start(self, ep)

            if self.stop_training:  # check for stop_training signal
                break

            for batch_num, (pos_batch, neg_batch) in enumerate(zip(pos_batches,
                                                               neg_batches)):
                callbacks.on_batch_start(self, ep, batch_num)

                all_grads = self.compute_batch_gradients(k, pos_batch,
                                                         neg_batch)
                optimizer.zero_grad()  # clear any cached gradients

                # assign all available gradients to the corresponding parameter
                for net in self.nn_state.networks:
                    rbm = getattr(self.nn_state, net)
                    for param in all_grads[net].keys():
                        getattr(rbm, param).grad = all_grads[net][param]

                optimizer.step()  # tell the optimizer to apply the gradients

                callbacks.on_batch_end(self, ep, batch_num)

            callbacks.on_epoch_end(self, ep)

        callbacks.on_train_end(self)

