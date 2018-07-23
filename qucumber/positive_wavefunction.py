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
#from itertools import chain
#
#import numpy as np
#from math import sqrt
import torch
#from torch import nn
#from torch.nn import functional as F
#from torch.utils.data import DataLoader
#from tqdm import tqdm, tqdm_notebook

import qucumber.cplx as cplx
from qucumber.samplers import Sampler
from qucumber.callbacks import CallbackList
from binary_rbm import BinaryRBM

__all__ = [
    "PositiveWavefunction"
]

class PositiveWavefunction(Sampler):
    def __init__(self, num_visible, num_hidden=None, gpu=True, seed=None):
        super(PositiveWavefunction, self).__init__()
        self.num_visible = int(num_visible)
        self.num_hidden = (int(num_hidden)
                           if num_hidden is not None
                           else self.num_visible)
        self.rbm = BinaryRBM(self.num_visible, self.num_hidden,
                                            gpu=gpu, seed=seed)
        self.networks = ["rbm_am"]
        self.stop_training = False
        self.num_pars = self.rbm.num_pars
        
        self.visible_state = torch.zeros(1,self.num_visible,
                                         device=self.rbm.device,
                                         dtype=torch.double)
        self.hidden_state = torch.zeros(1,self.num_hidden,
                                         device=self.rbm.device,
                                         dtype=torch.double)


    def set_visible_layer(self,v):
        self.visible_state.resize_(v.shape)
        self.hidden_state.resize_(v.shape[0],self.num_hidden)
        self.visible_state = v

    def psi(self,v):
        return (-self.rbm.effective_energy(v)).exp().sqrt()

    def gradient(self,v):
        return {"rbm_am": self.rbm.effective_energy_gradient(v)} 

    def sample(self, k):
        """Performs k steps of Block Gibbs sampling given an initial visible
        state v0.

        :param k: Number of Block Gibbs steps.
        :type k: int
        :param v0: The initial visible state.
        :type v0: torch.Tensor

        :returns: Tuple containing the initial visible state, v0,
                  the hidden state sampled from v0,
                  the visible state sampled after k steps,
                  the hidden state sampled after k steps and its corresponding

                  probability vector.
        :rtype: tuple(torch.Tensor, torch.Tensor,
                      torch.Tensor, torch.Tensor,
                      torch.Tensor)
        """
        for _ in range(k):
            self.hidden_state = self.rbm.sample_h_given_v(self.visible_state)
            self.visible_state = self.rbm.sample_v_given_h(self.hidden_state)

    def save(self, location, metadata={}):
        """Saves the RBM parameters to the given location along with
        any given metadata.

        :param location: The location to save the RBM parameters + metadata
        :type location: str or file
        :param metadata: Any extra metadata to store alongside the RBM
                         parameters
        :type metadata: dict
        """
        # add extra metadata to dictionary before saving it to disk
        data = {**self.rbm.state_dict(), **metadata}
        torch.save(data, location)

    def load(self, location):
        """Loads the RBM parameters from the given location ignoring any
        metadata stored in the file. Overwrites the RBM's parameters.

        .. note::
            The RBM object on which this function is called must
            have the same shape as the one who's parameters are being
            loaded.

        :param location: The location to load the RBM parameters from
        :type location: str or file
        """

        try:
            state_dict = torch.load(location)
        except AssertionError as e:
            state_dict = torch.load(location, lambda storage, loc: 'cpu')

        self.rbm.load_state_dict(state_dict, strict=False)

    #@staticmethod
    #def autoload(location, gpu=False):
    #    """Initializes an RBM from the parameters in the given location,
    #    ignoring any metadata stored in the file.

    #    :param location: The location to load the RBM parameters from
    #    :type location: str or file

    #    :returns: A new RBM initialized from the given parameters
    #    :rtype: BinomialRBM
    #    """
    #    '''
    #    _warn_on_missing_gpu(gpu)
    #    gpu = gpu and torch.cuda.is_available()

    #    if gpu:
    #        state_dict = torch.load(location, lambda storage, loc: 'cuda')
    #    else:
    #        state_dict = torch.load(location, lambda storage, loc: 'cpu')
    #    '''
    #    state_dict = torch.load(location)
    #    
    #    rbm = BinomialRBM(num_visible=len(state_dict['visible_bias']),
    #                      num_hidden=len(state_dict['hidden_bias']),
    #                      gpu=gpu,
    #                      seed=None)
    #    rbm.rbm.load_state_dict(state_dict, strict=False)

    #    return rbm

#    def compute_batch_gradients(self, k, pos_batch, neg_batch):
#        """This function will compute the gradients of a batch of the training
#        data (data_file) given the basis measurements (chars_file).
#
#        :param k: Number of contrastive divergence steps in training.
#        :type k: int
#        :param pos_batch: Batch of the input data for the positive phase.
#        :type pos_batch: torch.Tensor
#        :param neg_batch: Batch of the input data for the negative phase.
#        :type neg_batch: torch.Tensor
#
#        :returns: Dictionary containing all the gradients of the parameters.
#        :rtype: dict
#        """
#        grad = {}
#        pos_batch_size = float(len(pos_batch))
#        neg_batch_size = float(len(neg_batch))
#        
#        # Positive Phase
#        grad_data = self.gradient(pos_batch)
#        
#        # Negative Phase
#        self.set_visible_layer(neg_batch)
#        self.sample(k)
#        grad_model =self.gradient(self.visible_state)
#       
#        for net in self.networks:
#            tmp = {}
#            for par in grad_data[net].keys():
#                tmp[par] = grad_data[net][par]/pos_batch_size - grad_model[net][par]/neg_batch_size
#            grad[net] = tmp
#        
#        # Return negative gradients to match up nicely with the usual
#        # parameter update rules, which *subtract* the gradient from
#        # the parameters. This is in contrast with the RBM update
#        # rules which ADD the gradients (scaled by the learning rate)
#        # to the parameters.
#        return grad
#
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

        data = torch.tensor(data, device=self.rbm.device,
                            dtype=torch.double)
        optimizer = torch.optim.SGD([self.rbm.weights,
                                     self.rbm.visible_bias,
                                     self.rbm.hidden_bias],
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
                for name, grads in all_grads.items():
                    selected_RBM = getattr(self, name)
                    for param in grads.keys():
                        getattr(selected_RBM, param).grad = grads[param]

                optimizer.step()  # tell the optimizer to apply the gradients

                callbacks.on_batch_end(self, ep, batch_num)

            callbacks.on_epoch_end(self, ep)

        callbacks.on_train_end(self)


