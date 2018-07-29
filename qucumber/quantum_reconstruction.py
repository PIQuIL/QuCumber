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

import numpy as np
from math import sqrt
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm, tqdm_notebook
import time
import utils.cplx as cplx
from qucumber.samplers import Sampler
from qucumber.callbacks import CallbackList
from positive_wavefunction import PositiveWavefunction
from complex_wavefunction import ComplexWavefunction

__all__ = [
    "QuantumReconstruction"
]

class QuantumReconstruction(Sampler):
    def __init__(self, nn_state):
        super(QuantumReconstruction, self).__init__()
        self.nn_state = nn_state 
        self.num_visible = nn_state.num_visible
        self.stop_training = False
        unitary_dict = {}

    def compute_batch_gradients(self, k, samples_batch,bases_batch=None):
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
        grad_data = {}
        for net in self.nn_state.networks:
            tmp = {}
            rbm = getattr(self.nn_state, net)
            for par in rbm.state_dict():
                tmp[par]=0.0    
            grad[net] = tmp
            grad_data[net] = tmp
        
        if bases_batch is None:
            grad_data = self.nn_state.gradient(samples_batch)
        else:
            # Positive Phase
            for i in range(samples_batch.shape[0]):
            #for i in range(samples_batch['samples'].shape[0])
                b_flag = 0
                for j in range(self.nn_state.num_visible):
                    if (bases_batch[i][j] != 'Z'):
                        b_flag = 1
                if (b_flag == 0):
                    for par in getattr(self.nn_state, 'rbm_am').state_dict():
                        grad_data['rbm_am'][par] += self.nn_state.gradient(samples_batch[i])['rbm_am'][par]
                else:
                    rotated_grad = self.nn_state.rotate_grad(bases_batch[i],samples_batch[i])
                    for net in self.nn_state.networks:
                        for par in getattr(self.nn_state, net).state_dict():
                            grad_data[net][par] += rotated_grad[net][par]
            
        for net in self.nn_state.networks:
            for par in grad_data[net].keys():
                grad[net][par] = grad_data[net][par]/float(samples_batch.shape[0])# - grad_model[net][par]/float(self.nn_state.visible_state.shape[0])

        #vis = self.generate_visible_space()
        #Z = self.partition(vis)
        #for i in range(len(vis)):
        #    for par in grad_data[net].keys():
        #        grad['rbm_am'][par] -= ((self.nn_state.amplitude(vis[i])**2)/Z)*self.nn_state.gradient(vis[i])['rbm_am'][par] 

        self.nn_state.sample(k)
        grad_model = self.nn_state.gradient(self.nn_state.visible_state)
        for par in grad_data['rbm_am'].keys():
            grad['rbm_am'][par] -= grad_model['rbm_am'][par]/float(self.nn_state.visible_state.shape[0])
        return grad
        
    def fit(self, train_samples,epochs, pos_batch_size, neg_batch_size,
            k, lr,train_bases = None, progbar=False,callbacks=[],observer=None):
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

        data_samples = torch.tensor(train_samples, device=self.nn_state.device,
                            dtype=torch.double)
        #par_list = []
        #for net in self.nn_state.networks:
        #    rbm = getattr(self.nn_state, net) 
        #    for par in rbm.state_dict():
        #        par_list.append(getattr(rbm, par))
        #        #print(getattr(rbm, par))
        #print(par_list)
        #check_list = [self.nn_state.rbm_am.weights,
        #              self.nn_state.rbm_am.visible_bias,
        #              self.nn_state.rbm_am.hidden_bias]
        #print(check_list)
        ##optimizer = torch.optim.SGD(par_list,lr=lr)
        
        if (len(self.nn_state.networks) >1):
            optimizer = torch.optim.SGD([self.nn_state.rbm_am.weights,
                                         self.nn_state.rbm_am.visible_bias,
                                         self.nn_state.rbm_am.hidden_bias,
                                         self.nn_state.rbm_ph.weights,
                                         self.nn_state.rbm_ph.visible_bias,
                                         self.nn_state.rbm_ph.hidden_bias],
                                         lr=lr)

        else:
            optimizer = torch.optim.SGD([self.nn_state.rbm_am.weights,
                                         self.nn_state.rbm_am.visible_bias,
                                         self.nn_state.rbm_am.hidden_bias],lr=lr)

        callbacks.on_train_start(self)
        
        #t0 = time.time()

        for ep in progress_bar(range(1,epochs+1), desc="Epochs ",
                               disable=disable_progbar):
            pos_batches = DataLoader(data_samples, batch_size=pos_batch_size,
                                     shuffle=True)
            multiplier = int((neg_batch_size / pos_batch_size) + 0.5)
            neg_batches = [DataLoader(data_samples, batch_size=neg_batch_size,
                                      shuffle=True)
                           for i in range(multiplier)]
            neg_batches = chain(*neg_batches)

            callbacks.on_epoch_start(self, ep)

            if self.stop_training:  # check for stop_training signal
                break
            
            # FULL GRADIENT
            #self.nn_state.set_visible_layer(neg_batches)
            #self.nn_state.set_visible_layer(train_samples[0:100])
            #all_grads = self.compute_batch_gradients(k, data_samples,train_bases)
            #optimizer.zero_grad()  # clear any cached gradients
            ###assign all available gradients to the corresponding parameter
            #for net in self.nn_state.networks:
            #    rbm = getattr(self.nn_state, net)
            #    for param in all_grads[net].keys():
            #        getattr(rbm, param).grad = all_grads[net][param]

            #optimizer.step()  # tell the optimizer to apply the gradients
            
            for batch_num, (pos_batch, neg_batch) in enumerate(zip(pos_batches,
                                                               neg_batches)):
                callbacks.on_batch_start(self, ep, batch_num)

                self.nn_state.set_visible_layer(neg_batch)
                all_grads = self.compute_batch_gradients(k, pos_batch)
                optimizer.zero_grad()  # clear any cached gradients
                # assign all available gradients to the corresponding parameter
                for net in self.nn_state.networks:
                    rbm = getattr(self.nn_state, net)
                    for param in all_grads[net].keys():
                        getattr(rbm, param).grad = all_grads[net][param]
                optimizer.step()  # tell the optimizer to apply the gradients

                callbacks.on_batch_end(self, ep, batch_num)
            if ((ep % observer.frequency) == 0): 
                #if target_psi is not None:
                stat = observer.scan(ep,self.nn_state)

            callbacks.on_epoch_end(self, ep)
        #F = self.fidelity(target_psi,vis)
        #print(F.item())
        #callbacks.on_train_end(self)
        #t1 = time.time()
        #print("\nElapsed time = %.2f" %(t1-t0)) 

