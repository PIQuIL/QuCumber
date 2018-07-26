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

#import qucumber.cplx as cplx
import utils.cplx as cplx
from qucumber.samplers import Sampler
from qucumber.callbacks import CallbackList
from positive_wavefunction import PositiveWavefunction
from complex_wavefunction import ComplexWavefunction

__all__ = [
    "QuantumReconstruction"
]

class QuantumReconstruction:
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
            
        self.nn_state.sample(k)
        grad_model = self.nn_state.gradient(self.nn_state.visible_state)
        for net in self.nn_state.networks:
            for par in grad_data[net].keys():
                grad[net][par] = grad_data[net][par]/float(samples_batch.shape[0])# - grad_model[net][par]/float(self.nn_state.visible_state.shape[0])
        for par in grad_data['rbm_am'].keys():
            grad['rbm_am'][par] -= grad_model['rbm_am'][par]/float(self.nn_state.visible_state.shape[0])
        return grad
        
    def fit(self, train_samples,epochs, pos_batch_size, neg_batch_size,
            k, lr,train_bases = None, progbar=False, target_psi=None,callbacks=[]):
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
        par_list = []
        #for net in self.nn_state.networks:
        #    rbm = getattr(self.nn_state, net) 
        #    for par in rbm.state_dict():
        #        par_list.append(getattr(rbm, par))
        #        #print(getattr(rbm, par))
        
        #optimizer = torch.optim.SGD(par_list,lr=lr)
        optimizer = torch.optim.SGD([self.nn_state.rbm_am.weights,
                                     self.nn_state.rbm_am.visible_bias,
                                     self.nn_state.rbm_am.hidden_bias,#],lr=lr)
                                     self.nn_state.rbm_ph.weights,
                                     self.nn_state.rbm_ph.visible_bias,
                                     self.nn_state.rbm_ph.hidden_bias],
                                    lr=lr)

        callbacks.on_train_start(self)
        
        vis = self.generate_visible_space()

        for ep in progress_bar(range(epochs), desc="Epochs ",
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
            self.nn_state.set_visible_layer(train_samples[0:100])
            all_grads = self.compute_batch_gradients(k, train_samples,train_bases)
            optimizer.zero_grad()  # clear any cached gradients
            ##assign all available gradients to the corresponding parameter
            for net in self.nn_state.networks:
                #print(net)
                rbm = getattr(self.nn_state, net)
                #print(rbm)
                for param in all_grads[net].keys():
                    #print(param)
                    #print(all_grads[net][param])
                    getattr(rbm, param).grad = all_grads[net][param]

            optimizer.step()  # tell the optimizer to apply the gradients
            if target_psi is not None:
                F = self.fidelity(target_psi,vis)
                print(F.item())
            #for batch_num, (pos_batch, neg_batch) in enumerate(zip(pos_batches,
            #                                                   neg_batches)):
            #    callbacks.on_batch_start(self, ep, batch_num)

            #    self.nn_state.set_visible_layer(neg_batch)
            #    all_grads = self.compute_batch_gradients(k, pos_batch)
            #    optimizer.zero_grad()  # clear any cached gradients

            #    # assign all available gradients to the corresponding parameter
            #    for net in self.nn_state.networks:
            #        rbm = getattr(self.nn_state, net)
            #        for param in all_grads[net].keys():
            #            getattr(rbm, param).grad = all_grads[net][param]

            #    optimizer.step()  # tell the optimizer to apply the gradients

            #    callbacks.on_batch_end(self, ep, batch_num)

            callbacks.on_epoch_end(self, ep)

        callbacks.on_train_end(self)
    
    def generate_visible_space(self):
        """Generates all possible visible states.
    
        :returns: A tensor of all possible spin configurations.
        :rtype: torch.Tensor
        """
        space = torch.zeros((1 << self.num_visible, self.num_visible),
                            device="cpu", dtype=torch.double)
        for i in range(1 << self.num_visible):
            d = i
            for j in range(self.num_visible):
                d, r = divmod(d, 2)
                space[i, self.num_visible - j - 1] = int(r)
    
        return space

    def partition(self,visible_space):
        """The natural logarithm of the partition function of the RBM.
    
        :param visible_space: A rank 2 tensor of the entire visible space.
        :type visible_space: torch.Tensor
    
        :returns: The natural log of the partition function.
        :rtype: torch.Tensor
        """
        free_energies = -self.nn_state.rbm_am.effective_energy(visible_space)
        max_free_energy = free_energies.max()
    
        f_reduced = free_energies - max_free_energy
        logZ = max_free_energy + f_reduced.exp().sum().log()
        return logZ.exp()
        
        #return logZ
    def fidelity(self,target_psi,vis):
        F = torch.tensor([0., 0.], dtype=torch.double) 
        Z = self.partition(vis)
        #print(psi)
        for i in range(len(vis)):
            psi = self.nn_state.psi(vis[i])/Z.sqrt()
            #print(target_psi)
            F[0] += target_psi[0,i]*psi[0]+target_psi[1,i]*psi[1]
            F[1] += target_psi[0,i]*psi[1]-target_psi[1,i]*psi[0]
            #F += cplx.scalar_mult(target_psi[:,i],psi[:])/Z.sqrt()
        return cplx.norm(F)




