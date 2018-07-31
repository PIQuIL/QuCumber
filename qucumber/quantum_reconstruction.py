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
from torch.nn.utils import parameters_to_vector
from tqdm import tqdm, tqdm_notebook
import time
import utils.cplx as cplx
from qucumber.samplers import Sampler
from qucumber.callbacks import CallbackList
from positive_wavefunction import PositiveWavefunction
from complex_wavefunction import ComplexWavefunction
import math as m
__all__ = [
    "QuantumReconstruction"
]

class QuantumReconstruction(Sampler):
    def __init__(self, nn_state):
        super(QuantumReconstruction, self).__init__()
        self.nn_state = nn_state 
        self.num_visible = nn_state.num_visible
        self.stop_training = False
#        self.unitary_dict = {}
        self.grad = {}

    def reset_gradient(self):
        for net in self.nn_state.networks:
            tmp = {}
            rbm = getattr(self.nn_state, net)
            for par in rbm.state_dict():
                tmp[par]=0.0    
            self.grad[net] = tmp
            #grad_data[net] = tmp

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
        #grad = torch.zeros(nn_state.num_pars)
        #grad_data = {}
        #for net in self.nn_state.networks:
        #    tmp = {}
        #    rbm = getattr(self.nn_state, net)
        #    for par in rbm.state_dict():
        #        tmp[par]=0.0    
        #    grad[net] = tmp
        #    grad_data[net] = tmp
        self.nn_state.sample(k)
        grad_model = self.nn_state.rbm_am.effective_energy_gradient(self.nn_state.visible_state)

        if bases_batch is None:
            grad = [0.0]
            #grad_data = torch.zeros(self.nn_state.rbm_am.num_pars,dtype=torch.double)
            #for i in range(samples_batch.shape[0]):
            #    grad_data += self.nn_state.gradient(samples_batch[i])
            grad_data = self.nn_state.gradient(samples_batch)
            grad[0] = grad_data/float(samples_batch.shape[0]) - grad_model/float(self.nn_state.visible_state.shape[0])
        else:
            grad=[0.0,0.0]
            # Positive Phase
            #TODO THIS LOOP IS THE MAIN BOTTLENECK
            grad_data = [torch.zeros(self.nn_state.rbm_am.num_pars,dtype=torch.double),torch.zeros(self.nn_state.rbm_ph.num_pars,dtype=torch.double)]
            for i in range(samples_batch.shape[0]):
                data_gradient = self.nn_state.gradient(bases_batch[i],samples_batch[i])
                grad_data[0] += data_gradient[0]#/float(samples_batch.shape[0])
                grad_data[1] += data_gradient[1]#/float(samples_batch.shape[1])
                #for net in self.nn_state.networks:
                #    for par in getattr(self.nn_state, net).state_dict():
                #        grad_data[net][par] += data_gradient[net][par]
            
            grad[0] = grad_data[0]/float(samples_batch.shape[0]) - grad_model/float(self.nn_state.visible_state.shape[0])
            grad[1] = grad_data[1]/float(samples_batch.shape[0])
        #for net in self.nn_state.networks:
        #    for par in grad_data[net].keys():
        #        grad[net][par] = grad_data[net][par]/float(samples_batch.shape[0])

        
        #grad[0] = grad_data[0]/float(samples_batch.shape[0]) - grad_model/float(self.nn_state.visible_state.shape[0])
        #grad = grad_data/float(samples_batch.shape[0]) - grad_model/float(self.nn_state.visible_state.shape[0]) 
        
        #grad_model = {'rbm_am': self.nn_state.rbm_am.effective_energy_gradient(self.nn_state.visible_state)}
        #for par in grad_data['rbm_am'].keys():
        #    grad['rbm_am'][par] -= grad_model['rbm_am'][par]/float(self.nn_state.visible_state.shape[0])
        return grad
        
    def fit(self,input_samples,epochs,pos_batch_size, neg_batch_size,k,lr,
            observer=None,
            input_bases = None,z_samples = None,
            progbar=False,callbacks=[]):
            
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

        train_samples = torch.tensor(input_samples, device=self.nn_state.device,
                            dtype=torch.double)
        
        #TODO How to shuffle two datasets simultaneously using the same permutation in Torch
        #     Until then, we do not shuffle the dataset if there are bases

        #if(len(self.nn_state.networks) >1):
        #    shf = False
        #else:
        #    shf = True
        #    pos_batch_bases = None
       
        #TODO make this iterative
        if (len(self.nn_state.networks) >1):
            optimizer = torch.optim.SGD(list(self.nn_state.rbm_am.parameters())+list(self.nn_state.rbm_ph.parameters()),lr=lr)
            #optimizer = torch.optim.SGD([self.nn_state.rbm_am.weights,
            #                             self.nn_state.rbm_am.visible_bias,
            #                             self.nn_state.rbm_am.hidden_bias,
            #                             self.nn_state.rbm_ph.weights,
            #                             self.nn_state.rbm_ph.visible_bias,
            #                             self.nn_state.rbm_ph.hidden_bias],
            #                             lr=lr)

        else:
#            parameters = parameters_to_vector(self.nn_state.rbm_am.parameters())
            optimizer = torch.optim.SGD(self.nn_state.rbm_am.parameters(),lr=lr)
            #optimizer = torch.optim.SGD([self.nn_state.rbm_am.weights,
            #                             self.nn_state.rbm_am.visible_bias,
            #                             self.nn_state.rbm_am.hidden_bias],lr=lr)
            batch_bases = None
        callbacks.on_train_start(self)
        t0 = time.time()
        batch_num = m.ceil(train_samples.shape[0] / pos_batch_size)
        for ep in progress_bar(range(1,epochs+1), desc="Epochs ",
                               disable=disable_progbar):

            random_permutation      = torch.randperm(train_samples.shape[0])
            shuffled_samples        = train_samples[random_permutation]
            # List of all the batches for positive phase.
            pos_batches = [shuffled_samples[batch_start:(batch_start + pos_batch_size)]
                           for batch_start in range(0, len(train_samples), pos_batch_size)]

            if input_bases is not None:
                shuffled_bases = input_bases[random_permutation]
                pos_batches_bases = [shuffled_bases[batch_start:(batch_start + pos_batch_size)]
                           for batch_start in range(0, len(train_samples), pos_batch_size)]
             
            
            # DATALOADER
            #pos_batches = DataLoader(train_samples, batch_size=pos_batch_size,
            #                         shuffle=shf)
            #multiplier = int((neg_batch_size / pos_batch_size) + 0.5)
            # 
            #neg_batches = [DataLoader(train_samples, batch_size=neg_batch_size,
            #                          shuffle=True)
            #               for i in range(multiplier)]
            #neg_batches = chain(*neg_batches)
            callbacks.on_epoch_start(self, ep)

            if self.stop_training:  # check for stop_training signal
                break
            
            #for batch_num, (pos_batch,neg_batch) in enumerate(zip(pos_batches,
            #                                                   neg_batches)):
            for b in range(batch_num):
                callbacks.on_batch_start(self, ep, b)
                #if input_bases is not None:
                #    pos_batch_bases = input_bases[batch_num*pos_batch_size:(batch_num+1)*;pos_batch_size]

                if input_bases is None:
                    random_permutation      = torch.randperm(train_samples.shape[0])
                    neg_batch        = train_samples[random_permutation][0:neg_batch_size]
                
                #neg_batches = [shuffled_samples[batch_start:(batch_start + neg_batch_size)]
                #           for batch_start in range(0, len(train_samples), neg_batch_size)]

                #if input_bases is not None:
                else:
                    random_permutation      = torch.randperm(z_samples.shape[0])
                    neg_batch = z_samples[random_permutation][0:neg_batch_size]
                    batch_bases = pos_batches_bases[b]
                self.nn_state.set_visible_layer(neg_batch)
                all_grads = self.compute_batch_gradients(k, pos_batches[b],batch_bases)
                optimizer.zero_grad()  # clear any cached gradients
                
                
                for p,net in enumerate(self.nn_state.networks):
                    rbm = getattr(self.nn_state,net)
                    vector_to_grads(all_grads[p],rbm.parameters())
                #vector_to_grads(all_grads[0],self.nn_state.rbm_am.parameters())
                #vector_to_grads(all_grads[1],self.nn_state.rbm_ph.parameters())
                # assign all available gradients to the corresponding parameter
                #for net in self.nn_state.networks:
                
                #for net in self.nn_state.networks:
                #    rbm = getattr(self.nn_state, net)
                #    for param in all_grads[net].keys():
                #        getattr(rbm, param).grad = all_grads[net][param]
                optimizer.step()  # tell the optimizer to apply the gradients

                callbacks.on_batch_end(self, ep, b)
            if observer is not None:
                if ((ep % observer.frequency) == 0): 
                    #if target_psi is not None:
                    stat = observer.scan(ep,self.nn_state)

            callbacks.on_epoch_end(self, ep)
        #F = self.fidelity(target_psi,vis)
        #print(F.item())
        #callbacks.on_train_end(self)
        t1 = time.time()
        print("\nElapsed time = %.2f" %(t1-t0)) 


def vector_to_grads(vec, parameters):
    r"""Convert one vector to the parameters

    Arguments:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    """
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError('expected torch.Tensor, but got: {}'
                        .format(torch.typename(vec)))
    # Flag for the device where the parameter is located
    param_device = None

    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        # The length of the parameter
        num_param = torch.prod(torch.LongTensor(list(param.size())))
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.grad = vec[pointer:pointer + num_param].view(param.size()).data

        # Increment the pointer
        pointer += num_param


def _check_param_device(param, old_param_device):
    r"""This helper function is to check if the parameters are located
    in the same device. Currently, the conversion between model parameters
    and single vector form is not supported for multiple allocations,
    e.g. parameters in different GPUs, or mixture of CPU/GPU.

    Arguments:
        param ([Tensor]): a Tensor of a parameter of a model
        old_param_device (int): the device where the first parameter of a
                                model is allocated.

    Returns:
        old_param_device (int): report device for the first time
    """

    # Meet the first parameter
    if old_param_device is None:
        old_param_device = param.get_device() if param.is_cuda else -1
    else:
        warn = False
        if param.is_cuda:  # Check if in same GPU
            warn = (param.get_device() != old_param_device)
        else:  # Check if in CPU
            warn = (old_param_device != -1)
        if warn:
            raise TypeError('Found two parameters on different devices, '
                            'this is currently not supported.')
    return old_param_device


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

