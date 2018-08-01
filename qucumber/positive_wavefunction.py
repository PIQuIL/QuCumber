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

import warnings
from itertools import chain

import numpy as np
from math import sqrt
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm, tqdm_notebook

import utils.cplx as cplx
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
        self.rbm_am = BinaryRBM(self.num_visible, self.num_hidden,
                                            gpu=gpu, seed=seed)
        
        self.size_cut=20
        self.space = None
        self.Z = 0.0
        self.num_pars = self.rbm_am.num_pars
        self.networks = ["rbm_am"]
        self.device = self.rbm_am.device 
        self.visible_state = torch.zeros(1,self.num_visible,
                                         device=self.rbm_am.device,
                                         dtype=torch.double)
        self.hidden_state = torch.zeros(1,self.num_hidden,
                                         device=self.rbm_am.device,
                                         dtype=torch.double)

    def randomize(self):
        """Randomize the parameters of the amplitude RBM"""
        self.rbm_am.randomize()

    def set_visible_layer(self,v):
        r""" Set the visible state to a given vector/matrix
        
        :param v: State to initialize the wavefunction to
        :type v: torch.Tensor
        """
        self.visible_state = v
        if (self.visible_state.shape != v.shape):
            raise RuntimeError ('Error in set_visible_layer')
    
    def amplitude(self,v):
        r""" Compute the amplitude of a given vector/matrix of visible states

        :param v: visible states
        :type v: torch.tensor

        :returns Matrix/vector containing the amplitudes of v
        :rtype torch.tensor
        """
        return (-self.rbm_am.effective_energy(v)).exp().sqrt()
    
    def psi(self,v):
        r""" Compute the wavefunction coefficient  of a given vector/matrix of visible states

        :param v: visible states
        :type v: torch.tensor

        :returns Complex object containing the wavefunction coefficients of v
        :rtype torch.tensor
        """
        psi = torch.zeros(2, dtype=torch.double)
        psi[0] = self.amplitude(v)
        psi[1] = 0.0
        return psi

    def gradient(self,v):
        r"""Compute the gradient for a batch of visible states v

        :param v: visible states
        :type v: torch.tensor

        :returns dictionary with one key (rbm_am)
        :rtype  dictionary(dictionary(torch.tensor,torch.tensor,torch.tensor)
        """
        return self.rbm_am.effective_energy_gradient(v)

    def sample(self, k_cd):
        """Performs k steps of Block Gibbs sampling 
        

        :param k: Number of Block Gibbs steps.
        :type k: int
        """
        for _ in range(k_cd):
            self.hidden_state = self.rbm_am.sample_h_given_v(self.visible_state)
            self.visible_state = self.rbm_am.sample_v_given_h(self.hidden_state)

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
        data = {"rbm_am":self.rbm_am.state_dict(), **metadata}
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

        self.rbm_am.load_state_dict(state_dict, strict=False)

    def generate_Hilbert_space(self,size):
        """Generates all possible visible states.
    
        :returns: A tensor of all possible spin configurations.
        :rtype: torch.Tensor
        """
        if (size > self.size_cut):
            raise ValueError('Size of the Hilbert space too large!')
        else: 
            self.space = torch.zeros((1 << size, size),
                                device=self.device, dtype=torch.double)
            for i in range(1 << size):
                d = i
                for j in range(size):
                    d, r = divmod(d, 2)
                    self.space[i, size - j - 1] = int(r)
            #return space

    def compute_normalization(self):
        """Compute the normalization constant of the wavefunction.
    
        :param space: A rank 2 tensor of the entire visible space.
        :type space: torch.Tensor

        """
        if (self.space is None):
            raise ValueError('Missing Hilbert space')
        else:
            self.Z = self.rbm_am.compute_partition_function(self.space)

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

