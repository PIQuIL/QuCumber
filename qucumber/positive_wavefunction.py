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
        self.networks = ["rbm_am"]
        self.device = self.rbm_am.device 
        self.visible_state = torch.zeros(1,self.num_visible,
                                         device=self.rbm_am.device,
                                         dtype=torch.double)
        self.hidden_state = torch.zeros(1,self.num_hidden,
                                         device=self.rbm_am.device,
                                         dtype=torch.double)

    def randomize(self):
        self.rbm_am.randomize()

    def set_visible_layer(self,v):
        #NOTE double check this
        #self.visible_state.resize_(v.shape)
        #self.hidden_state.resize_(v.shape[0],self.num_hidden)
        self.visible_state = v
    def amplitude(self,v):
        return (-self.rbm_am.effective_energy(v)).exp().sqrt()
    
    def psi(self,v):
        psi = torch.zeros(2, dtype=torch.double)
        psi[0] = self.amplitude(v)
        psi[1] = 0.0
        return psi
        #return (-self.rbm_am.effective_energy(v)).exp().sqrt()

    def gradient(self,v):
        return {"rbm_am": self.rbm_am.effective_energy_gradient(v)} 

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

