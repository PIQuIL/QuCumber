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

import qucumber.cplx as cplx
from qucumber.samplers import Sampler
from qucumber.callbacks import CallbackList
from binary_rbm import BinaryRBM
import qucumber.cplx as cplx
#from qucumber.samplers import Sampler
#from qucumber.callbacks import CallbackList

__all__ = [
    "ComplexWavefunction"
]

class ComplexWavefunction(Sampler):
    # NOTE: In development. Might be unstable.
    # NOTE: The 'full_unitaries' argument is not needed for training/sampling.
    # This is only here for debugging the gradients. Delete this argument when
    # gradients have been debugged.
    def __init__(self, full_unitaries, psi_dictionary, num_visible,
                 num_hidden, gpu=True,
                 seed=1234):
        super(ComplexWavefunction, self).__init__()
        self.num_visible = int(num_visible)
        self.num_hidden = int(num_hidden)
        #self.full_unitaries = full_unitaries
        #self.psi_dictionary = psi_dictionary
        self.rbm_am = BinaryRBM(num_visible, num_hidden, gpu=gpu,
                                  seed=seed)
        self.rbm_ph = BinaryRBM(num_visible, num_hidden, gpu=gpu,
                                  seed=seed+72938)
        self.networks = ["rbm_am","rbm_ph"]
        self.device = self.rbm_am.device

        self.visible_state = torch.zeros(1,self.num_visible,
                                         device=self.rbm_am.device,
                                         dtype=torch.double)
        self.hidden_state = torch.zeros(1,self.num_hidden,
                                         device=self.rbm_am.device,
                                         dtype=torch.double)
    def set_visible_layer(self,v):
        #self.visible_state.resize_(v.shape)
        #self.hidden_state.resize_(v.shape[0],self.num_hidden)
        self.visible_state = v
   
    def amplitude(self,v):
        return (-self.rbm_am.effective_energy(v)).exp().sqrt()
    
    def phase(self,v):
        return -self.rbm_ph.effective_energy(v)
   
    def gradient(self,v):
        return {'rbm_am': self.rbm_am.effective_energy_gradient(v),'rbm_ph': self.rbm_am.effective_energy_gradient(v)}

    def psi(self,v):
        #v_prime = v.view(-1, self.num_visible)
        #temp1 = (self.unnormalized_probability_amp(v_prime)).sqrt()
        cos_phase = (0.5*self.phase(v)).cos() 
        sin_phase = (0.5*self.phase(v)).sin() 
        #temp2 = ((self.unnormalized_probability_phase(v_prime)).log())*0.5
        psi = torch.zeros(2, dtype=torch.double)
        psi[0] = self.amplitude(v)*cos_phase 
        psi[1] = self.amplitude(v)*sin_phase
        return psi

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
        data = {"rbm_am":self.rbm_am.state_dict(),"rbm_ph":self.rbm_ph.state_dict(), **metadata}
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

        self.rbm_am.load_state_dict(state_dict['rbm_am'], strict=False)
        self.rbm_ph.load_state_dict(state_dict['rbm_ph'], strict=False)

