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

from qucumber.callbacks import CallbackList
from qucumber.binary_rbm import BinaryRBM
import qucumber.utils.cplx as cplx
import qucumber.utils.unitaries as unitaries
from qucumber.callbacks import CallbackList

__all__ = [
    "ComplexWavefunction"
]

class ComplexWavefunction(object):
    
    def __init__(self, unitary_dict, num_visible,
                 num_hidden, gpu=True,
                 seed=1234):
        super(ComplexWavefunction, self).__init__()
        self.num_visible = int(num_visible)
        self.num_hidden = int(num_hidden)
        self.rbm_am = BinaryRBM(num_visible, num_hidden, gpu=gpu,
                                  seed=seed)
        self.rbm_ph = BinaryRBM(num_visible, num_hidden, gpu=gpu,
                                  seed=seed+72938)
              
        self.size_cut=20
        self.space = None
        self.Z = 0.0
 
        self.networks = ["rbm_am","rbm_ph"]
        self.device = self.rbm_am.device
        self.unitary_dict = unitary_dict 
        self.visible_state = torch.zeros(1,self.num_visible,
                                         device=self.rbm_am.device,
                                         dtype=torch.double)
        self.hidden_state = torch.zeros(1,self.num_hidden,
                                         device=self.rbm_am.device,
                                         dtype=torch.double)

    def randomize(self):
        """Randomize the parameters of the amplitude and phase RBM"""
        self.rbm_am.randomize()
        self.rbm_ph.randomize()
        
    def set_visible_layer(self,v):
        r""" Set the visible state to a given vector/matrix
        
        :param v: State to initialize the wavefunction to
        :type v: torch.Tensor
        """
        self.visible_state = v
   
    def amplitude(self,v):
        r""" Compute the amplitude of a given vector/matrix of visible states

        :param v: visible states
        :type v: torch.tensor

        :returns Matrix/vector containing the amplitudes of v
        :rtype torch.tensor
        """
        return (-self.rbm_am.effective_energy(v)).exp().sqrt()
    
    def phase(self,v):
        r""" Compute the phase of a given vector/matrix of visible states

        :param v: visible states
        :type v: torch.tensor

        :returns Matrix/vector containing the phases of v
        :rtype torch.tensor
        """
        return -0.5*self.rbm_ph.effective_energy(v)
   
    def psi(self,v):
        r""" Compute the wavefunction coefficient  of a given vector/matrix of visible states

        :param v: visible states
        :type v: torch.tensor

        :returns Complex object containing the wavefunction coefficients of v
        :rtype torch.tensor
        """
        cos_phase = (self.phase(v)).cos() 
        sin_phase = (self.phase(v)).sin() 
        psi = torch.zeros(2, dtype=torch.double, device=self.device)
        psi[0] = self.amplitude(v)*cos_phase 
        psi[1] = self.amplitude(v)*sin_phase
        return psi

    def gradient(self,basis,sample):
        r"""Compute the gradient of a set (v_state) of samples, measured
        in different bases
        
        :param basis: A set of basis, (i.e.vector of strings)
        :type basis: np.array
        """
        num_U = 0               # Number of 1-local unitary rotations
        rotated_sites = []      # List of site where the rotations are applied
        grad = []               # Gradient
        
        # Read where the unitary rotations are applied
        for j in range(self.num_visible):
            if (basis[j] != 'Z'):
                num_U += 1
                rotated_sites.append(j)

        # If the basis is the reference one ('ZZZ..Z')
        if (num_U == 0):
            grad.append(self.rbm_am.effective_energy_gradient(sample))  # Real 
            grad.append(0.0)                                            # Imaginary
        
        else:
            # Initialize
            vp = torch.zeros(self.num_visible, dtype=torch.double, device = self.device)
            rotated_grad = [torch.zeros(2,self.rbm_am.num_pars,dtype=torch.double, device = self.device),torch.zeros(2,self.rbm_ph.num_pars,dtype=torch.double, device = self.device)]
            Upsi = torch.zeros(2, dtype=torch.double, device = self.device)
            
            # Sum over the full subspace where the rotation are applied
            #sub_state = self.generate_visible_space(num_U)
            sub_space = self.generate_Hilbert_space(num_U)
            for x in range(1<<num_U):
                # Create the correct state for the full system (given the data)
                cnt = 0
                for j in range(self.num_visible):
                    if (basis[j] != 'Z'):
                        #vp[j]=sub_state[x][cnt] # This site sums (it is rotated)
                        vp[j] = sub_space[x][cnt]
                        cnt += 1
                    else:
                        vp[j]=sample[j]         # This site is left unchanged

                U = torch.tensor([1., 0.], dtype=torch.double, device = self.device) #Product of the matrix elements of the unitaries
                for ii in range(num_U):
                    tmp = self.unitary_dict[basis[rotated_sites[ii]]][:,int(sample[rotated_sites[ii]]),int(vp[rotated_sites[ii]])]
                    U = cplx.scalar_mult(U,tmp.to(self.device))
                
                # Gradient on the current configuration
                grad_vp = [self.rbm_am.effective_energy_gradient(vp),self.rbm_ph.effective_energy_gradient(vp)]
                
                # NN state rotated in this bases
                Upsi_v = cplx.scalar_mult(U,self.psi(vp))
                
                Upsi += Upsi_v            
                rotated_grad[0] += cplx.scalar_mult(Upsi_v,cplx.make_complex(grad_vp[0],torch.zeros_like(grad_vp[0])))
                rotated_grad[1] += cplx.scalar_mult(Upsi_v,cplx.make_complex(grad_vp[1],torch.zeros_like(grad_vp[1]))) 

            grad.append(cplx.scalar_divide(rotated_grad[0],Upsi)[0,:])
            grad.append(-cplx.scalar_divide(rotated_grad[1],Upsi)[1,:])

        return grad

    def sample(self, k):
        """Performs k steps of Block Gibbs sampling given an initial visible
        state v0.

        :param k: Number of Block Gibbs steps.
        :type k: int
        """
        for _ in range(k):
            self.hidden_state = self.rbm_am.sample_h_given_v(self.visible_state)
            self.visible_state = self.rbm_am.sample_v_given_h(self.hidden_state)

    def generate_Hilbert_space(self,size):
        """Generates Hilbert space of dimension 2^size.
    
        :returns: A tensor with all the basis states of the Hilbert space.
        :rtype: torch.Tensor
        """
        if (size > self.size_cut):
            raise ValueError('Size of the Hilbert space too large!')
        else: 
            space = torch.zeros((1 << size, size),
                                device=self.device, dtype=torch.double)
            for i in range(1 << size):
                d = i
                for j in range(size):
                    d, r = divmod(d, 2)
                    space[i, size - j - 1] = int(r)
            return space

    def compute_normalization(self):
        """Compute the normalization constant of the wavefunction.
    
        :param space: A rank 2 tensor of the entire visible space.
        :type space: torch.Tensor

        """
        if (self.space is None):
            raise ValueError('Missing Hilbert space')
        else:
            self.Z = self.rbm_am.compute_partition_function(self.space)

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
