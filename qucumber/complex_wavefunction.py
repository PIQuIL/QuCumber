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

from qucumber.samplers import Sampler
from qucumber.callbacks import CallbackList
from binary_rbm import BinaryRBM
#import qucumber.cplx as cplx
from utils import cplx
from utils import unitaries
from qucumber.samplers import Sampler
from qucumber.callbacks import CallbackList
from qucumber import unitaries

__all__ = [
    "ComplexWavefunction"
]

class ComplexWavefunction(Sampler):
    
    def __init__(self,num_visible,
                 num_hidden, gpu=True,
                 seed=1234):
        super(ComplexWavefunction, self).__init__()
        self.num_visible = int(num_visible)
        self.num_hidden = int(num_hidden)
        self.rbm_am = BinaryRBM(num_visible, num_hidden, gpu=gpu,
                                  seed=seed)
        self.rbm_ph = BinaryRBM(num_visible, num_hidden, gpu=gpu,
                                  seed=seed+72938)
        self.networks = ["rbm_am","rbm_ph"]
        self.device = self.rbm_am.device
        self.unitary_dict = unitaries.create_dict()
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
        #self.visible_state.resize_(v.shape)
        #self.hidden_state.resize_(v.shape[0],self.num_hidden)
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
        #NOTE Why
        #v_prime = v.view(-1, self.num_visible)
        cos_phase = (self.phase(v)).cos() 
        sin_phase = (self.phase(v)).sin() 
        psi = torch.zeros(2, dtype=torch.double)
        psi[0] = self.amplitude(v)*cos_phase 
        psi[1] = self.amplitude(v)*sin_phase
        return psi

    def gradient(self,basis,v_state):
        num_nontrivial_U = 0
        nontrivial_sites = []
        final_grad = []
        # Check how many local bases rotations appear in basis
        for j in range(self.num_visible):
            if (basis[j] != 'Z'):
                num_nontrivial_U += 1
                nontrivial_sites.append(j)
        if (num_nontrivial_U == 0):
            final_grad.append(self.rbm_am.effective_energy_gradient(v_state)) 
            final_grad.append(0.0)
        else:
            v = torch.zeros(self.num_visible, dtype=torch.double)
            rotated_grad = [torch.zeros(2,self.rbm_am.num_pars,dtype=torch.double),torch.zeros(2,self.rbm_ph.num_pars,dtype=torch.double)]
            #for net in self.networks:
            #    rbm = getattr(self, net)
            #    tmp = {}
            #    for par in rbm.state_dict():
            #        tmp[par] = 0
            #    rotated_grad[net] = tmp
            #    final_grad[net] = tmp
            
            Upsi = torch.zeros(2, dtype=torch.double)
            sub_state = self.generate_visible_space(num_nontrivial_U)
    
            for x in range(1<<num_nontrivial_U):
                cnt = 0
                for j in range(self.num_visible):
                    if (basis[j] != 'Z'):
                        v[j]=sub_state[x][cnt]
                        cnt += 1
                    else:
                        v[j]=v_state[j]
                U = torch.tensor([1., 0.], dtype=torch.double)
                for ii in range(num_nontrivial_U):
                    tmp = self.unitary_dict[basis[nontrivial_sites[ii]]][:,int(v_state[nontrivial_sites[ii]]),int(v[nontrivial_sites[ii]])]
                    U = cplx.scalar_mult(U,tmp)
                
                grad = [self.rbm_am.effective_energy_gradient(v),self.rbm_ph.effective_energy_gradient(v)]
                
                Upsi_v = cplx.scalar_mult(U,self.psi(v))
                
                Upsi += Upsi_v            
                rotated_grad[0] += cplx.scalar_mult(Upsi_v,cplx.make_complex(grad[0],torch.zeros_like(grad[0])))
                rotated_grad[1] += cplx.scalar_mult(Upsi_v,cplx.make_complex(grad[1],torch.zeros_like(grad[1]))) 

            final_grad.append(cplx.divide(rotated_grad[0],Upsi)[0,:])
            final_grad.append(-cplx.divide(rotated_grad[1],Upsi)[1,:])
           #     for net in self.networks:
           #         Upsi_v = cplx.scalar_mult(U,self.psi(v))
           #         tmp = cplx.make_complex_matrix(grad[net]['weights'],torch.zeros(grad[net]['weights'].shape[0],grad[net]['weights'].shape[1],dtype=torch.double))
           #         rotated_grad[net]['weights']+=cplx.MS_mult(Upsi_v,tmp)
           #         tmp = cplx.make_complex_vector(grad[net]['visible_bias'],torch.zeros(grad[net]['visible_bias'].shape[0],dtype=torch.double))
           #         rotated_grad[net]['visible_bias']+=cplx.VS_mult(Upsi_v,tmp)
           #         tmp = cplx.make_complex_vector(grad[net]['hidden_bias'],torch.zeros(grad[net]['hidden_bias'].shape[0],dtype=torch.double))
           #         rotated_grad[net]['hidden_bias']+=cplx.VS_mult(Upsi_v,tmp)

           # final_grad['rbm_am']['weights'] = cplx.MS_divide(rotated_grad['rbm_am']['weights'],Upsi)[0,:,:] 
           # final_grad['rbm_am']['visible_bias'] = cplx.VS_divide(rotated_grad['rbm_am']['visible_bias'],Upsi)[0,:]  
           # final_grad['rbm_am']['hidden_bias'] = cplx.VS_divide(rotated_grad['rbm_am']['hidden_bias'],Upsi)[0,:]  
           # final_grad['rbm_ph']['weights'] = -cplx.MS_divide(rotated_grad['rbm_ph']['weights'],Upsi)[1,:,:] 
           # final_grad['rbm_ph']['visible_bias'] = -cplx.VS_divide(rotated_grad['rbm_ph']['visible_bias'],Upsi)[1,:]  
           # final_grad['rbm_ph']['hidden_bias'] = -cplx.VS_divide(rotated_grad['rbm_ph']['hidden_bias'],Upsi)[1,:]  

        return final_grad

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


    def generate_visible_space(self,size):
        """Generates all possible visible states.

        :returns: A tensor of all possible spin configurations.
        :rtype: torch.Tensor
        """
        space = torch.zeros((1 << size, size),
                            device="cpu", dtype=torch.double)
        for i in range(1 << size):
            d = i
            for j in range(size):
                d, r = divmod(d, 2)
                space[i, size - j - 1] = int(r)
        
        return space

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

