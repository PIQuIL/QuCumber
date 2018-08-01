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

import torch
import numpy as np
import cplx
import unitaries

def fidelity(nn_state,target_psi):
    nn_state.compute_normalization() 
    F = torch.tensor([0., 0.], dtype=torch.double) 
    target_psi = target_psi.t()
    for i in range(len(nn_state.space)):
        psi = nn_state.psi(nn_state.space[i])/(nn_state.Z).sqrt()
        F[0] += target_psi[0,i]*psi[0]+target_psi[1,i]*psi[1]
        F[1] += target_psi[0,i]*psi[1]-target_psi[1,i]*psi[0]
    return  cplx.norm(F)


class TrainingStatistics(object):

    def __init__(self,N,frequency=10):
        self.N = N
        self.frequency = frequency
        self.vis = self.generate_visible_space(self.N)
        self.Z = 0.0
        self.bases = []
        self.target_psi = torch.zeros(1<<self.N,dtype=torch.double)
        self.target_psi_dict = {}
        self.unitaries = unitaries.create_dict()

        self.F  = torch.tensor([0., 0.], dtype=torch.double)
        self.KL = 0.0
        self.NLL = 0.0

    def scan(self,epoch,nn_state):
        self.partition(nn_state)
        self.fidelity(nn_state)
        #self.compute_KL(nn_state)
        print('Epoch = %d \tFidelity = ' % epoch,end="")
        print('%.6f' % self.F.item(),end="")
        #print('\tKL = ',end="") 
        #print('%.8f' %self.KL.item(),end="")
        print()

    def fidelity(self,nn_state):
        F = torch.tensor([0., 0.], dtype=torch.double) 
        for i in range(len(self.vis)):
            psi = nn_state.psi(self.vis[i])/self.Z.sqrt()
            F[0] += self.target_psi[0,i]*psi[0]+self.target_psi[1,i]*psi[1]
            F[1] += self.target_psi[0,i]*psi[1]-self.target_psi[1,i]*psi[0]
        self.F = cplx.norm(F)

    def rotate_psi(self,nn_state,basis):
        v = torch.zeros(self.N, dtype=torch.double)
        psi_r = torch.zeros(2,1<<self.N,dtype=torch.double)
        
        for x in range(1<<self.N):
            Upsi = torch.zeros(2, dtype=torch.double)
            num_nontrivial_U = 0
            nontrivial_sites = []
            for j in range(self.N):
                if (basis[j] is not 'Z'):
                    num_nontrivial_U += 1
                    nontrivial_sites.append(j)
            sub_state = generate_visible_space(num_nontrivial_U)
             
            for xp in range(1<<num_nontrivial_U):
                cnt = 0
                for j in range(self.N):
                    if (basis[j] is not 'Z'):
                        v[j]=sub_state[xp][cnt] 
                        cnt += 1
                    else:
                        v[j]=vis[x,j]
                U = torch.tensor([1., 0.], dtype=torch.double)
                for ii in range(num_nontrivial_U):
                    tmp = self.unitaries[basis[nontrivial_sites[ii]]][:,int(vis[x][nontrivial_sites[ii]]),int(v[nontrivial_sites[ii]])]
                    U = cplx.scalar_mult(U,tmp)
                Upsi += cplx.scalar_mult(U,nn_state.psi(v))
            psi_r[:,x] = Upsi
        return psi_r


    def compute_KL(self,nn_state):
        psi_r = torch.zeros(2,1<<self.N,dtype=torch.double)
        KL = 0.0
        for i in range(len(self.vis)):
            KL += cplx.norm(self.target_psi[:,i])*cplx.norm(self.target_psi[:,i]).log()/float(len(self.bases))
            KL -= cplx.norm(self.target_psi[:,i])*(self.probability(nn_state,self.vis[i])).log().item()/float(len(self.bases))
        for b in range(1,len(self.bases)):
            psi_r = rotate_psi(nn_state,self.bases[b])
            for ii in range(len(self.vis)):
                if(cplx.norm(self.psi_dict[self.bases[b]][:,ii])>0.0):
                    KL += cplx.norm(self.psi_dict[self.bases[b]][:,ii])*cplx.norm(self.psi_dict[bases[b]][:,ii]).log()/float(len(self.bases))
                KL -= cplx.norm(self.psi_dict[self.bases[b]][:,ii])*cplx.norm(psi_r[:,ii]).log().item()/float(len(self.bases))
                KL += cplx.norm(self.psi_dict[self.bases[b]][:,ii])*self.Z.log()/float(len(self.bases))
        self.KL = KL    

    def generate_visible_space(self,n):
        """Generates all possible visible states.
    
        :returns: A tensor of all possible spin configurations.
        :rtype: torch.Tensor
        """
        space = torch.zeros((1 << n, n),
                            device="cpu", dtype=torch.double)
        for i in range(1 << n):
            d = i
            for j in range(n):
                d, r = divmod(d, 2)
                space[i, n - j - 1] = int(r)
    
        return space

    def partition(self,nn_state):
        """The natural logarithm of the partition function of the RBM.
    
        :param visible_space: A rank 2 tensor of the entire visible space.
        :type visible_space: torch.Tensor
    
        :returns: The natural log of the partition function.
        :rtype: torch.Tensor
        """
        free_energies = -nn_state.rbm_am.effective_energy(self.vis)
        max_free_energy = free_energies.max()
    
        f_reduced = free_energies - max_free_energy
        logZ = max_free_energy + f_reduced.exp().sum().log()
        self.Z = logZ.exp()        
       
    def probability(self,nn_state,v):
        """Evaluates the probability of the given vector(s) of visible
        units; NOT RECOMMENDED FOR RBMS WITH A LARGE # OF VISIBLE UNITS

        :param v: The visible states.
        :type v: torch.Tensor
        :param Z: The partition function.
        :type Z: float

        :returns: The probability of the given vector(s) of visible units.
        :rtype: torch.Tensor
        """
        return (nn_state.amplitude(v))**2 / self.Z


    def load_target_psi(self,psi_data):
        D = 1<<self.N
        psi=torch.zeros(2,D, dtype=torch.double)
        if (len(psi_data.shape)<2):
            psi[0] = torch.tensor(psi_data,dtype=torch.double)
            psi[1] = torch.zeros(D,dtype=torch.double)
        else:
            psi_real = torch.tensor(psi_data[0:D,0],dtype=torch.double)
            psi_imag = torch.tensor(psi_data[0:D,1],dtype=torch.double)
            psi[0]   = psi_real
            psi[1]   = psi_imag
        self.target_psi = psi  

    def load_bases(self,bases):
        for i in range(len(bases)):
            tmp = ""
            for j in range(len(bases[i])):
                if bases[i][j] is not " ":
                    tmp += bases[i][j]
            self.bases.append(tmp)
    
    def load_psi_dict(self,psi_dict):
        D = int(len(psi_dict)/float(len(self.bases)))
        for b in range(len(self.bases)):
            psi      = torch.zeros(2,D, dtype=torch.double)
            psi_real = torch.tensor(psi_dict[b*D:D*(b+1),0], dtype=torch.double)
            psi_imag = torch.tensor(psi_dict[b*D:D*(b+1),1], dtype=torch.double)
            psi[0]   = psi_real
            psi[1]   = psi_imag
            
            self.target_psi_dict[self.bases[b]] = psi
        self.target_psi = self.target_psi_dict[self.bases[0]]
    
    def load(self,target_psi=None,bases=None,target_psi_dict=None):
        if(target_psi is not None):
            self.load_target_psi(target_psi)
            basis = ""
            for j in range(self.N):
                basis += 'Z'
            self.bases.append(basis)
        else:
            self.load_bases(bases)
            self.load_psi_dict(target_psi_dict)
            


