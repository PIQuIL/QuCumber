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

import pickle
import unittest
import numpy as np

import torch
from torch import nn

from qucumber.quantum_reconstruction import QuantumReconstruction
from qucumber.positive_wavefunction import PositiveWavefunction
from qucumber.complex_wavefunction import ComplexWavefunction

from qucumber.callbacks import MetricEvaluator
from qucumber.callbacks import CallbackList

import qucumber.utils.training_statistics as ts
import qucumber.utils.data as data
import qucumber.utils.cplx as cplx
import qucumber.utils.unitaries as unitaries


class TestTraining(unittest.TestCase):

    def test_trainingpositive(self):
        print ('Positive Wavefunction')
        print ('---------------------')

        train_samples_path = '../examples/01_Ising/tfim1d_train_samples.txt'
        psi_path           = '../examples/01_Ising/tfim1d_psi.txt'

        train_samples, target_psi = data.load_data(train_samples_path, psi_path)

        nv = train_samples.shape[-1]
        nh = nv

        fidelities = []
        KLs        = []

        print ("Training 10 times and checking fidelity and KL at 5 epochs...\n")
        for i in range(10):
            print ('Iteration: ', i+1)

            nn_state = PositiveWavefunction(num_visible = nv,
                                            num_hidden = nh,
                                            gpu=False)

            epochs     = 5
            batch_size = 100
            num_chains = 200
            CD         = 10
            lr         = 0.1
            log_every  = 5

            nn_state.space = nn_state.generate_hilbert_space(nv)
            callbacks = [MetricEvaluator(log_every,
                                         {'Fidelity':ts.fidelity,'KL':ts.KL},
                                         target_psi = target_psi, 
                                         verbose=True)] 

            qr = QuantumReconstruction(nn_state)

            self.initialize_posreal_params(nn_state)

            qr.fit(train_samples, epochs, batch_size, num_chains, CD, lr, 
                   progbar = False, callbacks=callbacks)

            fidelities.append(ts.fidelity(nn_state, target_psi).item())
            KLs.append(ts.KL(nn_state, target_psi).item())

        print ('\nStatistics')
        print ('----------')
        print ('Fidelity: ',np.average(fidelities),'+/-',np.std(fidelities)/np.sqrt(10),'\n')
        print ('KL: ',np.average(KLs),'+/-',np.std(KLs)/np.sqrt(10),'\n')

        self.assertTrue(abs(np.average(fidelities) - 0.85) < 0.02)
        self.assertTrue(abs(np.average(KLs) - 0.29) < 0.05)
        self.assertTrue((np.std(fidelities)/np.sqrt(len(fidelities))) < 0.01) 
        self.assertTrue((np.std(KLs)/np.sqrt(len(KLs))) < 0.01)
    
    def test_trainingcomplex(self):
        print ('Complex Wavefunction')
        print ('--------------------')

        train_samples_path = '../examples/02_qubits/qubits_train_samples.txt'
        train_bases_path   = '../examples/02_qubits/qubits_train_bases.txt'
        bases_path         =  '../examples/02_qubits/qubits_bases.txt'
        psi_path           = '../examples/02_qubits/qubits_psi.txt'

        train_samples, target_psi, train_bases, bases = data.load_data(train_samples_path,
                                                                       psi_path,
                                                                       train_bases_path, 
                                                                       bases_path)

        unitary_dict = unitaries.create_dict()
        nv = train_samples.shape[-1]
        nh = nv

        fidelities = []
        KLs        = []

        print ("Training 10 times and checking fidelity and KL at 5 epochs...\n")
        for i in range(10):
            print ('Iteration: ', i+1)

            nn_state = ComplexWavefunction(unitary_dict, num_visible = nv,
                                            num_hidden = nh,
                                            gpu=False)

            epochs     = 5
            batch_size = 50
            num_chains = 10
            CD         = 10
            lr         = 0.1
            log_every  = 5

            nn_state.space = nn_state.generate_hilbert_space(nv)
            callbacks = [MetricEvaluator(log_every,
                                         {'Fidelity':ts.fidelity,'KL':ts.KL},
                                         target_psi = target_psi,bases=bases,
                                         verbose=True)]

            z_samples = data.extract_refbasis_samples(train_samples, train_bases) 

            qr = QuantumReconstruction(nn_state)
            
            self.initialize_complex_params(nn_state) 

            qr.fit(train_samples, epochs, batch_size, num_chains, CD, lr, 
                   input_bases=train_bases, progbar = False, callbacks=callbacks,
                   z_samples = z_samples)

            fidelities.append(ts.fidelity(nn_state, target_psi).item())
            KLs.append(ts.KL(nn_state, target_psi, bases=bases).item())

        print ('\nStatistics')
        print ('----------')
        print ('Fidelity: ',np.average(fidelities),'+/-',np.std(fidelities)/np.sqrt(10),'\n')
        print ('KL: ',np.average(KLs),'+/-',np.std(KLs)/np.sqrt(10),'\n')

        self.assertTrue(abs(np.average(fidelities) - 0.38) < 0.05)
        self.assertTrue(abs(np.average(KLs) - 0.33) < 0.05) 
        self.assertTrue((np.std(fidelities)/np.sqrt(len(fidelities))) < 0.01) 
        self.assertTrue((np.std(KLs)/np.sqrt(len(KLs))) < 0.01)

    def initialize_posreal_params(self, nn_state):
        nn_state.rbm_am.weights = nn.Parameter( 
        torch.tensor([[-0.0753, -0.3096, -0.0811, -0.2651, -1.5460, -0.9804, -0.8723, -0.1330, 1.1215, -0.0898],
                      [ 2.4220,  0.4530, -1.5482, -0.0703, -0.7792,  0.8163,  0.5810,  0.0671, -0.9139,  0.3976],
                      [-1.6763,  0.5341, -1.4518, -1.9420,  0.6738, -0.6341, -1.4337, -0.6102, 0.6756,  1.3573],
                      [ 0.1702,  0.5211, -1.2175,  1.2129,  0.1714,  1.0136,  0.3437, -1.6570, -0.2229,  0.9149],
                      [ 1.9086,  1.6654,  0.1610,  0.0149,  0.5397, -1.3111, -0.2058,  0.7036, -0.7754,  0.2784],
                      [ 0.6267, -1.0399,  1.1952,  0.8137, -1.3016, -0.2017, -0.6997,  1.3262, 1.5532, -0.1674],
                      [-0.7907,  1.0205,  1.1567, -0.4956,  1.0089, -0.7910,  0.6909, -1.8196, -0.4850,  0.0799],
                      [-0.0350, -0.4999, -0.2218, -0.8704,  1.3338,  0.5471,  1.7503,  0.8141, 1.2716, -1.3127],
                      [ 0.7965,  2.3641, -1.5205, -1.2001,  0.9513,  0.8850,  0.4363,  1.9064, -0.0556, -0.0072],
                      [-0.2320,  0.8995, -0.4102,  0.1374,  1.1064, -1.1422,  0.8807, -0.8379, -0.6466, -1.5447]],
                     device = nn_state.device, dtype = torch.double), requires_grad=True
        )
    
        nn_state.rbm_am.visible_bias = nn.Parameter( 
        torch.tensor([-1.8245, 1.1097, -1.0773, 0.2703, 2.5470, 1.0985, 1.0781, -0.9696, -0.5660, 0.7058], 
                     device = nn_state.device, dtype = torch.double), requires_grad=True
        )
    
        nn_state.rbm_am.hidden_bias  = nn.Parameter( 
        torch.tensor([0.3472, -0.9928, -0.7945, 0.8348, -0.5590, 0.0083, -1.1400, 0.9709, 1.9565, -1.2171],
                     device = nn_state.device, dtype = torch.double), requires_grad=True
        )
                
    def initialize_complex_params(self, nn_state):
        nn_state.rbm_am.weights = nn.Parameter( 
        torch.tensor([[-0.0001, -1.2932],
                      [ 0.1503,  0.1440]],
                     device = nn_state.device, dtype = torch.double), requires_grad=True 
        )
    
        nn_state.rbm_am.visible_bias = nn.Parameter( 
        torch.tensor([0.1141, 1.7245], 
                     device = nn_state.device, dtype = torch.double), requires_grad=True
        )
    
        nn_state.rbm_am.hidden_bias  = nn.Parameter( 
        torch.tensor([-0.0992,  0.8556],
                     device = nn_state.device, dtype = torch.double), requires_grad=True
        )
        
        nn_state.rbm_ph.weights = nn.Parameter( 
        torch.tensor([[ 1.7191,  0.6447],
                      [-0.1314,  0.4127]],
                     device = nn_state.device, dtype = torch.double), requires_grad=True 
        )
    
        nn_state.rbm_ph.visible_bias = nn.Parameter(
        torch.tensor([0.3236, 0.7000], 
                     device = nn_state.device, dtype = torch.double), requires_grad=True
        )
    
        nn_state.rbm_ph.hidden_bias  = nn.Parameter(
        torch.tensor([ 1.0623, -1.2753],
                     device = nn_state.device, dtype = torch.double), requires_grad=True
        )


if __name__ == '__main__':
    unittest.main()
