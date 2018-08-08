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

import unittest

import pickle

import torch

from grads_utils import PosGradsUtils, ComplexGradsUtils

import qucumber
from qucumber.positive_wavefunction import PositiveWavefunction
from qucumber.complex_wavefunction import ComplexWavefunction
from qucumber.quantum_reconstruction import QuantumReconstruction
from qucumber.utils import cplx, unitaries


class TestAllGrads(unittest.TestCase):

    def percent_diff(self, a, b): # for NLL
        numerator   = torch.abs(a-b)*100.
        denominator = torch.abs(0.5*(a+b))
        return numerator / denominator


    # assertion functions
    def assertAlmostEqual(self, a, b, tol, msg=None):
        result = torch.ge(tol*torch.ones_like(torch.abs(a-b)), torch.abs(a-b))
        expect = torch.ones_like(torch.abs(a-b), dtype = torch.uint8)
        self.assertTrue(torch.equal(result, expect), msg=msg)


    def assertPercentDiff(self, a, b, pdiff, msg=None):
        result = torch.ge(pdiff*torch.ones_like(self.percent_diff(a, b)), self.percent_diff(a,b))
        expect = torch.ones_like(result, dtype = torch.uint8)
        self.assertTrue(torch.equal(result, expect), msg=msg)


    # test grads for positive real wavefunction
    def test_posgrads(self):
        print ('\nTesting gradients for positive-real wavefunction.')

        k          = 10
        num_chains = 100
        seed       = 1234
        eps        = 1.e-6

        tol   = torch.tensor(1e-9, dtype = torch.double)
        pdiff = torch.tensor(100, dtype = torch.double)
        
        with open('test_data.pkl', 'rb') as fin:
            test_data = pickle.load(fin)

        qucumber.set_random_seed(seed, cpu=True, gpu=True, quiet=True)

        data        = torch.tensor(test_data['tfim1d']['train_samples'],
                                   dtype=torch.double)
        target_psi  = torch.tensor(test_data['tfim1d']['target_psi'],
                                   dtype=torch.double)

        num_visible = data.shape[-1]
        num_hidden  = num_visible

        nn_state = PositiveWavefunction(num_visible,num_hidden,gpu=False)
        PGU      = PosGradsUtils(nn_state)

        qr         = QuantumReconstruction(nn_state)
        data       = data.to(device=nn_state.device)
        vis        = PGU.generate_visible_space(num_visible) 
        target_psi = target_psi.to(device=nn_state.device)
 
        alg_grad_kl  = PGU.algorithmic_gradKL(target_psi, vis)
        alg_grad_nll = PGU.algorithmic_gradNLL(qr, data, k)

        num_grad_kl  = PGU.numeric_gradKL(target_psi,
                                          nn_state.rbm_am.weights.view(-1),
                                          vis, eps)
        num_grad_nll = PGU.numeric_gradNLL(nn_state.rbm_am.weights.view(-1),
                                           data, vis, eps)

        counter = 0
        print("\nTesting weights...")
        print("numerical kl\talg kl\t\t\tnumerical nll\talg nll")
        for i in range(len(nn_state.rbm_am.weights.view(-1))):
            print("{: 10.8f}\t{: 10.8f}\t\t"
                  .format(num_grad_kl[i], alg_grad_kl[counter].item()),
                  end="", flush=True)
            print("{: 10.8f}\t{: 10.8f}\t\t"
                  .format(num_grad_nll[i], alg_grad_nll[0][i].item()))
            counter += 1 

        self.assertAlmostEqual(num_grad_kl,
                               alg_grad_kl[:len(nn_state.rbm_am.weights.view(-1))],
                               tol,
                               msg="KL grads are not close enough for weights!"
                              )

        self.assertPercentDiff(num_grad_nll, 
                               alg_grad_nll[0][:len(nn_state.rbm_am.weights.view(-1))],
                               pdiff,
                               msg="NLL grads are not close enough for weights!"
                              )
                               

        num_grad_kl  = PGU.numeric_gradKL(target_psi,
                                          nn_state.rbm_am.visible_bias, vis, eps)
        num_grad_nll = PGU.numeric_gradNLL(nn_state.rbm_am.visible_bias,
                                           data, vis, eps)

        print("\nTesting visible bias...")
        print("numerical kl\talg kl\t\t\tnumerical nll\talg nll")
        for i in range(len(nn_state.rbm_am.visible_bias)):
            print("{: 10.8f}\t{: 10.8f}\t\t"
                  .format(num_grad_kl[i], alg_grad_kl[counter].item()),
                  end="", flush=True)
            print("{: 10.8f}\t{: 10.8f}\t\t"
                  .format(num_grad_nll[i], alg_grad_nll[0][counter].item()))
            counter += 1

        self.assertAlmostEqual(num_grad_kl,
                               alg_grad_kl[len(nn_state.rbm_am.weights.view(-1)):counter],
                               tol,
                               msg="KL grads are not close enough for visible biases!"
                              )

        self.assertPercentDiff(num_grad_nll, 
                               alg_grad_nll[0][len(nn_state.rbm_am.weights.view(-1)):counter],
                               pdiff,
                               msg="NLL grads are not close enough for visible biases!"
                              )
         
        num_grad_kl  = PGU.numeric_gradKL(target_psi,
                                         nn_state.rbm_am.hidden_bias, vis, eps)
        num_grad_nll = PGU.numeric_gradNLL(nn_state.rbm_am.hidden_bias,
                                           data, vis, eps)
    
        print("\nTesting hidden bias...")
        print("numerical kl\talg kl\t\t\tnumerical nll\talg nll")
        for i in range(len(nn_state.rbm_am.hidden_bias)):
            print("{: 10.8f}\t{: 10.8f}\t\t"
                  .format(num_grad_kl[i], alg_grad_kl[counter].item()),
                  end="", flush=True)
            print("{: 10.8f}\t{: 10.8f}\t\t"
                  .format(num_grad_nll[i], alg_grad_nll[0][counter].item()))
            counter += 1
       
        self.assertAlmostEqual(num_grad_kl,
                               alg_grad_kl[(len(nn_state.rbm_am.weights.view(-1))+
                                            len(nn_state.rbm_am.visible_bias)):counter],
                               tol,
                               msg="KL grads are not close enough for hidden biases!"
                              )

        self.assertPercentDiff(num_grad_nll, 
                               alg_grad_nll[0][(len(nn_state.rbm_am.weights.view(-1))+
                                                len(nn_state.rbm_am.visible_bias)):counter],
                               pdiff,
                               msg="NLL grads are not close enough for hidden biases!"
                              )

    
    # test grads for complex wavefunction
    def test_complexgrads(self):
        print ('\nTesting gradients for complex wavefunction.')

        k          = 2
        num_chains = 10
        seed       = 1234
        eps        = 1.e-6

        tol   = torch.tensor(1e-9, dtype = torch.double)
        pdiff = torch.tensor(100, dtype = torch.double)

        with open('test_data.pkl', 'rb') as fin:
            test_data = pickle.load(fin)
    
        qucumber.set_random_seed(seed, cpu=True, gpu=True, quiet=True)

        train_bases   = test_data['2qubits']['train_bases']
        train_samples = torch.tensor(test_data['2qubits']['train_samples'],
                                     dtype=torch.double)

        bases_data     = test_data['2qubits']['bases']
        target_psi_tmp = torch.tensor(test_data['2qubits']['target_psi'],
                                      dtype=torch.double)

        num_visible = train_samples.shape[-1]
        num_hidden  = num_visible

        unitary_dict = unitaries.create_dict()
        nn_state     = ComplexWavefunction(unitary_dict, num_visible,
                                           num_hidden, gpu=False)
        CGU          = ComplexGradsUtils(nn_state)

        bases = CGU.transform_bases(bases_data)

        psi_dict = CGU.load_target_psi(bases, target_psi_tmp)
        vis      = CGU.generate_visible_space(num_visible)
     
        qr            = QuantumReconstruction(nn_state)
        device        = qr.nn_state.device
        train_samples = train_samples.to(device=device)
        vis           = vis.to(device=device)
    
        unitary_dict = {b: v.to(device=device) for b, v in unitary_dict.items()}
        psi_dict = {b: v.to(device=device) for b, v in psi_dict.items()}
   
        alg_grad_nll = CGU.algorithmic_gradNLL(qr, train_samples, train_bases, k)
        alg_grad_kl  = CGU.algorithmic_gradKL(psi_dict, vis, unitary_dict, bases)

        for n, net in enumerate(qr.nn_state.networks):
            counter = 0
            print('\nRBM: %s' % net)
            rbm = getattr(qr.nn_state, net)
            
            num_grad_kl  = CGU.numeric_gradKL(rbm.weights.view(-1), psi_dict,
                                              vis, unitary_dict, bases, eps)
            num_grad_nll = CGU.numeric_gradNLL(train_samples, train_bases, unitary_dict,
                                               rbm.weights.view(-1), vis, eps)
    
            print("\nTesting weights...")
            print("Numerical KL\tAlg KL\t\t\tNumerical NLL\tAlg NLL")
            for i in range(len(rbm.weights.view(-1))):
                print("{: 10.8f}\t{: 10.8f}\t\t"
                      .format(num_grad_kl[i], alg_grad_kl[n][counter].item()),
                      end="", flush=True)
                print("{: 10.8f}\t{: 10.8f}\t\t"
                      .format(num_grad_nll[i], alg_grad_nll[n][i].item()))
                counter += 1

            #print (num_grad_kl)
            #print (alg_grad_kl[n][:counter])

            self.assertAlmostEqual(num_grad_kl,
                                   alg_grad_kl[n][:counter],
                                   tol,
                                   msg="KL grads are not close enough for weights!"
                                  )

            self.assertPercentDiff(num_grad_nll, 
                                   alg_grad_nll[n][:counter],
                                   pdiff,
                                   msg="NLL grads are not close enough for weights!"
                                  )

            num_grad_kl  = CGU.numeric_gradKL(rbm.visible_bias, psi_dict,
                                              vis, unitary_dict, bases, eps)
            num_grad_nll = CGU.numeric_gradNLL(train_samples, train_bases,
                                               unitary_dict, rbm.visible_bias,
                                               vis, eps)
    
            print("\nTesting visible bias...")
            print("Numerical KL\tAlg KL\t\t\tNumerical NLL\tAlg NLL")
            for i in range(len(rbm.visible_bias)):
                print("{: 10.8f}\t{: 10.8f}\t\t"
                      .format(num_grad_kl[i], alg_grad_kl[n][counter].item()),
                      end="", flush=True)
                print("{: 10.8f}\t{: 10.8f}\t\t"
                      .format(num_grad_nll[i], alg_grad_nll[n][counter].item()))
                counter += 1
   
            self.assertAlmostEqual(num_grad_kl,
                                   alg_grad_kl[n][len(rbm.weights.view(-1)):counter],
                                   tol,
                                   msg="KL grads are not close enough for visible biases!"
                                  )

            self.assertPercentDiff(num_grad_nll, 
                                   alg_grad_nll[n][len(rbm.weights.view(-1)):counter],
                                   pdiff,
                                   msg="NLL grads are not close enough for visible biases!"
                                  )
         
 
            num_grad_kl  = CGU.numeric_gradKL(rbm.hidden_bias, psi_dict,
                                             vis, unitary_dict, bases, eps)
            num_grad_nll = CGU.numeric_gradNLL(train_samples, train_bases,
                                               unitary_dict, rbm.hidden_bias,
                                               vis, eps)

            print("\nTesting hidden bias...")
            print("Numerical KL\tAlg KL\t\t\tNumerical NLL\tAlg NLL")
            for i in range(len(rbm.hidden_bias)):
                print("{: 10.8f}\t{: 10.8f}\t\t"
                      .format(num_grad_kl[i], alg_grad_kl[n][counter].item()),
                      end="", flush=True)
                print("{: 10.8f}\t{: 10.8f}\t\t"
                      .format(num_grad_nll[i], alg_grad_nll[n][counter].item()))
                counter += 1

            self.assertAlmostEqual(num_grad_kl,
                                   alg_grad_kl[n][(len(rbm.weights.view(-1))+
                                                len(rbm.visible_bias)):counter],
                                   tol,
                                   msg="KL grads are not close enough for hidden biases!"
                                  )

            self.assertPercentDiff(num_grad_nll, 
                                   alg_grad_nll[n][(len(rbm.weights.view(-1))+
                                                    len(rbm.visible_bias)):counter],
                                   pdiff,
                                   msg="NLL grads are not close enough for hidden biases!"
                                  )  

    
        print('')
       

if __name__ == '__main__':
    unittest.main()

