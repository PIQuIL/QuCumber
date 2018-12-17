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
import torch
from math import isclose
import qucumber.observables as observables
from qucumber.nn_states import WaveFunctionBase


class TestWaveFunction(WaveFunctionBase):

    _rbm_am = None
    _device = None

    def __init__(self, nqubits, state=None):
        self.nqubits = nqubits
        self.device = "cpu"

    def sample(self, num_samples=1):
        dist = torch.distributions.Bernoulli(probs=0.5)
        sample_size = torch.Size((num_samples, self.nqubits))
        initial_state = dist.sample(sample_size).to(
            device=self.device, dtype=torch.double
        )

        return initial_state

    @staticmethod
    def autoload(location, gpu=False):
        pass

    def gradient(self, v):
        return 0.0

    @property
    def rbm_am(self):
        return self._rbm_am

    @rbm_am.setter
    def rbm_am(self, new_val):
        self._rbm_am = new_val

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, new_val):
        self._device = new_val

    @property
    def networks(self):
        return ["rbm_am"]

    def phase(self, v):
        if v.dim() == 1:
            v = v.unsqueeze(0)
            unsqueezed = True
        else:
            unsqueezed = False

        phase = torch.zeros(v.shape[0])

        if unsqueezed:
            return phase.squeeze_(0)
        else:
            return phase

    def amplitude(self, v):
        return 1 / torch.sqrt(torch.tensor(float(self.nqubits))) * torch.ones(v.size(0))

    def psi(self, v):
        # vector/tensor of shape (len(v),)
        amplitude = self.amplitude(v)

        # complex vector; shape: (2, len(v))
        psi = torch.zeros((2,) + amplitude.shape).to(
            dtype=torch.double, device=self.device
        )
        psi[0] = amplitude

        # squeeze down to complex scalar if there was only one visible state
        return psi.squeeze()


# TODO: add assertions


class TestPauli(unittest.TestCase):
    def test_spinflip(self):
        test_psi = TestWaveFunction(2)
        test_sample = test_psi.sample(num_samples=1000)
        observables.pauli.flip_spin(1, test_sample)

    def test_apply(self):
        test_psi = TestWaveFunction(2)
        test_sample = test_psi.sample(num_samples=100000)
        X = observables.SigmaX()

        measure_X = float(X.apply(test_psi, test_sample).mean())
        self.assertTrue(1.0 == measure_X, msg="measure Pauli X failed")

        Y = observables.SigmaY()
        measure_Y = float(Y.apply(test_psi, test_sample).mean())
        self.assertTrue(0.0 == measure_Y, msg="measure Pauli Y failed")

        Z = observables.SigmaZ()

        measure_Z = float(Z.apply(test_psi, test_sample).mean())
        self.assertTrue(
            isclose(0.0, measure_Z, abs_tol=1e-2), msg="measure Pauli Z failed"
        )


if __name__ == "__main__":
    unittest.main()
