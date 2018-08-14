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
from torch.nn.utils import parameters_to_vector

import qucumber
from qucumber.nn_states import PositiveWavefunction
from qucumber.quantum_reconstruction import QuantumReconstruction


SEED = 1234


def test_positive_wavefunction():
    qucumber.set_random_seed(SEED, cpu=True)

    nn_state = PositiveWavefunction(10, gpu=False)
    qr = QuantumReconstruction(nn_state)

    old_params = parameters_to_vector(nn_state.rbm_am.parameters())

    data = torch.ones(100, 10)

    qr.fit(data, epochs=1, pos_batch_size=10, neg_batch_size=10)

    new_params = parameters_to_vector(nn_state.rbm_am.parameters())

    msg = "PositiveWavefunction's parameters did not change!"

    assert not torch.equal(old_params, new_params), msg
