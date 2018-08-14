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


import os.path

import torch

import qucumber
from qucumber.nn_states import PositiveWavefunction
from . import __location__

INIT_SEED = 1234  # seed to initialize model params with
SAMPLING_SEED = 1337  # seed to draw samples from the model with


def test_model_saving_and_loading_pos():
    # some CUDA ops are non-deterministic; don't test on GPU.
    qucumber.set_random_seed(INIT_SEED, cpu=True, gpu=False, quiet=True)
    nn_state = PositiveWavefunction(10, gpu=False)

    model_path = os.path.join(__location__, "positive_wavefunction")

    nn_state.save(model_path)

    qucumber.set_random_seed(SAMPLING_SEED, cpu=True, gpu=False, quiet=True)
    # don't worry about floating-point wonkyness
    orig_sample = nn_state.sample(k=10).to(dtype=torch.uint8)

    nn_state2 = PositiveWavefunction(10, gpu=False)
    nn_state2.load(model_path)

    qucumber.set_random_seed(SAMPLING_SEED, cpu=True, gpu=False, quiet=True)
    post_load_sample = nn_state2.sample(k=10).to(dtype=torch.uint8)

    msg = "Got different sample after reloading model!"
    assert torch.equal(orig_sample, post_load_sample), msg

    nn_state3 = PositiveWavefunction.autoload(model_path, gpu=False)

    qucumber.set_random_seed(SAMPLING_SEED, cpu=True, gpu=False, quiet=True)
    post_autoload_sample = nn_state3.sample(k=10).to(dtype=torch.uint8)

    msg = "Got different sample after autoloading model!"
    assert torch.equal(orig_sample, post_autoload_sample), msg

    os.remove(model_path)
