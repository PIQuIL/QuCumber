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
from torch.nn.utils import parameters_to_vector
import pytest

import qucumber
from qucumber.nn_states import PositiveWaveFunction, ComplexWaveFunction
from . import __tests_location__

INIT_SEED = 1234  # seed to initialize model params with
SAMPLING_SEED = 1337  # seed to draw samples from the model with


@pytest.mark.parametrize("wvfn_type", [PositiveWaveFunction, ComplexWaveFunction])
def test_model_saving_and_loading(wvfn_type):
    # some CUDA ops are non-deterministic; don't test on GPU.
    qucumber.set_random_seed(INIT_SEED, cpu=True, gpu=False, quiet=True)
    nn_state = wvfn_type(10, gpu=False)

    model_path = os.path.join(__tests_location__, "wavefunction")

    nn_state.save(model_path)

    qucumber.set_random_seed(SAMPLING_SEED, cpu=True, gpu=False, quiet=True)
    # don't worry about floating-point wonkyness
    orig_sample = nn_state.sample(k=10).to(dtype=torch.uint8)

    nn_state2 = wvfn_type(10, gpu=False)
    nn_state2.load(model_path)

    qucumber.set_random_seed(SAMPLING_SEED, cpu=True, gpu=False, quiet=True)
    post_load_sample = nn_state2.sample(k=10).to(dtype=torch.uint8)

    msg = "Got different sample after reloading model!"
    assert torch.equal(orig_sample, post_load_sample), msg

    nn_state3 = wvfn_type.autoload(model_path, gpu=False)

    qucumber.set_random_seed(SAMPLING_SEED, cpu=True, gpu=False, quiet=True)
    post_autoload_sample = nn_state3.sample(k=10).to(dtype=torch.uint8)

    msg = "Got different sample after autoloading model!"
    assert torch.equal(orig_sample, post_autoload_sample), msg

    os.remove(model_path)


@pytest.mark.parametrize("wvfn_type", [PositiveWaveFunction, ComplexWaveFunction])
def test_model_saving_bad_metadata_key(wvfn_type):
    # some CUDA ops are non-deterministic; don't test on GPU.
    qucumber.set_random_seed(INIT_SEED, cpu=True, gpu=False, quiet=True)
    nn_state = wvfn_type(10, gpu=False)

    model_path = os.path.join(__tests_location__, "wavefunction")

    msg = "Metadata with invalid key should raise an error."
    with pytest.raises(ValueError, message=msg):
        nn_state.save(model_path, metadata={"rbm_am": 1337})


def test_positive_wavefunction_phase():
    nn_state = PositiveWaveFunction(10, gpu=False)

    vis_state = torch.ones(10).to(dtype=torch.double)
    actual_phase = nn_state.phase(vis_state).to(vis_state)
    expected_phase = torch.zeros(1).to(vis_state)

    msg = "PositiveWaveFunction is giving a non-zero phase for single visible state!"
    assert torch.equal(actual_phase, expected_phase), msg

    vis_state = torch.ones(10, 10).to(dtype=torch.double)
    actual_phase = nn_state.phase(vis_state).to(vis_state)
    expected_phase = torch.zeros(10).to(vis_state)

    msg = "PositiveWaveFunction is giving a non-zero phase for batch of visible states!"
    assert torch.equal(actual_phase, expected_phase), msg


def test_positive_wavefunction_psi():
    nn_state = PositiveWaveFunction(10, gpu=False)

    vis_state = torch.ones(10).to(dtype=torch.double)
    actual_psi = nn_state.psi(vis_state)[1].to(vis_state)
    expected_psi = torch.zeros(1).to(vis_state)

    msg = "PositiveWaveFunction is giving a non-zero imaginary part!"
    assert torch.equal(actual_psi, expected_psi), msg


def test_single_positive_sample():
    nn_state = PositiveWaveFunction(10, 7, gpu=False)

    sample = nn_state.sample(k=10).squeeze()
    h_sample = nn_state.sample_h_given_v(sample)
    v_prob = nn_state.prob_v_given_h(h_sample)

    msg = "Single hidden sample should give a "
    assert v_prob.dim() == 1, msg


def test_sampling_with_overwrite():
    nn_state = PositiveWaveFunction(10, gpu=False)

    old_state = torch.empty(100, 10).bernoulli_().to(dtype=torch.double)
    initial_state = old_state.clone()

    sample = nn_state.sample(k=10, initial_state=initial_state, overwrite=True)

    assert torch.equal(sample, initial_state), "initial_state did not get overwritten!"
    assert not torch.equal(sample, old_state), "Markov Chain did not get updated!"


def test_bad_stop_training_val():
    nn_state = PositiveWaveFunction(10, gpu=False)

    msg = "Setting stop_training to a non-boolean value should have raised an error."
    with pytest.raises(ValueError, message=msg):
        nn_state.stop_training = "foobar"


@pytest.mark.parametrize("wvfn_type", [PositiveWaveFunction, ComplexWaveFunction])
def test_parameter_reinitialization(wvfn_type):
    # some CUDA ops are non-deterministic; don't test on GPU.
    qucumber.set_random_seed(INIT_SEED, cpu=True, gpu=False, quiet=True)
    nn_state = wvfn_type(10, gpu=False)

    old_params = parameters_to_vector(nn_state.rbm_am.parameters())
    nn_state.reinitialize_parameters()
    new_params = parameters_to_vector(nn_state.rbm_am.parameters())

    msg = "Model parameters did not get reinitialized!"
    assert not torch.equal(old_params, new_params), msg


@pytest.mark.parametrize("wvfn_type", [PositiveWaveFunction, ComplexWaveFunction])
def test_large_hilbert_space_fail(wvfn_type):
    qucumber.set_random_seed(INIT_SEED, cpu=True, gpu=False, quiet=True)

    nn_state = wvfn_type(10, gpu=False)
    max_size = nn_state.max_size

    msg = "Generating full Hilbert Space for more than {} qubits should fail.".format(
        max_size
    )
    with pytest.raises(ValueError, message=msg):
        nn_state.generate_hilbert_space(size=max_size + 1)
