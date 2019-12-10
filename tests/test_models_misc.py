# Copyright 2019 PIQuIL - All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os.path
from functools import reduce

import torch
from torch.nn.utils import parameters_to_vector
import pytest

import qucumber
from qucumber.nn_states import PositiveWaveFunction, ComplexWaveFunction, DensityMatrix
from qucumber.utils import cplx
from qucumber.utils.unitaries import create_dict
from qucumber.utils.training_statistics import rotate_psi, rotate_rho

from test_grads import assertAlmostEqual


INIT_SEED = 1234  # seed to initialize model params with
SAMPLING_SEED = 1337  # seed to draw samples from the model with

TOL = 1e-6


@pytest.mark.parametrize("wvfn_type", [PositiveWaveFunction, ComplexWaveFunction])
def test_model_saving_and_loading(tmpdir, wvfn_type):
    # some CUDA ops are non-deterministic; don't test on GPU.
    qucumber.set_random_seed(INIT_SEED, cpu=True, gpu=False, quiet=True)
    nn_state = wvfn_type(10, gpu=False)

    model_path = str(tmpdir.mkdir("wvfn").join("params.pt").realpath())

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


gpu_availability = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU required"
)

devices = [
    pytest.param(False, id="cpu"),
    pytest.param(True, id="gpu", marks=[gpu_availability, pytest.mark.gpu]),
]


@pytest.mark.parametrize("wvfn_type", [PositiveWaveFunction, ComplexWaveFunction])
@pytest.mark.parametrize("is_src_gpu", devices)
@pytest.mark.parametrize("is_dest_gpu", devices)
def test_autoloading(tmpdir, wvfn_type, is_src_gpu, is_dest_gpu):
    model_path = str(tmpdir.mkdir("wvfn").join("params.pt").realpath())

    nn_state = wvfn_type(10, gpu=is_src_gpu)
    nn_state.save(model_path)

    nn_state2 = wvfn_type(10, gpu=is_dest_gpu)
    nn_state2.load(model_path)

    os.remove(model_path)


@pytest.mark.parametrize("wvfn_type", [PositiveWaveFunction, ComplexWaveFunction])
def test_model_saving_bad_metadata_key(tmpdir, wvfn_type):
    # some CUDA ops are non-deterministic; don't test on GPU.
    qucumber.set_random_seed(INIT_SEED, cpu=True, gpu=False, quiet=True)
    nn_state = wvfn_type(10, gpu=False)

    model_path = str(tmpdir.mkdir("wvfn").join("params.pt").realpath())

    msg = "Metadata with invalid key should raise an error."
    with pytest.raises(ValueError):
        nn_state.save(model_path, metadata={"rbm_am": 1337})
        pytest.fail(msg)


def test_positive_wavefunction_phase():
    nn_state = PositiveWaveFunction(10, gpu=False)

    vis_state = torch.ones(10).to(dtype=torch.double)
    actual_phase = nn_state.phase(vis_state).to(vis_state)
    expected_phase = torch.zeros(1).to(vis_state).squeeze()

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


def test_density_matrix_hermiticity():
    nn_state = DensityMatrix(5, 5, 5, gpu=False)

    space = nn_state.generate_hilbert_space(5)
    Z = nn_state.normalization(space)
    rho = nn_state.rho(space, space) / Z

    assert torch.equal(rho, cplx.conjugate(rho)), "DensityMatrix should be Hermitian!"


def test_density_matrix_tr1():
    nn_state = DensityMatrix(5, 5, 5, gpu=False)

    v_space = nn_state.generate_hilbert_space(5)
    matrix = nn_state.rho(v_space, v_space) / nn_state.normalization(v_space)

    msg = f"Trace of density matrix is not within {TOL} of 1!"
    assertAlmostEqual(torch.trace(matrix[0]), torch.Tensor([1]), TOL, msg=msg)


def test_gamma_plus():
    nn_state = DensityMatrix(5, gpu=False)
    v = nn_state.generate_hilbert_space(5)
    vp = v[:4, :]

    gamma_p = nn_state.rbm_am.gamma_plus(v, vp)

    assert gamma_p.shape == (v.shape[0], vp.shape[0])


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
    with pytest.raises(ValueError):
        nn_state.stop_training = "foobar"
        pytest.fail(msg)


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
    with pytest.raises(ValueError):
        nn_state.generate_hilbert_space(size=max_size + 1)
        pytest.fail(msg)


@pytest.mark.parametrize("num_visible", [1, 2, 7])
@pytest.mark.parametrize("wvfn_type", [PositiveWaveFunction, ComplexWaveFunction])
def test_rotate_psi(num_visible, wvfn_type):
    nn_state = wvfn_type(num_visible, gpu=False)
    basis = "X" * num_visible
    unitary_dict = create_dict()

    space = nn_state.generate_hilbert_space()
    psi = nn_state.psi(space)

    psi_r_fast = rotate_psi(nn_state, basis, space, unitary_dict, psi=psi)

    U = reduce(cplx.kronecker_prod, [unitary_dict[b] for b in basis])
    psi_r_correct = cplx.matmul(U, psi)

    assertAlmostEqual(psi_r_fast, psi_r_correct, msg="Fast psi rotation failed!")


@pytest.mark.parametrize("num_visible", [1, 2, 7])
@pytest.mark.parametrize("state_type", [DensityMatrix])
def test_rotate_rho(num_visible, state_type):
    nn_state = state_type(num_visible, gpu=False)
    basis = "X" * num_visible
    unitary_dict = create_dict()

    space = nn_state.generate_hilbert_space()
    rho = nn_state.rho(space, space)

    rho_r_fast = rotate_rho(nn_state, basis, space, unitary_dict, rho=rho)

    U = reduce(cplx.kronecker_prod, [unitary_dict[b] for b in basis])
    rho_r_correct = cplx.matmul(U, rho)
    rho_r_correct = cplx.matmul(rho_r_correct, cplx.conjugate(U))

    assertAlmostEqual(rho_r_fast, rho_r_correct, msg="Fast rho rotation failed!")
