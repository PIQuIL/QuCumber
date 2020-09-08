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
import pickle
from collections import namedtuple

import torch
import pytest

import qucumber
from qucumber.nn_states import PositiveWaveFunction, ComplexWaveFunction, DensityMatrix
from qucumber.utils import cplx

from grads_utils import ComplexGradsUtils, PosGradsUtils, DensityGradsUtils
from conftest import all_state_types, assertAlmostEqual, TOL


SEED = 1234
EPS = 1e-6


def positive_wavefunction_data(request, gpu, num_hidden):
    with open(
        os.path.join(request.fspath.dirname, "data", "test_grad_data.pkl"), "rb"
    ) as f:
        test_data = pickle.load(f)

    qucumber.set_random_seed(SEED, cpu=True, gpu=gpu, quiet=True)

    data = torch.tensor(test_data["tfim1d"]["train_samples"], dtype=torch.double)
    target = torch.tensor(test_data["tfim1d"]["target_psi"], dtype=torch.double).t()

    target = cplx.make_complex(target[0])

    num_visible = data.shape[-1]

    nn_state = PositiveWaveFunction(num_visible, num_hidden, gpu=gpu)
    PGU = PosGradsUtils(nn_state)

    data = data.to(device=nn_state.device)
    space = nn_state.generate_hilbert_space()
    target = target.to(device=nn_state.device)

    PositiveWaveFunctionFixture = namedtuple(
        "PositiveWaveFunctionFixture",
        ["data_samples", "target", "grad_utils", "nn_state", "space"],
    )

    return PositiveWaveFunctionFixture(
        data_samples=data, target=target, grad_utils=PGU, nn_state=nn_state, space=space
    )


def complex_wavefunction_data(request, gpu, num_hidden):
    with open(
        os.path.join(request.fspath.dirname, "data", "test_grad_data.pkl"), "rb"
    ) as f:
        test_data = pickle.load(f)

    qucumber.set_random_seed(SEED, cpu=True, gpu=gpu, quiet=True)

    data_bases = test_data["2qubits"]["train_bases"]
    data_samples = torch.tensor(
        test_data["2qubits"]["train_samples"], dtype=torch.double
    )

    all_bases = test_data["2qubits"]["bases"]
    target_psi_tmp = torch.tensor(
        test_data["2qubits"]["target_psi"], dtype=torch.double
    ).t()

    target_psi_tmp = cplx.make_complex(target_psi_tmp[0], target_psi_tmp[1])

    num_visible = data_samples.shape[-1]

    nn_state = ComplexWaveFunction(num_visible, num_hidden, gpu=gpu)
    unitary_dict = nn_state.unitary_dict

    CGU = ComplexGradsUtils(nn_state)

    all_bases = CGU.transform_bases(all_bases)

    target = CGU.load_target_psi(all_bases, target_psi_tmp)
    target = {b: v.to(device=nn_state.device) for b, v in target.items()}

    space = nn_state.generate_hilbert_space()
    data_samples = data_samples.to(device=nn_state.device)

    ComplexWaveFunctionFixture = namedtuple(
        "ComplexWaveFunctionFixture",
        [
            "data_samples",
            "data_bases",
            "grad_utils",
            "all_bases",
            "target",
            "space",
            "nn_state",
            "unitary_dict",
        ],
    )

    return ComplexWaveFunctionFixture(
        data_samples=data_samples,
        data_bases=data_bases,
        grad_utils=CGU,
        all_bases=all_bases,
        target=target,
        space=space,
        nn_state=nn_state,
        unitary_dict=unitary_dict,
    )


def density_matrix_data(request, gpu, num_hidden):
    with open(
        os.path.join(request.fspath.dirname, "data", "test_grad_data.pkl"), "rb"
    ) as f:
        test_data = pickle.load(f)

    qucumber.set_random_seed(SEED, cpu=True, gpu=gpu, quiet=True)

    data_bases = test_data["density_matrix"]["train_bases"]
    data_samples = torch.tensor(
        test_data["density_matrix"]["train_samples"], dtype=torch.double
    )

    all_bases = test_data["density_matrix"]["bases"]
    target = torch.tensor(
        test_data["density_matrix"]["density_matrix"], dtype=torch.double
    )

    target = cplx.make_complex(target[0], target[1])

    num_visible = data_samples.shape[-1]
    num_aux = num_visible + 1

    nn_state = DensityMatrix(num_visible, num_hidden, num_aux, gpu=gpu)
    unitary_dict = nn_state.unitary_dict

    DGU = DensityGradsUtils(nn_state)

    all_bases = DGU.transform_bases(all_bases)

    space = nn_state.generate_hilbert_space()
    data_samples = data_samples.to(device=nn_state.device)
    target = target.to(device=nn_state.device)

    DensityMatrixFixture = namedtuple(
        "DensityMatrixFixture",
        [
            "data_samples",
            "data_bases",
            "grad_utils",
            "all_bases",
            "target",
            "space",
            "nn_state",
            "unitary_dict",
        ],
    )

    return DensityMatrixFixture(
        data_samples=data_samples,
        data_bases=data_bases,
        grad_utils=DGU,
        all_bases=all_bases,
        target=target,
        space=space,
        nn_state=nn_state,
        unitary_dict=unitary_dict,
    )


hidden_layer_sizes = [pytest.param(9, id="9", marks=[pytest.mark.extra]), 10]
grad_types = ["KL", "NLL"]


@pytest.fixture(scope="module", params=all_state_types)
def quantum_state_constructor(request):
    nn_state_type = request.param
    if nn_state_type == PositiveWaveFunction:
        return positive_wavefunction_data
    elif nn_state_type == ComplexWaveFunction:
        return complex_wavefunction_data
    elif nn_state_type == DensityMatrix:
        return density_matrix_data
    else:
        raise ValueError(
            f"invalid test config: {nn_state_type} is not a valid quantum state type"
        )


@pytest.fixture(scope="module", params=hidden_layer_sizes)
def quantum_state_data(request, quantum_state_constructor, quantum_state_device):
    return quantum_state_constructor(request, quantum_state_device, request.param)


@pytest.fixture(scope="module", params=grad_types)
def quantum_state_graddata(request, quantum_state_data):
    grad_type = request.param
    nn_state, grad_utils = quantum_state_data.nn_state, quantum_state_data.grad_utils

    if grad_type == "KL":
        alg_grad_fn = grad_utils.algorithmic_gradKL
        num_grad_fn = grad_utils.numeric_gradKL
    else:
        alg_grad_fn = grad_utils.algorithmic_gradNLL
        num_grad_fn = grad_utils.numeric_gradNLL

    alg_grads = alg_grad_fn(**quantum_state_data._asdict())
    num_grads = [None for _ in nn_state.networks]

    for n, net in enumerate(nn_state.networks):
        rbm = getattr(nn_state, net)
        num_grad = torch.tensor([]).to(device=rbm.device, dtype=torch.double)
        for param in rbm.parameters():
            num_grad = torch.cat(
                (
                    num_grad,
                    num_grad_fn(
                        param=param.view(-1), eps=EPS, **quantum_state_data._asdict()
                    ).to(num_grad),
                )
            )
        num_grads[n] = num_grad

    return nn_state, alg_grads, num_grads, grad_type, TOL


def get_param_status(i, param_ranges):
    """Get parameter name of the parameter in param_ranges which contains the index i.

    Also return whether i is pointing to the first index of the parameter.
    """
    for p, rng in param_ranges.items():
        if i in rng:
            return p, i == rng[0]


def test_grads(quantum_state_graddata):
    nn_state, alg_grads, num_grads, grad_type, test_tol = quantum_state_graddata

    print(
        "\nTesting {} gradients for {} on {}.".format(
            grad_type, nn_state.__class__.__name__, nn_state.device
        )
    )

    for n, net in enumerate(nn_state.networks):
        print("\nRBM: %s" % net)
        rbm = getattr(nn_state, net)

        param_ranges = {}
        counter = 0
        for param_name, param in rbm.named_parameters():
            param_ranges[param_name] = range(counter, counter + param.numel())
            counter += param.numel()

        for i, grad in enumerate(num_grads[n]):
            p_name, at_start = get_param_status(i, param_ranges)
            if at_start:
                print(f"\nTesting {p_name}...")
                print(f"Numerical {grad_type}\tAlg {grad_type}")

            print("{: 10.8f}\t{: 10.8f}\t\t".format(grad, alg_grads[n][i].item()))

        assertAlmostEqual(
            num_grads[n],
            alg_grads[n],
            test_tol,
            msg=f"{grad_type} grads are not close enough for {net}!",
        )
