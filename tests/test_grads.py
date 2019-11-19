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
from qucumber.utils import unitaries

from .grads_utils import ComplexGradsUtils, PosGradsUtils, DensityGradsUtils
from . import __tests_location__


K = 10
SEED = 1234
EPS = 1.0e-6

TOL = torch.tensor(1.00, dtype=torch.double)
# NLL grad tests are a bit too random tbh
PDIFF = torch.tensor(100, dtype=torch.double)


def percent_diff(a, b):  # for NLL
    numerator = torch.abs(a - b) * 100.0
    denominator = torch.abs(0.5 * (a + b))
    return numerator / denominator


# assertion functions
def assertAlmostEqual(a, b, tol, msg=None):
    a = a.to(device=torch.device("cpu"))
    b = b.to(device=torch.device("cpu"))
    result = torch.ge(tol * torch.ones_like(torch.abs(a - b)), torch.abs(a - b))
    expect = torch.ones_like(torch.abs(a - b), dtype=torch.uint8)
    assert torch.equal(result, expect), msg


def assertPercentDiff(a, b, pdiff, msg=None):
    a = a.to(device=torch.device("cpu"))
    b = b.to(device=torch.device("cpu"))
    result = torch.ge(pdiff * torch.ones_like(percent_diff(a, b)), percent_diff(a, b))
    expect = torch.ones_like(result, dtype=torch.uint8)
    assert torch.equal(result, expect), msg


def positive_wavefunction_data(gpu, num_hidden):
    with open(
        os.path.join(__tests_location__, "data", "test_grad_data.pkl"), "rb"
    ) as f:
        test_data = pickle.load(f)

    qucumber.set_random_seed(SEED, cpu=True, gpu=gpu, quiet=True)

    data = torch.tensor(test_data["tfim1d"]["train_samples"], dtype=torch.double)
    target_psi = torch.tensor(test_data["tfim1d"]["target_psi"], dtype=torch.double)

    num_visible = data.shape[-1]

    nn_state = PositiveWaveFunction(num_visible, num_hidden, gpu=gpu)
    PGU = PosGradsUtils(nn_state)

    data = data.to(device=nn_state.device)
    vis = nn_state.generate_hilbert_space(num_visible)
    target_psi = target_psi.to(device=nn_state.device)

    PositiveWaveFunctionFixture = namedtuple(
        "PositiveWaveFunctionFixture",
        ["data", "target_psi", "grad_utils", "nn_state", "vis"],
    )

    return PositiveWaveFunctionFixture(
        data=data, target_psi=target_psi, grad_utils=PGU, nn_state=nn_state, vis=vis
    )


def complex_wavefunction_data(gpu, num_hidden):
    with open(
        os.path.join(__tests_location__, "data", "test_grad_data.pkl"), "rb"
    ) as f:
        test_data = pickle.load(f)

    qucumber.set_random_seed(SEED, cpu=True, gpu=gpu, quiet=True)

    data_bases = test_data["2qubits"]["train_bases"]
    data_samples = torch.tensor(
        test_data["2qubits"]["train_samples"], dtype=torch.double
    )

    bases_data = test_data["2qubits"]["bases"]
    target_psi_tmp = torch.tensor(
        test_data["2qubits"]["target_psi"], dtype=torch.double
    )

    num_visible = data_samples.shape[-1]

    unitary_dict = unitaries.create_dict()
    nn_state = ComplexWaveFunction(
        num_visible, num_hidden, unitary_dict=unitary_dict, gpu=gpu
    )
    CGU = ComplexGradsUtils(nn_state)

    bases = CGU.transform_bases(bases_data)

    psi_dict = CGU.load_target_psi(bases, target_psi_tmp)
    vis = nn_state.generate_hilbert_space(num_visible)

    data_samples = data_samples.to(device=nn_state.device)

    unitary_dict = {b: v.to(device=nn_state.device) for b, v in unitary_dict.items()}
    psi_dict = {b: v.to(device=nn_state.device) for b, v in psi_dict.items()}

    ComplexWaveFunctionFixture = namedtuple(
        "ComplexWaveFunctionFixture",
        [
            "data_samples",
            "data_bases",
            "grad_utils",
            "bases",
            "psi_dict",
            "vis",
            "nn_state",
            "unitary_dict",
        ],
    )

    return ComplexWaveFunctionFixture(
        data_samples=data_samples,
        data_bases=data_bases,
        grad_utils=CGU,
        bases=bases,
        psi_dict=psi_dict,
        vis=vis,
        nn_state=nn_state,
        unitary_dict=unitary_dict,
    )


def density_matrix_data(gpu, num_hidden):
    with open(
        os.path.join(__tests_location__, "data", "test_grad_data.pkl"), "rb"
    ) as f:
        test_data = pickle.load(f)

    qucumber.set_random_seed(SEED, cpu=True, gpu=gpu, quiet=True)

    data_bases = test_data["density_matrix"]["train_bases"]
    data_samples = torch.tensor(
        test_data["density_matrix"]["train_samples"], dtype=torch.double
    )

    bases_data = test_data["density_matrix"]["bases"]
    target_matrix = torch.tensor(
        test_data["density_matrix"]["density_matrix"], dtype=torch.double
    )

    num_visible = data_samples.shape[-1]
    num_aux = num_hidden + 1  # this is not a rule, will change with data

    unitary_dict = unitaries.create_dict()
    nn_state = DensityMatrix(
        num_visible, num_hidden, num_aux, unitary_dict=unitary_dict, gpu=gpu
    )
    DGU = DensityGradsUtils(nn_state)

    bases = DGU.transform_bases(bases_data)

    v_space = nn_state.generate_hilbert_space(num_visible)
    a_space = nn_state.generate_hilbert_space(num_aux)

    data_samples = data_samples.to(device=nn_state.device)

    unitary_dict = {b: v.to(device=nn_state.device) for b, v in unitary_dict.items()}

    DensityMatrixFixture = namedtuple(
        "DensityMatrixFixture",
        [
            "data_samples",
            "data_bases",
            "grad_utils",
            "bases",
            "target",
            "v_space",
            "a_space",
            "nn_state",
            "unitary_dict",
        ],
    )

    return DensityMatrixFixture(
        data_samples=data_samples,
        data_bases=data_bases,
        grad_utils=DGU,
        bases=bases,
        target=target_matrix,
        v_space=v_space,
        a_space=a_space,
        nn_state=nn_state,
        unitary_dict=unitary_dict,
    )


gpu_availability = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU required"
)
wavefunction_types = ["positive", "complex", "density_matrix"]
devices = [
    pytest.param(False, id="cpu"),
    pytest.param(True, id="gpu", marks=[gpu_availability, pytest.mark.gpu]),
]
hidden_layer_sizes = [pytest.param(9, id="9", marks=[pytest.mark.extra]), 10]
grad_types = [
    "KL",
    pytest.param("NLL", id="NLL", marks=[pytest.mark.nll, pytest.mark.slow]),
]


@pytest.fixture(scope="module", params=wavefunction_types)
def wavefunction_constructor(request):
    wvfn_type = request.param
    if wvfn_type == "positive":
        return positive_wavefunction_data
    elif wvfn_type == "complex":
        return complex_wavefunction_data
    elif wvfn_type == "density_matrix":
        return density_matrix_data
    else:
        raise ValueError(
            "invalid test config: {} is not a valid wavefunction type".format(wvfn_type)
        )


@pytest.fixture(scope="module", params=devices)
def wavefunction_device(request):
    return request.param


@pytest.fixture(scope="module", params=hidden_layer_sizes)
def wavefunction_data(request, wavefunction_constructor, wavefunction_device):
    return wavefunction_constructor(wavefunction_device, request.param)


@pytest.fixture(scope="module", params=grad_types)
def wavefunction_graddata(request, wavefunction_data):
    grad_type = request.param
    nn_state, grad_utils = wavefunction_data.nn_state, wavefunction_data.grad_utils

    if grad_type == "KL":
        alg_grad_fn = grad_utils.algorithmic_gradKL
        num_grad_fn = grad_utils.numeric_gradKL
        test_tol = TOL
    else:
        alg_grad_fn = grad_utils.algorithmic_gradNLL
        num_grad_fn = grad_utils.numeric_gradNLL
        test_tol = PDIFF

    alg_grads = alg_grad_fn(k=K, **wavefunction_data._asdict())
    num_grads = [None for _ in nn_state.networks]

    for n, net in enumerate(nn_state.networks):
        rbm = getattr(nn_state, net)
        num_grad = torch.tensor([]).to(device=rbm.device, dtype=torch.double)
        for param in rbm.parameters():
            num_grad = torch.cat(
                (
                    num_grad,
                    num_grad_fn(
                        param=param.view(-1), eps=EPS, **wavefunction_data._asdict()
                    ).to(num_grad),
                )
            )
        num_grads[n] = num_grad

    return nn_state, alg_grads, num_grads, grad_type, test_tol


def get_param_status(i, param_ranges):
    """Get parameter name of the parameter in param_ranges which contains the index i.

    Also return whether i is pointing to the first index of the parameter.
    """
    for p, rng in param_ranges.items():
        if i in rng:
            return p, i == rng[0]


def test_grads(wavefunction_graddata):
    nn_state, alg_grads, num_grads, grad_type, test_tol = wavefunction_graddata

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
                print("\nTesting {}...".format(p_name))
                print("Numerical {}\tAlg {}".format(grad_type, grad_type))

            print("{: 10.8f}\t{: 10.8f}\t\t".format(grad, alg_grads[n][i].item()))

        assertAlmostEqual(
            num_grads[n],
            alg_grads[n],
            test_tol,
            msg="{} grads are not close enough for {}!".format(grad_type, net),
        )
