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
import pickle
from collections import namedtuple

import torch
import pytest

import qucumber
from qucumber.nn_states import PositiveWavefunction, ComplexWavefunction
from qucumber.quantum_reconstruction import QuantumReconstruction
from qucumber.utils import unitaries
from .grads_utils import ComplexGradsUtils, PosGradsUtils
from . import __location__


K = 10
SEED = 1234
EPS = 1.e-6

TOL = torch.tensor(1e-9, dtype=torch.double)
PDIFF = torch.tensor(100, dtype=torch.double)


def percent_diff(a, b):  # for NLL
    numerator = torch.abs(a - b) * 100.
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


@pytest.fixture(scope="module", params=[
    False,
    pytest.param(True,
                 marks=[
                     pytest.mark.skipif(not torch.cuda.is_available(),
                                        reason="GPU required"),
                     pytest.mark.gpu,
                 ]),
])
def positive_wavefunction_data(request):
    with open(os.path.join(__location__, "test_data.pkl"), "rb") as fin:
        test_data = pickle.load(fin)

    qucumber.set_random_seed(SEED, cpu=True, gpu=request.param, quiet=True)

    data = torch.tensor(test_data["tfim1d"]["train_samples"], dtype=torch.double)
    target_psi = torch.tensor(test_data["tfim1d"]["target_psi"], dtype=torch.double)

    num_visible = data.shape[-1]
    num_hidden = num_visible

    nn_state = PositiveWavefunction(num_visible, num_hidden, gpu=request.param)
    PGU = PosGradsUtils(nn_state)

    qr = QuantumReconstruction(nn_state)
    data = data.to(device=nn_state.device)
    vis = nn_state.generate_hilbert_space(num_visible)
    target_psi = target_psi.to(device=nn_state.device)

    PositiveWavefunctionFixture = namedtuple(
        "PositiveWavefunctionFixture", ["data", "target_psi", "PGU", "qr", "vis"]
    )

    return PositiveWavefunctionFixture(
        data=data, target_psi=target_psi, PGU=PGU, qr=qr, vis=vis
    )


@pytest.mark.skip(reason="doesn't give consistent results")
def test_posgrad_nll(positive_wavefunction_data):
    print("\nTesting NLL gradients for positive-real wavefunction.")
    data, target_psi, PGU, qr, vis = positive_wavefunction_data

    alg_grad_nll = PGU.algorithmic_gradNLL(qr, data, K)
    num_grad_nll = PGU.numeric_gradNLL(
        qr.nn_state.rbm_am.weights.view(-1), data, vis, EPS
    )

    counter = 0
    print("\nTesting weights...")
    print("numerical nll\talg nll")
    for i in range(len(qr.nn_state.rbm_am.weights.view(-1))):
        print(
            "{: 10.8f}\t{: 10.8f}\t\t".format(
                num_grad_nll[i], alg_grad_nll[0][i].item()
            )
        )
        counter += 1

    assertPercentDiff(
        num_grad_nll,
        alg_grad_nll[0][: len(qr.nn_state.rbm_am.weights.view(-1))],
        PDIFF,
        msg="NLL grads are not close enough for weights!",
    )

    num_grad_nll = PGU.numeric_gradNLL(qr.nn_state.rbm_am.visible_bias, data, vis, EPS)

    print("\nTesting visible bias...")
    print("numerical nll\talg nll")
    for i in range(len(qr.nn_state.rbm_am.visible_bias)):
        print(
            "{: 10.8f}\t{: 10.8f}\t\t".format(
                num_grad_nll[i], alg_grad_nll[0][counter].item()
            )
        )
        counter += 1

    assertPercentDiff(
        num_grad_nll,
        alg_grad_nll[0][len(qr.nn_state.rbm_am.weights.view(-1)) : counter],
        PDIFF,
        msg="NLL grads are not close enough for visible biases!",
    )

    num_grad_nll = PGU.numeric_gradNLL(qr.nn_state.rbm_am.hidden_bias, data, vis, EPS)

    print("\nTesting hidden bias...")
    print("numerical nll\talg nll")
    for i in range(len(qr.nn_state.rbm_am.hidden_bias)):
        print(
            "{: 10.8f}\t{: 10.8f}\t\t".format(
                num_grad_nll[i], alg_grad_nll[0][counter].item()
            )
        )
        counter += 1

    assertPercentDiff(
        num_grad_nll,
        alg_grad_nll[0][
            (
                len(qr.nn_state.rbm_am.weights.view(-1))
                + len(qr.nn_state.rbm_am.visible_bias)
            ) : counter
        ],
        PDIFF,
        msg="NLL grads are not close enough for hidden biases!",
    )


def test_posgrad_kl(positive_wavefunction_data):
    print("\nTesting KL gradients for positive-real wavefunction.")
    data, target_psi, PGU, qr, vis = positive_wavefunction_data

    alg_grad_kl = PGU.algorithmic_gradKL(target_psi, vis)
    num_grad_kl = PGU.numeric_gradKL(
        target_psi, qr.nn_state.rbm_am.weights.view(-1), vis, EPS
    )

    counter = 0
    print("\nTesting weights...")
    print("numerical kl\talg kl")
    for i in range(len(qr.nn_state.rbm_am.weights.view(-1))):
        print(
            "{: 10.8f}\t{: 10.8f}\t\t".format(
                num_grad_kl[i], alg_grad_kl[counter].item()
            )
        )
        counter += 1

    assertAlmostEqual(
        num_grad_kl,
        alg_grad_kl[: len(qr.nn_state.rbm_am.weights.view(-1))],
        TOL,
        msg="KL grads are not close enough for weights!",
    )

    num_grad_kl = PGU.numeric_gradKL(
        target_psi, qr.nn_state.rbm_am.visible_bias, vis, EPS
    )

    print("\nTesting visible bias...")
    print("numerical kl\talg kl")
    for i in range(len(qr.nn_state.rbm_am.visible_bias)):
        print(
            "{: 10.8f}\t{: 10.8f}\t\t".format(
                num_grad_kl[i], alg_grad_kl[counter].item()
            )
        )
        counter += 1

    assertAlmostEqual(
        num_grad_kl,
        alg_grad_kl[len(qr.nn_state.rbm_am.weights.view(-1)) : counter],
        TOL,
        msg="KL grads are not close enough for visible biases!",
    )

    num_grad_kl = PGU.numeric_gradKL(
        target_psi, qr.nn_state.rbm_am.hidden_bias, vis, EPS
    )

    print("\nTesting hidden bias...")
    print("numerical kl\talg kl")
    for i in range(len(qr.nn_state.rbm_am.hidden_bias)):
        print(
            "{: 10.8f}\t{: 10.8f}\t\t".format(
                num_grad_kl[i], alg_grad_kl[counter].item()
            )
        )
        counter += 1

    assertAlmostEqual(
        num_grad_kl,
        alg_grad_kl[
            (
                len(qr.nn_state.rbm_am.weights.view(-1))
                + len(qr.nn_state.rbm_am.visible_bias)
            ) : counter
        ],
        TOL,
        msg="KL grads are not close enough for hidden biases!",
    )


@pytest.fixture(scope="module", params=[
    False,
    pytest.param(True,
                 marks=[
                     pytest.mark.skipif(not torch.cuda.is_available(),
                                        reason="GPU required"),
                     pytest.mark.gpu,
                 ]),
])
def complex_wavefunction_data(request):
    with open(os.path.join(__location__, "test_data.pkl"), "rb") as fin:
        test_data = pickle.load(fin)

    qucumber.set_random_seed(SEED, cpu=True, gpu=request.param, quiet=True)

    train_bases = test_data["2qubits"]["train_bases"]
    train_samples = torch.tensor(
        test_data["2qubits"]["train_samples"], dtype=torch.double
    )

    bases_data = test_data["2qubits"]["bases"]
    target_psi_tmp = torch.tensor(
        test_data["2qubits"]["target_psi"], dtype=torch.double
    )

    num_visible = train_samples.shape[-1]
    num_hidden = num_visible

    unitary_dict = unitaries.create_dict()
    nn_state = ComplexWavefunction(
        num_visible, num_hidden, unitary_dict=unitary_dict, gpu=request.param
    )
    CGU = ComplexGradsUtils(nn_state)

    bases = CGU.transform_bases(bases_data)

    psi_dict = CGU.load_target_psi(bases, target_psi_tmp)
    vis = nn_state.generate_hilbert_space(num_visible)

    qr = QuantumReconstruction(nn_state)
    device = qr.nn_state.device
    train_samples = train_samples.to(device=device)
    vis = vis.to(device=device)

    unitary_dict = {b: v.to(device=device) for b, v in unitary_dict.items()}
    psi_dict = {b: v.to(device=device) for b, v in psi_dict.items()}

    ComplexWavefunctionFixture = namedtuple(
        "ComplexWavefunctionFixture",
        [
            "train_samples",
            "train_bases",
            "CGU",
            "bases",
            "psi_dict",
            "vis",
            "qr",
            "unitary_dict",
        ],
    )

    return ComplexWavefunctionFixture(
        train_samples=train_samples,
        train_bases=train_bases,
        CGU=CGU,
        bases=bases,
        psi_dict=psi_dict,
        vis=vis,
        qr=qr,
        unitary_dict=unitary_dict,
    )


@pytest.mark.skip(reason="doesn't give consistent results")
def test_complexgrads_nll(complex_wavefunction_data):
    print("\nTesting NLL gradients for complex wavefunction.")
    train_samples, train_bases, CGU, bases, psi_dict, vis, qr, unitary_dict = (
        complex_wavefunction_data
    )

    alg_grad_nll = CGU.algorithmic_gradNLL(qr, train_samples, train_bases, K)

    for n, net in enumerate(qr.nn_state.networks):
        counter = 0
        print("\nRBM: %s" % net)
        rbm = getattr(qr.nn_state, net)

        num_grad_nll = CGU.numeric_gradNLL(
            train_samples, train_bases, unitary_dict, rbm.weights.view(-1), vis, EPS
        )

        print("\nTesting weights...")
        print("Numerical NLL\tAlg NLL")
        for i in range(len(rbm.weights.view(-1))):
            print(
                "{: 10.8f}\t{: 10.8f}\t\t".format(
                    num_grad_nll[i], alg_grad_nll[n][i].item()
                )
            )
            counter += 1

        assertPercentDiff(
            num_grad_nll,
            alg_grad_nll[n][:counter],
            PDIFF,
            msg="NLL grads are not close enough for {} weights!".format(net),
        )

        num_grad_nll = CGU.numeric_gradNLL(
            train_samples, train_bases, unitary_dict, rbm.visible_bias, vis, EPS
        )

        print("\nTesting visible bias...")
        print("Numerical NLL\tAlg NLL")
        for i in range(len(rbm.visible_bias)):
            print(
                "{: 10.8f}\t{: 10.8f}\t\t".format(
                    num_grad_nll[i], alg_grad_nll[n][counter].item()
                )
            )
            counter += 1

        assertPercentDiff(
            num_grad_nll,
            alg_grad_nll[n][len(rbm.weights.view(-1)) : counter],
            PDIFF,
            msg="NLL grads are not close enough for {} visible biases!".format(net),
        )

        num_grad_nll = CGU.numeric_gradNLL(
            train_samples, train_bases, unitary_dict, rbm.hidden_bias, vis, EPS
        )

        print("\nTesting hidden bias...")
        print("Numerical NLL\tAlg NLL")
        for i in range(len(rbm.hidden_bias)):
            print(
                "{: 10.8f}\t{: 10.8f}\t\t".format(
                    num_grad_nll[i], alg_grad_nll[n][counter].item()
                )
            )
            counter += 1

        assertPercentDiff(
            num_grad_nll,
            alg_grad_nll[n][
                (len(rbm.weights.view(-1)) + len(rbm.visible_bias)) : counter
            ],
            PDIFF,
            msg="NLL grads are not close enough for {} hidden biases!".format(net),
        )


def test_complexgrads_kl(complex_wavefunction_data):
    print("\nTesting KL gradients for complex wavefunction.")
    train_samples, train_bases, CGU, bases, psi_dict, vis, qr, unitary_dict = (
        complex_wavefunction_data
    )

    alg_grad_kl = CGU.algorithmic_gradKL(psi_dict, vis, unitary_dict, bases)

    for n, net in enumerate(qr.nn_state.networks):
        counter = 0
        print("\nRBM: %s" % net)
        rbm = getattr(qr.nn_state, net)

        num_grad_kl = CGU.numeric_gradKL(
            rbm.weights.view(-1), psi_dict, vis, unitary_dict, bases, EPS
        )

        print("\nTesting weights...")
        print("Numerical KL\tAlg KL")
        for i in range(len(rbm.weights.view(-1))):
            print(
                "{: 10.8f}\t{: 10.8f}\t\t".format(
                    num_grad_kl[i], alg_grad_kl[n][counter].item()
                )
            )
            counter += 1

        assertAlmostEqual(
            num_grad_kl,
            alg_grad_kl[n][:counter],
            TOL,
            msg="KL grads are not close enough for {} weights!".format(net),
        )

        num_grad_kl = CGU.numeric_gradKL(
            rbm.visible_bias, psi_dict, vis, unitary_dict, bases, EPS
        )

        print("\nTesting visible bias...")
        print("Numerical KL\tAlg KL")
        for i in range(len(rbm.visible_bias)):
            print(
                "{: 10.8f}\t{: 10.8f}\t\t".format(
                    num_grad_kl[i], alg_grad_kl[n][counter].item()
                )
            )
            counter += 1

        assertAlmostEqual(
            num_grad_kl,
            alg_grad_kl[n][len(rbm.weights.view(-1)) : counter],
            TOL,
            msg="KL grads are not close enough for {} visible biases!".format(net),
        )

        num_grad_kl = CGU.numeric_gradKL(
            rbm.hidden_bias, psi_dict, vis, unitary_dict, bases, EPS
        )

        print("\nTesting hidden bias...")
        print("Numerical KL\tAlg KL")
        for i in range(len(rbm.hidden_bias)):
            print(
                "{: 10.8f}\t{: 10.8f}\t\t".format(
                    num_grad_kl[i], alg_grad_kl[n][counter].item()
                )
            )
            counter += 1

        assertAlmostEqual(
            num_grad_kl,
            alg_grad_kl[n][
                (len(rbm.weights.view(-1)) + len(rbm.visible_bias)) : counter
            ],
            TOL,
            msg="KL grads are not close enough for {} hidden biases!".format(net),
        )
