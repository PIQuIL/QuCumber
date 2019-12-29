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

import pytest
import numpy as np
import torch
from torch.nn.utils import parameters_to_vector

import qucumber
import qucumber.utils.data as data
import qucumber.utils.training_statistics as ts
from qucumber.callbacks import LambdaCallback
from qucumber.nn_states import ComplexWaveFunction, PositiveWaveFunction, DensityMatrix

from test_models_misc import all_state_types
from test_grads import devices, quantum_state_types

SEED = 1234


@pytest.mark.gpu
def test_complex_warn_on_gpu():
    with pytest.warns(ResourceWarning):
        ComplexWaveFunction(10, gpu=True)


@pytest.mark.parametrize("gpu", devices)
@pytest.mark.parametrize("state_type", all_state_types)
def test_neural_state(gpu, state_type):
    qucumber.set_random_seed(SEED, cpu=True, gpu=gpu, quiet=True)
    np.random.seed(SEED)

    nn_state = state_type(10, gpu=gpu)

    old_params = torch.cat(
        [
            parameters_to_vector(getattr(nn_state, net).parameters())
            for net in nn_state.networks
        ]
    )

    data = torch.ones(100, 10)

    # generate sample bases randomly, with probability 0.9 of being 'Z', otherwise 'X'
    bases = np.where(np.random.binomial(1, 0.9, size=(100, 10)), "Z", "X")

    nn_state.fit(data, epochs=1, pos_batch_size=10, input_bases=bases)

    new_params = torch.cat(
        [
            parameters_to_vector(getattr(nn_state, net).parameters())
            for net in nn_state.networks
        ]
    )

    msg = f"{state_type.__name__}'s parameters did not change!"
    assert not torch.equal(old_params, new_params), msg


def test_complex_training_without_bases_fail():
    qucumber.set_random_seed(SEED, cpu=True, gpu=False, quiet=True)

    nn_state = ComplexWaveFunction(10, gpu=False)

    data = torch.ones(100, 10)

    msg = "Training ComplexWaveFunction without providing bases should fail!"
    with pytest.raises(ValueError):
        nn_state.fit(data, epochs=1, pos_batch_size=10, input_bases=None)
        pytest.fail(msg)


@pytest.mark.parametrize("gpu", devices)
def test_stop_training(gpu):
    qucumber.set_random_seed(SEED, cpu=True, gpu=gpu, quiet=True)

    nn_state = PositiveWaveFunction(10, gpu=gpu)

    old_params = parameters_to_vector(nn_state.rbm_am.parameters())
    data = torch.ones(100, 10)

    nn_state.stop_training = True
    nn_state.fit(data)

    new_params = parameters_to_vector(nn_state.rbm_am.parameters())

    msg = "stop_training didn't work!"
    assert torch.equal(old_params, new_params), msg


def set_stop_training(nn_state):
    nn_state.stop_training = True


@pytest.mark.parametrize("gpu", devices)
def test_stop_training_in_batch(gpu):
    qucumber.set_random_seed(SEED, cpu=True, gpu=gpu, quiet=True)

    nn_state = PositiveWaveFunction(10, gpu=gpu)

    data = torch.ones(100, 10)

    callbacks = [
        LambdaCallback(on_batch_end=lambda nn_state, ep, b: set_stop_training(nn_state))
    ]

    nn_state.fit(data, callbacks=callbacks)

    msg = "stop_training wasn't set!"
    assert nn_state.stop_training, msg


@pytest.mark.parametrize("gpu", devices)
def test_stop_training_in_epoch(gpu):
    qucumber.set_random_seed(SEED, cpu=True, gpu=gpu, quiet=True)

    nn_state = PositiveWaveFunction(10, gpu=gpu)

    data = torch.ones(100, 10)

    callbacks = [
        LambdaCallback(on_epoch_end=lambda nn_state, ep: set_stop_training(nn_state))
    ]

    nn_state.fit(data, callbacks=callbacks)

    msg = "stop_training wasn't set!"
    assert nn_state.stop_training, msg


@pytest.fixture(scope="module", params=quantum_state_types)
def quantum_state_training_data(request):
    nn_state_type = request.param

    if nn_state_type == "positive":

        root = os.path.join(
            request.fspath.dirname,
            "..",
            "examples",
            "Tutorial1_TrainPosRealWaveFunction",
        )

        train_samples, target = data.load_data(
            tr_samples_path=os.path.join(root, "tfim1d_data.txt"),
            tr_psi_path=os.path.join(root, "tfim1d_psi.txt"),
        )
        train_bases, bases = None, None

        nn_state = PositiveWaveFunction(num_visible=train_samples.shape[-1], gpu=False)

        batch_size, num_chains = 100, 200
        fid_target, kl_target = 0.85, 0.29

        reinit_params_fn = initialize_posreal_params

    elif nn_state_type == "complex":

        root = os.path.join(
            request.fspath.dirname,
            "..",
            "examples",
            "Tutorial2_TrainComplexWaveFunction",
        )

        train_samples, target, train_bases, bases = data.load_data(
            tr_samples_path=os.path.join(root, "qubits_train.txt"),
            tr_psi_path=os.path.join(root, "qubits_psi.txt"),
            tr_bases_path=os.path.join(root, "qubits_train_bases.txt"),
            bases_path=os.path.join(root, "qubits_bases.txt"),
        )

        nn_state = ComplexWaveFunction(num_visible=train_samples.shape[-1], gpu=False)

        batch_size, num_chains = 50, 10
        fid_target, kl_target = 0.38, 0.33

        reinit_params_fn = initialize_complex_params

    elif nn_state_type == "density_matrix":

        root = os.path.join(
            request.fspath.dirname, "..", "examples", "Tutorial3_TrainDensityMatrix"
        )

        train_samples, target, train_bases, bases = data.load_data_DM(
            tr_samples_path=os.path.join(root, "N2_W_state_100_samples_data.txt"),
            tr_mtx_real_path=os.path.join(root, "N2_W_state_target_real.txt"),
            tr_mtx_imag_path=os.path.join(root, "N2_W_state_target_imag.txt"),
            tr_bases_path=os.path.join(root, "N2_W_state_100_samples_bases.txt"),
            bases_path=os.path.join(root, "N2_IC_bases.txt"),
        )

        nn_state = DensityMatrix(num_visible=train_samples.shape[-1], gpu=False)

        batch_size, num_chains = 100, 10
        fid_target, kl_target = 0.45, 0.42

        def reinit_params_fn(req, nn_state):
            nn_state.reinitialize_parameters()

    else:
        raise ValueError(
            f"invalid test config: {nn_state_type} is not a valid quantum state type"
        )

    return {
        "nn_state": nn_state,
        "data": train_samples,
        "input_bases": train_bases,
        "target": target,
        "bases": bases,
        "epochs": 5,
        "pos_batch_size": batch_size,
        "neg_batch_size": num_chains,
        "k": 10,
        "lr": 0.1,
        "space": nn_state.generate_hilbert_space(),
        "fid_target": fid_target,
        "kl_target": kl_target,
        "reinit_params_fn": reinit_params_fn,
    }


@pytest.mark.slow
def test_training(request, quantum_state_training_data):
    qucumber.set_random_seed(SEED, cpu=True, gpu=False, quiet=True)

    fidelities = []
    KLs = []

    qstd = quantum_state_training_data

    print("Training 10 times and checking fidelity and KL at 5 epochs...\n")
    for i in range(10):
        print(f"Iteration: {i + 1}")

        qstd["reinit_params_fn"](request, qstd["nn_state"])

        qstd["nn_state"].fit(time=True, progbar=False, **qstd)

        fidelities.append(ts.fidelity(**qstd))
        KLs.append(ts.KL(**qstd))
        print(f"Fidelity: {fidelities[-1]}; KL: {KLs[-1]}.")

    print("\nStatistics")
    print("----------")
    print(
        "Fidelity: ",
        np.average(fidelities),
        "+/-",
        np.std(fidelities) / np.sqrt(len(fidelities)),
        "\n",
    )
    print("KL: ", np.average(KLs), "+/-", np.std(KLs) / np.sqrt(len(KLs)), "\n")

    assert abs(np.average(fidelities) - qstd["fid_target"]) < 0.02
    assert abs(np.average(KLs) - qstd["kl_target"]) < 0.02
    assert (np.std(fidelities) / np.sqrt(len(fidelities))) < 0.02
    assert (np.std(KLs) / np.sqrt(len(KLs))) < 0.02


def initialize_posreal_params(request, nn_state):
    with open(
        os.path.join(
            request.fspath.dirname, "data", "test_training_init_pos_params.npz"
        ),
        "rb",
    ) as f:
        x = np.load(f)
        for p in x.files:
            getattr(nn_state.rbm_am, p).data = torch.tensor(x[p]).to(
                getattr(nn_state.rbm_am, p)
            )


def initialize_complex_params(request, nn_state):
    with open(
        os.path.join(
            request.fspath.dirname, "data", "test_training_init_complex_params.npz"
        ),
        "rb",
    ) as f:
        x = np.load(f)
        for p in x.files:
            if p.startswith("am"):
                rbm = nn_state.rbm_am
            elif p.startswith("ph"):
                rbm = nn_state.rbm_ph

            q = p.split("_", maxsplit=1)[-1]
            getattr(rbm, q).data = torch.tensor(x[p]).to(getattr(rbm, q))
