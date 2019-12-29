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


from math import isclose

import pytest
import torch

from qucumber import observables, set_random_seed
from qucumber.nn_states import WaveFunctionBase, NeuralStateBase
from qucumber.utils import cplx, auto_unsqueeze_args

from conftest import all_state_types


SEED = 1234


class MockWaveFunction(WaveFunctionBase):
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

    @auto_unsqueeze_args()
    def phase(self, v):
        return torch.zeros(v.shape[0], dtype=torch.double, device=self.device)

    def amplitude(self, v):
        return torch.ones(v.shape[0], dtype=torch.double, device=self.device)

    def psi(self, v):
        return cplx.make_complex(self.amplitude(v))


class MockDensityMatrix(NeuralStateBase):
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

    def probability(self, v):
        return torch.ones(v.shape[0], dtype=torch.double, device=self.device)

    def importance_sampling_numerator(self, iter_sample, drawn_sample):
        out = torch.zeros(iter_sample.shape[0], dtype=torch.double, device=self.device)
        idx = torch.all(iter_sample == drawn_sample, dim=-1)
        out[idx] = self.probability(iter_sample[idx, :])
        return cplx.make_complex(out)

    def importance_sampling_denominator(self, drawn_sample):
        return cplx.make_complex(self.probability(drawn_sample))


@pytest.fixture(scope="module", params=[MockWaveFunction, MockDensityMatrix])
def mock_state_samples(request):
    set_random_seed(SEED, cpu=True, gpu=False, quiet=True)
    state_type = request.param

    pauli_expectations = {
        observables.SigmaX: (1.0, 1.0),
        observables.SigmaY: (0.0, 0.0),
        observables.SigmaZ: (0.0, 0.5),
    }

    if state_type == MockDensityMatrix:
        pauli_expectations[observables.SigmaX] = (0.0, 0.0)

    nn_state = state_type(2)
    test_sample = nn_state.sample(num_samples=100000)
    return nn_state, test_sample, pauli_expectations


def test_spinflip(mock_state_samples):
    _, test_sample, _ = mock_state_samples
    samples = test_sample.clone()
    observables.pauli.flip_spin(1, samples)  # flip spin
    observables.pauli.flip_spin(1, samples)  # flip it back
    assert torch.equal(samples, test_sample)


paulis = [
    pytest.param(observables.SigmaX, id="X"),
    pytest.param(observables.SigmaY, id="Y"),
    pytest.param(observables.SigmaZ, id="Z"),
]


@pytest.mark.parametrize("pauli", paulis)
@pytest.mark.parametrize(
    "absolute", [pytest.param(True, id="absolute"), pytest.param(False, id="signed")]
)
def test_pauli(mock_state_samples, pauli, absolute):
    nn_state, test_sample, pauli_expectations = mock_state_samples
    expectations = pauli_expectations[pauli]

    if absolute:
        mag = expectations[1]
        prefix = "(absolute) "
    else:
        mag = expectations[0]
        prefix = ""

    obs = pauli(absolute=absolute)
    measure_O = float(obs.apply(nn_state, test_sample).mean())

    assert isclose(
        mag, measure_O, abs_tol=1e-2
    ), "measure {}-magnetization failed".format(prefix + obs.symbol)


@pytest.fixture(scope="module", params=[2, 7])
def nn_state_num_visible(request):
    return request.param


@pytest.fixture(scope="module", params=all_state_types)
def nn_state(request, quantum_state_device, nn_state_num_visible):
    set_random_seed(SEED, cpu=True, gpu=quantum_state_device, quiet=True)
    state_type = request.param

    return state_type(nn_state_num_visible, gpu=quantum_state_device)


observable_constructors = [
    observables.SWAP,
    observables.SigmaX,
    observables.SigmaY,
    observables.SigmaZ,
    observables.NeighbourInteraction,
]

observable_constructors = [(cls.__name__, cls) for cls in observable_constructors]

observable_constructors.append(
    (
        "AlgebraicallyCombined",
        lambda: -observables.NeighbourInteraction() - (3 * observables.SigmaX()) + 1,
    )
)


@pytest.fixture(scope="module", params=observable_constructors, ids=lambda p: p[0])
def nn_state_observable_fixture(request, nn_state):
    obs_name, obs_constructor = request.param

    if obs_name == "SWAP":
        obs = obs_constructor(list(range(nn_state.num_visible // 2)))
    else:
        obs = obs_constructor()

    return nn_state, obs


@pytest.fixture(scope="module")
def nn_state_observable_samples(request, nn_state_observable_fixture):
    nn_state, obs = nn_state_observable_fixture

    test_sample = nn_state.sample(k=1, num_samples=1000)
    return nn_state, obs, test_sample


def test_sanity_check_observable_apply(request, nn_state_observable_samples):
    nn_state, obs, test_sample = nn_state_observable_samples
    applied = obs.apply(nn_state, test_sample)

    msg = "Output of `apply` should have the same number of samples as `test_sample`!"
    assert applied.shape[0] == test_sample.shape[0], msg


def test_sanity_check_observable_statistics_from_samples(
    request, nn_state_observable_samples
):
    nn_state, obs, test_sample = nn_state_observable_samples
    obs.statistics_from_samples(nn_state, test_sample)


def test_sanity_check_observable_statistics(request, nn_state_observable_fixture):
    nn_state, obs = nn_state_observable_fixture
    obs.statistics(nn_state, 1000)
