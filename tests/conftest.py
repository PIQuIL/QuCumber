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


import pytest
import torch

from qucumber.nn_states import PositiveWaveFunction, ComplexWaveFunction, DensityMatrix

torch.set_printoptions(precision=10)

all_state_types = [PositiveWaveFunction, ComplexWaveFunction, DensityMatrix]

gpu_availability = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU required"
)
devices = [
    pytest.param(False, id="cpu"),
    pytest.param(True, id="gpu", marks=[gpu_availability, pytest.mark.gpu]),
]


@pytest.fixture(scope="module", params=devices)
def quantum_state_device(request):
    return request.param


TOL = torch.tensor(2e-8, dtype=torch.double)


def assertAlmostEqual(a, b, tol=TOL, msg=None):
    a = a.to(device=torch.device("cpu"))
    b = b.to(device=torch.device("cpu"))
    diff = torch.abs(a - b)
    result = torch.ge(tol * torch.ones_like(diff), diff)
    assert torch.all(result).squeeze().item(), msg
