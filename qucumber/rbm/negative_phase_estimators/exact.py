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

import torch

from .negative_phase_estimator import NegativePhaseEstimatorBase


class ExactEstimator(NegativePhaseEstimatorBase):
    def __init__(self):
        super().__init__()

    def __call__(self, nn_state, space):
        probs = nn_state.probability(space, Z=1.0)  # unnormalized probs
        Z = probs.sum()
        probs /= Z

        all_grads = nn_state.rbm_am.effective_energy_gradient(space, reduce=False)
        return torch.mv(
            all_grads.t(), probs
        )  # average the gradients, weighted by probs
