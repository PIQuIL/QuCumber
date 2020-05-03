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

from .negative_phase_estimator import NegativePhaseEstimatorBase


class PCDEstimator(NegativePhaseEstimatorBase):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.state = None

    def __call__(self, nn_state, samples):
        if self.state is None:
            self.state = samples.clone()

        nn_state.rbm_am.gibbs_steps(self.k, self.state, overwrite=True)

        grad_model = nn_state.rbm_am.effective_energy_gradient(self.state)
        return grad_model / float(self.state.shape[0])
