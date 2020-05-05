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


class UCDEstimator(NegativePhaseEstimatorBase):
    def __init__(self, k=1, T_max=100):
        super().__init__()
        if k < 1:
            raise ValueError("k must be >= 1!")
        self.k = k
        self.T_max = T_max

    def __call__(self, nn_state, samples):
        rbm = nn_state.rbm_am

        zeta = rbm.sample_full_state_given_v(samples)
        eta = tuple(x.clone() for x in zeta)
        zeta = rbm._propagate_state_(zeta)

        acc_grad = torch.zeros(samples.shape[0], rbm.num_pars).to(
            dtype=torch.double, device=rbm.device
        )

        for t in range(2, self.T_max + 1):
            zeta_proposal = rbm._propagate_state(zeta)

            U = torch.rand(samples.shape[0])

            numer = rbm.prob_v_given_h(*eta[1:], v=zeta_proposal[0])
            denom = rbm.prob_v_given_h(*zeta[1:], v=zeta_proposal[0])
            if torch.all(U <= (numer / denom)) or t == self.T_max:
                zeta = eta = zeta_proposal
                break
            else:
                while True:
                    eta_proposal = rbm._propagate_state(eta)
                    U = torch.rand(samples.shape[0])
                    numer = rbm.prob_v_given_h(*zeta[1:], v=eta_proposal[0])
                    denom = rbm.prob_v_given_h(*eta[1:], v=eta_proposal[0])

                    if torch.all(U > (numer / denom)):
                        break

                zeta = zeta_proposal
                eta = eta_proposal

            if t >= self.k:
                acc_grad += rbm.effective_energy_gradient(zeta[0], reduce=False)
                if t > self.k:
                    acc_grad -= rbm.effective_energy_gradient(eta[0], reduce=False)

        grad_model = acc_grad
        return grad_model / float(samples.shape[0])
