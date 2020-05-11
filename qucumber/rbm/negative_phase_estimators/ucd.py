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

    def _select_states(self, states, indices):
        return tuple(x[indices, ...] for x in states)

    def _assign_states_(self, src, dest, indices):
        for s, d in zip(src, dest):
            d[indices, ...] = s[indices, ...].clone()

    def _coupling_step(self, rbm, zeta, eta):
        zeta = rbm.sample_full_state_given_h(zeta[1])

        accept = (
            (
                rbm.prob_v_given_h(*eta[1:], v=zeta[0])
                / rbm.prob_v_given_h(*zeta[1:], v=zeta[0])
            )
            .clamp_(min=0, max=1)
            .bernoulli()
        )

        accepted_idx = accept == 1

        if torch.all(accepted_idx):
            zeta = rbm.sample_full_state_given_v(zeta[0])
            return (zeta, tuple(z.clone() for z in zeta), True)

        rejected_idx = accept == 0

        zeta[1][accepted_idx, :] = rbm.sample_h_given_v(zeta[0][accepted_idx, :])

        self._assign_states_(zeta, eta, accepted_idx)

        rejected_zeta = rejected_idx.clone()
        rejected_eta = rejected_idx.clone()
        U_h = torch.rand_like(zeta[1][rejected_idx, :])

        bool_type = rejected_zeta.dtype
        rejected_zeta_any = rejected_zeta.any()
        rejected_eta_any = rejected_eta.any()

        while rejected_zeta_any or rejected_eta_any:
            # we're generating too many random numbers here, should optimize this eventually
            U_v = torch.rand_like(zeta[0])

            if rejected_zeta_any:
                zeta[0][rejected_zeta, :] = rbm._sample_v_given_h_from_u(
                    U_v[rejected_zeta, :], zeta[1][rejected_zeta, :],
                )

                rejected_zeta[rejected_zeta] = (
                    (
                        rbm.prob_v_given_h(
                            eta[1][rejected_zeta, :], v=zeta[0][rejected_zeta, :]
                        )
                        / rbm.prob_v_given_h(
                            zeta[1][rejected_zeta, :], v=zeta[0][rejected_zeta, :]
                        )
                    )
                    .clamp_(min=0, max=1)
                    .bernoulli()
                    .to(dtype=bool_type)
                )

                rejected_zeta_any = rejected_zeta.any()

            if rejected_eta_any:
                eta[0][rejected_eta, :] = rbm._sample_v_given_h_from_u(
                    U_v[rejected_eta, :], eta[1][rejected_eta, :]
                )

                rejected_eta[rejected_eta] = (
                    (
                        rbm.prob_v_given_h(
                            zeta[1][rejected_eta, :], v=eta[0][rejected_eta, :]
                        )
                        / rbm.prob_v_given_h(
                            eta[1][rejected_eta, :], v=eta[0][rejected_eta, :]
                        )
                    )
                    .clamp_(min=0, max=1)
                    .bernoulli()
                    .to(dtype=bool_type)
                )

                rejected_eta_any = rejected_eta.any()

        zeta[1][rejected_idx, :] = rbm._sample_h_given_v_from_u(
            U_h, zeta[0][rejected_idx, :]
        )
        eta[1][rejected_idx, :] = rbm._sample_h_given_v_from_u(
            U_h, eta[0][rejected_idx, :]
        )

        return (zeta, eta, False)

    def __call__(self, nn_state, samples):
        rbm = nn_state.rbm_am

        acc_grad = torch.zeros(rbm.num_pars, dtype=torch.double, device=rbm.device)

        eta = rbm.sample_full_state_given_v(samples.clone())
        zeta = rbm._propagate_state(eta)

        if self.k == 1:
            acc_grad += rbm.effective_energy_gradient(zeta[0])

        for t in range(2, self.T_max):
            zeta, eta, breakout = self._coupling_step(rbm, zeta, eta)

            if t >= self.k:
                acc_grad += rbm.effective_energy_gradient(zeta[0])

            if t > self.k:
                acc_grad -= rbm.effective_energy_gradient(eta[0])

            if breakout and t > self.k:
                break

        grad_model = acc_grad / samples.shape[0]
        return grad_model
