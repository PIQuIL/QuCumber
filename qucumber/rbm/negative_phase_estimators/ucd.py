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
        self.all_t = []
        self.all_rejection_steps = []

    def _select_states(self, states, indices):
        return tuple(x[indices, ...] for x in states)

    def _assign_states_(self, src, dest, indices):
        for s, d in zip(src, dest):
            d[indices, ...] = s[indices, ...].clone()

    @staticmethod
    def cond_density_ratio(v, p, q):
        temp = (v * (p / q)) + (1 - v) * ((1 - p) / (1 - q))
        return torch.prod(temp, dim=-1)

    def _rejection_step(
        self,
        zeta,
        eta,
        pv_zeta,
        pv_eta,
        rejected_zeta,
        rejected_eta,
        rejected_zeta_any,
        rejected_eta_any,
    ):
        U_v = torch.rand_like(zeta[0])
        if rejected_zeta_any:
            temp = (
                pv_zeta[rejected_zeta, :]
                .gt(U_v[rejected_zeta, :])
                .to(dtype=zeta[0].dtype)
                .contiguous()
            )
            zeta[0][rejected_zeta, :] = temp

            rejected_zeta[rejected_zeta] = (
                self.cond_density_ratio(
                    temp, pv_eta[rejected_zeta, :], pv_zeta[rejected_zeta, :]
                )
                .gt(torch.rand(rejected_zeta.sum(), dtype=zeta[0].dtype))
                .to(dtype=rejected_zeta.dtype)
            )

            rejected_zeta_any = rejected_zeta.any()

        if rejected_eta_any:
            temp = (
                pv_eta[rejected_eta, :]
                .gt(U_v[rejected_eta, :])
                .to(dtype=eta[0].dtype)
                .contiguous()
            )
            eta[0][rejected_eta, :] = temp

            rejected_eta[rejected_eta] = (
                self.cond_density_ratio(
                    temp, pv_zeta[rejected_eta, :], pv_eta[rejected_eta, :]
                )
                .gt(torch.rand(rejected_eta.sum(), dtype=eta[0].dtype))
                .to(dtype=rejected_eta.dtype)
            )

            rejected_eta_any = rejected_eta.any()

        return (rejected_zeta_any, rejected_eta_any)

    def _coupling_step(self, rbm, zeta, eta):
        zeta = rbm.sample_full_state_given_h(zeta[1])

        pv_zeta = rbm.prob_v_given_h(zeta[1])
        pv_eta = rbm.prob_v_given_h(eta[1])

        accept = self.cond_density_ratio(zeta[0], pv_eta, pv_zeta).gt(
            torch.rand(zeta[0].shape[0], dtype=zeta[0].dtype)
        )

        accepted_idx = accept == 1

        if torch.all(accepted_idx):
            zeta = rbm.sample_full_state_given_v(zeta[0])
            return (zeta, tuple(z.clone() for z in zeta), True, None, None)

        rejected_idx = accept == 0

        zeta[1][accepted_idx, :] = rbm.sample_h_given_v(zeta[0][accepted_idx, :])

        eta[0][accepted_idx, :] = zeta[0][accepted_idx, :]
        eta[1][accepted_idx, :] = zeta[1][accepted_idx, :]

        rejected_zeta = rejected_idx.clone()
        rejected_eta = rejected_idx.clone()
        U_h = torch.rand_like(zeta[1][rejected_idx, :])

        rejected_zeta_any = rejected_zeta.any()
        rejected_eta_any = rejected_eta.any()

        rejection_steps = 0
        while rejected_zeta_any or rejected_eta_any:
            rejection_steps += rejected_eta | rejected_zeta

            rejected_zeta_any, rejected_eta_any = self._rejection_step(
                zeta,
                eta,
                pv_zeta,
                pv_eta,
                rejected_zeta,
                rejected_eta,
                rejected_zeta_any,
                rejected_eta_any,
            )

        zeta[1][rejected_idx, :] = rbm._sample_h_given_v_from_u(
            U_h, zeta[0][rejected_idx, :]
        )
        eta[1][rejected_idx, :] = rbm._sample_h_given_v_from_u(
            U_h, eta[0][rejected_idx, :]
        )

        return (zeta, eta, False, rejected_idx, rejection_steps)

    def __call__(self, nn_state, samples):
        rbm = nn_state.rbm_am

        with torch.no_grad():
            eta = rbm.sample_full_state_given_v(samples.clone())
            zeta = rbm._propagate_state(eta)

            for t in range(1, self.k):
                zeta, eta, _, _, _ = self._coupling_step(rbm, zeta, eta)

            acc_grad = rbm.effective_energy_gradient(zeta[0])

            for t in range(self.k, self.T_max):
                (
                    zeta,
                    eta,
                    breakout,
                    rejected_idx,
                    rejection_steps,
                ) = self._coupling_step(rbm, zeta, eta)

                if breakout:
                    break
                elif rejected_idx is not None:  # drop chains that have converged
                    zeta = (
                        zeta[0][rejected_idx, :].contiguous(),
                        zeta[1][rejected_idx, :].contiguous(),
                    )
                    eta = (
                        eta[0][rejected_idx, :].contiguous(),
                        eta[1][rejected_idx, :].contiguous(),
                    )

                acc_grad += rbm.effective_energy_gradient(
                    zeta[0]
                ) - rbm.effective_energy_gradient(eta[0])

                self.all_rejection_steps.append(rejection_steps)

            self.all_t.append(t)

            grad_model = acc_grad / samples.shape[0]
            return grad_model
