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
            d[indices, ...] = s[indices, ...]

    @staticmethod
    def _coupling_step(rbm, zeta_v, zeta_h, eta_v, eta_h):
        zeta_v = rbm.sample_v_given_h(zeta_h, out=zeta_v)

        accept = (
            (rbm.prob_v_given_h(eta_h, v=zeta_v) / rbm.prob_v_given_h(zeta_h, v=zeta_v))
            .clamp_(min=0, max=1)
            .bernoulli()
        )

        accepted_idx = accept == 1

        if torch.all(accepted_idx):
            zeta_h = rbm.sample_h_given_v(zeta_v, out=zeta_h)
            return (zeta_v, zeta_h, zeta_v.clone(), zeta_h.clone(), True)

        rejected_idx = accept == 0

        zeta_h[accepted_idx, :] = rbm.sample_h_given_v(zeta_v[accepted_idx, :])
        eta_v[accepted_idx, :] = zeta_v[accepted_idx, :].clone()
        eta_h[accepted_idx, :] = zeta_h[accepted_idx, :].clone()

        rejected_zeta = rejected_idx.clone()
        rejected_eta = rejected_idx.clone()
        U_h = torch.rand_like(zeta_h[rejected_idx, :])

        bool_type = rejected_zeta.dtype
        rejected_zeta_any = rejected_zeta.any()
        rejected_eta_any = rejected_eta.any()

        while rejected_zeta_any or rejected_eta_any:
            # we're generating too many random numbers here, should optimize this eventually
            U_v = torch.rand_like(zeta_v)

            if rejected_zeta_any:
                zeta_v[rejected_zeta, :] = rbm._sample_v_given_h_from_u(
                    zeta_h[rejected_zeta, :], U_v[rejected_zeta, :]
                )

                rejected_zeta[rejected_zeta] = (
                    (
                        rbm.prob_v_given_h(
                            eta_h[rejected_zeta, :], v=zeta_v[rejected_zeta, :]
                        )
                        / rbm.prob_v_given_h(
                            zeta_h[rejected_zeta, :], v=zeta_v[rejected_zeta, :]
                        )
                    )
                    .clamp_(min=0, max=1)
                    .bernoulli()
                    .to(dtype=bool_type)
                )

                rejected_zeta_any = rejected_zeta.any()

            if rejected_eta_any:
                eta_v[rejected_eta, :] = rbm._sample_v_given_h_from_u(
                    eta_h[rejected_eta, :], U_v[rejected_eta, :]
                )

                rejected_eta[rejected_eta] = (
                    (
                        rbm.prob_v_given_h(
                            zeta_h[rejected_eta, :], v=eta_v[rejected_eta, :]
                        )
                        / rbm.prob_v_given_h(
                            eta_h[rejected_eta, :], v=eta_v[rejected_eta, :]
                        )
                    )
                    .clamp_(min=0, max=1)
                    .bernoulli()
                    .to(dtype=bool_type)
                )

                rejected_eta_any = rejected_eta.any()

        zeta_h[rejected_idx, :] = rbm._sample_h_given_v_from_u(
            zeta_v[rejected_idx, :], U_h
        )
        eta_h[rejected_idx, :] = rbm._sample_h_given_v_from_u(
            eta_v[rejected_idx, :], U_h
        )

        return (zeta_v, zeta_h, eta_v, eta_h, False)

    def __call__(self, nn_state, samples):
        rbm = nn_state.rbm_am

        acc_grad = torch.zeros(rbm.num_pars, dtype=torch.double, device=rbm.device)

        eta_v = samples.clone()
        eta_h = rbm.sample_h_given_v(eta_v)

        zeta_v = rbm.sample_v_given_h(eta_h)
        zeta_h = rbm.sample_h_given_v(eta_v)

        if self.k == 1:
            acc_grad += rbm.effective_energy_gradient(zeta_v)

        for t in range(2, self.T_max):
            zeta_v, zeta_h, eta_v, eta_h, breakout = self._coupling_step(
                rbm, zeta_v, zeta_h, eta_v, eta_h
            )

            if t >= self.k:
                acc_grad += rbm.effective_energy_gradient(zeta_v)

            if t > self.k:
                acc_grad -= rbm.effective_energy_gradient(eta_v)

            if breakout and t > self.k:
                break

        grad_model = acc_grad / eta_v.shape[0]
        return grad_model
