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
        zeta_v = rbm.sample_v_given_h(zeta_h.clone())

        accept = bool(
            (rbm.prob_v_given_h(eta_h, v=zeta_v) / rbm.prob_v_given_h(zeta_h, v=zeta_v))
            .clamp_(min=0, max=1)
            .bernoulli()
            .item()
        )

        if accept:
            zeta_h = rbm.sample_h_given_v(zeta_v)
            return (zeta_v, zeta_h, zeta_v, zeta_h, True)
        else:
            reject_zeta = True
            reject_eta = True
            U_h = torch.rand_like(zeta_h)

            while reject_zeta or reject_eta:
                U_v = torch.rand_like(zeta_v)

                if reject_zeta:
                    zeta_v = rbm._sample_v_given_h_from_u(zeta_h, U_v)

                    reject_zeta = bool(
                        (
                            rbm.prob_v_given_h(eta_h, v=zeta_v)
                            / rbm.prob_v_given_h(zeta_h, v=zeta_v)
                        )
                        .clamp_(min=0, max=1)
                        .bernoulli()
                        .item()
                    )

                if reject_eta:
                    eta_v = rbm._sample_v_given_h_from_u(eta_h, U_v)

                    reject_eta = bool(
                        (
                            rbm.prob_v_given_h(zeta_h, v=eta_v)
                            / rbm.prob_v_given_h(eta_h, v=eta_v)
                        )
                        .clamp_(min=0, max=1)
                        .bernoulli()
                        .item()
                    )

            zeta_h = rbm._sample_h_given_v_from_u(zeta_v, U_h)
            eta_h = rbm._sample_h_given_v_from_u(eta_v, U_h)

            return (zeta_v, zeta_h, eta_v, eta_h, False)

    def __call__(self, nn_state, samples):
        rbm = nn_state.rbm_am

        acc_grad = torch.zeros(
            samples.shape[0], rbm.num_pars, dtype=torch.double, device=rbm.device
        )

        eta_v = samples
        eta_h = rbm.sample_h_given_v(eta_v)

        zeta_v = rbm.sample_v_given_h(eta_h)
        zeta_h = rbm.sample_h_given_v(eta_v)

        if self.k == 1:
            acc_grad += rbm.effective_energy_gradient(zeta_v, reduce=False)

        for i in range(samples.shape[0]):
            for t in range(2, self.T_max):
                (
                    zeta_v[i, :],
                    zeta_h[i, :],
                    eta_v[i, :],
                    eta_h[i, :],
                    breakout,
                ) = self._coupling_step(
                    rbm, zeta_v[i, :], zeta_h[i, :], eta_v[i, :], eta_h[i, :]
                )

                if t >= self.k:
                    acc_grad[i, :] += rbm.effective_energy_gradient(zeta_v[i, :])

                if t > self.k:
                    acc_grad[i, :] -= rbm.effective_energy_gradient(eta_v[i, :])

                if breakout and t > self.k:
                    break

        return torch.mean(acc_grad, dim=0)
