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

from qucumber.utils import cplx
from qucumber.utils import training_statistics as ts
from qucumber.utils.unitaries import rotate_psi_inner_prod, rotate_rho_probs, rotate_rho


class PosGradsUtils:
    def __init__(self, nn_state):
        self.nn_state = nn_state

    def compute_numerical_KL(self, target, space, all_bases=None):
        return ts.KL(self.nn_state, target, space, bases=all_bases)

    def compute_numerical_KL_norot(self, target, space):
        return ts.KL(self.nn_state, target, space)

    def compute_numerical_NLL(self, data_samples, space, data_bases=None):
        return ts.NLL(self.nn_state, data_samples, space, sample_bases=data_bases)

    def algorithmic_gradKL(self, target, space, **kwargs):
        Z = self.nn_state.normalization(space)
        grad_KL = torch.zeros(
            self.nn_state.rbm_am.num_pars,
            dtype=torch.double,
            device=self.nn_state.device,
        )
        for i in range(len(space)):
            sample_grad = self.nn_state.gradient(space[i])[0]
            grad_KL += (cplx.absolute_value(target[i]) ** 2) * sample_grad
            grad_KL -= self.nn_state.probability(space[i], Z) * sample_grad
        return [grad_KL]

    def algorithmic_gradKL_norot(self, target, space, **kwargs):
        return self.algorithmic_gradKL(target, space, **kwargs)

    def algorithmic_gradNLL(self, data_samples, space, data_bases=None, **kwargs):
        return self.nn_state.compute_exact_gradients(
            data_samples, space, bases_batch=data_bases
        )

    def numeric_gradKL(self, target, param, space, eps, all_bases=None, **kwargs):
        num_gradKL = []
        for i in range(len(param)):
            param[i] += eps
            KL_p = self.compute_numerical_KL(target, space, all_bases=all_bases)

            param[i] -= 2 * eps
            KL_m = self.compute_numerical_KL(target, space, all_bases=all_bases)

            param[i] += eps
            num_gradKL.append((KL_p - KL_m) / (2 * eps))

        return torch.tensor(num_gradKL, dtype=torch.double).to(param)

    def numeric_gradKL_norot(self, target, param, space, eps, **kwargs):
        num_gradKL = []
        for i in range(len(param)):
            param[i] += eps
            KL_p = self.compute_numerical_KL_norot(target, space)

            param[i] -= 2 * eps
            KL_m = self.compute_numerical_KL_norot(target, space)

            param[i] += eps
            num_gradKL.append((KL_p - KL_m) / (2 * eps))

        return torch.tensor(num_gradKL, dtype=torch.double).to(param)

    def numeric_gradNLL(
        self, param, data_samples, space, eps, data_bases=None, **kwargs
    ):
        num_gradNLL = []
        for i in range(len(param)):
            param[i] += eps
            NLL_p = self.compute_numerical_NLL(
                data_samples, space, data_bases=data_bases
            )

            param[i] -= 2 * eps
            NLL_m = self.compute_numerical_NLL(
                data_samples, space, data_bases=data_bases
            )

            param[i] += eps
            num_gradNLL.append((NLL_p - NLL_m) / (2 * eps))

        return torch.tensor(num_gradNLL, dtype=torch.double).to(param)


class ComplexGradsUtils(PosGradsUtils):
    def load_target_psi(self, bases, psi_data):
        if isinstance(psi_data, torch.Tensor):
            psi_data = psi_data.clone().detach().to(dtype=torch.cdouble)
        else:
            psi_data = torch.tensor(psi_data, dtype=torch.cdouble)

        psi_dict = {}
        D = int(psi_data.shape[0] / float(len(bases)))

        for b in range(len(bases)):
            psi = torch.zeros(D, dtype=torch.cdouble)
            psi = psi_data[b * D : (b + 1) * D]
            psi_dict[bases[b]] = psi

        return psi_dict

    def transform_bases(self, bases_data):
        bases_strs = ["".join(b for b in basis if b != " ") for basis in bases_data]
        return bases_strs

    def algorithmic_gradKL(self, target, space, all_bases, **kwargs):
        grad_KL = [
            torch.zeros(
                self.nn_state.rbm_am.num_pars,
                dtype=torch.double,
                device=self.nn_state.device,
            ),
            torch.zeros(
                self.nn_state.rbm_ph.num_pars,
                dtype=torch.double,
                device=self.nn_state.device,
            ),
        ]
        Z = self.nn_state.normalization(space)

        if all_bases is None:
            if isinstance(target, dict):
                unrotated_basis = [k for k in target.keys() if all(c == "Z" for c in k)]
                target_r = cplx.absolute_value(target[unrotated_basis[0]]) ** 2
            else:
                target_r = cplx.absolute_value(target) ** 2

            probs = self.nn_state.probability(space, Z)
            all_grads = self.nn_state.rbm_am.effective_energy_gradient(
                space, reduce=False
            )

            grad_KL[0] += torch.mv(
                all_grads.t(), target_r - probs
            )  # average the gradients, weighted by probs

            return grad_KL

        for b in range(len(all_bases)):
            if isinstance(target, dict):
                target_r = target[all_bases[b]]
            else:
                target_r = rotate_psi_inner_prod(
                    self.nn_state, all_bases[b], space, psi=target
                )
            target_r = cplx.absolute_value(target_r) ** 2

            for i in range(len(space)):
                rotated_grad = self.nn_state.gradient(space[i], all_bases[b])
                grad_KL[0] += target_r[i] * rotated_grad[0] / float(len(all_bases))
                grad_KL[1] += target_r[i] * rotated_grad[1] / float(len(all_bases))

        probs = self.nn_state.probability(space, Z)
        all_grads = self.nn_state.rbm_am.effective_energy_gradient(space, reduce=False)
        grad_KL[0] -= torch.mv(
            all_grads.t(), probs
        )  # average the gradients, weighted by probs

        return grad_KL


class DensityGradsUtils(ComplexGradsUtils):
    def algorithmic_gradKL(self, target, space, all_bases, **kwargs):
        grad_KL = [
            torch.zeros(
                self.nn_state.rbm_am.num_pars,
                dtype=torch.double,
                device=self.nn_state.device,
            ),
            torch.zeros(
                self.nn_state.rbm_ph.num_pars,
                dtype=torch.double,
                device=self.nn_state.device,
            ),
        ]
        Z = self.nn_state.normalization(space)

        if all_bases is None:
            if isinstance(target, dict):
                unrotated_basis = [k for k in target.keys() if all(c == "Z" for c in k)]
                target_r = torch.diagonal(cplx.real(target[unrotated_basis[0]]))
            else:
                target_r = torch.diagonal(cplx.real(target))

            probs = self.nn_state.probability(space, Z)
            all_grads = self.nn_state.rbm_am.effective_energy_gradient(
                space, reduce=False
            )

            grad_KL[0] += torch.mv(
                all_grads.t(), target_r - probs
            )  # average the gradients, weighted by probs

            return grad_KL

        for b in range(len(all_bases)):
            if isinstance(target, dict):
                target_r = target[all_bases[b]]
            else:
                target_r = rotate_rho(self.nn_state, all_bases[b], space, rho=target)
            target_r = torch.diagonal(cplx.real(target_r))

            for i in range(len(space)):
                rotated_grad = self.nn_state.gradient(space[i], all_bases[b])
                grad_KL[0] += target_r[i] * rotated_grad[0] / float(len(all_bases))
                grad_KL[1] += target_r[i] * rotated_grad[1] / float(len(all_bases))

        probs = self.nn_state.probability(space, Z)
        all_grads = self.nn_state.rbm_am.effective_energy_gradient(space, reduce=False)
        grad_KL[0] -= torch.mv(
            all_grads.t(), probs
        )  # average the gradients, weighted by probs

        return grad_KL
