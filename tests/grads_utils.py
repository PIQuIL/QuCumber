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


class PosGradsUtils:
    def __init__(self, nn_state):
        self.nn_state = nn_state

    def compute_numerical_kl(self, target_psi, vis):
        return ts.KL(self.nn_state, target_psi, vis)

    def compute_numerical_NLL(self, data, vis):
        return ts.NLL(self.nn_state, data, vis)

    def algorithmic_gradKL(self, target_psi, vis, **kwargs):
        Z = self.nn_state.compute_normalization(vis)
        grad_KL = torch.zeros(
            self.nn_state.rbm_am.num_pars,
            dtype=torch.double,
            device=self.nn_state.device,
        )
        for i in range(len(vis)):
            grad_KL += ((target_psi[0, i]) ** 2) * self.nn_state.gradient(vis[i])
            grad_KL -= self.nn_state.probability(vis[i], Z) * self.nn_state.gradient(
                vis[i]
            )
        return [grad_KL]

    def algorithmic_gradNLL(self, data, k, **kwargs):
        return self.nn_state.compute_batch_gradients(k, data, data)

    def numeric_gradKL(self, target_psi, param, vis, eps, **kwargs):
        num_gradKL = []
        for i in range(len(param)):
            param[i] += eps
            KL_p = self.compute_numerical_kl(target_psi, vis)

            param[i] -= 2 * eps
            KL_m = self.compute_numerical_kl(target_psi, vis)

            param[i] += eps
            num_gradKL.append((KL_p - KL_m) / (2 * eps))

        return torch.tensor(num_gradKL, dtype=torch.double).to(param)

    def numeric_gradNLL(self, param, data, vis, eps, **kwargs):
        num_gradNLL = []
        for i in range(len(param)):
            param[i] += eps
            NLL_p = self.compute_numerical_NLL(data, vis)

            param[i] -= 2 * eps
            NLL_m = self.compute_numerical_NLL(data, vis)

            param[i] += eps
            num_gradNLL.append((NLL_p - NLL_m) / (2 * eps))

        return torch.tensor(num_gradNLL, dtype=torch.double).to(param)


class ComplexGradsUtils:
    def __init__(self, nn_state):
        self.nn_state = nn_state

    def load_target_psi(self, bases, psi_data):
        if isinstance(psi_data, torch.Tensor):
            psi_data = psi_data.clone().detach().to(dtype=torch.double)
        else:
            psi_data = torch.tensor(psi_data, dtype=torch.double)

        psi_dict = {}
        D = int(psi_data.shape[1] / float(len(bases)))

        for b in range(len(bases)):
            psi = torch.zeros(2, D, dtype=torch.double)
            psi[0, ...] = psi_data[0, b * D : (b + 1) * D]
            psi[1, ...] = psi_data[1, b * D : (b + 1) * D]
            psi_dict[bases[b]] = psi

        return psi_dict

    def transform_bases(self, bases_data):
        bases = []
        for i in range(len(bases_data)):
            tmp = ""
            for j in range(len(bases_data[i])):
                if bases_data[i][j] != " ":
                    tmp += bases_data[i][j]
            bases.append(tmp)
        return bases

    def rotate_psi(self, basis, unitary_dict, vis):
        return ts.rotate_psi(self.nn_state, basis, vis, unitary_dict)

    def compute_numerical_NLL(self, data_samples, data_bases, vis):
        return ts.NLL(self.nn_state, data_samples, vis, bases=data_bases)

    def compute_numerical_kl(self, psi_dict, vis, bases):
        return ts.KL(self.nn_state, psi_dict, vis, bases=bases)

    def algorithmic_gradNLL(self, data_samples, data_bases, k, **kwargs):
        return self.nn_state.compute_batch_gradients(
            k, data_samples, data_samples, data_bases
        )

    def numeric_gradNLL(self, data_samples, data_bases, param, vis, eps, **kwargs):
        num_gradNLL = []
        for i in range(len(param)):
            param[i] += eps
            NLL_p = self.compute_numerical_NLL(data_samples, data_bases, vis)

            param[i] -= 2 * eps
            NLL_m = self.compute_numerical_NLL(data_samples, data_bases, vis)

            param[i] += eps
            num_gradNLL.append((NLL_p - NLL_m) / (2 * eps))

        return torch.tensor(num_gradNLL, dtype=torch.double).to(param)

    def numeric_gradKL(self, param, psi_dict, vis, bases, eps, **kwargs):
        num_gradKL = []
        for i in range(len(param)):
            param[i] += eps
            KL_p = self.compute_numerical_kl(psi_dict, vis, bases)

            param[i] -= 2 * eps
            KL_m = self.compute_numerical_kl(psi_dict, vis, bases)

            param[i] += eps
            num_gradKL.append((KL_p - KL_m) / (2 * eps))

        return torch.tensor(num_gradKL).to(param)

    def algorithmic_gradKL(self, psi_dict, vis, unitary_dict, bases, **kwargs):
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
        Z = self.nn_state.compute_normalization(vis).to(device=self.nn_state.device)

        for i in range(len(vis)):
            grad_KL[0] += (
                cplx.norm_sqr(psi_dict[bases[0]][:, i])
                * self.nn_state.rbm_am.effective_energy_gradient(vis[i])
                / float(len(bases))
            )
            grad_KL[0] -= (
                self.nn_state.probability(vis[i], Z)
                * self.nn_state.rbm_am.effective_energy_gradient(vis[i])
                / float(len(bases))
            )

        for b in range(1, len(bases)):
            for i in range(len(vis)):
                rotated_grad = self.nn_state.gradient(bases[b], vis[i])
                grad_KL[0] += (
                    cplx.norm_sqr(psi_dict[bases[b]][:, i])
                    * rotated_grad[0]
                    / float(len(bases))
                )
                grad_KL[1] += (
                    cplx.norm_sqr(psi_dict[bases[b]][:, i])
                    * rotated_grad[1]
                    / float(len(bases))
                )
                grad_KL[0] -= (
                    self.nn_state.probability(vis[i], Z)
                    * self.nn_state.rbm_am.effective_energy_gradient(vis[i])
                    / float(len(bases))
                )
        return grad_KL


class DensityGradsUtils:
    def __init__(self, nn_state):
        self.nn_state = nn_state

    def transform_bases(self, bases_data):
        bases = []
        for i in range(len(bases_data)):
            tmp = ""
            for j in range(len(bases_data[i])):
                if bases_data[i][j] != " ":
                    tmp += bases_data[i][j]
            bases.append(tmp)
        return bases

    # NOTE: Not sure if compute_numerical_NLL function for
    # density matrix is useful, below is only for pure states
    # def compute_numerical_NLL(self, data_samples, data_bases, Z, unitary_dict, vis):
    #     NLL = 0
    #     batch_size = len(data_samples)
    #     b_flag = 0
    #     for i in range(batch_size):
    #         bitstate = []
    #         for j in range(self.nn_state.num_visible):
    #             ind = 0
    #             if data_bases[i][j] != "Z":
    #                 b_flag = 1
    #             bitstate.append(int(data_samples[i, j].item()))
    #         ind = int("".join(str(i) for i in bitstate), 2)
    #         if b_flag == 0:
    #             NLL -= (
    #                 self.nn_state.probability(data_samples[i], Z)
    #             ).log().item() / batch_size
    #         else:
    #             psi_r = self.rotate_psi(data_bases[i], unitary_dict, vis)
    #             NLL -= (
    #                 cplx.norm_sqr(psi_r[:, ind]).log() - Z.log()
    #             ).item() / batch_size
    #     return NLL

    # NOTE: Not sure if the NLL will ever be used. Keeping these
    # here, commented out, for possible future reference

    # def algorithmic_gradNLL(self, data_samples, data_bases, k, **kwargs):
    #     return self.nn_state.compute_batch_gradients(
    #         k, data_samples, data_samples, data_bases
    #     )

    # def numeric_gradNLL(
    #     self, data_samples, data_bases, unitary_dict, param, vis, eps, **kwargs
    # ):
    #     num_gradNLL = []
    #     for i in range(len(param)):
    #         param[i] += eps

    #         Z = self.nn_state.compute_normalization(vis)
    #         NLL_p = self.compute_numerical_NLL(
    #             data_samples, data_bases, Z, unitary_dict, vis
    #         )
    #         param[i] -= 2 * eps

    #         Z = self.nn_state.compute_normalization(vis)
    #         NLL_m = self.compute_numerical_NLL(
    #             data_samples, data_bases, Z, unitary_dict, vis
    #         )

    #         param[i] += eps

    #         num_gradNLL.append((NLL_p - NLL_m) / (2 * eps))

    #     return torch.tensor(np.array(num_gradNLL), dtype=torch.double).to(param)

    def compute_numerical_KL(self, target, bases, space):
        return ts.density_matrix_KL(self.nn_state, target, bases, space)

    def algorithmic_gradKL(self, data_samples, data_bases, space, **kwargs):
        return self.nn_state.compute_exact_grads(data_samples, data_bases, space)

    def numeric_gradKL(self, param, target, space, bases, eps, **kwargs):
        num_gradKL = []
        param_shape = param.shape
        param = param.view(-1)

        for i in range(len(param)):
            param[i] += eps
            KL_p = self.compute_numerical_KL(target, bases, space)

            param[i] -= 2 * eps
            KL_m = self.compute_numerical_KL(target, bases, space)

            param[i] += eps
            num_gradKL.append((KL_p - KL_m) / (2 * eps))

        param = param.reshape(param_shape)
        return torch.tensor(num_gradKL, dtype=torch.double)

    def all_num_grads(self, target, space, bases, eps):
        num_grads = []
        for _n, net in enumerate(self.nn_state.networks):
            rbm = getattr(self.nn_state, net)
            num_grad = torch.tensor([]).to(dtype=torch.double)
            counter = 0
            for param in rbm.parameters():
                num_grad = torch.cat(
                    (num_grad, self.numeric_gradKL(param, target, space, bases, eps),)
                )
                counter += 1
            num_grads.append(num_grad)

        return num_grads
