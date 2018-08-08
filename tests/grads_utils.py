# Copyright 2018 PIQuIL - All Rights Reserved

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import numpy as np
import torch

from qucumber.utils import cplx


class PosGradsUtils():

    def __init__(self, nn_state):
        self.nn_state = nn_state

    def generate_visible_space(self, N):
        """Generates all possible visible states.

        :returns: A tensor of all possible spin configurations.
        :rtype: torch.Tensor
        """
        space = torch.zeros((2**N, N), dtype=torch.double,
                            device=self.nn_state.device)
        for i in range(1 << N):
            d = i
            for j in range(N):
                d, r = divmod(d, 2)
                space[i, N - j - 1] = int(r)

        return space

    def partition(self, visible_space):
        """The natural logarithm of the partition function of the RBM.

        :param visible_space: A rank 2 tensor of the entire visible space.
        :type visible_space: torch.Tensor

        :returns: The natural log of the partition function.
        :rtype: torch.Tensor
        """
        return torch.tensor(
            self.nn_state.rbm_am.compute_partition_function(visible_space),
            dtype=torch.double, device=self.nn_state.device
        )

    def probability(self, v, Z):
        """Evaluates the probability of the given vector(s) of visible
        units; NOT RECOMMENDED FOR RBMS WITH A LARGE # OF VISIBLE UNITS

        :param v: The visible states.
        :type v: torch.Tensor
        :param Z: The partition function.
        :type Z: float

        :returns: The probability of the given vector(s) of visible units.
        :rtype: torch.Tensor
        """
        return (self.nn_state.psi(v.to(device=self.nn_state.device))[0])**2 / Z

    def compute_numerical_kl(self, target_psi, vis, Z):
        KL = 0.0
        for i in range(len(vis)):
            KL += ((target_psi[i, 0])**2)*((target_psi[i, 0])**2).log()
            KL -= (((target_psi[i, 0])**2)
                   * (self.probability(vis[i], Z)).log().item())
        return KL

    def compute_numerical_NLL(self, data, Z):
        NLL = 0
        batch_size = len(data)

        for i in range(batch_size):
            NLL -= (self.probability(data[i], Z).log().item()
                    / float(batch_size))

        return NLL

    def algorithmic_gradKL(self, target_psi, vis):
        Z = self.partition(vis)
        grad_KL = torch.zeros(self.nn_state.rbm_am.num_pars,
                              dtype=torch.double, device=self.nn_state.device)
        for i in range(len(vis)):
            grad_KL += ((target_psi[i, 0])**2) * self.nn_state.gradient(vis[i])
            grad_KL -= (self.probability(vis[i], Z)
                        * self.nn_state.gradient(vis[i]))
        return grad_KL

    def algorithmic_gradNLL(self, qr, data, k):
        return qr.compute_batch_gradients(k, data, data)

    def numeric_gradKL(self, target_psi, param, vis, eps):
        num_gradKL = []
        for i in range(len(param)):
            param[i] += eps

            Z = self.partition(vis)
            KL_p = self.compute_numerical_kl(target_psi, vis, Z)

            param[i] -= 2*eps

            Z = self.partition(vis)
            KL_m = self.compute_numerical_kl(target_psi, vis, Z)

            param[i] += eps

            num_gradKL.append((KL_p - KL_m) / (2*eps))

        return torch.stack(num_gradKL)

    def numeric_gradNLL(self, param, data, vis, eps):
        num_gradNLL = []
        for i in range(len(param)):
            param[i] += eps

            Z = self.partition(vis)
            NLL_p = self.compute_numerical_NLL(data, Z)

            param[i] -= 2*eps

            Z = self.partition(vis)
            NLL_m = self.compute_numerical_NLL(data, Z)

            param[i] += eps

            num_gradNLL.append((NLL_p - NLL_m) / (2*eps))

        return torch.tensor(np.array(num_gradNLL), dtype=torch.double)


class ComplexGradsUtils():

    def __init__(self, nn_state):
        self.nn_state = nn_state

    def generate_visible_space(self, N):
        """Generates all possible visible states.

        :returns: A tensor of all possible spin configurations.
        :rtype: torch.Tensor
        """
        space = torch.zeros((2**N, N), dtype=torch.double,
                            device=self.nn_state.device)
        for i in range(1 << N):
            d = i
            for j in range(N):
                d, r = divmod(d, 2)
                space[i, N - j - 1] = int(r)

        return space

    def partition(self, visible_space):
        """The natural logarithm of the partition function of the RBM.

        :param visible_space: A rank 2 tensor of the entire visible space.
        :type visible_space: torch.Tensor

        :returns: The natural log of the partition function.
        :rtype: torch.Tensor
        """
        return torch.tensor(
            self.nn_state.rbm_am.compute_partition_function(visible_space),
            dtype=torch.double, device=self.nn_state.device
        )

    def probability(self, v, Z):
        """Evaluates the probability of the given vector(s) of visible
        units; NOT RECOMMENDED FOR RBMS WITH A LARGE # OF VISIBLE UNITS

        :param v: The visible states.
        :type v: torch.Tensor
        :param Z: The partition function.
        :type Z: float

        :returns: The probability of the given vector(s) of visible units.
        :rtype: torch.Tensor
        """
        return (self.nn_state.amplitude(
            v.to(device=self.nn_state.device)
        ))**2 / Z

    def load_target_psi(self, bases, psi_data):
        psi_dict = {}
        D = int(len(psi_data)/float(len(bases)))

        for b in range(len(bases)):
            psi = torch.zeros(2, D, dtype=torch.double)
            psi_real = torch.tensor(psi_data[b*D:D*(b+1), 0],
                                    dtype=torch.double)
            psi_imag = torch.tensor(psi_data[b*D:D*(b+1), 1],
                                    dtype=torch.double)
            psi[0] = psi_real
            psi[1] = psi_imag
            psi_dict[bases[b]] = psi

        return psi_dict

    def transform_bases(self, bases_data):
        bases = []
        for i in range(len(bases_data)):
            tmp = ""
            for j in range(len(bases_data[i])):
                if bases_data[i][j] is not " ":
                    tmp += bases_data[i][j]
            bases.append(tmp)
        return bases

    def rotate_psi_full(self, basis, full_unitary_dict, psi):
        U = full_unitary_dict[basis]
        Upsi = cplx.matmul(U, psi)
        return Upsi

    def rotate_psi(self, basis, unitary_dict, vis):
        N = self.nn_state.num_visible
        v = torch.zeros(N, dtype=torch.double, device=self.nn_state.device)
        psi_r = torch.zeros(2, 1 << N, dtype=torch.double,
                            device=self.nn_state.device)

        for x in range(1 << N):
            Upsi = torch.zeros(2, dtype=torch.double,
                               device=self.nn_state.device)
            num_nontrivial_U = 0
            nontrivial_sites = []
            for j in range(N):
                if (basis[j] is not 'Z'):
                    num_nontrivial_U += 1
                    nontrivial_sites.append(j)
            sub_state = self.generate_visible_space(num_nontrivial_U)

            for xp in range(1 << num_nontrivial_U):
                cnt = 0
                for j in range(N):
                    if (basis[j] is not 'Z'):
                        v[j] = sub_state[xp][cnt]
                        cnt += 1
                    else:
                        v[j] = vis[x, j]

                U = torch.tensor([1., 0.], dtype=torch.double,
                                 device=self.nn_state.device)
                for ii in range(num_nontrivial_U):
                    tmp = unitary_dict[basis[nontrivial_sites[ii]]]
                    tmp = tmp[:,
                              int(vis[x][nontrivial_sites[ii]]),
                              int(v[nontrivial_sites[ii]])]
                    U = cplx.scalar_mult(U, tmp)

                Upsi += cplx.scalar_mult(U, self.nn_state.psi(v))

            psi_r[:, x] = Upsi
        return psi_r

    def compute_numerical_NLL(self, data_samples, data_bases, Z,
                              unitary_dict, vis):
        NLL = 0
        batch_size = len(data_samples)
        b_flag = 0
        for i in range(batch_size):
            bitstate = []
            for j in range(self.nn_state.num_visible):
                ind = 0
                if (data_bases[i][j] != 'Z'):
                    b_flag = 1
                bitstate.append(int(data_samples[i, j].item()))
            ind = int("".join(str(i) for i in bitstate), 2)
            if (b_flag == 0):
                NLL -= ((self.probability(data_samples[i], Z)).log().item()
                        / batch_size)
            else:
                psi_r = self.rotate_psi(data_bases[i], unitary_dict, vis)
                NLL -= ((cplx.norm(psi_r[:, ind]).log()-Z.log()).item()
                        / batch_size)
        return NLL

    def compute_numerical_kl(self, psi_dict, vis, Z, unitary_dict, bases):
        N = self.nn_state.num_visible
        psi_r = torch.zeros(2, 1 << N, dtype=torch.double,
                            device=self.nn_state.device)
        KL = 0.0
        for i in range(len(vis)):
            KL += (cplx.norm(psi_dict[bases[0]][:, i])
                   * cplx.norm(psi_dict[bases[0]][:, i]).log()
                   / float(len(bases)))
            KL -= (cplx.norm(psi_dict[bases[0]][:, i])
                   * self.probability(vis[i], Z).log().item()
                   / float(len(bases)))

        for b in range(1, len(bases)):
            psi_r = self.rotate_psi(bases[b], unitary_dict, vis)
            for ii in range(len(vis)):
                if(cplx.norm(psi_dict[bases[b]][:, ii]) > 0.0):
                    KL += (cplx.norm(psi_dict[bases[b]][:, ii])
                           * cplx.norm(psi_dict[bases[b]][:, ii]).log()
                           / float(len(bases)))

                KL -= (cplx.norm(psi_dict[bases[b]][:, ii])
                       * cplx.norm(psi_r[:, ii]).log()
                       / float(len(bases)))
                KL += (cplx.norm(psi_dict[bases[b]][:, ii])
                       * Z.log()
                       / float(len(bases)))

        return KL

    def algorithmic_gradNLL(self, qr, data_samples, data_bases, k):
        return qr.compute_batch_gradients(k, data_samples,
                                          data_samples, data_bases)

    def numeric_gradNLL(self, data_samples, data_bases, unitary_dict, param,
                        vis, eps):
        num_gradNLL = []
        for i in range(len(param)):
            param[i] += eps

            Z = self.partition(vis)
            NLL_p = self.compute_numerical_NLL(data_samples, data_bases, Z,
                                               unitary_dict, vis)
            param[i] -= 2*eps

            Z = self.partition(vis)
            NLL_m = self.compute_numerical_NLL(data_samples, data_bases, Z,
                                               unitary_dict, vis)

            param[i] += eps

            num_gradNLL.append((NLL_p - NLL_m) / (2*eps))

        return torch.tensor(np.array(num_gradNLL), dtype=torch.double)

    def numeric_gradKL(self, param, psi_dict, vis, unitary_dict, bases, eps):
        num_gradKL = []
        for i in range(len(param)):
            param[i] += eps

            Z = self.partition(vis)
            KL_p = self.compute_numerical_kl(psi_dict, vis, Z,
                                             unitary_dict, bases)

            param[i] -= 2*eps

            Z = self.partition(vis)
            KL_m = self.compute_numerical_kl(psi_dict, vis, Z,
                                             unitary_dict, bases)
            param[i] += eps

            num_gradKL.append((KL_p - KL_m) / (2*eps))

        return torch.stack(num_gradKL)

    def algorithmic_gradKL(self, psi_dict, vis, unitary_dict, bases):
        grad_KL = [torch.zeros(self.nn_state.rbm_am.num_pars,
                               dtype=torch.double,
                               device=self.nn_state.device),
                   torch.zeros(self.nn_state.rbm_ph.num_pars,
                               dtype=torch.double,
                               device=self.nn_state.device)]
        Z = self.partition(vis).to(device=self.nn_state.device)

        for i in range(len(vis)):
            grad_KL[0] += (
                cplx.norm(psi_dict[bases[0]][:, i])
                * self.nn_state.rbm_am.effective_energy_gradient(vis[i])
                / float(len(bases))
            )
            grad_KL[0] -= (
                self.probability(vis[i], Z)
                * self.nn_state.rbm_am.effective_energy_gradient(vis[i])
                / float(len(bases))
            )

        for b in range(1, len(bases)):
            for i in range(len(vis)):
                rotated_grad = self.nn_state.gradient(bases[b], vis[i])
                grad_KL[0] += (cplx.norm(psi_dict[bases[b]][:, i])
                               * rotated_grad[0]
                               / float(len(bases)))
                grad_KL[1] += (cplx.norm(psi_dict[bases[b]][:, i])
                               * rotated_grad[1]
                               / float(len(bases)))
                grad_KL[0] -= (
                    self.probability(vis[i], Z)
                    * self.nn_state.rbm_am.effective_energy_gradient(vis[i])
                    / float(len(bases))
                )
        return grad_KL
