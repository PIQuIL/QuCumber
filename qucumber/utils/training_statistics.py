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

import qucumber.utils.cplx as cplx
import qucumber.utils.unitaries as unitaries


def fidelity(nn_state, target_psi, space, **kwargs):
    r"""Calculates the square of the overlap (fidelity) between the reconstructed
    wavefunction and the true wavefunction (both in the computational basis).

    :param nn_state: The neural network state (i.e. complex wavefunction or
                     positive wavefunction).
    :type nn_state: WaveFunction
    :param target_psi: The true wavefunction of the system.
    :type target_psi: torch.Tensor
    :param space: The hilbert space of the system.
    :type space: torch.Tensor
    :param \**kwargs: Extra keyword arguments that may be passed. Will be ignored.

    :returns: The fidelity.
    :rtype: torch.Tensor
    """
    Z = nn_state.compute_normalization(space)
    F = torch.tensor([0.0, 0.0], dtype=torch.double, device=nn_state.device)
    target_psi = target_psi.to(nn_state.device)
    for i in range(len(space)):
        psi = nn_state.psi(space[i]) / Z.sqrt()
        F[0] += target_psi[0, i] * psi[0] + target_psi[1, i] * psi[1]
        F[1] += target_psi[0, i] * psi[1] - target_psi[1, i] * psi[0]
    return cplx.norm_sqr(F)


def rotate_psi(nn_state, basis, space, unitaries, psi=None):
    r"""A function that rotates the reconstructed wavefunction to a different
    basis.

    :param nn_state: The neural network state (i.e. complex wavefunction or
                     positive wavefunction).
    :type nn_state: WaveFunction
    :param basis: The basis to rotate the wavefunction to.
    :type basis: str
    :param space: The hilbert space of the system.
    :type space: torch.Tensor
    :param unitaries: A dictionary of (2x2) unitary operators.
    :type unitaries: dict
    :param psi: A wavefunction that the user can input to override the neural
                network state's wavefunction.
    :type psi: torch.Tensor

    :returns: A wavefunction in a new basis.
    :rtype: torch.Tensor
    """
    N = nn_state.num_visible
    v = torch.zeros(N, dtype=torch.double, device=nn_state.device)
    psi_r = torch.zeros(2, 1 << N, dtype=torch.double, device=nn_state.device)
    for x in range(1 << N):
        Upsi = torch.zeros(2, dtype=torch.double, device=nn_state.device)
        num_nontrivial_U = 0
        nontrivial_sites = []
        for jj in range(N):
            if basis[jj] != "Z":
                num_nontrivial_U += 1
                nontrivial_sites.append(jj)
        sub_state = nn_state.generate_hilbert_space(num_nontrivial_U)

        for xp in range(1 << num_nontrivial_U):
            cnt = 0
            for j in range(N):
                if basis[j] != "Z":
                    v[j] = sub_state[xp][cnt]
                    cnt += 1
                else:
                    v[j] = space[x, j]
            U = torch.tensor([1.0, 0.0], dtype=torch.double, device=nn_state.device)
            for ii in range(num_nontrivial_U):
                tmp = unitaries[basis[nontrivial_sites[ii]]]
                tmp = tmp[
                    :, int(space[x][nontrivial_sites[ii]]), int(v[nontrivial_sites[ii]])
                ].to(nn_state.device)
                U = cplx.scalar_mult(U, tmp)
            if psi is None:
                Upsi += cplx.scalar_mult(U, nn_state.psi(v))
            else:
                index = 0
                for k in range(len(v)):
                    index = (index << 1) | int(v[k].item())
                Upsi += cplx.scalar_mult(U, psi[:, index])
        psi_r[:, x] = Upsi
    return psi_r


def NLL(nn_state, samples, space, train_bases=None, **kwargs):
    r"""A function for calculating the negative log-likelihood.

    :param nn_state: The neural network state (i.e. complex wavefunction or
                     positive wavefunction).
    :type nn_state: WaveFunction
    :param samples: Samples to compute the NLL on.
    :type samples: torch.Tensor
    :param space: The hilbert space of the system.
    :type space: torch.Tensor
    :param train_bases: An array of bases where measurements were taken.
    :type train_bases: np.array(dtype=str)
    :param \**kwargs: Extra keyword arguments that may be passed. Will be ignored.

    :returns: The Negative Log-Likelihood.
    :rtype: torch.Tensor
    """
    psi_r = torch.zeros(
        2, 1 << nn_state.num_visible, dtype=torch.double, device=nn_state.device
    )
    NLL = 0.0
    unitary_dict = unitaries.create_dict()
    Z = nn_state.compute_normalization(space)
    eps = 0.000001
    if train_bases is None:
        for i in range(len(samples)):
            NLL -= (cplx.norm_sqr(nn_state.psi(samples[i])) + eps).log()
            NLL += Z.log()
    else:
        for i in range(len(samples)):
            # Check whether the sample was measured the reference basis
            is_reference_basis = True
            # b_ID = 0
            for j in range(nn_state.num_visible):
                if train_bases[i][j] != "Z":
                    is_reference_basis = False
                    break
            if is_reference_basis is True:
                NLL -= (cplx.norm_sqr(nn_state.psi(samples[i])) + eps).log()
                NLL += Z.log()
            else:
                psi_r = rotate_psi(nn_state, train_bases[i], space, unitary_dict)
                # Get the index value of the sample state
                ind = 0
                for j in range(nn_state.num_visible):
                    if samples[i, nn_state.num_visible - j - 1] == 1:
                        ind += pow(2, j)
                NLL -= cplx.norm_sqr(psi_r[:, ind]).log().item()
                NLL += Z.log()
    return NLL / float(len(samples))


def KL(nn_state, target_psi, space, bases=None, **kwargs):
    r"""A function for calculating the total KL divergence.

    :param nn_state: The neural network state (i.e. complex wavefunction or
                     positive wavefunction).
    :type nn_state: WaveFunction
    :param target_psi: The true wavefunction of the system.
    :type target_psi: torch.Tensor
    :param space: The hilbert space of the system.
    :type space: torch.Tensor
    :param bases: An array of unique bases.
    :type bases: np.array(dtype=str)
    :param \**kwargs: Extra keyword arguments that may be passed. Will be ignored.

    :returns: The KL divergence.
    :rtype: torch.Tensor
    """
    psi_r = torch.zeros(
        2, 1 << nn_state.num_visible, dtype=torch.double, device=nn_state.device
    )
    KL = 0.0
    unitary_dict = unitaries.create_dict()
    target_psi = target_psi.to(nn_state.device)
    Z = nn_state.compute_normalization(space)
    eps = 0.000001
    if bases is None:
        num_bases = 1
        for i in range(len(space)):
            KL += (
                cplx.norm_sqr(target_psi[:, i])
                * (cplx.norm_sqr(target_psi[:, i]) + eps).log()
            )
            KL -= (
                cplx.norm_sqr(target_psi[:, i])
                * (cplx.norm_sqr(nn_state.psi(space[i])) + eps).log()
            )
            KL += cplx.norm_sqr(target_psi[:, i]) * Z.log()
    else:
        num_bases = len(bases)
        for b in range(1, len(bases)):
            psi_r = rotate_psi(nn_state, bases[b], space, unitary_dict)
            target_psi_r = rotate_psi(
                nn_state, bases[b], space, unitary_dict, target_psi
            )
            for ii in range(len(space)):
                if cplx.norm_sqr(target_psi_r[:, ii]) > 0.0:
                    KL += (
                        cplx.norm_sqr(target_psi_r[:, ii])
                        * cplx.norm_sqr(target_psi_r[:, ii]).log()
                    )
                KL -= (
                    cplx.norm_sqr(target_psi_r[:, ii])
                    * cplx.norm_sqr(psi_r[:, ii]).log().item()
                )
                KL += cplx.norm_sqr(target_psi_r[:, ii]) * Z.log()
    return KL / float(num_bases)
