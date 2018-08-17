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

import torch

import qucumber.utils.cplx as cplx
import qucumber.utils.unitaries as unitaries


def fidelity(nn_state, target_psi, space):
    """Calculates the square of the overlap (fidelity) between the reconstructed
    wavefunction and the true wavefunction (both in the computational basis).

    :param nn_state: The neural network state (i.e. complex wavefunction or 
                     positive wavefunction).
    :type nn_state: Wavefunction
    :param target_psi: The true wavefunction of the system.
    :type target_psi: torch.Tensor
    :param space: The hilbert space of the system.
    :type space: torch.Tensor
    
    :returns: The fidelity.
    :rtype: torch.Tensor 
    """
    Z = nn_state.compute_normalization(space)
    F = torch.tensor([0., 0.], dtype=torch.double, device=nn_state.device)
    target_psi = target_psi.to(nn_state.device)
    for i in range(len(space)):
        psi = nn_state.psi(space[i]) / Z.sqrt()
        F[0] += target_psi[0, i] * psi[0] + target_psi[1, i] * psi[1]
        F[1] += target_psi[0, i] * psi[1] - target_psi[1, i] * psi[0]
    return cplx.norm(F)


def rotate_psi(nn_state, basis, space, unitaries, psi=None):
    """A function that rotates the reconstructed wavefunction to a different
    basis.

    :param nn_state: The neural network state (i.e. complex wavefunction or 
                     positive wavefunction).
    :type nn_state: Wavefunction 
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
            U = torch.tensor([1., 0.], dtype=torch.double, device=nn_state.device)
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


def KL(nn_state, target_psi, space, bases=None):
    """A function for calculating the total KL divergence.

    :param nn_state: The neural network state (i.e. complex wavefunction or 
                     positive wavefunction).
    :type nn_state: Wavefunction
    :param target_psi: The true wavefunction of the system.
    :type target_psi: torch.Tensor
    :param space: The hilbert space of the system.
    :type space: torch.Tensor
    :param bases: An array of unique bases.
    :type bases: np.array(dtype=str)
    """
    psi_r = torch.zeros(
        2, 1 << nn_state.num_visible, dtype=torch.double, device=nn_state.device
    )
    KL = 0.0
    unitary_dict = unitaries.create_dict()
    target_psi = target_psi.to(nn_state.device)
    Z = nn_state.compute_normalization(space)
    if bases is None:
        num_bases = 1
        for i in range(len(space)):
            KL += cplx.norm(target_psi[:, i]) * cplx.norm(target_psi[:, i]).log()
            KL -= cplx.norm(target_psi[:, i]) * cplx.norm(nn_state.psi(space[i])).log()
            KL += cplx.norm(target_psi[:, i]) * Z.log()
    else:
        num_bases = len(bases)
        for b in range(1, len(bases)):
            psi_r = rotate_psi(nn_state, bases[b], space, unitary_dict)
            target_psi_r = rotate_psi(
                nn_state, bases[b], space, unitary_dict, target_psi
            )
            for ii in range(len(space)):
                if cplx.norm(target_psi_r[:, ii]) > 0.0:
                    KL += (
                        cplx.norm(target_psi_r[:, ii])
                        * cplx.norm(target_psi_r[:, ii]).log()
                    )
                KL -= (
                    cplx.norm(target_psi_r[:, ii])
                    * cplx.norm(psi_r[:, ii]).log().item()
                )
                KL += cplx.norm(target_psi_r[:, ii]) * Z.log()
    return KL / float(num_bases)
