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
    :rtype: float
    """
    Z = nn_state.compute_normalization(space)
    target_psi = target_psi.to(nn_state.device)
    psi = nn_state.psi(space) / Z.sqrt()
    F = cplx.inner_prod(target_psi, psi)
    return cplx.absolute_value(F).pow_(2).item()


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
    psi = (
        nn_state.psi(space)
        if psi is None
        else psi.to(dtype=torch.double, device=nn_state.device)
    )

    unitaries = {k: v.to(device=nn_state.device) for k, v in unitaries.items()}

    l, r = psi.shape[-1], 1
    psi_r = psi.clone()
    for s in range(N)[::-1]:
        l //= N  # noqa: E741
        m = unitaries[basis[s]]
        for k in range(l):
            for i in range(r):
                slc = slice(k * N * r + i, (k + 1) * N * r + i, r)
                U = psi_r[:, slc]
                psi_r[:, slc] = cplx.matmul(m, U)
        r *= N

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
    :rtype: float
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
    return (NLL / float(len(samples))).item()


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
    :rtype: float
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
    return (KL / float(num_bases)).item()
