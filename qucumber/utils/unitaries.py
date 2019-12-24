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
import numpy as np

from qucumber.utils import cplx


def create_dict(**kwargs):
    r"""A function that creates a dictionary of unitary operators.

    By default, the dictionary contains the unitaries which perform a change of
    basis from the computational basis (Pauli-Z) to one of the other Pauli
    bases. The default keys (`X`, `Y`, and `Z`) denote the target basis.

    :param \**kwargs: Keyword arguments of any unitary operators to add to the
                      resulting dictionary. The given operators will overwrite
                      the default matrices if they share the same key.

    :returns: A dictionary of unitaries.
    :rtype: dict
    """
    dictionary = {
        "X": torch.tensor(
            [[[1.0, 1.0], [1.0, -1.0]], [[0.0, 0.0], [0.0, 0.0]]], dtype=torch.double
        )
        / np.sqrt(2),
        "Y": torch.tensor(
            [[[1.0, 0.0], [1.0, 0.0]], [[0.0, -1.0], [0.0, 1.0]]], dtype=torch.double
        )
        / np.sqrt(2),
        "Z": torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]], dtype=torch.double
        ),
    }

    dictionary.update(
        {
            name: (
                matrix.clone().detach()
                if isinstance(matrix, torch.Tensor)
                else torch.tensor(matrix)
            ).to(dtype=torch.double)
            for name, matrix in kwargs.items()
        }
    )

    return dictionary


def _kron_mult(matrices, x):
    n = [m.size()[0] for m in matrices]
    l, r = np.prod(n), 1  # noqa: E741

    if l != x.shape[1]:  # noqa: E741
        raise ValueError("Incompatible sizes!")

    y = x.clone()
    for s in reversed(range(len(n))):
        l //= n[s]  # noqa: E741
        m = matrices[s]

        for k in range(l):
            for i in range(r):
                slc = slice(k * n[s] * r + i, (k + 1) * n[s] * r + i, r)
                temp = y[:, slc, ...]
                y[:, slc, ...] = cplx.matmul(m, temp)
        r *= n[s]

    return y


def rotate_psi(nn_state, basis, space, unitaries=None, psi=None):
    r"""A function that rotates the reconstructed wavefunction to a different
    basis.

    :param nn_state: The neural network state (i.e. complex wavefunction or
                     positive wavefunction).
    :type nn_state: qucumber.nn_states.WaveFunctionBase
    :param basis: The basis to rotate the wavefunction to.
    :type basis: str
    :param space: The basis elements of the Hilbert space of the system :math:`\mathcal{H}`.
    :type space: torch.Tensor
    :param unitaries: A dictionary of (2x2) unitary operators.
    :type unitaries: dict(str, torch.Tensor)
    :param psi: A wavefunction that the user can input to override the neural
                network state's wavefunction.
    :type psi: torch.Tensor

    :returns: A wavefunction in a new basis.
    :rtype: torch.Tensor
    """
    psi = (
        nn_state.psi(space)
        if psi is None
        else psi.to(dtype=torch.double, device=nn_state.device)
    )

    unitaries = unitaries if unitaries else nn_state.unitary_dict
    unitaries = {k: v.to(device=nn_state.device) for k, v in unitaries.items()}
    us = [unitaries[b] for b in basis]
    return _kron_mult(us, psi)


def rotate_rho(nn_state, basis, space, unitaries=None, rho=None):
    r"""Computes the density matrix rotated into some basis.

    :param nn_state: The density matrix neural network state.
    :type nn_state: qucumber.nn_states.DensityMatrix
    :param basis: The basis to rotate the density matrix to.
    :type basis: str
    :param space: The basis elements of the Hilbert space of the system :math:`\mathcal{H}`.
    :type space: torch.Tensor
    :param unitaries: A dictionary of unitary matrices associated with
                        rotation into each basis
    :type unitaries: dict(str, torch.Tensor)
    :param rho: A density matrix that the user can input to override the neural
                network state's density matrix.
    :type rho: torch.Tensor

    :returns: The rotated density matrix
    :rtype: torch.Tensor
    """
    rho = (
        nn_state.rho(space, space)
        if rho is None
        else rho.to(dtype=torch.double, device=nn_state.device)
    )

    unitaries = unitaries if unitaries else nn_state.unitary_dict
    unitaries = {k: v.to(device=nn_state.device) for k, v in unitaries.items()}
    us = [unitaries[b] for b in basis]

    rho_r = _kron_mult(us, rho)
    rho_r = _kron_mult(us, cplx.conjugate(rho_r))

    return rho_r


# TODO: make this a generator function
def _rotate_basis_state(nn_state, basis, states, unitaries=None):
    unitaries = unitaries if unitaries else nn_state.unitary_dict
    unitaries = {k: v.to(device="cpu") for k, v in unitaries.items()}

    basis = np.array(list(basis))
    sites = np.where(basis != "Z")[0]

    if sites.size != 0:
        Us = torch.stack([unitaries[b] for b in basis[sites]]).cpu().numpy()

        reps = [1 for _ in states.shape]
        v = states.unsqueeze(0).repeat(2 ** sites.size, *reps)
        v[..., sites] = nn_state.generate_hilbert_space(size=sites.size).unsqueeze(1)
        v = v.contiguous()

        int_sample = states[..., sites].round().int().cpu().numpy()
        ints_size = np.arange(sites.size)

        # overwrite rotated elements
        int_vp = v[..., sites].long().cpu().numpy()
        all_Us = Us[ints_size, :, int_sample, int_vp]

        Ut = np.prod(all_Us[..., 0] + (1j * all_Us[..., 1]), axis=-1)
    else:
        v = states.unsqueeze(0)
        Ut = np.ones(v.shape[:-1], dtype=complex)

    return Ut, v


def _convert_basis_element_to_index(states):
    powers = (2 ** (torch.arange(states.shape[-1], 0, -1) - 1)).to(states)
    return torch.matmul(states, powers)


def rotate_psi_inner_prod(
    nn_state, basis, states, unitaries=None, psi=None, include_extras=False
):
    r"""A function that rotates the wavefunction to a different
    basis and then computes the resulting amplitude of a batch of basis elements.

    :param nn_state: The neural network state (i.e. complex wavefunction or
                     positive wavefunction).
    :type nn_state: qucumber.nn_states.WaveFunctionBase
    :param basis: The basis to rotate the wavefunction to.
    :type basis: str
    :param states: The batch of basis elements to compute the amplitudes of.
    :type states: torch.Tensor
    :param unitaries: A dictionary of (2x2) unitary operators.
    :type unitaries: dict(str, torch.Tensor)
    :param psi: A wavefunction that the user can input to override the neural
                network state's wavefunction.
    :type psi: torch.Tensor
    :param include_extras: Whether to include all the terms of the summation as
                           well as the expanded basis states in the output.
    :type include_extras: bool

    :returns: Amplitude of the states wrt the rotated wavefunction, and
              possibly some extras.
    :rtype: torch.Tensor or tuple(torch.Tensor)
    """
    Ut, v = _rotate_basis_state(nn_state, basis, states, unitaries=unitaries)

    if psi is None:
        psi = nn_state.psi(v).detach()
    else:
        # pick out the entries of psi that we actually need
        idx = _convert_basis_element_to_index(v).long()
        psi = psi[:, idx]

    psi = cplx.numpy(psi.cpu())
    Ut *= psi

    Upsi_v = cplx.make_complex(Ut).to(dtype=torch.double, device=nn_state.device)
    Upsi = torch.sum(Upsi_v, dim=1)

    if include_extras:
        return Upsi, Upsi_v, v
    else:
        return Upsi


def rotate_rho_probs(
    nn_state, basis, states, unitaries=None, rho=None, include_extras=False
):
    r"""A function that rotates the wavefunction to a different
    basis and then computes the resulting Born rule probability of a basis
    element.

    :param nn_state: The density matrix neural network state.
    :type nn_state: qucumber.nn_states.DensityMatrix
    :param basis: The basis to rotate the density matrix to.
    :type basis: str
    :param states: The batch of basis elements to compute the probabilities of.
    :type states: torch.Tensor
    :param unitaries: A dictionary of (2x2) unitary operators.
    :type unitaries: dict(str, torch.Tensor)
    :param rho: A density matrix that the user can input to override the neural
                network state's density matrix.
    :type rho: torch.Tensor
    :param include_extras: Whether to include all the terms of the summation as
                           well as the expanded basis states in the output.
    :type include_extras: bool

    :returns: Probability of the states wrt the rotated density matrix, and
              possibly some extras.
    :rtype: torch.Tensor or tuple(torch.Tensor)
    """
    Ut, v = _rotate_basis_state(nn_state, basis, states, unitaries=unitaries)
    Ut = np.einsum("ib,jb->ijb", Ut, np.conj(Ut))

    if rho is None:
        rho = nn_state.rho(v).detach()
    else:
        # pick out the entries of rho that we actually need
        idx = _convert_basis_element_to_index(v).long()
        rho = rho[:, idx.unsqueeze(0), idx.unsqueeze(1)]

    rho = cplx.numpy(rho.cpu())
    Ut *= rho

    UrhoU_v = cplx.make_complex(Ut).to(dtype=torch.double, device=nn_state.device)
    UrhoU = torch.sum(
        cplx.real(UrhoU_v), dim=(0, 1)
    )  # imaginary parts will cancel out anyway

    if include_extras:
        return UrhoU, UrhoU_v, v
    else:
        return UrhoU
