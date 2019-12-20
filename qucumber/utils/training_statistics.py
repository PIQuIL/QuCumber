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
from torch.distributions.utils import probs_to_logits
import numpy as np
from scipy.linalg import sqrtm

from qucumber.nn_states import WaveFunctionBase
import qucumber.utils.cplx as cplx


def _kron_mult(matrices, x):
    n = [m.size()[0] for m in matrices]
    l, r = np.prod(n), 1  # noqa: E741

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


def rotate_psi_prob(nn_state, basis, state, unitaries=None, psi=None):
    r"""A function that rotates the wavefunction to a different
    basis and then computes the resulting Born rule probability of a spin
    configuration.

    :param nn_state: The neural network state (i.e. complex wavefunction or
                     positive wavefunction).
    :type nn_state: qucumber.nn_states.WaveFunctionBase
    :param basis: The basis to rotate the wavefunction to.
    :type basis: str
    :param state: The basis element to compute the probability of.
    :type state: torch.Tensor
    :param unitaries: A dictionary of (2x2) unitary operators.
    :type unitaries: dict(str, torch.Tensor)
    :param psi: A wavefunction that the user can input to override the neural
                network state's wavefunction.
    :type psi: torch.Tensor

    :returns: Probability of the state wrt the rotated wavefunction.
    :rtype: torch.Tensor
    """
    if unitaries:
        unitaries = {k: v.to(device=nn_state.device) for k, v in unitaries.items()}
    else:
        unitaries = nn_state.unitary_dict

    basis = np.array(list(basis))  # list is silly, but works for now
    sites = np.where(basis != "Z")[0]
    Us = torch.stack([unitaries[b] for b in basis[sites]]).cpu().numpy()

    vp = state.round().clone().unsqueeze(0).repeat(2 ** sites.size, 1)
    vp[:, sites] = nn_state.generate_hilbert_space(size=sites.size)
    vp = vp.contiguous()

    int_sample = state[sites].round().int().cpu().numpy()
    ints_size = np.arange(sites.size)

    # overwrite rotated elements
    int_vp = vp[:, sites].long().cpu().numpy()
    all_Us = Us[ints_size, :, int_sample, int_vp]

    Ut = np.prod(all_Us[..., 0] + (1j * all_Us[..., 1]), axis=1)
    U = (
        cplx.make_complex(torch.tensor(Ut.real), torch.tensor(Ut.imag))
        .to(vp)
        .contiguous()
    )

    psi = (
        nn_state.psi(vp).detach()
        if psi is None
        else psi.to(dtype=torch.double, device=nn_state.device)
    )

    Upsi_v = cplx.scalar_mult(U, psi)
    Upsi = torch.sum(Upsi_v, dim=1)

    return cplx.norm_sqr(Upsi)


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


def rotate_rho_prob(nn_state, basis, state, unitaries=None, rho=None):
    r"""A function that rotates the wavefunction to a different
    basis and then computes the resulting Born rule probability of a spin
    configuration.

    :param nn_state: The density matrix neural network state.
    :type nn_state: qucumber.nn_states.DensityMatrix
    :param basis: The basis to rotate the density matrix to.
    :type basis: str
    :param state: The basis element to compute the probability of.
    :type state: torch.Tensor
    :param unitaries: A dictionary of (2x2) unitary operators.
    :type unitaries: dict(str, torch.Tensor)
    :param rho: A density matrix that the user can input to override the neural
                network state's density matrix.
    :type rho: torch.Tensor

    :returns: Probability of the state wrt the rotated density matrix.
    :rtype: torch.Tensor
    """
    if unitaries:
        unitaries = {k: v.to(device=nn_state.device) for k, v in unitaries.items()}
    else:
        unitaries = nn_state.unitary_dict

    basis = np.array(list(basis))  # list is silly, but works for now
    sites = np.where(basis != "Z")[0]
    Us = torch.stack([unitaries[b] for b in basis[sites]]).cpu().numpy()

    v = state.round().clone().unsqueeze(0).repeat(2 ** sites.size, 1)
    v[:, sites] = nn_state.generate_hilbert_space(size=sites.size)
    v = v.contiguous()

    int_sample = state[sites].round().int().cpu().numpy()
    ints_size = np.arange(sites.size)

    int_v = v[:, sites].int().cpu().numpy()
    all_Us = Us[ints_size, :, int_sample, int_v]
    Ut = np.prod(all_Us[..., 0] + (1j * all_Us[..., 1]), axis=1)
    Ut = np.outer(Ut, np.conj(Ut))
    U = (
        cplx.make_complex(torch.tensor(Ut.real), torch.tensor(Ut.imag))
        .to(state)
        .contiguous()
    )

    rho = (
        nn_state.rho(v).detach()
        if rho is None
        else rho.to(dtype=torch.double, device=nn_state.device)
    )

    UrhoU_v = cplx.scalar_mult(U, rho)
    UrhoU = torch.sum(UrhoU_v, dim=(1, 2))

    return cplx.real(UrhoU)


def fidelity(nn_state, target, space, **kwargs):
    r"""Calculates the square of the overlap (fidelity) between the reconstructed
    state and the true state (both in the computational basis).

    .. math::

        F = \vert \langle \psi_{RBM} \vert \psi_{target} \rangle \vert ^2
          = \left( \tr \lbrack \sqrt{ \sqrt{\rho_{RBM}} \rho_{target} \sqrt{\rho_{RBM}} } \rbrack \right) ^ 2

    :param nn_state: The neural network state.
    :type nn_state: qucumber.nn_states.NeuralStateBase
    :param target: The true state of the system.
    :type target: torch.Tensor
    :param space: The basis elements of the Hilbert space of the system :math:`\mathcal{H}`.
                  The ordering of the basis elements must match with the ordering of the
                  coefficients given in `target`.
    :type space: torch.Tensor
    :param \**kwargs: Extra keyword arguments that may be passed. Will be ignored.

    :returns: The fidelity.
    :rtype: float
    """
    Z = nn_state.normalization(space)
    target = target.to(nn_state.device)

    if isinstance(nn_state, WaveFunctionBase):
        assert target.dim() == 2, "target must be a complex vector!"

        psi = nn_state.psi(space) / Z.sqrt()
        F = cplx.inner_prod(target, psi)
        return cplx.absolute_value(F).pow_(2).item()
    else:
        assert target.dim() == 3, "target must be a complex matrix!"

        rho = nn_state.rho(space, space) / Z
        arg_real = cplx.real(rho).numpy()
        arg_imag = cplx.imag(rho).numpy()

        rho_rbm_ = arg_real + 1j * arg_imag

        arg_real = cplx.real(target).numpy()
        arg_imag = cplx.imag(target).numpy()

        target_ = arg_real + 1j * arg_imag

        sqrt_rho_rbm = sqrtm(rho_rbm_)
        prod = np.matmul(sqrt_rho_rbm, np.matmul(target_, sqrt_rho_rbm))

        # Instead of sqrt'ing then taking the trace, we compute the eigenvals,
        #  sqrt those, and then sum them up. This is a bit more efficient.
        eigvals = np.linalg.eigvalsh(prod)
        trace = np.sum(np.sqrt(eigvals).real)  # imaginary parts should be zero
        return trace ** 2


def NLL(nn_state, samples, space, bases=None, **kwargs):
    r"""A function for calculating the negative log-likelihood (NLL).

    :param nn_state: The neural network state.
    :type nn_state: qucumber.nn_states.NeuralStateBase
    :param samples: Samples to compute the NLL on.
    :type samples: torch.Tensor
    :param space: The basis elements of the Hilbert space of the system :math:`\mathcal{H}`.
    :type space: torch.Tensor
    :param bases: An array of bases where measurements were taken.
    :type bases: numpy.ndarray
    :param \**kwargs: Extra keyword arguments that may be passed. Will be ignored.

    :returns: The Negative Log-Likelihood.
    :rtype: float
    """
    Z = nn_state.normalization(space)

    if bases is None:
        nn_probs = nn_state.probability(samples, Z)
        NLL_ = -torch.mean(probs_to_logits(nn_probs)).item()
        return NLL_
    else:
        NLL_ = 0.0

        for i in range(len(samples)):
            # Check whether the sample was measured the reference basis
            is_reference_basis = True
            for j in range(nn_state.num_visible):
                if bases[i][j] != "Z":
                    is_reference_basis = False
                    break

            if is_reference_basis is True:
                nn_probs = nn_state.probability(samples[i], Z)
                NLL_ -= probs_to_logits(nn_probs).item()
            else:
                if isinstance(nn_state, WaveFunctionBase):
                    probs_r = rotate_psi_prob(nn_state, bases[i], samples[i]) / Z
                    NLL_ -= probs_to_logits(probs_r).item()
                else:
                    probs_r = rotate_rho_prob(nn_state, bases[i], samples[i]) / Z
                    NLL_ -= probs_to_logits(probs_r).item()

        return NLL_ / float(len(samples))


def _single_basis_KL(target_probs, nn_probs):
    return torch.sum(target_probs * probs_to_logits(target_probs)) - torch.sum(
        target_probs * probs_to_logits(nn_probs)
    )


def KL(nn_state, target, space, bases=None, **kwargs):
    r"""A function for calculating the KL divergence averaged over every given
    basis.

    .. math:: KL(P_{target} \vert P_{RBM}) = -\sum_{x \in \mathcal{H}} P_{target}(x)\log(\frac{P_{RBM}(x)}{P_{target}(x)})

    :param nn_state: The neural network state.
    :type nn_state: qucumber.nn_states.NeuralStateBase
    :param target: The true state (wavefunction or density matrix) of the system.
                   Can be a dictionary with each value being the state
                   represented in a different basis, and the key identifying the basis.
    :type target: torch.Tensor or dict(str, torch.Tensor)
    :param space: The basis elements of the Hilbert space of the system :math:`\mathcal{H}`.
                  The ordering of the basis elements must match with the ordering of the
                  coefficients given in `target`.
    :type space: torch.Tensor
    :param bases: An array of unique bases. If given, the KL divergence will be
                  computed for each basis and the average will be returned.
    :type bases: numpy.ndarray
    :param \**kwargs: Extra keyword arguments that may be passed. Will be ignored.

    :returns: The KL divergence.
    :rtype: float
    """
    KL = 0.0

    if isinstance(target, dict):
        target = {k: v.to(nn_state.device) for k, v in target.items()}
        if bases is None:
            bases = list(target.keys())
        else:
            assert set(bases) == set(
                target.keys()
            ), "Given bases must match the keys of the target_psi dictionary."
    else:
        target = target.to(nn_state.device)

    Z = nn_state.normalization(space)

    if bases is None:
        target_probs = cplx.absolute_value(target) ** 2
        nn_probs = nn_state.probability(space, Z)

        KL += _single_basis_KL(target_probs, nn_probs)

    elif isinstance(nn_state, WaveFunctionBase):
        for basis in bases:
            if isinstance(target, dict):
                target_psi_r = target[basis]
                assert target_psi_r.dim() == 2, "target must be a complex vector!"
            else:
                assert target.dim() == 2, "target must be a complex vector!"
                target_psi_r = rotate_psi(nn_state, basis, space, psi=target)

            psi_r = rotate_psi(nn_state, basis, space)
            nn_probs_r = (cplx.absolute_value(psi_r) ** 2) / Z
            target_probs_r = cplx.absolute_value(target_psi_r) ** 2

            KL += _single_basis_KL(target_probs_r, nn_probs_r)

        KL /= float(len(bases))
    else:
        for basis in bases:
            if isinstance(target, dict):
                target_rho_r = target[basis]
                assert target_rho_r.dim() == 3, "target must be a complex matrix!"
            else:
                assert target.dim() == 3, "target must be a complex matrix!"
                target_rho_r = rotate_rho(nn_state, basis, space, rho=target)

            rho_r = rotate_rho(nn_state, basis, space)
            nn_probs_r = torch.diagonal(cplx.real(rho_r)) / Z
            target_probs_r = torch.diagonal(cplx.real(target_rho_r))

            KL += _single_basis_KL(target_probs_r, nn_probs_r)

        KL /= float(len(bases))

    return KL.item()
