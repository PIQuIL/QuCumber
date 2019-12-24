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
from qucumber.utils import cplx, deprecated_kwarg
from qucumber.utils.unitaries import rotate_psi, rotate_psi_inner_prod, rotate_rho_probs


@deprecated_kwarg(target_psi="target", target_rho="target")
def fidelity(nn_state, target, space=None, **kwargs):
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
                  coefficients given in `target`. If `None`, will generate them using
                  the provided `nn_state`.
    :type space: torch.Tensor
    :param \**kwargs: Extra keyword arguments that may be passed. Will be ignored.

    :returns: The fidelity.
    :rtype: float
    """
    space = space if space is not None else nn_state.generate_hilbert_space()
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
        rho_rbm_ = cplx.numpy(rho)
        target_ = cplx.numpy(target)

        sqrt_rho_rbm = sqrtm(rho_rbm_)
        prod = np.matmul(sqrt_rho_rbm, np.matmul(target_, sqrt_rho_rbm))

        # Instead of sqrt'ing then taking the trace, we compute the eigenvals,
        #  sqrt those, and then sum them up. This is a bit more efficient.
        eigvals = np.linalg.eigvals(prod).real  # imaginary parts should be zero
        eigvals = np.abs(eigvals)
        trace = np.sum(np.sqrt(eigvals))

        return trace ** 2


def NLL(nn_state, samples, space=None, sample_bases=None, **kwargs):
    r"""A function for calculating the negative log-likelihood (NLL).

    :param nn_state: The neural network state.
    :type nn_state: qucumber.nn_states.NeuralStateBase
    :param samples: Samples to compute the NLL on.
    :type samples: torch.Tensor
    :param space: The basis elements of the Hilbert space of the system :math:`\mathcal{H}`.
                  If `None`, will generate them using the provided `nn_state`.
    :type space: torch.Tensor
    :param sample_bases: An array of bases where measurements were taken.
    :type sample_bases: numpy.ndarray
    :param \**kwargs: Extra keyword arguments that may be passed. Will be ignored.

    :returns: The Negative Log-Likelihood.
    :rtype: float
    """
    space = space if space is not None else nn_state.generate_hilbert_space()
    Z = nn_state.normalization(space)

    if sample_bases is None:
        nn_probs = nn_state.probability(samples, Z)
        NLL_ = -torch.mean(probs_to_logits(nn_probs)).item()
        return NLL_
    else:
        NLL_ = 0.0

        unique_bases, indices = np.unique(sample_bases, axis=0, return_inverse=True)
        indices = torch.Tensor(indices).to(samples)

        for i in range(unique_bases.shape[0]):
            basis = unique_bases[i, :]
            rot_sites = np.where(basis != "Z")[0]

            if rot_sites.size != 0:
                if isinstance(nn_state, WaveFunctionBase):
                    Upsi = rotate_psi_inner_prod(
                        nn_state, basis, samples[indices == i, :]
                    )
                    nn_probs = (cplx.absolute_value(Upsi) ** 2) / Z
                else:
                    nn_probs = (
                        rotate_rho_probs(nn_state, basis, samples[indices == i, :]) / Z
                    )
            else:
                nn_probs = nn_state.probability(samples[indices == i, :], Z)

            NLL_ -= torch.sum(probs_to_logits(nn_probs))

        return NLL_ / float(len(samples))


def _single_basis_KL(target_probs, nn_probs):
    return torch.sum(target_probs * probs_to_logits(target_probs)) - torch.sum(
        target_probs * probs_to_logits(nn_probs)
    )


@deprecated_kwarg(target_psi="target", target_rho="target")
def KL(nn_state, target, space=None, bases=None, **kwargs):
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
                  coefficients given in `target`. If `None`, will generate them using
                  the provided `nn_state`.
    :type space: torch.Tensor
    :param bases: An array of unique bases. If given, the KL divergence will be
                  computed for each basis and the average will be returned.
    :type bases: numpy.ndarray
    :param \**kwargs: Extra keyword arguments that may be passed. Will be ignored.

    :returns: The KL divergence.
    :rtype: float
    """
    space = space if space is not None else nn_state.generate_hilbert_space()
    Z = nn_state.normalization(space)

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

    KL = 0.0

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
                target_probs_r = torch.diagonal(cplx.real(target_rho_r))
            else:
                assert target.dim() == 3, "target must be a complex matrix!"
                target_probs_r = rotate_rho_probs(nn_state, basis, space, rho=target)

            rho_r = rotate_rho_probs(nn_state, basis, space)
            nn_probs_r = rho_r / Z

            KL += _single_basis_KL(target_probs_r, nn_probs_r)

        KL /= float(len(bases))

    return KL.item()
