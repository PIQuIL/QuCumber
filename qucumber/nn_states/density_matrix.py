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
from itertools import chain
from math import ceil

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector
from tqdm import tqdm
from tqdm import tqdm_notebook

from qucumber.callbacks import CallbackList
from qucumber.callbacks import Timer
from qucumber.rbm import PurificationRBM
from qucumber.utils import cplx
from qucumber.utils import training_statistics as ts
from qucumber.utils import unitaries
from qucumber.utils.data import extract_refbasis_samples
from qucumber.utils.gradients_utils import vector_to_grads


class DensityMatrix:
    r"""
    :param num_visible: The number of visible units, i.e. the size of the system
    :type num_visible: int
    :param num_hidden: The number of units in the hidden layer
    :type num_hidden: int
    :param num_aux: The number of units in the purification layer
    :type num_aux: int
    :param unitary_dict: A dictionary associating bases with their unitary rotations
    :type unitary_dict: dict[str, torch.Tensor]
    :param gpu: Whether to perform computations on the default gpu.
    :type gpu: bool
    """

    _rbm_am = None
    _rbm_ph = None
    _device = None

    def __init__(self, num_visible, num_hidden, num_aux, unitary_dict=None, gpu=False):
        self.rbm_am = PurificationRBM(
            int(num_visible), int(num_hidden), int(num_aux), gpu=gpu
        )

        self.rbm_ph = PurificationRBM(
            int(num_visible), int(num_hidden), int(num_aux), gpu=gpu
        )

        self.num_visible = int(num_visible)
        self.num_hidden = int(num_hidden)
        self.num_aux = int(num_aux)

        self.device = self.rbm_am.device

        self.unitary_dict = unitary_dict if unitary_dict else unitaries.create_dict()
        self.unitary_dict = {k: v for k, v in self.unitary_dict.items()}

    @property
    def networks(self):
        return ["rbm_am", "rbm_ph"]

    @property
    def rbm_am(self):
        return self._rbm_am

    @rbm_am.setter
    def rbm_am(self, new_val):
        self._rbm_am = new_val

    @property
    def rbm_ph(self):
        """RBM used to learn the wavefunction phase."""
        return self._rbm_ph

    @rbm_ph.setter
    def rbm_ph(self, new_val):
        self._rbm_ph = new_val

    def generate_hilbert_space(self, n, device=None):
        r"""Generates Hilbert space of dimension :math:`2^n`.

        :param n: The size of each element in the Hilbert space
        :type n: int
        :returns: A tensor with the basis states of the Hilbert space
        :rtype: torch.Tensor
        """
        device = device if device is not None else self.device
        dim = np.arange(2 ** n)
        space = ((dim[:, None] & (1 << np.arange(n))) > 0)[:, ::-1]
        space = space.astype(int)
        return torch.tensor(space, dtype=torch.double, device=device)

    def subspace_vector(self, num, size, device=None):
        r"""Generates a single vector from Hilbert space of
            dimension :math:`2^{n}`

        :param num: The decimal representation of the desired
                    vector of the Hilbert space
        :type num: int
        :param size: The size of each element of the Hilbert space
        :type size: int
        :returns: A single vector from the Hilbert space
        :rtype: torch.Tensor
        """
        device = device if device is not None else self.device
        space = ((num & (1 << np.arange(size))) > 0)[::-1]
        space = space.astype(int)
        return torch.tensor(space, dtype=torch.double, device=device)

    def am_grads(self, v, vp):
        r"""Computes the gradients of the amplitude RBM for given input states

        :param v: The first input state, :math:`\sigma`
        :type v: torch.Tensor
        :param vp: The second input state, :math:`\sigma'`
        :type vp: torch.Tensor
        :returns: The gradients of all amplitude RBM parameters
        :rtype: torch.Tensor
        """
        return self.rbm_am.GammaP_grad(v, vp) + self.Pi_grad_am(v, vp)

    def ph_grads(self, v, vp):
        r"""Computes the gradients of the phase RBM for given input states

        :param v: The first input state, :math:`\sigma`
        :type v: torch.Tensor
        :param vp: The second input state, :math:`\sigma'`
        :type vp: torch.Tensor
        :returns: The gradients of all phase RBM parameters
        :rtype: torch.Tensor
        """
        return cplx.scalar_mult(  # need to multiply Gamma- by i
            self.rbm_ph.GammaM_grad(v, vp), torch.Tensor([0, 1])
        ) + self.Pi_grad_ph(v, vp)

    def Pi(self, v, vp):
        r"""Calculates an element of the :math:`\Pi` matrix

        :param v: One of the visible states, :math:`\sigma`
        :type v: torch.Tensor
        :param vp: The other visible state, :math`\sigma'`
        :type vp: torch.Tensor
        :returns: The matrix element given by :math:`\langle\sigma|\Pi|\sigma'\rangle`
        :rtype: torch.Tensor
        """
        if len(v.shape) < 2 and len(vp.shape) < 2:

            exp_arg = self.rbm_am.mixing_term(v + vp)
            phase = self.rbm_ph.mixing_term(v - vp)

            log_term = (
                (1 + 2 * exp_arg.exp() * phase.cos() + (2 * exp_arg).exp()).sqrt().log()
            )

            phase_term = (
                (exp_arg.exp() * phase.sin()) / (1 + exp_arg.exp() * phase.cos())
            ).atan()

            return cplx.make_complex(log_term.sum(), phase_term.sum())

        else:
            out = torch.zeros(
                2,
                2 ** self.num_visible,
                2 ** self.num_visible,
                dtype=torch.double,
                device=self.device,
            )
            for i in range(2 ** self.num_visible):
                for j in range(2 ** self.num_visible):
                    exp_arg = self.rbm_am.mixing_term(v[i] + vp[j])
                    phase = self.rbm_ph.mixing_term(v[i] - vp[j])

                    log_term = (
                        (1 + 2 * exp_arg.exp() * phase.cos() + (2 * exp_arg).exp())
                        .sqrt()
                        .log()
                    )

                    phase_term = (
                        (exp_arg.exp() * phase.sin())
                        / (1 + exp_arg.exp() * phase.cos())
                    ).atan()

                    out[0][i][j] = log_term.sum()
                    out[1][i][j] = phase_term.sum()

            return out

    def Pi_grad_am(self, v, vp, reduce=False):
        r"""Calculates the gradient of the :math:`\Pi` matrix with
            respect to the amplitude RBM parameters for two input states

        :param v: One of the visible states, :math:`\sigma`
        :type v: torch.Tensor
        :param vp: The other visible state, :math`\sigma'`
        :type vp: torch.Tensor
        :returns: The matrix element of the gradient given by
                  :math:`\langle\sigma|\nabla_\lambda\Pi|\sigma'\rangle`
        :rtype: torch.Tensor
        """
        argReal = self.rbm_am.mixing_term(v + vp)
        argIm = self.rbm_ph.mixing_term(v - vp)
        sigReal = cplx.sigmoid(argReal, argIm)[0]
        sigIm = cplx.sigmoid(argReal, argIm)[1]

        if v.dim() < 2:
            W_grad_real = torch.zeros_like(self.rbm_am.weights_W)
            W_grad_imag = W_grad_real.clone()
            vb_grad_real = torch.zeros_like(self.rbm_am.visible_bias)
            vb_grad_imag = vb_grad_real.clone()
            hb_grad_real = torch.zeros_like(self.rbm_am.hidden_bias)
            hb_grad_imag = hb_grad_real.clone()
            U_grad_real = 0.5 * torch.ger(sigReal, (v + vp))
            U_grad_imag = 0.5 * torch.ger(sigIm, (v + vp))
            ab_grad_real = sigReal
            ab_grad_imag = sigIm

            return cplx.make_complex(
                parameters_to_vector(
                    [W_grad_real, U_grad_real, vb_grad_real, hb_grad_real, ab_grad_real]
                ),
                parameters_to_vector(
                    [W_grad_imag, U_grad_imag, vb_grad_imag, hb_grad_imag, ab_grad_imag]
                ),
            )

        else:
            W_grad_real = torch.zeros(
                v.shape[0],
                self.rbm_am.weights_W.shape[0],
                self.rbm_am.weights_W.shape[1],
                dtype=torch.double,
            )
            W_grad_imag = W_grad_real.clone()
            vb_grad_real = torch.zeros(
                v.shape[0], self.rbm_am.num_visible, dtype=torch.double
            )
            vb_grad_imag = vb_grad_real.clone()
            hb_grad_real = torch.zeros(
                v.shape[0], self.rbm_am.num_hidden, dtype=torch.double
            )
            hb_grad_imag = hb_grad_real.clone()
            U_grad_real = 0.5 * (torch.einsum("ij,ik->ijk", sigReal, (v + vp)))
            U_grad_imag = 0.5 * (torch.einsum("ij,ik->ijk", sigIm, (v + vp)))
            ab_grad_real = sigReal
            ab_grad_imag = sigIm

            vec_real = [
                W_grad_real.view(v.size()[0], -1),
                U_grad_real.view(v.size()[0], -1),
                vb_grad_real,
                hb_grad_real,
                ab_grad_real,
            ]
            vec_imag = [
                W_grad_imag.view(v.size()[0], -1),
                U_grad_imag.view(v.size()[0], -1),
                vb_grad_imag,
                hb_grad_imag,
                ab_grad_imag,
            ]
            return cplx.make_complex(
                torch.cat(vec_real, dim=1), torch.cat(vec_imag, dim=1)
            )

    def Pi_grad_ph(self, v, vp, reduce=False):
        r"""Calculates the gradient of the :math:`\Pi` matrix with
            respect to the phase RBM parameters for two input states

        :param v: One of the visible states, :math:`\sigma`
        :type v: torch.Tensor
        :param vp: The other visible state, :math`\sigma'`
        :type vp: torch.Tensor
        :returns: The matrix element of the gradient given by
                  :math:`\langle\sigma|\nabla_\mu\Pi|\sigma'\rangle`
        :rtype: torch.Tensor
        """
        argReal = self.rbm_am.mixing_term(v + vp)
        argIm = self.rbm_ph.mixing_term(v - vp)
        sigmoid_term = cplx.sigmoid(argReal, argIm)
        sigReal = sigmoid_term[0]
        sigIm = sigmoid_term[1]

        if v.dim() < 2:
            W_grad_real = torch.zeros_like(self.rbm_ph.weights_W)
            W_grad_imag = W_grad_real.clone()
            vb_grad_real = torch.zeros_like(self.rbm_ph.visible_bias)
            vb_grad_imag = vb_grad_real.clone()
            hb_grad_real = torch.zeros_like(self.rbm_ph.hidden_bias)
            hb_grad_imag = hb_grad_real.clone()
            ab_grad_real = torch.zeros_like(self.rbm_ph.aux_bias)
            ab_grad_imag = ab_grad_real.clone()
            U_grad_real = -0.5 * torch.ger(sigIm, (v - vp))
            U_grad_imag = 0.5 * torch.ger(sigReal, (v - vp))

            return cplx.make_complex(
                parameters_to_vector(
                    [W_grad_real, U_grad_real, vb_grad_real, hb_grad_real, ab_grad_real]
                ),
                parameters_to_vector(
                    [W_grad_imag, U_grad_imag, vb_grad_imag, hb_grad_imag, ab_grad_imag]
                ),
            )

        else:
            W_grad_real = torch.zeros(
                v.shape[0],
                self.rbm_ph.weights_W.shape[0],
                self.rbm_ph.weights_W.shape[1],
                dtype=torch.double,
            )
            W_grad_imag = W_grad_real.clone()
            vb_grad_real = torch.zeros(
                v.shape[0], self.rbm_ph.num_visible, dtype=torch.double
            )
            vb_grad_imag = vb_grad_real.clone()
            hb_grad_real = torch.zeros(
                v.shape[0], self.rbm_ph.num_hidden, dtype=torch.double
            )
            hb_grad_imag = hb_grad_real.clone()
            ab_grad_real = torch.zeros(
                v.shape[0], self.rbm_ph.num_aux, dtype=torch.double
            )
            ab_grad_imag = ab_grad_real.clone()
            U_grad_real = -0.5 * (torch.einsum("ij,ik->ijk", sigIm, (v - vp)))
            U_grad_imag = 0.5 * (torch.einsum("ij,ik->ijk", sigReal, (v - vp)))

            vec_real = [
                W_grad_real.view(v.size()[0], -1),
                U_grad_real.view(v.size()[0], -1),
                vb_grad_real,
                hb_grad_real,
                ab_grad_real,
            ]
            vec_imag = [
                W_grad_imag.view(v.size()[0], -1),
                U_grad_imag.view(v.size()[0], -1),
                vb_grad_imag,
                hb_grad_imag,
                ab_grad_imag,
            ]
            return cplx.make_complex(
                torch.cat(vec_real, dim=1), torch.cat(vec_imag, dim=1)
            )

    def rhoRBM_tilde(self, v, vp):
        r"""Computes the matrix elements of the current density matrix
        excluding the partition function

        :param v: One of the visible states, :math:`\sigma`
        :type v: torch.Tensor
        :param vp: The other visible state, :math`\sigma'`
        :type vp: torch.Tensor
        :returns: The element of the current density matrix
                  :math:`\langle\sigma|\widetilde{\rho}|\sigma'\rangle`
        :rtype: torch.Tensor
        """
        if len(v.shape) < 2 and len(vp.shape) < 2:
            out = torch.zeros(2, dtype=torch.double)
        else:
            out = torch.zeros(
                2, 2 ** self.num_visible, 2 ** self.num_visible, dtype=torch.double
            )
        pi_ = self.Pi(v, vp)
        amp = (self.rbm_am.GammaP(v, vp) + pi_[0]).exp()
        phase = self.rbm_ph.GammaM(v, vp) + pi_[1]

        out[0] = amp * (phase.cos())
        out[1] = amp * (phase.sin())

        return out

    def rhoRBM(self, v, vp):
        r"""Computes the matrix elements of the current density matrix

        :param v: One of the visible states, :math:`\sigma`
        :type v: torch.Tensor
        :param vp: The other visible state, :math`\sigma'`
        :type vp: torch.Tensor
        :returns: The element of the current density matrix
                  :math:`\langle\sigma|\rho|\sigma'\rangle`
        :rtype: torch.Tensor
        """
        return self.rhoRBM_tilde(v, vp) / torch.trace(self.rhoRBM_tilde(v, vp)[0])

    def init_gradient(self, basis, sites):
        r"""Initalizes all required variables for gradient computation

        :param basis: The bases of the measurements
        :type basis: np.array
        :param sites: The sites where the measurements are not
                      in the computational basis
        """
        UrhoU = torch.zeros(2, dtype=torch.double)
        v = torch.zeros(self.num_visible, dtype=torch.double)
        vp = torch.zeros(self.num_visible, dtype=torch.double)
        Us = torch.stack([self.unitary_dict[b] for b in basis[sites]]).numpy()
        Us_dag = torch.stack(
            [cplx.conjugate(self.unitary_dict[b]) for b in basis[sites]]
        ).numpy()

        rotated_grad = [
            torch.zeros(2, getattr(self, net).num_pars, dtype=torch.double)
            for net in self.networks
        ]

        return UrhoU, v, vp, Us, Us_dag, rotated_grad

    def rotated_gradient(self, basis, sites, sample):
        r"""Computes the gradients rotated into the measurement basis

        :param basis: The bases in which the measurement is made
        :type basis: np.array
        :param sites: The sites where the measurements are not made
                      in the computational basis
        :type sites: np.array
        :param sample: The measurement (either 0 or 1)
        :type sample: torch.Tensor
        :returns: A list of two tensors, representing the rotated gradients
                  of the amplitude and phase RBMS
        :rtype: list[torch.Tensor, torch.Tensor]
        """
        UrhoU, v, vp, Us, Us_dag, rotated_grad = self.init_gradient(basis, sites)
        int_sample = sample[sites].round().int().numpy()
        ints_size = np.arange(sites.size)

        U_ = torch.tensor([1.0, 1.0], dtype=torch.double)
        UrhoU = torch.zeros(2, dtype=torch.double)
        UrhoU_ = torch.zeros_like(UrhoU)

        grad_size = (
            self.num_visible * self.num_hidden
            + self.num_visible * self.num_aux
            + self.num_visible
            + self.num_hidden
            + self.num_aux
        )
        Z2 = torch.zeros((2, grad_size), dtype=torch.double)

        v = sample.round().clone()
        vp = sample.round().clone()

        for x in range(2 ** sites.size):
            v = sample.round().clone()
            v[sites] = self.subspace_vector(x, sites.size)
            int_v = v[sites].int().numpy()
            all_Us = Us[ints_size, :, int_sample, int_v]

            for y in range(2 ** sites.size):
                vp = sample.round().clone()
                vp[sites] = self.subspace_vector(y, sites.size)
                int_vp = vp[sites].int().numpy()
                all_Us_dag = Us[ints_size, :, int_sample, int_vp]

                Ut = np.prod(all_Us[:, 0] + (1j * all_Us[:, 1]))
                Ut *= np.prod(np.conj(all_Us_dag[:, 0] + (1j * all_Us_dag[:, 1])))
                U_[0] = Ut.real
                U_[1] = Ut.imag

                cplx.scalar_mult(U_, self.rhoRBM_tilde(v, vp), out=UrhoU_)
                UrhoU += UrhoU_

                grad0 = self.am_grads(v, vp)
                grad1 = self.ph_grads(v, vp)

                rotated_grad[0] += cplx.scalar_mult(UrhoU_, grad0, out=Z2)
                rotated_grad[1] += cplx.scalar_mult(UrhoU_, grad1, out=Z2)

        grad = [
            cplx.scalar_divide(rotated_grad[0], UrhoU),
            cplx.scalar_divide(rotated_grad[1], UrhoU),
        ]

        return grad

    def gradient(self, basis, sample):
        r"""Computes the gradient of the amplitude and phase RBM parameters

        :param basis: The bases in which the measurements are made
        :type basis: np.array
        :param sample: The measurements
        :type sample: torch.Tensor
        :returns: A list containing the amplitude and phase
                  RBM parameter gradients
        :rtype: list[torch.Tensor, torch.Tensor]
        """
        basis = np.array(list(basis))
        rot_sites = np.where(basis != "Z")[0]

        if rot_sites.size == 0:
            grad = [cplx.real(self.am_grads(sample, sample)), 0.0]

        else:
            grad = self.rotated_gradient(basis, rot_sites, sample)

        return grad

    def compute_exact_grads(self, train_samples, train_bases, Z):
        r"""Computes the gradients of the parameters, usign exact sampling
        for the negative phase update instead of Gibbs sampling

        :param train_samples: The measurements
        :type train_samples: torch.Tensor
        :param train_bases: The bases in which the measurements are made
        :type train_bases: np.array
        :param Z: The partition function
        :type Z: torch.Tensor
        :returns: A list containing the amplitude and phase RBM gradients
                  calculated with exact sampling for negative phase update
        :rtype: list[torch.Tensor, torch.Tensor]
        """
        grad = [0.0, 0.0]

        grad_data = [
            torch.zeros(2, getattr(self, net).num_pars, dtype=torch.double)
            for net in self.networks
        ]

        grad_model = [torch.zeros(self.rbm_am.num_pars, dtype=torch.double)]

        v_space = self.generate_hilbert_space(self.num_visible)
        rho_rbm = self.rhoRBM(v_space, v_space)

        for i in range(train_samples.shape[0]):
            data_gradient = self.gradient(train_bases[i], train_samples[i])
            grad_data[0] += data_gradient[0]
            grad_data[1] += data_gradient[1]

        # Can just take the real parts since the imaginary parts will have cancelled out
        grad[0] = -cplx.real(grad_data[0]) / float(train_samples.shape[0])
        grad[1] = -cplx.real(grad_data[1]) / float(train_samples.shape[0])

        for i in range(2 ** self.num_visible):
            grad_model[0] += rho_rbm[0][i][i] * cplx.real(
                self.am_grads(v_space[i], v_space[i])
            )

        grad[0] += grad_model[0]

        return grad

    def compute_batch_gradients(self, k, samples_batch, neg_batch, bases_batch):
        r"""Compute the gradients of a batch of training data

        :param k: The number of contrastive divergence steps
        :type k: int
        :param samples_batch: Batch of input samples
        :type samples_batch: torch.Tensor
        :param neg_batch: Batch to be used in the calculation of the negative phase
        :type neg_batch: torch.Tensor
        :param bases_batch: The bases in which the measurements are taken, which
                            correspond to the measurements in samples_batch
        :returns: List containing the gradients of amplitude and phase RBM parameters
        :rtype: list[torch.Tensor, torch.Tensor]
        """
        vk = self.rbm_am.gibbs_steps(k, neg_batch)
        grad_model = cplx.real(self.am_grads(vk, vk)).sum(0)

        grad = [0.0, 0.0]

        grad_data = [
            torch.zeros(2, getattr(self, net).num_pars, dtype=torch.double)
            for net in self.networks
        ]

        for i in range(samples_batch.shape[0]):

            data_gradient = self.gradient(bases_batch[i], samples_batch[i])
            grad_data[0] += data_gradient[0]
            grad_data[1] += data_gradient[1]

        # Can just take the real parts now since the imaginary parts have cancelled
        grad[0] = -cplx.real(grad_data[0]) / float(samples_batch.shape[0])
        grad[0] += grad_model / float(neg_batch.shape[0])
        grad[1] = -cplx.real(grad_data[1]) / float(samples_batch.shape[0])

        return grad

    def get_param_status(self, i, param_ranges):
        for p, rng in param_ranges.items():
            if i in rng:
                return p, i == rng[0]

    def rotate_rho(self, basis, space, Z, unitaries, rho=None):
        r"""Computes the density matrix rotated into some basis

        :param basis: The basis into which to rotate the density matrix
        :type basis: np.array
        :param space: The Hilbert space of the system
        :type space: torch.Tensor
        :param unitaries: A dictionary of unitary matrices associated with
                          rotation into each basis
        :type unitaries: dict[str, torch.Tensor]
        :returns: The rotated denstiy matrix
        :rtype: torch.Tensor
        """
        rho = self.rhoRBM(space, space) if rho is None else rho

        unitaries = {k: v for k, v in unitaries.items()}
        us = [unitaries[b] for b in basis]

        # After ensuring there is more than one measurement, compute the
        # composite unitary by repeated Kronecker products

        if len(us) != 1:
            for index in range(len(us) - 1):
                us[index + 1] = cplx.kronecker_prod(us[index], us[index + 1])

        U = us[-1]
        U_dag = cplx.conjugate(U)

        rot_rho = cplx.matmul(rho, U_dag)
        rot_rho_ = cplx.matmul(U, rot_rho)

        return rot_rho_

    def _shuffle_data(
        self,
        pos_batch_size,
        neg_batch_size,
        num_batches,
        train_samples,
        input_bases,
        z_samples,
    ):
        pos_batch_perm = torch.randperm(train_samples.shape[0])

        shuffled_pos_samples = train_samples[pos_batch_perm]
        if input_bases is None:
            if neg_batch_size == pos_batch_size:
                neg_batch_perm = pos_batch_perm
            else:
                neg_batch_perm = torch.randint(
                    train_samples.shape[0],
                    size=(num_batches * neg_batch_size,),
                    dtype=torch.long,
                )
            shuffled_neg_samples = train_samples[neg_batch_perm]
        else:
            neg_batch_perm = torch.randint(
                z_samples.shape[0],
                size=(num_batches * neg_batch_size,),
                dtype=torch.long,
            )
            shuffled_neg_samples = z_samples[neg_batch_perm]

        # List of all the batches for positive phase.
        pos_batches = [
            shuffled_pos_samples[batch_start : (batch_start + pos_batch_size)]
            for batch_start in range(0, len(shuffled_pos_samples), pos_batch_size)
        ]

        neg_batches = [
            shuffled_neg_samples[batch_start : (batch_start + neg_batch_size)]
            for batch_start in range(0, len(shuffled_neg_samples), neg_batch_size)
        ]

        if input_bases is not None:
            shuffled_pos_bases = input_bases[pos_batch_perm]
            pos_batches_bases = [
                shuffled_pos_bases[batch_start : (batch_start + pos_batch_size)]
                for batch_start in range(0, len(train_samples), pos_batch_size)
            ]
            return zip(pos_batches, neg_batches, pos_batches_bases)
        else:
            return zip(pos_batches, neg_batches)

    def fit(
        self,
        data,
        input_bases,
        target=None,
        epochs=100,
        pos_batch_size=100,
        neg_batch_size=None,
        k=1,
        lr=1,
        progbar=False,
        starting_epoch=1,
        callbacks=None,
        time=False,
        optimizer=torch.optim.Adadelta,
        scheduler=torch.optim.lr_scheduler.MultiStepLR,
        lr_drop_epoch=50,
        lr_drop_factor=1.0,
        bases=None,
        train_to_fid=False,
        track_fid=False,
        **kwargs,
    ):
        r"""Trains the density matrix

        :param data: The training samples
        :type data: np.array
        :param input_bases: The measurement bases for each sample
        :type input_bases: np.array
        :param target: The density matrix you are trying to train towards
        :type target: torch.Tensor
        :param epochs: The number of epochs to train for
        :type epochs: int
        :param pos_batch_size: The size of batches for the positive phase
        :type pos_batch_size: int
        :param neg_batch_size: The size of batches for the negative phase
        :type neg_batch_size: int
        :param k: Number of contrastive divergence steps
        :type k: int
        :param lr: Learning rate - different meaning depending on optimizer!
        :type lr: float
        :param progbar: Whether or note to use a progress bar. Pass "notebook"
                        for a Jupyter notebook-friendly version
        :type progbar: bool or str
        :param starting_epoch: The epoch to start from
        :type starting_epoch: int
        :param callbacks: Callbacks to run while training
        :type callbacks: list[qucumber.callbacks.CallbackBase]
        :param optimizer: The constructor of a torch optimizer
        :type optimizer: torch.optim.Optimizer
        :param scheduler: The constructor of a torch scheduler
        :type scheduler: torch.optim.LRScheduler
        :param lr_drop_epoch: The epoch, or list of epochs, at which the
                              base learning rate is dropped
        :type lr_drop_epoch: int or list[int]
        :param lr_drop_factor: The factor by which the scheduler will decrease the
                               learning after the prescribed number of steps
        :type lr_drop_factor: float
        :param bases: All bases in which a measurement is made. Used to check gradients
        :type bases: np.array
        :param train_to_fid: Instructs the RBM to end training prematurely if the
                             specified fidelity is reached. If it is never reached,
                             training will continue until the specified epoch
        :type train_to_fid: float or bool
        :param track_fid: A file to which to write fidelity at every epoch.
                          Useful for keeping track of training run in background
        :type track_fid: str or bool
        """
        disable_progbar = progbar is False
        progress_bar = tqdm_notebook if progbar == "notebook" else tqdm
        lr_drop_epoch = (
            [lr_drop_epoch] if isinstance(lr_drop_epoch, int) else lr_drop_epoch
        )

        callbacks = CallbackList(callbacks if callbacks else [])
        if time:
            callbacks.append(Timer())

        train_samples = data.clone().detach().double()

        neg_batch_size = neg_batch_size if neg_batch_size else pos_batch_size

        all_params = [getattr(self, net).parameters() for net in self.networks]
        all_params = list(chain(*all_params))

        optimizer = optimizer(all_params, lr=lr, **kwargs)
        scheduler = scheduler(optimizer, lr_drop_epoch, gamma=lr_drop_factor)

        z_samples = extract_refbasis_samples(train_samples, input_bases)

        num_batches = ceil(train_samples.shape[0] / pos_batch_size)

        # here for now to test shit

        callbacks.on_train_start(self)

        for ep in progress_bar(
            range(starting_epoch, epochs + 1), desc="Epochs ", disable=disable_progbar
        ):

            data_iterator = self._shuffle_data(
                pos_batch_size,
                neg_batch_size,
                num_batches,
                train_samples,
                input_bases,
                z_samples,
            )
            callbacks.on_epoch_start(self, ep)

            for b, batch in enumerate(data_iterator):
                callbacks.on_batch_start(self, ep, b)
                all_grads = self.compute_batch_gradients(k, *batch)
                optimizer.zero_grad()

                for i, net in enumerate(self.networks):

                    rbm = getattr(self, net)
                    vector_to_grads(all_grads[i], rbm.parameters())

                optimizer.step()

                callbacks.on_batch_end(self, ep, b)

            callbacks.on_epoch_end(self, ep)

            scheduler.step()

            if train_to_fid or track_fid:
                v_space = self.generate_hilbert_space(self.num_visible)
                fidel = ts.density_matrix_fidelity(self, target, v_space)

            if track_fid:
                f = open(track_fid, "a")
                f.write("Epoch: {0}\tFidelity: {1}\n".format(ep, fidel))
                f.close()

            if train_to_fid:
                if fidel >= train_to_fid:
                    print(
                        "\n\nTarget fidelity of", train_to_fid, "reached or exceeded!"
                    )
                    break

        callbacks.on_train_end(self)

    def save(self, location, metadata=None):
        """Saves the DensityMatrix parameters to the given location along with
        any given metadata.

        :param location: The location to save the data.
        :type location: str or file
        :param metadata: Any extra metadata to store alongside the DensityMatrix
                         parameters.
        :type metadata: dict
        """
        # add extra metadata to dictionary before saving it to disk
        metadata = metadata if metadata else {}
        metadata["unitary_dict"] = self.unitary_dict

        # validate metadata
        for net in self.networks:
            if net in metadata.keys():
                raise ValueError(
                    "Invalid key in metadata; '{}' cannot be a key!".format(net)
                )

        data = {net: getattr(self, net).state_dict() for net in self.networks}
        data.update(**metadata)
        torch.save(data, location)

    def load(self, location):
        """Loads the DensityMatrix parameters from the given location ignoring any
        metadata stored in the file. Overwrites the DensityMatrix's parameters.

        .. note::
            The DensityMatrix object on which this function is called must
            have the same parameter shapes as the one who's parameters are being
            loaded.

        :param location: The location to load the DensityMatrix parameters from.
        :type location: str or file
        """
        state_dict = torch.load(location, map_location=self.device)

        for net in self.networks:
            getattr(self, net).load_state_dict(state_dict[net])
