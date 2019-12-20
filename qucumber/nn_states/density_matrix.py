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


import warnings

import numpy as np
import torch
from torch.nn import functional as F

from qucumber import _warn_on_missing_gpu
from qucumber.utils import cplx, unitaries
from qucumber.rbm import PurificationRBM
from .neural_state import NeuralStateBase


class DensityMatrix(NeuralStateBase):
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

    def __init__(
        self,
        num_visible,
        num_hidden=None,
        num_aux=None,
        unitary_dict=None,
        gpu=False,
        module=None,
    ):
        if gpu and torch.cuda.is_available():
            warnings.warn(
                "Using DensityMatrix on GPU is not recommended due to poor performance compared to CPU.",
                ResourceWarning,
                2,
            )
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if module is None:
            self.rbm_am = PurificationRBM(num_visible, num_hidden, num_aux, gpu=gpu)
            self.rbm_ph = PurificationRBM(num_visible, num_hidden, num_aux, gpu=gpu)
        else:
            _warn_on_missing_gpu(gpu)
            self.rbm_am = module.to(self.device)
            self.rbm_am.device = self.device
            self.rbm_ph = module.to(self.device).clone()
            self.rbm_ph.device = self.device

        self.num_visible = self.rbm_am.num_visible
        self.num_hidden = self.rbm_am.num_hidden
        self.num_aux = self.rbm_am.num_aux
        self.device = self.rbm_am.device

        self.unitary_dict = unitary_dict if unitary_dict else unitaries.create_dict()
        self.unitary_dict = {
            k: v.to(device=self.device) for k, v in self.unitary_dict.items()
        }

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

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, new_val):
        self._device = new_val

    def pi(self, v, vp, expand=True):
        r"""Calculates elements of the :math:`\Pi` matrix.
        If `expand` is `True`, will return a complex matrix
        :math:`A_{ij} = \langle\sigma_i|\Pi|\sigma'_j\rangle`.
        Otherwise will return a complex vector
        :math:`A_{i} = \langle\sigma_i|\Pi|\sigma'_i\rangle`.

        :param v: A batch of visible states, :math:`\sigma`.
        :type v: torch.Tensor
        :param vp: The other batch of visible state, :math`\sigma'`.
        :type vp: torch.Tensor
        :param expand: Whether to return a matrix (`True`) or a vector (`False`).
        :type expand: bool

        :returns: The matrix elements given by :math:`\langle\sigma|\Pi|\sigma'\rangle`
        :rtype: torch.Tensor
        """
        m_am = F.linear(v, self.rbm_am.weights_U, self.rbm_am.aux_bias)
        mp_am = F.linear(vp, self.rbm_am.weights_U, self.rbm_am.aux_bias)

        m_ph = F.linear(v, self.rbm_ph.weights_U)
        mp_ph = F.linear(vp, self.rbm_ph.weights_U)

        if expand and v.dim() >= 2:
            m_am = m_am.unsqueeze_(1)
            m_ph = m_ph.unsqueeze_(1)
        if expand and vp.dim() >= 2:
            mp_am = mp_am.unsqueeze_(0)
            mp_ph = mp_ph.unsqueeze_(0)

        exp_arg = (m_am + mp_am) / 2
        phase = (m_ph - mp_ph) / 2

        real = (
            (1 + 2 * exp_arg.exp() * phase.cos() + (2 * exp_arg).exp())
            .sqrt()
            .log()
            .sum(-1)
        )

        imag = torch.atan2(
            (exp_arg.exp() * phase.sin()), (1 + exp_arg.exp() * phase.cos())
        ).sum(-1)

        return cplx.make_complex(real, imag)

    def pi_grad_am(self, v, vp):
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
        unsqueezed = v.dim() < 2 or vp.dim() < 2
        v = (v.unsqueeze(0) if v.dim() < 2 else v).to(self.rbm_am.weights_W)
        vp = (vp.unsqueeze(0) if vp.dim() < 2 else vp).to(self.rbm_am.weights_W)

        arg_real = self.rbm_am.mixing_term(v + vp)
        arg_imag = self.rbm_ph.mixing_term(v - vp)
        sig = cplx.sigmoid(arg_real, arg_imag)
        sig_real = cplx.real(sig)
        sig_imag = cplx.imag(sig)

        W_grad = torch.zeros_like(self.rbm_am.weights_W).expand(v.shape[0], -1, -1)
        vb_grad = torch.zeros_like(self.rbm_am.visible_bias).expand(v.shape[0], -1)
        hb_grad = torch.zeros_like(self.rbm_am.hidden_bias).expand(v.shape[0], -1)
        U_grad_real = 0.5 * (torch.einsum("ij,ik->ijk", sig_real, (v + vp)))
        U_grad_imag = 0.5 * (torch.einsum("ij,ik->ijk", sig_imag, (v + vp)))
        ab_grad_real = sig_real
        ab_grad_imag = sig_imag

        vec_real = [
            W_grad.view(v.size()[0], -1),
            U_grad_real.view(v.size()[0], -1),
            vb_grad,
            hb_grad,
            ab_grad_real,
        ]
        vec_imag = [
            W_grad.view(v.size()[0], -1).clone(),
            U_grad_imag.view(v.size()[0], -1),
            vb_grad.clone(),
            hb_grad.clone(),
            ab_grad_imag,
        ]

        if unsqueezed:
            vec_real = [grad.squeeze_(0) for grad in vec_real]
            vec_imag = [grad.squeeze_(0) for grad in vec_imag]

        return cplx.make_complex(
            torch.cat(vec_real, dim=-1), torch.cat(vec_imag, dim=-1)
        )

    def pi_grad_ph(self, v, vp):
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
        unsqueezed = v.dim() < 2 or vp.dim() < 2
        v = (v.unsqueeze(0) if v.dim() < 2 else v).to(self.rbm_ph.weights_W)
        vp = (vp.unsqueeze(0) if vp.dim() < 2 else vp).to(self.rbm_ph.weights_W)

        arg_real = self.rbm_am.mixing_term(v + vp)
        arg_imag = self.rbm_ph.mixing_term(v - vp)
        sig = cplx.sigmoid(arg_real, arg_imag)
        sig_real = cplx.real(sig)
        sig_imag = cplx.imag(sig)

        W_grad = torch.zeros_like(self.rbm_ph.weights_W).expand(v.shape[0], -1, -1)
        vb_grad = torch.zeros_like(self.rbm_ph.visible_bias).expand(v.shape[0], -1)
        hb_grad = torch.zeros_like(self.rbm_ph.hidden_bias).expand(v.shape[0], -1)
        ab_grad = torch.zeros_like(self.rbm_ph.aux_bias).expand(v.shape[0], -1)
        U_grad_real = -0.5 * (torch.einsum("ij,ik->ijk", sig_imag, (v - vp)))
        U_grad_imag = 0.5 * (torch.einsum("ij,ik->ijk", sig_real, (v - vp)))

        vec_real = [
            W_grad.view(v.size()[0], -1),
            U_grad_real.view(v.size()[0], -1),
            vb_grad,
            hb_grad,
            ab_grad,
        ]
        vec_imag = [
            W_grad.view(v.size()[0], -1).clone(),
            U_grad_imag.view(v.size()[0], -1),
            vb_grad.clone(),
            hb_grad.clone(),
            ab_grad.clone(),
        ]

        if unsqueezed:
            vec_real = [grad.squeeze_(0) for grad in vec_real]
            vec_imag = [grad.squeeze_(0) for grad in vec_imag]

        return cplx.make_complex(
            torch.cat(vec_real, dim=-1), torch.cat(vec_imag, dim=-1)
        )

    def rho(self, v, vp=None, expand=True):
        r"""Computes the matrix elements of the (unnormalized) density matrix.
        If `expand` is `True`, will return a complex matrix
        :math:`A_{ij} = \langle\sigma_i|\widetilde{\rho}|\sigma'_j\rangle`.
        Otherwise will return a complex vector
        :math:`A_{i} = \langle\sigma_i|\widetilde{\rho}|\sigma'_i\rangle`.

        :param v: One of the visible states, :math:`\sigma`.
        :type v: torch.Tensor
        :param vp: The other visible state, :math:`\sigma'`.
                   If `None`, will be set to `v`.
        :type vp: torch.Tensor
        :param expand: Whether to return a matrix (`True`) or a vector (`False`).
        :type expand: bool

        :returns: The elements of the current density matrix
                  :math:`\langle\sigma|\widetilde{\rho}|\sigma'\rangle`
        :rtype: torch.Tensor
        """
        if expand is False and vp is None:
            return cplx.make_complex(self.probability(v))

        pi_ = self.pi(v, vp, expand=expand)
        amp = (self.rbm_am.gamma_plus(v, vp, expand=expand) + cplx.real(pi_)).exp()
        phase = self.rbm_ph.gamma_minus(v, vp, expand=expand) + cplx.imag(pi_)

        return cplx.make_complex(amp * phase.cos(), amp * phase.sin())

    def importance_sampling_numerator(self, iter_sample, drawn_sample):
        return self.rho(drawn_sample, iter_sample, expand=False)

    def importance_sampling_denominator(self, drawn_sample):
        return self.probability(drawn_sample)

    def init_gradient(self, basis, sites):
        r"""Initializes all required variables for gradient computation

        :param basis: The bases of the measurements
        :type basis: numpy.ndarray
        :param sites: The sites where the measurements are not
                      in the computational basis
        """
        UrhoU = torch.zeros(2, dtype=torch.double, device=self.device)
        Us = torch.stack([self.unitary_dict[b] for b in basis[sites]]).cpu().numpy()

        rotated_grad = [
            torch.zeros(
                2, getattr(self, net).num_pars, dtype=torch.double, device=self.device
            )
            for net in self.networks
        ]

        return UrhoU, Us, rotated_grad

    def rotated_gradient(self, basis, sites, sample):
        r"""Computes the gradients rotated into the measurement basis

        :param basis: The bases in which the measurement is made
        :type basis: numpy.ndarray
        :param sites: The sites where the measurements are not made
                      in the computational basis
        :type sites: numpy.ndarray
        :param sample: The measurement (either 0 or 1)
        :type sample: torch.Tensor

        :returns: A list of two tensors, representing the rotated gradients
                  of the amplitude and phase RBMS
        :rtype: list[torch.Tensor, torch.Tensor]
        """
        UrhoU, Us, rotated_grad = self.init_gradient(basis, sites)
        int_sample = sample[sites].round().int().cpu().numpy()
        ints_size = np.arange(sites.size)

        # if the number of rotated sites is too large, fallback to loop
        #  since memory may be unable to store the entire expanded set of
        #  visible states
        if (2 * sites.size) > self.max_size or (
            hasattr(self, "debug_gradient_rotation") and self.debug_gradient_rotation
        ):
            U_ = torch.tensor([1.0, 1.0], dtype=torch.double, device=self.device)
            UrhoU_ = torch.zeros_like(UrhoU)
            Z2 = torch.zeros(
                (2, self.rbm_am.num_pars), dtype=torch.double, device=self.device
            )

            v = sample.round().clone()
            vp = sample.round().clone()

            for x in range(2 ** sites.size):
                v = sample.round().clone()
                v[sites] = self.subspace_vector(x, sites.size)
                int_v = v[sites].int().cpu().numpy()
                all_Us = Us[ints_size, :, int_sample, int_v]

                for y in range(2 ** sites.size):
                    vp = sample.round().clone()
                    vp[sites] = self.subspace_vector(y, sites.size)
                    int_vp = vp[sites].int().cpu().numpy()
                    all_Us_dag = Us[ints_size, :, int_sample, int_vp]

                    Ut = np.prod(all_Us[:, 0] + (1j * all_Us[:, 1]))
                    Ut *= np.prod(np.conj(all_Us_dag[:, 0] + (1j * all_Us_dag[:, 1])))
                    U_[0] = Ut.real
                    U_[1] = Ut.imag

                    cplx.scalar_mult(U_, self.rho(v, vp), out=UrhoU_)
                    UrhoU += UrhoU_

                    grad0 = self.am_grads(v, vp)
                    grad1 = self.ph_grads(v, vp)

                    rotated_grad[0] += cplx.scalar_mult(UrhoU_, grad0, out=Z2)
                    rotated_grad[1] += cplx.scalar_mult(UrhoU_, grad1, out=Z2)
        else:
            v = sample.round().clone().unsqueeze(0).repeat(2 ** sites.size, 1)
            v[:, sites] = self.generate_hilbert_space(size=sites.size)
            v = v.contiguous()

            int_v = v[:, sites].int().cpu().numpy()
            all_Us = Us[ints_size, :, int_sample, int_v]
            Ut = np.prod(all_Us[..., 0] + (1j * all_Us[..., 1]), axis=1)
            Ut = np.outer(Ut, np.conj(Ut))
            U = (
                cplx.make_complex(torch.tensor(Ut.real), torch.tensor(Ut.imag))
                .to(sample)
                .contiguous()
            )
            UrhoU_v = cplx.scalar_mult(U, self.rho(v, v).detach())
            UrhoU = torch.sum(UrhoU_v, dim=(1, 2))

            for i in range(v.shape[0]):
                for j in range(v.shape[0]):
                    rotated_grad[0] += cplx.scalar_mult(
                        UrhoU_v[:, i, j], self.am_grads(v[i, ...], v[j, ...])
                    )
                    rotated_grad[1] += cplx.scalar_mult(
                        UrhoU_v[:, i, j], self.ph_grads(v[i, ...], v[j, ...])
                    )

        grad = [
            -cplx.real(cplx.scalar_divide(rotated_grad[0], UrhoU)),
            -cplx.real(cplx.scalar_divide(rotated_grad[1], UrhoU)),
        ]

        return grad

    def am_grads(self, v, vp):
        r"""Computes the gradients of the amplitude RBM for given input states

        :param v: The first input state, :math:`\sigma`
        :type v: torch.Tensor
        :param vp: The second input state, :math:`\sigma'`
        :type vp: torch.Tensor
        :returns: The gradients of all amplitude RBM parameters
        :rtype: torch.Tensor
        """
        return self.rbm_am.gamma_plus_grad(v, vp) + self.pi_grad_am(v, vp)

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
            self.rbm_ph.gamma_minus_grad(v, vp), torch.Tensor([0, 1])
        ) + self.pi_grad_ph(v, vp)

    def fit(
        self,
        data,
        epochs=100,
        pos_batch_size=100,
        neg_batch_size=None,
        k=1,
        lr=1,
        input_bases=None,
        progbar=False,
        starting_epoch=1,
        time=False,
        callbacks=None,
        optimizer=torch.optim.SGD,
        optimizer_args=None,
        scheduler=None,
        scheduler_args=None,
        **kwargs,
    ):
        if input_bases is None:
            raise ValueError("input_bases must be provided to train a DensityMatrix!")
        else:
            super().fit(
                data=data,
                epochs=epochs,
                pos_batch_size=pos_batch_size,
                neg_batch_size=neg_batch_size,
                k=k,
                lr=lr,
                input_bases=input_bases,
                progbar=progbar,
                starting_epoch=starting_epoch,
                time=time,
                callbacks=callbacks,
                optimizer=optimizer,
                optimizer_args=optimizer_args,
                scheduler=scheduler,
                scheduler_args=scheduler_args,
                **kwargs,
            )

    def normalization(self, space):
        r"""Compute the normalization constant of the state.

        .. math::

            Z_{\bm{\lambda}}=
            \sum_{\bm{\sigma}} \rho_{\bm{\lambda}}(\bm{\sigma}, \bm{\sigma})
            \sum_{\bm{\sigma}} p_{\bm{\lambda}}(\bm{\sigma})

        :param space: A rank 2 tensor of the entire visible space.
        :type space: torch.Tensor

        """
        return super().normalization(space)

    @staticmethod
    def autoload(location, gpu=False):
        state_dict = torch.load(location)
        nn_state = DensityMatrix(
            unitary_dict=state_dict["unitary_dict"],
            num_visible=len(state_dict["rbm_am"]["visible_bias"]),
            num_hidden=len(state_dict["rbm_am"]["hidden_bias"]),
            num_aux=len(state_dict["rbm_am"]["aux_bias"]),
            gpu=gpu,
        )
        nn_state.load(location)
        return nn_state
