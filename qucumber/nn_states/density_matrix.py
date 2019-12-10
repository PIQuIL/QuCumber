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

from itertools import chain
from math import ceil

from torch.nn import functional as F

from tqdm import tqdm, tqdm_notebook

from qucumber import _warn_on_missing_gpu
from qucumber.utils import cplx, unitaries, training_statistics as ts
from qucumber.utils.data import extract_refbasis_samples
from qucumber.utils.gradients_utils import vector_to_grads
from qucumber.callbacks import CallbackList, Timer
from qucumber.rbm import PurificationRBM
from .neural_state import NeuralState


class DensityMatrix(NeuralState):
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

    def pi(self, v, vp):
        r"""Calculates an element of the :math:`\Pi` matrix

        :param v: One of the visible states, :math:`\sigma`
        :type v: torch.Tensor
        :param vp: The other visible state, :math`\sigma'`
        :type vp: torch.Tensor
        :returns: The matrix element given by :math:`\langle\sigma|\Pi|\sigma'\rangle`
        :rtype: torch.Tensor
        """
        m_am = F.linear(v, self.rbm_am.weights_U, self.rbm_am.aux_bias)
        mp_am = F.linear(vp, self.rbm_am.weights_U, self.rbm_am.aux_bias)

        m_ph = F.linear(v, self.rbm_ph.weights_U)
        mp_ph = F.linear(vp, self.rbm_ph.weights_U)

        if v.dim() >= 2:
            m_am = m_am.unsqueeze_(1)
            m_ph = m_ph.unsqueeze_(1)
        if vp.dim() >= 2:
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

    def rho(self, v, vp):
        r"""Computes the matrix elements of the (unnormalized) density matrix

        :param v: One of the visible states, :math:`\sigma`
        :type v: torch.Tensor
        :param vp: The other visible state, :math`\sigma'`
        :type vp: torch.Tensor
        :returns: The element of the current density matrix
                  :math:`\langle\sigma|\widetilde{\rho}|\sigma'\rangle`
        :rtype: torch.Tensor
        """
        pi_ = self.pi(v, vp)
        amp = (self.rbm_am.gamma_plus(v, vp) + cplx.real(pi_)).exp()
        phase = self.rbm_ph.gamma_minus(v, vp) + cplx.imag(pi_)

        return cplx.make_complex(amp * phase.cos(), amp * phase.sin())

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
        :type data: numpy.ndarray
        :param input_bases: The measurement bases for each sample
        :type input_bases: numpy.ndarray
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
        :param lr_drop_epoch: The epoch, or list of epochs, at which the
                              base learning rate is dropped
        :type lr_drop_epoch: int or list[int]
        :param lr_drop_factor: The factor by which the scheduler will decrease the
                               learning after the prescribed number of steps
        :type lr_drop_factor: float
        :param bases: All bases in which a measurement is made. Used to check gradients
        :type bases: numpy.ndarray
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

        train_samples = data.clone().detach().double().to(device=self.device)

        neg_batch_size = neg_batch_size if neg_batch_size else pos_batch_size

        all_params = [getattr(self, net).parameters() for net in self.networks]
        all_params = list(chain(*all_params))

        optimizer = optimizer(all_params, lr=lr, **kwargs)
        scheduler = scheduler(optimizer, lr_drop_epoch, gamma=lr_drop_factor)

        z_samples = extract_refbasis_samples(train_samples, input_bases)

        num_batches = ceil(train_samples.shape[0] / pos_batch_size)

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
                f.write(f"Epoch: {ep}\tFidelity: {fidel}\n")
                f.close()

            if train_to_fid:
                if fidel >= train_to_fid:
                    print(
                        "\n\nTarget fidelity of", train_to_fid, "reached or exceeded!"
                    )
                    break

        callbacks.on_train_end(self)

    def normalization(self, space):
        r"""Compute the normalization constant of the state.

        .. math::

            Z_{\bm{\lambda}}=
            \sqrt{\sum_{\bm{\sigma}}|\psi_{\bm{\lambda\mu}}|^2}=
            \sqrt{\sum_{\bm{\sigma}} p_{\bm{\lambda}}(\bm{\sigma})}

        :param space: A rank 2 tensor of the entire visible space.
        :type space: torch.Tensor

        """
        return self.rbm_am.partition(space)

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
                raise ValueError(f"Invalid key in metadata; '{net}' cannot be a key!")

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

    @staticmethod
    def autoload(location, gpu=True):
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
