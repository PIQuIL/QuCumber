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

import torch

from qucumber import _warn_on_missing_gpu
from qucumber.utils import cplx, unitaries
from qucumber.rbm import BinaryRBM
from .wavefunction import WaveFunctionBase


class ComplexWaveFunction(WaveFunctionBase):
    """Class capable of learning wavefunctions with a non-zero phase.

    :param num_visible: The number of visible units, ie. the size of the system being learned.
    :type num_visible: int
    :param num_hidden: The number of hidden units in both internal RBMs. Defaults to
                    the number of visible units.
    :type num_hidden: int
    :param unitary_dict: A dictionary mapping unitary names to their matrix representations.
    :type unitary_dict: dict[str, torch.Tensor]
    :param gpu: Whether to perform computations on the default GPU.
    :type gpu: bool
    :param module: An instance of a BinaryRBM module to use for density estimation;
                   The given RBM object will be used to estimate the amplitude of
                   the wavefunction, while a copy will be used to estimate
                   the phase of the wavefunction.
                   Will be copied to the default GPU if `gpu=True` (if it
                   isn't already there). If `None`, will initialize the BinaryRBMs
                   from scratch.
    :type module: qucumber.rbm.BinaryRBM
    """

    _rbm_am = None
    _rbm_ph = None
    _device = None

    def __init__(
        self, num_visible, num_hidden=None, unitary_dict=None, gpu=False, module=None
    ):
        if gpu and torch.cuda.is_available():
            warnings.warn(
                "Using ComplexWaveFunction on GPU is not recommended due to poor performance compared to CPU.",
                ResourceWarning,
                2,
            )
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if module is None:
            self.rbm_am = BinaryRBM(num_visible, num_hidden, gpu=gpu)
            self.rbm_ph = BinaryRBM(num_visible, num_hidden, gpu=gpu)
        else:
            _warn_on_missing_gpu(gpu)
            self.rbm_am = module.to(self.device)
            self.rbm_am.device = self.device
            self.rbm_ph = module.to(self.device).clone()
            self.rbm_ph.device = self.device

        self.num_visible = self.rbm_am.num_visible
        self.num_hidden = self.rbm_am.num_hidden
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

    def amplitude(self, v):
        r"""Compute the (unnormalized) amplitude of a given vector/matrix of visible states.

        .. math::

            \text{amplitude}(\bm{\sigma})=|\psi_{\bm{\lambda\mu}}(\bm{\sigma})|=
            e^{-\mathcal{E}_{\bm{\lambda}}(\bm{\sigma})/2}

        :param v: visible states :math:`\bm{\sigma}`.
        :type v: torch.Tensor

        :returns: Vector containing the amplitudes of the given states.
        :rtype: torch.Tensor
        """
        return super().amplitude(v)

    def phase(self, v):
        r"""Compute the phase of a given vector/matrix of visible states.

        .. math::

            \text{phase}(\bm{\sigma})=-\mathcal{E}_{\bm{\mu}}(\bm{\sigma})/2

        :param v: visible states :math:`\bm{\sigma}`.
        :type v: torch.Tensor

        :returns: Vector containing the phases of the given states.
        :rtype: torch.Tensor
        """
        return -0.5 * self.rbm_ph.effective_energy(v)

    def psi(self, v):
        r"""Compute the (unnormalized) wavefunction of a given vector/matrix of visible states.

        .. math::

            \psi_{\bm{\lambda\mu}}(\bm{\sigma})
                = e^{-[\mathcal{E}_{\bm{\lambda}}(\bm{\sigma})
                        + i\mathcal{E}_{\bm{\mu}}(\bm{\sigma})]/2}

        :param v: visible states :math:`\bm{\sigma}`
        :type v: torch.Tensor

        :returns: Complex object containing the value of the wavefunction for
                  each visible state
        :rtype: torch.Tensor
        """
        return super().psi(v)

    def rotated_gradient(self, basis, sample):
        r"""Computes the gradients rotated into the measurement basis

        :param basis: The bases in which the measurement is made
        :type basis: numpy.ndarray
        :param sample: The measurement (either 0 or 1)
        :type sample: torch.Tensor

        :returns: A list of two tensors, representing the rotated gradients
                  of the amplitude and phase RBMS
        :rtype: list[torch.Tensor, torch.Tensor]
        """
        Upsi, Upsi_v, v = unitaries.rotate_psi_inner_prod(
            self, basis, sample, include_extras=True
        )
        inv_Upsi = cplx.inverse(Upsi)

        raw_grads = [self.am_grads(v), self.ph_grads(v)]

        rotated_grad = [cplx.einsum("ib,ibg->bg", Upsi_v, g) for g in raw_grads]
        grad = [
            cplx.einsum("b,bg->g", inv_Upsi, rg, imag_part=False) for rg in rotated_grad
        ]

        return grad

    def am_grads(self, v):
        r"""Computes the gradients of the amplitude RBM for given input states

        :param v: The input state, :math:`\sigma`
        :type v: torch.Tensor

        :returns: The gradients of all amplitude RBM parameters
        :rtype: torch.Tensor
        """
        return cplx.make_complex(self.rbm_am.effective_energy_gradient(v, reduce=False))

    def ph_grads(self, v):
        r"""Computes the gradients of the phase RBM for given input states

        :param v: The input state, :math:`\sigma`
        :type v: torch.Tensor

        :returns: The gradients of all phase RBM parameters
        :rtype: torch.Tensor
        """
        return cplx.scalar_mult(
            cplx.make_complex(self.rbm_ph.effective_energy_gradient(v, reduce=False)),
            cplx.I,  # need to multiply phase gradient by i
        )

    def fit(
        self,
        data,
        epochs=100,
        pos_batch_size=100,
        neg_batch_size=None,
        k=1,
        lr=1e-3,
        input_bases=None,
        progbar=False,
        starting_epoch=1,
        time=False,
        callbacks=None,
        optimizer=torch.optim.SGD,
        optimizer_args=None,
        scheduler=None,
        scheduler_args=None,
        **kwargs
    ):
        if input_bases is None:
            raise ValueError(
                "input_bases must be provided to train a ComplexWaveFunction!"
            )
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
                **kwargs
            )

    @staticmethod
    def autoload(location, gpu=False):
        state_dict = torch.load(location)
        wvfn = ComplexWaveFunction(
            unitary_dict=state_dict["unitary_dict"],
            num_visible=len(state_dict["rbm_am"]["visible_bias"]),
            num_hidden=len(state_dict["rbm_am"]["hidden_bias"]),
            gpu=gpu,
        )
        wvfn.load(location)
        return wvfn
