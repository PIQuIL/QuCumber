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

from qucumber import _warn_on_missing_gpu
from qucumber.rbm import BinaryRBM
from qucumber.utils import cplx, auto_unsqueeze_args
from .wavefunction import WaveFunctionBase


class PositiveWaveFunction(WaveFunctionBase):
    """Class capable of learning wavefunctions with no phase.

    :param num_visible: The number of visible units, ie. the size of the system being learned.
    :type num_visible: int
    :param num_hidden: The number of hidden units in the internal RBM. Defaults to
                       the number of visible units.
    :type num_hidden: int
    :param gpu: Whether to perform computations on the default GPU.
    :type gpu: bool
    :param module: An instance of a BinaryRBM module to use for density estimation.
                   Will be copied to the default GPU if `gpu=True` (if it
                   isn't already there). If `None`, will initialize a BinaryRBM
                   from scratch.
    :type module: qucumber.rbm.BinaryRBM
    """

    _rbm_am = None
    _device = None

    def __init__(self, num_visible, num_hidden=None, gpu=True, module=None):
        if module is None:
            self.rbm_am = BinaryRBM(num_visible, num_hidden, gpu=gpu)
        else:
            _warn_on_missing_gpu(gpu)
            gpu = gpu and torch.cuda.is_available()
            device = torch.device("cuda") if gpu else torch.device("cpu")
            self.rbm_am = module.to(device)
            self.rbm_am.device = device

        self.num_visible = self.rbm_am.num_visible
        self.num_hidden = self.rbm_am.num_hidden
        self.device = self.rbm_am.device

    @property
    def networks(self):
        return ["rbm_am"]

    @property
    def rbm_am(self):
        return self._rbm_am

    @rbm_am.setter
    def rbm_am(self, new_val):
        self._rbm_am = new_val

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, new_val):
        self._device = new_val

    def amplitude(self, v):
        r"""Compute the (unnormalized) amplitude of a given vector/matrix of visible states.

        .. math::

            \text{amplitude}(\bm{\sigma})=|\psi_{\bm{\lambda}}(\bm{\sigma})|=
            e^{-\mathcal{E}_{\bm{\lambda}}(\bm{\sigma})/2}

        :param v: visible states :math:`\bm{\sigma}`
        :type v: torch.Tensor

        :returns: Matrix/vector containing the amplitudes of v
        :rtype: torch.Tensor
        """
        return super().amplitude(v)

    @auto_unsqueeze_args()
    def phase(self, v):
        r"""Compute the phase of a given vector/matrix of visible states.

        In the case of a PositiveWaveFunction, the phase is just zero.

        :param v: visible states :math:`\bm{\sigma}`
        :type v: torch.Tensor

        :returns: Matrix/vector containing the phases of v
        :rtype: torch.Tensor
        """
        return torch.zeros(v.shape[0], dtype=torch.double, device=self.device)

    def psi(self, v):
        r"""Compute the (unnormalized) wavefunction of a given vector/matrix of visible states.

        .. math::

            \psi_{\bm{\lambda}}(\bm{\sigma})
                = e^{-\mathcal{E}_{\bm{\lambda}}(\bm{\sigma})/2}

        :param v: visible states :math:`\bm{\sigma}`
        :type v: torch.Tensor

        :returns: Complex object containing the value of the wavefunction for
                  each visible state
        :rtype: torch.Tensor
        """
        # vector/tensor of shape (2, len(v))
        return cplx.make_complex(self.amplitude(v))

    def gradient(self, v, *args, **kwargs):
        r"""Compute the gradient of the effective energy for a batch of states.

        :math:`\nabla_{\bm{\lambda}}\mathcal{E}_{\bm{\lambda}}(\bm{\sigma})`

        :param v: visible states :math:`\bm{\sigma}`
        :type v: torch.Tensor
        :param \*args: Ignored.
        :param \**kwargs: Ignored.

        :returns: A two-element list containing the gradients of the
                  effective energy. The second element will always be zero.
        :rtype: list[torch.Tensor]
        """
        return super().gradient(v, bases=None)

    def positive_phase_gradients(self, samples_batch, *args, **kwargs):
        r"""Computes the positive phase of the gradients of the parameters.

        :param samples_batch: The measurements
        :type samples_batch: torch.Tensor
        :param \*args: Ignored.
        :param \**kwargs: Ignored.

        :returns: A two-element list containing the gradients of the
                  effective energy. The second element will always be zero.
        :rtype: list[torch.Tensor]
        """
        return super().positive_phase_gradients(samples_batch, bases_batch=None)

    def compute_exact_grads(self, samples_batch, space, *args, **kwargs):
        r"""Computes the gradients of the parameters, using exact sampling
        for the negative phase update instead of Gibbs sampling

        :param samples_batch: The measurements
        :type samples_batch: torch.Tensor
        :param space: A rank 2 tensor of the entire visible space.
        :type space: torch.Tensor
        :param \*args: Ignored.
        :param \**kwargs: Ignored.

        :returns: A single-element list containing the gradients calculated
                  with an exact negative phase update
        :rtype: list[torch.Tensor]
        """
        return super().compute_exact_grads(samples_batch, space, bases_batch=None)

    def compute_batch_gradients(self, k, samples_batch, neg_batch, *args, **kwargs):
        r"""Compute the gradients of a batch of the training data (`samples_batch`).

        :param k: Number of contrastive divergence steps in training.
        :type k: int
        :param samples_batch: Batch of the input samples.
        :type samples_batch: torch.Tensor
        :param neg_batch: Batch of the input samples for computing the
                          negative phase.
        :type neg_batch: torch.Tensor
        :param \*args: Ignored.
        :param \**kwargs: Ignored.

        :returns: A single-element list containing the gradients calculated
                  with a Gibbs sampled negative phase update
        :rtype: list[torch.Tensor]
        """
        return super().compute_batch_gradients(
            k, samples_batch, neg_batch, bases_batch=None
        )

    def fit(
        self,
        data,
        epochs=100,
        pos_batch_size=100,
        neg_batch_size=None,
        k=1,
        lr=1e-3,
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
        kwargs["input_bases"] = None
        return super().fit(
            data=data,
            epochs=epochs,
            pos_batch_size=pos_batch_size,
            neg_batch_size=neg_batch_size,
            k=k,
            lr=lr,
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
    def autoload(location, gpu=True):
        state_dict = torch.load(location)
        wvfn = PositiveWaveFunction(
            num_visible=len(state_dict["rbm_am"]["visible_bias"]),
            num_hidden=len(state_dict["rbm_am"]["hidden_bias"]),
            gpu=gpu,
        )
        wvfn.load(location)
        return wvfn
