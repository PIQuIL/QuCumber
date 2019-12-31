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


import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parameters_to_vector

from qucumber.utils import cplx, auto_unsqueeze_args
from qucumber import _warn_on_missing_gpu


class PurificationRBM(nn.Module):
    r"""An RBM with a hidden and "auxiliary" layer, each separately connected to the visible units

    :param num_visible: The number of visible units, i.e. the size of the system
    :type num_visible: int
    :param num_hidden: The number of units in the hidden layer
    :type num_hidden: int
    :param num_aux: The number of units in the auxiliary purification layer
    :type num_aux: int
    :param zero_weights: Whether or not to initialize the weights to zero
    :type zero_weights: bool
    :param gpu: Whether to perform computations on the default gpu.
    :type gpu: bool
    """

    def __init__(
        self, num_visible, num_hidden=None, num_aux=None, zero_weights=False, gpu=False
    ):
        super().__init__()
        self.num_visible = int(num_visible)
        self.num_hidden = (
            int(num_hidden) if num_hidden is not None else self.num_visible
        )
        self.num_aux = int(num_aux) if num_aux is not None else self.num_visible

        # Parameters are:
        # W: The weights of the visible-hidden edges
        # U: The weights of the visible-auxiliary edges
        # b: The biases of the visible nodes
        # c: The biases of the hidden nobdes
        # d: The biases of the auxiliary nodes

        # The auxiliary bias of the phase RBM is always zero

        self.num_pars = (
            (self.num_visible * self.num_hidden)
            + (self.num_visible * self.num_aux)
            + self.num_visible
            + self.num_hidden
            + self.num_aux
        )

        _warn_on_missing_gpu(gpu)
        self.gpu = gpu and torch.cuda.is_available()

        self.device = torch.device("cuda") if self.gpu else torch.device("cpu")

        self.initialize_parameters(zero_weights=zero_weights)

    def __repr__(self):
        return (
            f"PurificationBinaryRBM(num_visible={self.num_visible}, "
            f"num_hidden={self.num_hidden}, num_aux={self.num_aux}, gpu={self.gpu})"
        )

    def initialize_parameters(self, zero_weights=False):
        r"""Initialize the parameters of the RBM

        :param zero_weights: Whether or not to initialize the weights to zero
        :type zero_weights: bool
        """
        gen_tensor = torch.zeros if zero_weights else torch.randn

        self.weights_W = nn.Parameter(
            (
                gen_tensor(
                    self.num_hidden,
                    self.num_visible,
                    dtype=torch.double,
                    device=self.device,
                )
                / np.sqrt(self.num_visible)
            ),
            requires_grad=False,
        )

        self.weights_U = nn.Parameter(
            (
                gen_tensor(
                    self.num_aux,
                    self.num_visible,
                    dtype=torch.double,
                    device=self.device,
                )
                / np.sqrt(self.num_visible)
            ),
            requires_grad=False,
        )

        self.visible_bias = nn.Parameter(
            torch.zeros(self.num_visible, dtype=torch.double, device=self.device),
            requires_grad=False,
        )

        self.hidden_bias = nn.Parameter(
            torch.zeros(self.num_hidden, dtype=torch.double, device=self.device),
            requires_grad=False,
        )

        self.aux_bias = nn.Parameter(
            torch.zeros(self.num_aux, dtype=torch.double, device=self.device),
            requires_grad=False,
        )

    @auto_unsqueeze_args()
    def effective_energy(self, v, a=None):
        r"""Computes the equivalent of the "effective energy" for the RBM. If
        `a` is `None`, will analytically trace out the auxiliary units.

        :param v: The current state of the visible units. Shape (b, n_v) or (n_v,).
        :type v: torch.Tensor
        :param a: The current state of the auxiliary units. Shape (b, n_a) or (n_a,).
        :type a: torch.Tensor or None

        :returns: The "effective energy" of the RBM. Shape (b,) or (1,).
        :rtype: torch.Tensor
        """
        v = v.to(self.weights_W)

        vis_term = torch.matmul(v, self.visible_bias) + F.softplus(
            F.linear(v, self.weights_W, self.hidden_bias)
        ).sum(-1)

        if a is not None:
            a = (a.unsqueeze(0) if a.dim() < 2 else a).to(self.weights_W)

            aux_term = torch.matmul(a, self.aux_bias)
            mix_term = torch.einsum("...v,av,...a->...", v, self.weights_U.data, a)
            return -(vis_term + aux_term + mix_term)
        else:
            aux_term = F.softplus(F.linear(v, self.weights_U, self.aux_bias)).sum(-1)

            return -(vis_term + aux_term)

    def effective_energy_gradient(self, v, reduce=True):
        """The gradients of the effective energies for the given visible states.

        :param v: The visible states.
        :type v: torch.Tensor
        :param reduce: If `True`, will sum over the gradients resulting from
                       each visible state. Otherwise will return a batch of
                       gradient vectors.
        :type reduce: bool

        :returns: Will return a vector (or matrix if `reduce=False` and multiple
                  visible states were given as a matrix) containing the gradients
                  for all parameters (computed on the given visible states v).
        :rtype: torch.Tensor
        """
        v = (v.unsqueeze(0) if v.dim() < 2 else v).to(self.weights_W)
        ph = self.prob_h_given_v(v)
        pa = self.prob_a_given_v(v)

        if reduce:
            W_grad = -torch.matmul(ph.transpose(0, -1), v)
            U_grad = -torch.matmul(pa.transpose(0, -1), v)
            vb_grad = -torch.sum(v, 0)
            hb_grad = -torch.sum(ph, 0)
            ab_grad = -torch.sum(pa, 0)
            return parameters_to_vector([W_grad, U_grad, vb_grad, hb_grad, ab_grad])
        else:
            W_grad = -torch.einsum("...j,...k->...jk", ph, v).view(*v.shape[:-1], -1)
            U_grad = -torch.einsum("...j,...k->...jk", pa, v).view(*v.shape[:-1], -1)
            vb_grad = -v
            hb_grad = -ph
            ab_grad = -pa
            vec = [W_grad, U_grad, vb_grad, hb_grad, ab_grad]
            return torch.cat(vec, dim=-1)

    @auto_unsqueeze_args()
    def prob_h_given_v(self, v, out=None):
        r"""Given a visible unit configuration, compute the probability
        vector of the hidden units being on

        :param v: The visible units
        :type v: torch.Tensor
        :param out: The output tensor to write to
        :type out: torch.Tensor

        :returns: The probability of the hidden units being active
                  given the visible state
        :rtype torch.Tensor:
        """
        return (
            torch.matmul(v, self.weights_W.data.t(), out=out)
            .add_(self.hidden_bias.data)
            .sigmoid_()
            .clamp_(min=0, max=1)
        )

    @auto_unsqueeze_args()
    def prob_a_given_v(self, v, out=None):
        r"""Given a visible unit configuration, compute the probability
        vector of the auxiliary units being on

        :param v: The visible units
        :type v: torch.Tensor
        :param out: The output tensor to write to
        :type out: torch.Tensor

        :returns: The probability of the auxiliary units being active
                  given the visible state
        :rtype torch.Tensor:
        """
        return (
            torch.matmul(v, self.weights_U.data.t(), out=out)
            .add_(self.aux_bias.data)
            .sigmoid_()
            .clamp_(min=0, max=1)
        )

    @auto_unsqueeze_args(1, 2)
    def prob_v_given_ha(self, h, a, out=None):
        r"""Given a hidden and auxiliary unit configuration, compute
        the probability vector of the hidden units being on

        :param h: The hidden units
        :type h: torch.Tensor
        :param a: The auxiliary units
        :type a: torch.Tensor
        :param out: The output tensor to write to
        :type out: torch.Tensor

        :returns: The probability of the visible units being
                  active given the hidden and auxiliary states
        :rtype torch.Tensor:
        """
        return (
            torch.matmul(h, self.weights_W.data, out=out)
            .add_(self.visible_bias.data)
            .add_(torch.matmul(a, self.weights_U.data))
            .sigmoid_()
            .clamp_(min=0, max=1)
        )

    def sample_a_given_v(self, v, out=None):
        r"""Sample/generate an auxiliary state given a visible state

        :param v: The visible state
        :type v: torch.Tensor
        :param out: The output tensor to write to
        :type out: torch.Tensor

        :returns: The sampled auxiliary state
        :rtype: torch.Tensor
        """
        a = self.prob_a_given_v(v, out=out)
        a = torch.bernoulli(a, out=out)
        return a

    def sample_h_given_v(self, v, out=None):
        r"""Sample/generate a hidden state given a visible state

        :param v: The visible state
        :type v: torch.Tensor
        :param out: The output tensor to write to
        :type out: torch.Tensor

        :returns: The sampled hidden state
        :rtype: torch.Tensor
        """
        h = self.prob_h_given_v(v, out=out)
        h = torch.bernoulli(h, out=out)
        return h

    def sample_v_given_ha(self, h, a, out=None):
        r"""Sample/generate a visible state given the
        hidden and auxiliary states

        :param h: The hidden state
        :type h: torch.Tensor
        :param a: The auxiliary state
        :type a: torch.Tensor
        :param out: The output tensor to write to
        :type out: torch.Tensor

        :returns: The sampled visible state
        :rtype: torch.Tensor
        """
        v = self.prob_v_given_ha(h, a, out=out)
        v = torch.bernoulli(v, out=out)
        return v

    def gibbs_steps(self, k, initial_state, overwrite=False):
        r"""Perform k steps of Block Gibbs sampling. One step consists of
        sampling the hidden and auxiliary states from the visible state, and
        then sampling the visible state from the hidden and auxiliary states

        :param k: The number of Block Gibbs steps
        :type k: int
        :param initial_state: The initial visible state
        :type initial_state: torch.Tensor
        :param overwrite: Whether to overwrite the initial_state tensor.
                          Exception: If initial_state is not on the same device
                          as the RBM, it will NOT be overwritten.
        :type overwrite: bool

        :returns: Returns the visible states after k steps of
                  Block Gibbs sampling
        :rtype: torch.Tensor
        """
        v = (initial_state if overwrite else initial_state.clone()).to(self.weights_W)

        h = torch.zeros(*v.shape[:-1], self.num_hidden).to(self.weights_W)
        a = torch.zeros(*v.shape[:-1], self.num_aux).to(self.weights_W)

        for _ in range(k):
            self.sample_h_given_v(v, out=h)
            self.sample_a_given_v(v, out=a)
            self.sample_v_given_ha(h, a, out=v)

        return v

    @auto_unsqueeze_args()
    def mixing_term(self, v):
        r"""Describes the extent of mixing in the system,
            :math:`V_\theta = \frac{1}{2}U_\theta \bm{\sigma} + d_\theta`

        :param v: The visible state of the system
        :type v: torch.Tensor

        :returns: The term describing the mixing of the system
        :rtype: torch.Tensor
        """
        return F.linear(v, 0.5 * self.weights_U, self.aux_bias)

    def gamma(self, v, vp, eta=1, expand=True):
        r"""Calculates elements of the :math:`\Gamma^{(\eta)}` matrix,
        where :math:`\eta = \pm`.
        If `expand` is `True`, will return a complex matrix
        :math:`A_{ij} = \langle\sigma_i|\Gamma^{(\eta)}|\sigma'_j\rangle`.
        Otherwise will return a complex vector
        :math:`A_{i} = \langle\sigma_i|\Gamma^{(\eta)}|\sigma'_i\rangle`.

        :param v: A batch of visible states, :math:`\sigma`.
        :type v: torch.Tensor
        :param vp: The other batch of visible states, :math:`\sigma'`.
        :type vp: torch.Tensor
        :param eta: Determines which gamma matrix elements to compute.
        :type eta: int
        :param expand: Whether to return a matrix (`True`) or a vector (`False`).
                       Ignored if both inputs are vectors, in which case, a
                       scalar is returned.
        :type expand: bool

        :returns: The matrix element given by
                  :math:`\langle\sigma|\Gamma^{(\eta)}|\sigma'\rangle`
        :rtype: torch.Tensor
        """
        sign = np.sign(eta)
        if v.dim() < 2 and vp.dim() < 2:
            temp = torch.dot(v + sign * vp, self.visible_bias)
            temp += F.softplus(F.linear(v, self.weights_W, self.hidden_bias)).sum()
            temp += (
                sign * F.softplus(F.linear(vp, self.weights_W, self.hidden_bias)).sum()
            )
        else:
            temp1 = torch.matmul(v, self.visible_bias) + (
                F.softplus(F.linear(v, self.weights_W, self.hidden_bias)).sum(-1)
            )

            temp2 = torch.matmul(vp, self.visible_bias) + (
                F.softplus(F.linear(vp, self.weights_W, self.hidden_bias)).sum(-1)
            )

            if expand:
                temp = temp1.unsqueeze_(1) + (sign * temp2.unsqueeze_(0))
            else:
                temp = temp1 + (sign * temp2)

        return 0.5 * temp

    def gamma_grad(self, v, vp, eta=1, expand=False):
        r"""Calculates elements of the gradient of
            the :math:`\Gamma^{(\eta)}` matrix, where :math:`\eta = \pm`.

        :param v: A batch of visible states, :math:`\sigma`
        :type v: torch.Tensor
        :param vp: The other batch of visible states, :math:`\sigma'`
        :type vp: torch.Tensor
        :param eta: Determines which gamma matrix elements to compute.
        :type eta: int
        :param expand: Whether to return a rank-3 tensor (`True`) or a matrix (`False`).
        :type expand: bool

        :returns: The matrix element given by
                  :math:`\langle\sigma|\nabla_\lambda\Gamma^{(\eta)}|\sigma'\rangle`
        :rtype: torch.Tensor
        """
        sign = np.sign(eta)
        unsqueezed = v.dim() < 2 or vp.dim() < 2
        v = (v.unsqueeze(0) if v.dim() < 2 else v).to(self.weights_W)
        vp = (vp.unsqueeze(0) if vp.dim() < 2 else vp).to(self.weights_W)

        prob_h = self.prob_h_given_v(v)
        prob_hp = self.prob_h_given_v(vp)

        W_grad_ = torch.einsum("...j,...k->...jk", prob_h, v)
        W_grad_p = torch.einsum("...j,...k->...jk", prob_hp, vp)

        if expand:
            W_grad = 0.5 * (W_grad_.unsqueeze_(1) + sign * W_grad_p.unsqueeze_(0))
            vb_grad = 0.5 * (v.unsqueeze(1) + sign * vp.unsqueeze(0))
            hb_grad = 0.5 * (prob_h.unsqueeze_(1) + sign * prob_hp.unsqueeze_(0))
        else:
            W_grad = 0.5 * (W_grad_ + sign * W_grad_p)
            vb_grad = 0.5 * (v + sign * vp)
            hb_grad = 0.5 * (prob_h + sign * prob_hp)

        batch_sizes = (
            (v.shape[0], vp.shape[0], *v.shape[1:-1]) if expand else (*v.shape[:-1],)
        )
        U_grad = torch.zeros_like(self.weights_U).expand(*batch_sizes, -1, -1)
        ab_grad = torch.zeros_like(self.aux_bias).expand(*batch_sizes, -1)

        vec = [
            W_grad.view(*batch_sizes, -1),
            U_grad.view(*batch_sizes, -1),
            vb_grad,
            hb_grad,
            ab_grad,
        ]
        if unsqueezed and not expand:
            vec = [grad.squeeze_(0) for grad in vec]

        return cplx.make_complex(torch.cat(vec, dim=-1))

    def partition(self, space):
        r"""Computes the partition function

        :param space: The Hilbert space of the visible units
        :type space: torch.Tensor

        :returns: The partition function
        :rtype: torch.Tensor
        """
        logZ = (-self.effective_energy(space)).logsumexp(0)
        return logZ.exp()
