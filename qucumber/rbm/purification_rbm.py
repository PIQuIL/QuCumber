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

from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parameters_to_vector

from qucumber.utils import cplx
from qucumber import _warn_on_missing_gpu


class PurificationRBM(nn.Module):
    r"""An RBM with a hidden and "auxiliary" layer, each separately connected to the visible units

    :param num_visible: The number of visible units, i.e. the size of the system
    :type num_visible: int
    :param num_hidden: The number of units in the hidden layer
    :type num_hidden: int
    :param num_aux: The number of units in the auxilary purification layer
    :type num_aux: int
    :param zero_init: Whether or not to initialize the weights to zero
    :type zero_init: bool
    :param gpu: Whether to perform computations on the default gpu.
    :type gpu: bool
    """

    def __init__(self, num_visible, num_hidden, num_aux, zero_init=False, gpu=False):
        super().__init__()
        self.num_visible = int(num_visible)
        self.num_hidden = int(num_hidden)
        self.num_aux = int(num_aux)

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

        self.initialize_parameters(zero_init=zero_init)

    def initialize_parameters(self, zero_init=False):
        r"""Initialize the parameters of the RBM

        :param zero_init: Whether or not to initalize the weights to zero
        :type zero_init: bool
        """
        if zero_init:
            gen_tensor = torch.zeros
        else:
            gen_tensor = torch.rand

        self.weights_W = nn.Parameter(
            (
                0.01
                * gen_tensor(
                    self.num_hidden,
                    self.num_visible,
                    dtype=torch.double,
                    device=self.device,
                )
                - 0.005
            ),
            requires_grad=False,
        )

        self.weights_U = nn.Parameter(
            (
                0.01
                * gen_tensor(
                    self.num_aux,
                    self.num_visible,
                    dtype=torch.double,
                    device=self.device,
                )
                - 0.005
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

    def effective_energy(self, v, a):
        r"""Computes the equivalent of the "effective energy" for the RBM

        :param v: The current state of the visible units
        :type v: torch.Tensor
        :param a: The current state of the auxiliary units
        :type v: torch.Tensor
        :returns: The "effective energy" of the RBM
        :rtype: torch.Tensor
        """
        if len(v.shape) < 2 and len(a.shape) < 2:
            v = v.unsqueeze(0)
            a = a.unsqueeze(0)

            energy = torch.zeros(
                v.shape[0], a.shape[0], dtype=torch.double, device=self.device
            )

            vb_term = torch.mv(v, self.visible_bias)
            ab_term = torch.mv(a, self.aux_bias)
            vb_term += torch.mv(torch.matmul(v, self.weights_U.data.t()), a)
            other_term = F.softplus(F.linear(v, self.weights_W, self.hidden_bias)).sum(
                1
            )
            energy = vb_term + ab_term + other_term

        else:
            energy = torch.zeros(
                v.shape[0], a.shape[0], dtype=torch.double, device=self.device
            )

            for i in range(2 ** self.num_visible):
                for j in range(2 ** self.num_aux):
                    vb_term = torch.dot(v[i], self.visible_bias)
                    ab_term = torch.dot(a[j], self.aux_bias)
                    vb_term += torch.dot(torch.mv(self.weights_U, v[i]), self.aux_bias)
                    other_term = F.softplus(
                        F.linear(v[i], self.weights_W, self.hidden_bias)
                    ).sum(0)
                    energy[i][j] = vb_term + ab_term + other_term

        return energy

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
        if v.dim() < 2:  # create extra axis, if needed
            v = v.unsqueeze(0)
            unsqueezed = True
        else:
            unsqueezed = False

        p = torch.addmm(
            self.hidden_bias.data, v, self.weights_W.data.t(), out=out
        ).sigmoid_()

        if unsqueezed:
            return p.squeeze_(0)  # remove superfluous axis, if it exists
        else:
            return p

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
        if v.dim() < 2:  # create extra axis, if needed
            v = v.unsqueeze(0)
            unsqueezed = True
        else:
            unsqueezed = False

        p = torch.addmm(
            self.aux_bias.data, v, self.weights_U.data.t(), out=out
        ).sigmoid_()

        if unsqueezed:
            return p.squeeze_(0)  # remove superfluous axis, if it exists
        else:
            return p

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
        if h.dim() < 2 and a.dim() < 2:  # create extra axis, if needed
            h = h.unsqueeze(0)
            a = a.unsqueeze(0)
            unsqueezed = True
        else:
            unsqueezed = False

        p = torch.addmm(
            torch.addmm(self.visible_bias.data, h, self.weights_W.data),
            a,
            self.weights_U.data,
            out=out,
        ).sigmoid_()

        if unsqueezed:
            return p.squeeze_(0)  # remove superfluous axis, if it exists
        else:
            return p

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
        :param overwrite: Will overwrite initial_state tensor if True
        :type overwrite: bool
        :returns: Returns the visible state after k steps of
                  Block Gibbs sampling
        :rtype: torch.Tensor
        """
        v = initial_state.clone()

        h = torch.zeros(v.shape[0], self.num_hidden).to(self.weights_W)
        a = torch.zeros(v.shape[0], self.num_aux).to(self.weights_W)

        for _ in range(k):
            self.sample_h_given_v(v, out=h)
            self.sample_a_given_v(v, out=a)
            self.sample_v_given_ha(h, a, out=v)

        return v

    def mixing_term(self, v):
        r"""Describes the extent of mixing in the system,
            :math:`V_\theta = \frac{1}{2}U_\theta + d_\theta`

        :param v: The visible state of the system
        :type v: torch.Tensor
        :returns: The term describing the mixing of the system
        :rtype: torch.Tensor
        """
        if len(v.shape) < 2:
            v = v.unsqueeze(0)
            unsqueezed = True
        else:
            unsqueezed = False

        mixing_term = F.linear(v, 0.5 * self.weights_U, self.aux_bias)

        if unsqueezed:
            return mixing_term.squeeze(0)
        else:
            return mixing_term

    def GammaP(self, v, vp):
        r"""Calculates an element of the :math:`\Gamma^{(+)}` matrix

        :param v: One of the visible states, :math:`\sigma`
        :type v: torch.Tensor
        :param vp: The other visible state, :math`\sigma'`
        :type vp: torch.Tensor
        :returns: The matrix element given by
                  :math:`\langle\sigma|\Gamma^{(+)}|\sigma'\rangle`
        :rtype: torch.Tensor
        """
        if len(v.shape) < 2 and len(vp.shape) < 2:
            temp = torch.dot(v + vp, self.visible_bias)
            temp += F.softplus(F.linear(v, self.weights_W, self.hidden_bias)).sum()
            temp += F.softplus(F.linear(vp, self.weights_W, self.hidden_bias)).sum()

        # Computes the entrie matrix Gamma at once
        else:
            temp = torch.zeros(
                2 ** (self.num_visible),
                2 ** (self.num_visible),
                dtype=torch.double,
                device=self.device,
            )

            for i in range(2 ** self.num_visible):
                for j in range(2 ** self.num_visible):
                    temp[i][j] = torch.dot(v[i] + vp[j], self.visible_bias)
                    temp[i][j] += F.softplus(
                        F.linear(v[i], self.weights_W, self.hidden_bias)
                    ).sum()
                    temp[i][j] += F.softplus(
                        F.linear(vp[j], self.weights_W, self.hidden_bias)
                    ).sum()

        return 0.5 * temp

    def GammaM(self, v, vp):
        r"""Calculates an element of the :math:`\Gamma^{(-)}` matrix

        :param v: One of the visible states, :math:`\sigma`
        :type v: torch.Tensor
        :param vp: The other visible state, :math`\sigma'`
        :type vp: torch.Tensor
        :returns: The matrix element given by
                  :math:`\langle\sigma|\Gamma^{(-)}|\sigma'\rangle`
        :rtype: torch.Tensor
        """
        if len(v.shape) < 2 and len(vp.shape) < 2:
            temp = torch.dot(v - vp, self.visible_bias)
            temp += F.softplus(F.linear(v, self.weights_W, self.hidden_bias)).sum()
            temp -= F.softplus(F.linear(vp, self.weights_W, self.hidden_bias)).sum()

        # Computes the entire matrix Gamma- at once
        else:
            temp = torch.zeros(
                2 ** (self.num_visible),
                2 ** (self.num_visible),
                dtype=torch.double,
                device=self.device,
            )

            for i in range(2 ** self.num_visible):
                for j in range(2 ** self.num_visible):
                    temp[i][j] = torch.dot(v[i] - vp[j], self.visible_bias)
                    temp[i][j] += F.softplus(
                        F.linear(v[i], self.weights_W, self.hidden_bias)
                    ).sum()
                    temp[i][j] -= F.softplus(
                        F.linear(vp[j], self.weights_W, self.hidden_bias)
                    ).sum()

        return 0.5 * temp

    def GammaP_grad(self, v, vp, reduce=False):
        r"""Calculates an element of the gradient of
            the :math:`\Gamma^{(+)}` matrix

        :param v: One of the visible states, :math:`\sigma`
        :type v: torch.Tensor
        :param vp: The other visible state, :math`\sigma'`
        :type vp: torch.Tensor
        :returns: The matrix element given by
                  :math:`\langle\sigma|\nabla_\lambda\Gamma^{(+)}|\sigma'\rangle`
        :rtype: torch.Tensor
        """
        prob_h = self.prob_h_given_v(v)
        prob_hp = self.prob_h_given_v(vp)

        if v.dim() < 2:
            W_grad = 0.5 * (torch.ger(prob_h, v) + torch.ger(prob_hp, vp))
            U_grad = torch.zeros_like(self.weights_U)
            vb_grad = 0.5 * (v + vp)
            hb_grad = 0.5 * (prob_h + prob_hp)
            ab_grad = torch.zeros_like(self.aux_bias)

        else:
            # Don't think this works, but the 'else' option does
            if reduce:
                W_grad = 0.5 * (
                    torch.matmul(prob_h.t(), v) + torch.matmul(prob_hp.t(), vp)
                )
                U_grad = torch.zeros(
                    (self.weights_U.shape[0], self.weights_U.shape[1]),
                    dtype=torch.double,
                )
                vb_grad = 0.5 * torch.sum(v + vp, 0)
                hb_grad = 0.5 * torch.sum(prob_h + prob_hp, 0)
                ab_grad = torch.zeros(
                    (v.shape[0], self.num_aux), dtype=torch.double, device=self.device
                )

            else:
                W_grad = 0.5 * (
                    torch.einsum("ij,ik->ijk", prob_h, v)
                    + torch.einsum("ij,ik->ijk", prob_hp, vp)
                )
                U_grad = torch.zeros(
                    (v.shape[0], self.weights_U.shape[0], self.weights_U.shape[1]),
                    dtype=torch.double,
                    device=self.device,
                )
                vb_grad = 0.5 * (v + vp)
                hb_grad = 0.5 * (prob_h + prob_hp)
                ab_grad = torch.zeros(
                    (v.shape[0], self.num_aux), dtype=torch.double, device=self.device
                )
                vec = [
                    W_grad.view(v.size()[0], -1),
                    U_grad.view(v.size()[0], -1),
                    vb_grad,
                    hb_grad,
                    ab_grad,
                ]
                return cplx.make_complex(torch.cat(vec, dim=1))

        return cplx.make_complex(
            parameters_to_vector([W_grad, U_grad, vb_grad, hb_grad, ab_grad])
        )

    def GammaM_grad(self, v, vp, reduce=False):
        r"""Calculates an element of the gradient of
            the :math:`\Gamma^{(-)}` matrix

        :param v: One of the visible states, :math:`\sigma`
        :type v: torch.Tensor
        :param vp: The other visible state, :math`\sigma'`
        :type vp: torch.Tensor
        :returns: The matrix element given by
                  :math:`\langle\sigma|\nabla_\mu\Gamma^{(-)}|\sigma'\rangle`
        :rtype: torch.Tensor
        """
        prob_h = self.prob_h_given_v(v)
        prob_hp = self.prob_h_given_v(vp)

        if v.dim() < 2:
            W_grad = 0.5 * (torch.ger(prob_h.t(), v) - torch.ger(prob_hp.t(), vp))
            U_grad = torch.zeros_like(self.weights_U)
            vb_grad = 0.5 * (v - vp)
            hb_grad = 0.5 * (prob_h - prob_hp)
            ab_grad = torch.zeros_like(self.aux_bias)

        else:
            # Don't think this works, but the 'else' option does. Never use reduce
            if reduce:
                W_grad = 0.5 * (
                    torch.matmul(prob_h.t(), v) - torch.matmul(prob_hp.t(), vp)
                )
                U_grad = torch.zeros(
                    (v.shape[0], self.weights_U.shape[0], self.weights_U.shape[1]),
                    dtype=torch.double,
                    device=self.device,
                )
                vb_grad = 0.5 * torch.sum(v - vp, 0)
                hb_grad = 0.5 * torch.sum(prob_h - prob_hp, 0)
                ab_grad = torch.zeros(
                    (v.shape[0], self.num_aux), dtype=torch.double, device=self.device
                )

            else:
                W_grad = 0.5 * (
                    torch.einsum("ij,ik->ijk", prob_h, v)
                    - torch.einsum("ij,ik->ijk", prob_hp, vp)
                )
                U_grad = torch.zeros(
                    (v.shape[0], self.weights_U.shape[0], self.weights_U.shape[1]),
                    dtype=torch.double,
                    device=self.device,
                )
                vb_grad = 0.5 * (v - vp)
                hb_grad = 0.5 * (prob_h - prob_hp)
                ab_grad = torch.zeros(
                    (v.shape[0], self.num_aux), dtype=torch.double, device=self.device
                )
                vec = [
                    W_grad.view(v.size()[0], -1),
                    U_grad.view(v.size()[0], -1),
                    vb_grad,
                    hb_grad,
                    ab_grad,
                ]
                return cplx.make_complex(torch.cat(vec, dim=1))

        return cplx.make_complex(
            parameters_to_vector([W_grad, U_grad, vb_grad, hb_grad, ab_grad])
        )

    def probability(self, v, a):
        r"""Computes the probability of finding the system in a particular
            state of the visible and auxiliary units

        :param v: The visible units
        :type v: torch.Tensor
        :param a: The auxiliary units
        :type a: torch.Tensor
        :returns: The probability of the system having the
                  input visible and auxiliary states
        :rtype: torch.Tensor
        """
        return self.effective_energy(v, a).exp()

    def partition(self, v_space, a_space):
        r"""Computes the partition function

        :param v_space: The Hilbert space of the visible units
        :type v_space: torch.Tensor
        :param a_space: The Hilbert space of the auxiilary units
        :type a_space: torch.Tensor
        :returns: The partition function
        :rtype: torch.Tensor
        """
        logZ = self.effective_energy(v_space, a_space).logsumexp((0, 1))
        return logZ.exp()
