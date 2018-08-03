# Copyright 2018 PIQuIL - All Rights Reserved

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import warnings
from itertools import chain

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm, tqdm_notebook

from qucumber.callbacks import CallbackList

__all__ = [
    "BinomialRBMModule",
    "BinomialRBM"
]


def _warn_on_missing_gpu(gpu):
    if gpu and not torch.cuda.is_available():
        warnings.warn("Could not find GPU: will continue with CPU.",
                      ResourceWarning)


class BinomialRBMModule(nn.Module):
    def __init__(self, num_visible, num_hidden, zero_weights=False,
                 gpu=True, seed=None):
        super(BinomialRBMModule, self).__init__()
        self.num_visible = int(num_visible)
        self.num_hidden = int(num_hidden)

        _warn_on_missing_gpu(gpu)
        self.gpu = gpu and torch.cuda.is_available()

        if seed:
            if self.gpu:
                torch.cuda.manual_seed(seed)
            else:
                torch.manual_seed(seed)

        self.device = torch.device('cuda') if self.gpu else torch.device('cpu')

        if zero_weights:
            self.weights = nn.Parameter((torch.zeros(self.num_hidden,
                                                     self.num_visible,
                                                     device=self.device,
                                                     dtype=torch.double)),
                                        requires_grad=True)
        else:
            self.weights = nn.Parameter(
                (torch.randn(self.num_hidden, self.num_visible,
                             device=self.device, dtype=torch.double)
                 / np.sqrt(self.num_visible)),
                requires_grad=True)

        self.visible_bias = nn.Parameter(torch.zeros(self.num_visible,
                                                     device=self.device,
                                                     dtype=torch.double),
                                         requires_grad=True)
        self.hidden_bias = nn.Parameter(torch.zeros(self.num_hidden,
                                                    device=self.device,
                                                    dtype=torch.double),
                                        requires_grad=True)

    def __repr__(self):
        return ("BinomialRBMModule(num_visible={}, num_hidden={}, gpu={})"
                .format(self.num_visible, self.num_hidden, self.gpu))

    def effective_energy(self, v):
        r"""The effective energies of the given visible states.

        .. math::

            \mathcal{E}(\bm{v}) &= \sum_{j}b_j v_j
                        + \sum_{i}\log
                            \left\lbrack 1 +
                                  \exp\left(c_{i} + \sum_{j} W_{ij} v_j\right)
                            \right\rbrack

        :param v: The visible states.
        :type v: torch.Tensor

        :returns: The effective energies of the given visible states.
        :rtype: torch.Tensor
        """
        if len(v.shape) < 2:
            v = v.view(1, -1)
        visible_bias_term = torch.mv(v, self.visible_bias)
        hidden_bias_term = F.softplus(
            F.linear(v, self.weights, self.hidden_bias)
        ).sum(1)

        return visible_bias_term + hidden_bias_term

    def prob_v_given_h(self, h, out=None):
        """Given a hidden unit configuration, compute the probability
        vector of the visible units being on.

        :param h: The hidden unit
        :type h: torch.Tensor
        :param out: The output tensor to write to
        :type out: torch.Tensor

        :returns: The probability of visible units being active given the
                  hidden state.
        :rtype: torch.Tensor
        """
        p = torch.addmm(self.visible_bias.data, h,
                        self.weights.data, out=out)\
                 .sigmoid_()
        return p

    def prob_h_given_v(self, v, out=None):
        """Given a visible unit configuration, compute the probability
        vector of the hidden units being on.

        :param h: The hidden unit.
        :type h: torch.Tensor
        :param out: The output tensor to write to
        :type out: torch.Tensor

        :returns: The probability of hidden units being active given the
                  visible state.
        :rtype: torch.Tensor
        """
        p = torch.addmm(self.hidden_bias.data, v,
                        self.weights.data.t(), out=out) \
                 .sigmoid_()
        return p

    def sample_v_given_h(self, h):
        """Sample/generate a visible state given a hidden state.

        :param h: The hidden state.
        :type h: torch.Tensor

        :returns: Tuple containing prob_v_given_h(h) and the sampled visible
                  state.
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        p = self.prob_v_given_h(h)
        v = p.bernoulli()
        return p, v

    def sample_h_given_v(self, v):
        """Sample/generate a hidden state given a visible state.

        :param h: The visible state.
        :type h: torch.Tensor

        :returns: Tuple containing prob_h_given_v(v) and the sampled hidden
                  state.
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        p = self.prob_h_given_v(v)
        h = p.bernoulli()
        return p, h

    def gibbs_sampling(self, k, v0):
        """Performs k steps of Block Gibbs sampling given an initial visible
        state v0.

        :param k: Number of Block Gibbs steps.
        :type k: int
        :param v0: The initial visible state.
        :type v0: torch.Tensor

        :returns: Tuple containing the initial visible state, v0,
                  the hidden state sampled from v0,
                  the visible state sampled after k steps,
                  the hidden state sampled after k steps and its corresponding
                  probability vector.
        :rtype: tuple(torch.Tensor, torch.Tensor,
                      torch.Tensor, torch.Tensor,
                      torch.Tensor)
        """
        ph0 = self.prob_h_given_v(v0)
        v, h = torch.zeros_like(v0), torch.zeros_like(ph0)
        v.copy_(v0)
        h.copy_(ph0)
        for _ in range(k):
            self.prob_v_given_h(torch.bernoulli(h, out=h), out=v)
            self.prob_h_given_v(torch.bernoulli(v, out=v), out=h)
        if self.gpu:
            torch.cuda.empty_cache()
        return v0, ph0, v, h

    def sample(self, num_samples, k=10, initial_state=None):
        """Samples from the RBM using k steps of Block Gibbs sampling.

        :param num_samples: The number of samples to be generated
        :type num_samples: int
        :param k: Number of Block Gibbs steps.
        :type k: int
        :param initial_state: A set of samples to initialize the Markov Chains
                              with. If provided, `num_samples` is ignored, and
                              the number of samples returned will be equal to
                              `len(initial_state)`.
        :type initial_state: torch.Tensor

        :returns: Samples drawn from the RBM
        :rtype: torch.Tensor
        """
        if initial_state is None:
            dist = torch.distributions.bernoulli.Bernoulli(probs=0.5)
            v0 = (dist.sample(torch.Size([num_samples, self.num_visible]))
                      .to(device=self.device, dtype=torch.double))
        else:
            v0 = initial_state
        _, _, v, _ = self.gibbs_sampling(k, v0)
        return v

    def unnormalized_probability(self, v):
        r"""The unnormalized probabilities of the given visible states.

        .. math:: p(\bm{v}) = e^{\mathcal{E}(\bm{v})}

        :param v: The visible states.
        :type v: torch.Tensor

        :returns: The unnormalized probability of the given visible state(s).
        :rtype: torch.Tensor
        """
        return self.effective_energy(v).exp()

    def probability_ratio(self, a, b):
        return self.log_probability_ratio(a, b).exp()

    def log_probability_ratio(self, a, b):
        return self.effective_energy(a).sub(self.effective_energy(b))

    def generate_visible_space(self):
        """Generates all possible visible states.

        :returns: A tensor of all possible spin configurations.
        :rtype: torch.Tensor
        """
        space = torch.zeros((1 << self.num_visible, self.num_visible),
                            device=self.device, dtype=torch.double)
        for i in range(1 << self.num_visible):
            d = i
            for j in range(self.num_visible):
                d, r = divmod(d, 2)
                space[i, self.num_visible - j - 1] = int(r)

        return space

    def log_partition(self, visible_space):
        """The natural logarithm of the partition function of the RBM.

        :param visible_space: A rank 2 tensor of the entire visible space.
        :type visible_space: torch.Tensor

        :returns: The natural log of the partition function.
        :rtype: torch.Tensor
        """
        free_energies = self.effective_energy(visible_space)
        max_free_energy = free_energies.max()

        f_reduced = free_energies - max_free_energy
        logZ = max_free_energy + f_reduced.exp().sum().log()

        return logZ

    def partition(self, visible_space):
        """The partition function of the RBM.

        :param visible_space: A rank 2 tensor of the entire visible space.
        :type visible_space: torch.Tensor

        :returns: The partition function.
        :rtype: torch.Tensor
        """
        return self.log_partition(visible_space).exp()

    def probability(self, v, Z):
        """Evaluates the probability of the given vector(s) of visible
        units; NOT RECOMMENDED FOR RBMS WITH A LARGE # OF VISIBLE UNITS

        :param v: The visible states.
        :type v: torch.Tensor
        :param Z: The partition function.
        :type Z: float

        :returns: The probability of the given vector(s) of visible units.
        :rtype: torch.Tensor
        """
        return self.unnormalized_probability(v) / Z


class BinomialRBM:
    def __init__(self, num_visible, num_hidden=None, gpu=True, seed=None):
        super(BinomialRBM, self).__init__()
        self.num_visible = int(num_visible)
        self.num_hidden = (int(num_hidden)
                           if num_hidden is not None
                           else self.num_visible)
        self.rbm_module = BinomialRBMModule(self.num_visible, self.num_hidden,
                                            gpu=gpu, seed=seed)
        self.device = self.rbm_module.device
        self.stop_training = False

    def effective_energy(self, v):
        r"""The effective energies of the given visible states.

        .. math::

            \mathcal{E}(\bm{v}) &= \sum_{j}b_j v_j
                        + \sum_{i}\log
                            \left\lbrack 1 +
                                  \exp\left(c_{i} + \sum_{j} W_{ij} v_j\right)
                            \right\rbrack

        :param v: The visible states.
        :type v: torch.Tensor

        :returns: The effective energies of the given visible states.
        :rtype: torch.Tensor
        """
        return self.rbm_module.effective_energy(v)

    def save(self, location, metadata={}):
        """Saves the RBM parameters to the given location along with
        any given metadata.

        :param location: The location to save the RBM parameters + metadata
        :type location: str or file
        :param metadata: Any extra metadata to store alongside the RBM
                         parameters
        :type metadata: dict
        """
        # add extra metadata to dictionary before saving it to disk
        data = {**self.rbm_module.state_dict(), **metadata}
        torch.save(data, location)

    def load(self, location):
        """Loads the RBM parameters from the given location ignoring any
        metadata stored in the file. Overwrites the RBM's parameters.

        .. note::
            The RBM object on which this function is called must
            have the same shape as the one who's parameters are being
            loaded.

        :param location: The location to load the RBM parameters from
        :type location: str or file
        """

        try:
            state_dict = torch.load(location)
        except AssertionError as e:
            state_dict = torch.load(location, lambda storage, loc: 'cpu')

        if set(self.rbm_module.state_dict()) <= set(state_dict):
            self.rbm_module.load_state_dict(state_dict, strict=False)
        else:
            raise ValueError(
                "Given set of parameters is incomplete! "
                + "Has keys: {}; ".format(list(state_dict.keys()))
                + "Need: {}".format(self.rbm_module.named_parameters()))

    @staticmethod
    def autoload(location, gpu=True):
        """Initializes an RBM from the parameters in the given location,
        ignoring any metadata stored in the file.

        :param location: The location to load the RBM parameters from
        :type location: str or file

        :returns: A new RBM initialized from the given parameters
        :rtype: BinomialRBM
        """
        _warn_on_missing_gpu(gpu)
        gpu = gpu and torch.cuda.is_available()

        if gpu:
            state_dict = torch.load(location, lambda storage, loc: 'cuda')
        else:
            state_dict = torch.load(location, lambda storage, loc: 'cpu')

        rbm = BinomialRBM(num_visible=len(state_dict['visible_bias']),
                          num_hidden=len(state_dict['hidden_bias']),
                          gpu=gpu,
                          seed=None)

        if set(rbm.rbm_module.state_dict()) <= set(state_dict):
            rbm.rbm_module.load_state_dict(state_dict, strict=False)
        else:
            raise ValueError(
                "Given set of parameters is incomplete! "
                + "Has keys: {}; ".format(list(state_dict.keys()))
                + "Need: {}".format(rbm.rbm_module.named_parameters()))

        return rbm

    def compute_batch_gradients(self, k, pos_batch, neg_batch):
        """This function will compute the gradients of a batch of the training
        data (data_file) given the basis measurements (chars_file).

        :param k: Number of contrastive divergence steps in training.
        :type k: int
        :param pos_batch: Batch of the input data for the positive phase.
        :type pos_batch: torch.Tensor
        :param neg_batch: Batch of the input data for the negative phase.
        :type neg_batch: torch.Tensor

        :returns: Dictionary containing all the gradients of the parameters.
        :rtype: dict
        """

        v0, ph0, _, _ = self.rbm_module.gibbs_sampling(0, pos_batch)
        _, _, vk, phk = self.rbm_module.gibbs_sampling(k, neg_batch)

        pos_batch_size = float(len(pos_batch))
        neg_batch_size = float(len(neg_batch))

        w_grad = torch.einsum("ij,ik->jk", (ph0, v0))/pos_batch_size
        vb_grad = torch.einsum("ij->j", (v0,))/pos_batch_size
        hb_grad = torch.einsum("ij->j", (ph0,))/pos_batch_size

        w_grad -= torch.einsum("ij,ik->jk", (phk, vk))/neg_batch_size
        vb_grad -= torch.einsum("ij->j", (vk,))/neg_batch_size
        hb_grad -= torch.einsum("ij->j", (phk,))/neg_batch_size

        # Return negative gradients to match up nicely with the usual
        # parameter update rules, which *subtract* the gradient from
        # the parameters. This is in contrast with the RBM update
        # rules which ADD the gradients (scaled by the learning rate)
        # to the parameters.

        return {"rbm_module": {"weights": -w_grad,
                               "visible_bias": -vb_grad,
                               "hidden_bias": -hb_grad}}

    def fit(self, data, epochs=100, pos_batch_size=100, neg_batch_size=200,
            k=1, lr=1e-2, progbar=False, callbacks=[]):
        """Execute the training of the RBM.

        :param data: The actual training data
        :type data: torch.Tensor
        :param epochs: The number of parameter (i.e. weights and biases)
                       updates
        :type epochs: int
        :param pos_batch_size: The size of batches for the positive phase
                               taken from the data.
        :type pos_batch_size: int
        :param neg_batch_size: The size of batches for the negative phase
                               taken from the data
        :type neg_batch_size: int
        :param k: The number of contrastive divergence steps
        :type k: int
        :param lr: Learning rate
        :type lr: float
        :param progbar: Whether or not to display a progress bar. If "notebook"
                        is passed, will use a Jupyter notebook compatible
                        progress bar.
        :type progbar: bool or str
        :param callbacks: Callbacks to run while training.
        :type callbacks: list(qucumber.callbacks.Callback)
        """

        disable_progbar = (progbar is False)
        progress_bar = tqdm_notebook if progbar == "notebook" else tqdm
        callbacks = CallbackList(callbacks)

        data = torch.tensor(data, device=self.rbm_module.device,
                            dtype=torch.double)
        optimizer = torch.optim.SGD(self.rbm_module.parameters(), lr=lr)

        callbacks.on_train_start(self)

        for ep in progress_bar(range(epochs), desc="Epochs ",
                               disable=disable_progbar):
            pos_batches = DataLoader(data, batch_size=pos_batch_size,
                                     shuffle=True)

            multiplier = int((neg_batch_size / pos_batch_size) + 0.5)
            neg_batches = [DataLoader(data, batch_size=neg_batch_size,
                                      shuffle=True)
                           for i in range(multiplier)]
            neg_batches = chain(*neg_batches)

            callbacks.on_epoch_start(self, ep)

            if self.stop_training:  # check for stop_training signal
                break

            for batch_num, (pos_batch, neg_batch) in enumerate(zip(pos_batches,
                                                                   neg_batches)):
                callbacks.on_batch_start(self, ep, batch_num)

                all_grads = self.compute_batch_gradients(k, pos_batch,
                                                         neg_batch)
                optimizer.zero_grad()  # clear any cached gradients

                # assign all available gradients to the corresponding parameter
                for name, grads in all_grads.items():
                    selected_RBM = getattr(self, name)
                    for param in grads.keys():
                        getattr(selected_RBM, param).grad = grads[param]

                optimizer.step()  # tell the optimizer to apply the gradients

                callbacks.on_batch_end(self, ep, batch_num)

            callbacks.on_epoch_end(self, ep)

        callbacks.on_train_end(self)

    def sample(self, num_samples, k, initial_state=None):
        """Samples from the RBM using k steps of Block Gibbs sampling.

        :param num_samples: The number of samples to be generated
        :type num_samples: int
        :param k: Number of Block Gibbs steps.
        :type k: int
        :param initial_state: A set of samples to initialize the Markov Chains
                              with. If provided, `num_samples` is ignored, and
                              the number of samples returned will be equal to
                              `len(initial_state)`.
        :type initial_state: torch.Tensor

        :returns: Samples drawn from the RBM.
        :rtype: torch.Tensor
        """
        return self.rbm_module.sample(num_samples, k,
                                      initial_state=initial_state)

    def probability_ratio(self, a, b):
        return self.rbm_module.log_probability_ratio(a, b).exp()

    def log_probability_ratio(self, a, b):
        return self.rbm_module.effective_energy(a) \
                              .sub(self.effective_energy(b))
