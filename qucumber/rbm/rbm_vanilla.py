import warnings

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm, tqdm_notebook


class RBM(nn.Module):
    """Class to build the Restricted Boltzmann Machine.

    :ivar int num_visible: Number of visible units
                           (determined from the input training data).
    :ivar int num_hidden: Number of hidden units to learn the amplitude
                          (default = num_visible).
    :ivar bool gpu: Should the GPU be used for the training (default = True).
    :ivar int seed: Fix the random number seed to make results reproducable
                    (default = 1234).


    :param weights: The weight matrix for the amplitude RBM
                    (dims = num_hidden x num_visible).
                    Initialized to random numbers sampled from a
                    normal distribution.
    :type weights: torch.doubleTensor
    :param visible_bias: The visible bias vector for the amplitude RBM
                         (size = num_visible). Initialized to zeros.
    :type visible_bias: torch.doubleTensor
    :param hidden_bias: The hidden bias vector for the amplitude RBM
                        (size = num_hidden). Initialized to zeros.
    :type num_hidden: torch.doubleTensor

    :raises ResourceWarning: If a GPU could not be found;
                             continue running the program on the CPU.
    """
    def __init__(self, num_visible, num_hidden, gpu=True, seed=1234):
        super(RBM, self).__init__()
        self.num_visible = int(num_visible)
        self.num_hidden = int(num_hidden)
        self.stop_training = False

        if gpu and not torch.cuda.is_available():
            warnings.warn("Could not find GPU: will continue with CPU.",
                          ResourceWarning)

        self.gpu = gpu and torch.cuda.is_available()
        if self.gpu:
            torch.cuda.manual_seed(seed)
            self.device = torch.device('cuda')
        else:
            torch.manual_seed(seed)
            self.device = torch.device('cpu')

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
        return ("RBM(num_visible={}, num_hidden={}, gpu={})"
                .format(self.num_visible, self.num_hidden, self.gpu))

    def save(self, location, metadata={}):
        """Saves the RBM parameters to the given location along with
        any given metadata.

        :param location: The location to save the RBM parameters + metadata
        :type location: str or file-like
        :param metadata: Any extra metadata to store alongside the RBM
                         parameters
        :type metadata: dict
        """
        # add extra metadata to dictionary before saving it to disk
        rbm_data = {**self.state_dict(), **metadata}
        torch.save(rbm_data, location)

    def load(self, location):
        """Loads the RBM parameters from the given location ignoring any
        metadata stored in the file. Overwrites the RBM's parameters.

        .. note:: The RBM object on which this function is called must
                  have the same shape as the one who's parameters are being
                  loaded.

        :param location: The location to load the RBM parameters from
        :type location: str or file-like
        """
        self.load_state_dict(torch.load(location), strict=False)

    def prob_v_given_h(self, h):
        """Given a hidden unit configuration, compute the probability
        vector of the visible units being on.

        :param h: The hidden unit
        :type h: torch.doubleTensor

        :returns: The probability of visible units being active given the
                  hidden state.
        :rtype: torch.doubleTensor
        """
        p = F.sigmoid(F.linear(h, self.weights.t(), self.visible_bias))
        return p

    def prob_h_given_v(self, v):
        """Given a visible unit configuration, compute the probability
        vector of the hidden units being on.

        :param h: The hidden unit.
        :type h: torch.doubleTensor

        :returns: The probability of hidden units being active given the
                  visible state.
        :rtype: torch.doubleTensor
        """
        p = F.sigmoid(F.linear(v, self.weights, self.hidden_bias))
        return p

    def sample_v_given_h(self, h):
        """Sample/generate a visible state given a hidden state.

        :param h: The hidden state.
        :type h: torch.doubleTensor

        :returns: Tuple containing prob_v_given_h(h) and the sampled visible
                  state.
        :rtype: tuple(torch.doubleTensor, torch.doubleTensor)
        """
        p = self.prob_v_given_h(h)
        v = p.bernoulli()
        return p, v

    def sample_h_given_v(self, v):
        """Sample/generate a hidden state given a visible state.

        :param h: The visible state.
        :type h: torch.doubleTensor

        :returns: Tuple containing prob_h_given_v(v) and the sampled hidden
                  state.
        :rtype: tuple(torch.doubleTensor, torch.doubleTensor)
        """
        p = self.prob_h_given_v(v)
        h = p.bernoulli()
        return p, h

    def gibbs_sampling(self, k, v0):
        """Performs k steps of Block Gibbs sampling given an initial visible state v0.

        :param k: Number of Block Gibbs steps.
        :type k: int
        :param v0: The initial visible state.
        :type v0: torch.doubleTensor

        :returns: Tuple containing the initial visible state, v0,
                  the hidden state sampled from v0,
                  the visible state sampled after k steps,
                  the hidden state sampled after k steps and its corresponding
                  probability vector.
        :rtype: tuple(torch.doubleTensor, torch.doubleTensor,
                      torch.doubleTensor, torch.doubleTensor,
                      torch.doubleTensor)
        """
        ph, h0 = self.sample_h_given_v(v0)
        v, h = v0, h0
        for _ in range(k):
            pv, v = self.sample_v_given_h(h)
            ph, h = self.sample_h_given_v(v)
        return v0, h0, v, h, ph

    def sample(self, num_samples, k):
        """Samples from the RBM using k steps of Block Gibbs sampling.

        :param num_samples: The number of samples to be generated
        :type num_samples: int
        :param k: Number of Block Gibbs steps.
        :type k: int

        :returns: Samples drawn from the RBM
        :rtype: torch.doubleTensor
        """
        dist = torch.distributions.bernoulli.Bernoulli(probs=0.5)
        v0 = (dist.sample(torch.Size([num_samples, self.num_visible]))
                  .to(device=self.device, dtype=torch.double))
        _, _, v, _, _ = self.gibbs_sampling(k, v0)
        return v

    def regularize_weight_gradients(self, w_grad, l1_reg, l2_reg):
        r"""Applies regularization to the given weight gradient

        :param w_grad: The entire weight gradient matrix,
                       :math:`\nabla_{W}NLL`
        :type w_grad: torch.doubleTensor
        :param l1_reg: l1 regularization hyperparameter (default = 0.0).
        :type l1_reg: double
        :param l2_reg: l2 regularization hyperparameter (default = 0.0).
        :type l2_reg: double

        :returns:

        .. math::

            [\nabla_{W_{\lambda}}NLL]_{i,j}
                + l_1\sgn([W_{\lambda}]_{i,j})
                + l_2\vert[W_{\lambda}]_{i,j}\vert

        :rtype: torch.doubleTensor
        """
        return (w_grad
                + (l2_reg * self.weights)
                + (l1_reg * self.weights.sign()))

    def compute_batch_gradients(self, k, batch,
                                persistent=False,
                                pbatch=None,
                                l1_reg=0.0, l2_reg=0.0,
                                stddev=0.0):
        """This function will compute the gradients of a batch of the training
        data (data_file) given the basis measurements (chars_file).

        :param k: Number of contrastive divergence steps in training.
        :type k: int
        :param batch: Batch of the input data.
        :type batch: torch.doubleTensor
        :param l1_reg: L1 regularization hyperparameter (default = 0.0)
        :type l1_reg: double
        :param l2_reg: L2 regularization hyperparameter (default = 0.0)
        :type l2_reg: double
        :param stddev: Standard deviation of random noise that can be added to
                       the weights.	This is also a hyperparameter.
                       (default = 0.0)
        :type stddev: double

        :returns: Dictionary containing all the gradients.
        :rtype: dict
        """
        if len(batch) == 0:
            return (torch.zeros_like(self.weights,
                                     device=self.device,
                                     dtype=torch.double),
                    torch.zeros_like(self.visible_bias,
                                     device=self.device,
                                     dtype=torch.double),
                    torch.zeros_like(self.hidden_bias,
                                     device=self.device,
                                     dtype=torch.double))

        if not persistent:
            v0, h0, vk, hk, phk = self.gibbs_sampling(k, batch)
        else:
            # Positive phase comes from training data
            v0, h0, _, _, _ = self.gibbs_sampling(0, batch)
            # Negative phase comes from Markov chains
            _, _, vk, hk, phk = self.gibbs_sampling(k, pbatch)

        w_grad = torch.einsum("ij,ik->jk", (h0, v0))
        vb_grad = torch.einsum("ij->j", (v0,))
        hb_grad = torch.einsum("ij->j", (h0,))

        w_grad -= torch.einsum("ij,ik->jk", (phk, vk))
        vb_grad -= torch.einsum("ij->j", (vk,))
        hb_grad -= torch.einsum("ij->j", (phk,))

        w_grad /= float(len(batch))
        vb_grad /= float(len(batch))
        hb_grad /= float(len(batch))

        w_grad = self.regularize_weight_gradients(w_grad, l1_reg, l2_reg)

        if stddev != 0.0:
            w_grad += (stddev * torch.randn_like(w_grad, device=self.device))

        # Return negative gradients to match up nicely with the usual
        # parameter update rules, which *subtract* the gradient from
        # the parameters. This is in contrast with the RBM update
        # rules which ADD the gradients (scaled by the learning rate)
        # to the parameters.
        return (
            {"weights": -w_grad,
             "visible_bias": -vb_grad,
             "hidden_bias": -hb_grad},
            (vk if persistent else None)
        )

    def train(self, data, epochs, batch_size,
              k=10, persistent=False,
              lr=1e-3, momentum=0.0,
              l1_reg=0.0, l2_reg=0.0,
              initial_gaussian_noise=0.0, gamma=0.55,
              callbacks=[], progbar=False, starting_epoch=0):
        """Execute the training of the RBM.

        :param data: The actual training data
        :type data: array_like of doubles
        :param epochs: The number of parameter (i.e. weights and biases)
                       updates (default = 100).
        :type epochs: int
        :param batch_size: The size of batches taken from the data
                           (default = 100).
        :type batch_size: int
        :param k: The number of contrastive divergence steps (default = 10).
        :type k: int
        :param persistent: Whether to use persistent contrastive divergence
                           (aka. Stochastic Maximum Likelihood)
                           (default = False).
        :type persistent: bool
        :param lr: Learning rate (default = 1.0e-3).
        :type lr: double
        :param momentum: Momentum hyperparameter (default = 0.0).
        :type momentum: double
        :param l1_reg: L1 regularization hyperparameter (default = 0.0).
        :type l1_reg: double
        :param l2_reg: L2 regularization hyperparameter (default = 0.0).
        :type l2_reg: double
        :param initial_gaussian_noise: Initial gaussian noise used to calculate
                                       stddev of random noise added to weight
                                       gradients (default = 0.01).
        :type initial_gaussian_noise: double
        :param gamma: Parameter used to calculate stddev (default = 0.55).
        :type gamma: double
        :param callbacks: A list of Callback functions to call at the beginning
                          of each epoch
        :type callbacks: [Callback]
        :param progbar: Whether or not to display a progress bar. If "notebook"
                        is passed, will use a Jupyter notebook compatible
                        progress bar.
        :type progbar: bool or "notebook"
        :param starting_epoch: Which epoch to start from. Doesn't actually
                               affect the model, exists simply for book-keeping
                               when restarting training.

        :returns: None
        """

        disable_progbar = (progbar is False)
        progress_bar = tqdm_notebook if progbar == "notebook" else tqdm

        data = torch.tensor(data).to(device=self.device,
                                     dtype=torch.double)
        optimizer = torch.optim.SGD([self.weights,
                                     self.visible_bias,
                                     self.hidden_bias],
                                    lr=lr)

        if persistent:
            dist = torch.distributions.bernoulli.Bernoulli(probs=0.5)
            pbatch = (dist.sample(torch.Size([batch_size, self.num_visible]))
                          .to(device=self.device, dtype=torch.double))
        else:
            pbatch = None

        for ep in progress_bar(range(starting_epoch, epochs + 1),
                               desc="Epochs ", total=epochs,
                               disable=disable_progbar):
            batches = DataLoader(data, batch_size=batch_size, shuffle=True)

            for cb in callbacks:
                cb(self, ep)

            if self.stop_training:  # check for stop_training signal
                break

            if ep == epochs:
                break

            stddev = torch.tensor(
                [initial_gaussian_noise / ((1 + ep) ** gamma)],
                dtype=torch.double, device=self.device).sqrt()

            for batch in progress_bar(batches, desc="Batches",
                                      leave=False, disable=True):
                grads, pbatch = self.compute_batch_gradients(
                    k, batch,
                    pbatch=pbatch,
                    persistent=persistent,
                    l1_reg=l1_reg, l2_reg=l2_reg, stddev=stddev)

                optimizer.zero_grad()  # clear any cached gradients

                # assign all available gradients to the corresponding parameter
                for name in grads.keys():
                    getattr(self, name).grad = grads[name]

                optimizer.step()  # tell the optimizer to apply the gradients

    def effective_energy(self, v):
        r"""The effective energies of the given visible states.

        .. math::

           \mathcal{E}(\bm{v}) &= \sum_{j}b_j v_j
                       + \sum_{i}\log
                            \left\lbrack 1 +
                                  \exp\left(c_{i} + \sum_{j} W_{ij} v_j\right)
                            \right\rbrack

        :param v: The visible states.
        :type v: torch.doubleTensor

        :returns: The effective energies of the given visible states.
        :rtype: torch.doubleTensor
        """
        if len(v.shape) < 2:
            v = v.view(1, -1)
        visible_bias_term = torch.mv(v, self.visible_bias)
        hidden_bias_term = F.softplus(
            F.linear(v, self.weights, self.hidden_bias)
        ).sum(1)

        return visible_bias_term + hidden_bias_term

    def unnormalized_probability(self, v):
        r"""The unnormalized probabilities of the given visible states.

        .. math:: p(v) = \exp{\mathcal{E}(\bm{v})}

        :param v: The visible states.
        :type v: torch.doubleTensor

        :returns: The unnormalized probability of the given visible state(s).
        :rtype: torch.doubleTensor
        """
        return self.effective_energy(v).exp()

    def probability_ratio(self, a, b):
        """The probability ratio of two sets of visible states

        .. note:: `a` and `b` must be have the same shape

        :param a: The visible states for the numerator.
        :type a: torch.doubleTensor
        :param b: The visible states for the denominator.
        :type b: torch.doubleTensor

        :returns: The probability ratios of the given visible states
        :rtype: torch.doubleTensor
        """
        prob_a = self.unnormalized_probability(a)
        prob_b = self.unnormalized_probability(b)

        return prob_a.div(prob_b)

    def log_probability_ratio(self, a, b):
        """The natural logarithm of the probability ratio of
        two sets of visible states

        .. note:: `a` and `b` must be have the same shape

        :param a: The visible states for the numerator.
        :type a: torch.doubleTensor
        :param b: The visible states for the denominator.
        :type b: torch.doubleTensor

        :returns: The log-probability ratios of the given visible states
        :rtype: torch.doubleTensor
        """
        log_prob_a = self.effective_energy(a)
        log_prob_b = self.effective_energy(b)

        return log_prob_a.sub(log_prob_b)

    def generate_visible_space(self):
        """Generates all possible visible states.

        :returns: A tensor of all possible spin configurations.
        :rtype: torch.doubleTensor
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
        :type visible_space: torch.doubleTensor

        :returns: The natural log of the partition function.
        :rtype: torch.doubleTensor
        """
        free_energies = self.effective_energy(visible_space)
        max_free_energy = free_energies.max()

        f_reduced = free_energies - max_free_energy
        logZ = max_free_energy + f_reduced.exp().sum().log()

        return logZ

    def partition(self, visible_space):
        """The partition function of the RBM.

        :param visible_space: A rank 2 tensor of the entire visible space.
        :type visible_space: torch.doubleTensor

        :returns: The partition function.
        :rtype: torch.doubleTensor
        """
        return self.log_partition(visible_space).exp()

    def nll(self, data, visible_space):
        """The negative log likelihood of the given data.

        :param data: A rank 2 tensor of visible states.
        :type data: torch.doubleTensor
        :param visible_space: A rank 2 tensor of the entire visible space.
        :type visible_space: torch.doubleTensor

        :returns: The negative log likelihood of the given data.
        :rtype: torch.doubleTensor
        """
        total_free_energy = self.effective_energy(data).sum()
        logZ = self.log_partition(visible_space)
        return (len(data)*logZ) - total_free_energy
