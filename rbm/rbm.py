import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import tqdm, tqdm_notebook
import warnings


class RBM(nn.Module):
    def __init__(self, num_visible, num_hidden, gpu=True, seed=1234):
        super(RBM, self).__init__()
        self.num_visible = int(num_visible)
        self.num_hidden = int(num_hidden)

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

    def prob_v_given_h(self, h):
        p = F.sigmoid(F.linear(h, self.weights.t(), self.visible_bias))
        return p

    def prob_h_given_v(self, v):
        p = F.sigmoid(F.linear(v, self.weights, self.hidden_bias))
        return p

    def sample_v_given_h(self, h):
        p = self.prob_v_given_h(h)
        v = p.bernoulli()
        return p, v

    def sample_h_given_v(self, v):
        p = self.prob_h_given_v(v)
        h = p.bernoulli()
        return p, h

    def gibbs_sampling(self, k, v0):
        _, h0 = self.sample_h_given_v(v0)
        h = h0
        for _ in range(k):
            pv, v = self.sample_v_given_h(h)
            ph, h = self.sample_h_given_v(v)
        return v0, h0, v, h, ph

    def sample(self, k, num_samples):
        dist = torch.distributions.bernoulli.Bernoulli(probs=0.5)
        v0 = (dist.sample(torch.Size([num_samples, self.num_visible]))
                  .to(device=self.device, dtype=torch.double))
        _, _, v, _, _ = self.gibbs_sampling(k, v0)
        return v

    def regularize_weight_gradients(self, w_grad, l1_reg, l2_reg):
        return (w_grad
                + (l2_reg * self.weights)
                + (l1_reg * self.weights.sign()))

    def compute_batch_gradients(self, k, batch, l1_reg, l2_reg, stddev=0):
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

        v0, h0, vk, hk, phk = self.gibbs_sampling(k, batch)

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
        w_grad += (stddev * torch.randn_like(w_grad, device=self.device))
        # Return negative gradients to match up nicely with the usual
        # parameter update rules, which *subtract* the gradient from
        # the parameters. This is in contrast with the RBM update
        # rules which ADD the gradients (scaled by the learning rate)
        # to the parameters.
        return {"weights": -w_grad,
                "visible_bias": -vb_grad,
                "hidden_bias": -hb_grad}

    def train(self, data, epochs, batch_size,
              k=10, lr=1e-3, momentum=0.0,
              method='sgd', l1_reg=0.0, l2_reg=0.0,
              initial_gaussian_noise=0.01, gamma=0.55,
              callbacks=[], progbar=False,
              log_every=50,
              **kwargs):
        # callback_outputs = []
        disable_progbar = (progbar is False)
        progress_bar = tqdm_notebook if progbar == "notebook" else tqdm

        data = torch.tensor(data).to(device=self.device)
        optimizer = torch.optim.Adam([self.weights,
                                      self.visible_bias,
                                      self.hidden_bias],
                                     lr=lr)

        vis = self.generate_visible_space()
        for ep in progress_bar(range(epochs + 1), desc="Epochs ",
                               total=epochs, disable=disable_progbar):
            batches = DataLoader(data, batch_size=batch_size, shuffle=True)

            if ep % log_every == 0:
                logZ = self.log_partition(vis)
                nll = self.nll(data, logZ)
                tqdm.write("{}: {}".format(ep, nll.item() / len(data)))

            if ep == epochs:
                break

            stddev = torch.tensor(
                [initial_gaussian_noise / ((1 + ep) ** gamma)],
                dtype=torch.double, device=self.device).sqrt()

            for batch in progress_bar(batches, desc="Batches",
                                      leave=False, disable=True):
                grads = self.compute_batch_gradients(k, batch,
                                                     l1_reg, l2_reg,
                                                     stddev=stddev)

                optimizer.zero_grad()  # clear any cached gradients

                # assign all available gradients to the corresponding parameter
                for name in grads.keys():
                    getattr(self, name).grad = grads[name]

                optimizer.step()  # tell the optimizer to apply the gradients
            # TODO: run callbacks

    def free_energy(self, v):
        if len(v.shape) < 2:
            v = v.view(1, -1)
        visible_bias_term = torch.mv(v, self.visible_bias)
        hidden_bias_term = F.softplus(
            F.linear(v, self.weights, self.hidden_bias)
        ).sum(1)

        return visible_bias_term + hidden_bias_term

    def unnormalized_probability(self, v):
        return self.free_energy(v).exp()

    def generate_visible_space(self):
        space = torch.zeros((1 << self.num_visible, self.num_visible),
                            device=self.device, dtype=torch.double)
        for i in range(1 << self.num_visible):
            d = i
            for j in range(self.num_visible):
                d, r = divmod(d, 2)
                space[i, self.num_visible - j - 1] = int(r)

        return space

    def log_partition(self, visible_space):
        free_energies = self.free_energy(visible_space)
        max_free_energy = free_energies.max()

        f_reduced = free_energies - max_free_energy
        logZ = max_free_energy + f_reduced.exp().sum().log()

        return logZ

    def partition(self, visible_space):
        return self.log_partition(visible_space).exp()

    def nll(self, data, logZ):
        total_free_energy = self.free_energy(data).sum()

        return (len(data)*logZ) - total_free_energy
