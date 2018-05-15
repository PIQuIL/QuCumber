import numpy as np
from scipy import stats, special
from scipy.special import expit as sigmoid
from tqdm import tqdm_notebook, tqdm


class RBM:
    def __init__(self, num_visible, num_hidden, seed=1234):
        self.num_visible = num_visible
        self.num_hidden = num_hidden

        self.rand_state = np.random.RandomState(1234)
        try:
            self.cache = {}
            self.cache["visible_space"] = self.get_visible_space(num_visible)
        except MemoryError:
            self.cache = {}
            raise
            # a visible space that large will take too long;
            # better to just give up now

        self.weights = self.rand_state.randn(self.num_hidden, self.num_visible)
        self.visible_bias = self.rand_state.randn(self.num_visible)
        self.hidden_bias = self.rand_state.randn(self.num_hidden)

    def __repr__(self):
        return ("RBM(num_visible={}, num_hidden={})"
                .format(self.num_visible, self.num_hidden))

    def clear_cache(self, keep_visible_space=True):
        for key in list(self.cache.keys()):
            if key != "visible_space":
                del self.cache[key]

        if not keep_visible_space:
            del self.cache["visible_space"]

    @staticmethod
    @np.vectorize
    def ln1pexp(x):
        if x > 30:
            # maximum absolute error on order of 1e-13;
            # error -> 0 as x -> \infty
            return x
        else:
            return np.log1p(np.exp(x))

    def free_energy(self, v):
        free_energy_ = np.dot(self.visible_bias, v)
        free_energy_ += np.sum(self.ln1pexp(np.dot(self.weights, v)
                                            + self.hidden_bias))

        return free_energy_

    def prob_h_given_v(self, v):
        if len(v.shape) == 2:
            p = sigmoid(np.dot(self.weights, v.T)
                        + self.hidden_bias.reshape(-1, 1)).T
        else:
            p = sigmoid(np.dot(self.weights, v) + self.hidden_bias)
        return p

    def prob_v_given_h(self, h):
        if len(h.shape) == 2:
            p = sigmoid(np.dot(self.weights.T, h.T)
                        + self.visible_bias.reshape(-1, 1)).T
        else:
            p = sigmoid(np.dot(self.weights.T, h) + self.visible_bias)
        return p

    def sample_h_given_v(self, v):
        h = self.rand_state.binomial(1, self.prob_h_given_v(v))
        return h.astype(np.float32)

    def sample_v_given_h(self, h):
        v = self.rand_state.binomial(1, self.prob_v_given_h(h))
        return v.astype(np.float32)

    def gibbs_sampling(self, k, v0):
        h0 = self.sample_h_given_v(v0)
        v, h = v0, h0
        for _ in range(k):
            v = self.sample_v_given_h(h)
            h = self.sample_h_given_v(v)
        return v0, h0, v, h, self.prob_h_given_v(v)

    def single_batch_gradients(self, batch, k, first_batch=False, loop=False):
        w_grad = np.zeros_like(self.weights)
        v_b_grad = np.zeros_like(self.visible_bias)
        h_b_grad = np.zeros_like(self.hidden_bias)

        if len(batch) == 0:
            return w_grad, v_b_grad, h_b_grad

        for v0 in batch:
            v0, h0, vk, hk, phk = self.gibbs_sampling(k, v0)
            w_grad -= np.outer(h0, v0) - np.outer(phk, vk)
            v_b_grad -= v0 - vk
            h_b_grad -= h0 - phk

        w_grad /= len(batch)
        v_b_grad /= len(batch)
        h_b_grad /= len(batch)

        return w_grad, v_b_grad, h_b_grad

    def apply_to_visible_space(self, fn):
        if "visible_space" in self.cache:
            return np.apply_along_axis(fn, 1, self.cache["visible_space"])
        else:
            # for when the visible space is too big to fit in memory
            # (warning: slow)
            return np.fromiter(
                map(fn, self.visible_space_generator(self.num_visible)),
                dtype=np.float32
            )

    def compute_numerical_kl(self, target):
        KL = stats.entropy(np.power(target, 2),
                           self.apply_to_visible_space(self.probability))
        return KL

    def test_gradient(self, target, param, alg_grad, eps=1e-8):
        for i in range(len(param)):
            param[i] += eps
            self.clear_cache()
            KL_p = self.compute_numerical_kl(target)

            param[i] -= 2*eps
            self.clear_cache()
            KL_m = self.compute_numerical_kl(target)

            param[i] += eps
            self.clear_cache()

            num_grad = (KL_p - KL_m) / (2*eps)
            print("{: 10.8f}, {: 10.8f}, {: 10.8f}"
                  .format(num_grad, alg_grad[i], abs(num_grad - alg_grad[i])))

    def test_gradients(self, data, target, k, eps=1e-8):
        w_grad, v_b_grad, h_b_grad = self.single_batch_gradients(data, k)
        print("Testing visible bias...")
        self.test_gradient(target, self.visible_bias, v_b_grad)
        print("\nTesting hidden bias...")
        self.test_gradient(target, self.hidden_bias, h_b_grad)
        print("\nTesting weights...")
        self.test_gradient(target, self.weights.flat, w_grad.flat)

    def train(self, data, target, epochs, batch_size,
              k=1, lr=1e-3, log_every=10, notebook=False):
        nll_list, overlap_list = [], []
        prog_bar = tqdm_notebook if notebook else tqdm

        for ep in prog_bar(range(epochs), desc="Epochs "):
            self.rand_state.shuffle(data)
            self.clear_cache()

            if ep % log_every == 0:
                overlap = self.overlap(target)
                overlap_list.append(overlap)
                nll = self.negative_log_likelihood(data)
                nll_list.append(nll)
                tqdm.write(("NLL per training example = {: 10.8f};"
                            " Overlap = {: 10.8f}")
                           .format(nll / len(data), overlap))

            batches = [data[batch_start:(batch_start + batch_size)]
                       for batch_start in range(0, len(data), batch_size)]

            first_batch = True
            for batch in prog_bar(batches, desc="Batches", leave=False):
                w_grad, v_b_grad, h_b_grad = \
                    self.single_batch_gradients(batch, k,
                                                first_batch=first_batch)
                first_batch = False
                self.weights -= (lr * w_grad)
                self.visible_bias -= (lr * v_b_grad)
                self.hidden_bias -= (lr * h_b_grad)

        return nll_list, overlap_list

    @staticmethod
    def visible_space_generator(n):
        for i in range(1 << n):
            d = i
            arr = np.zeros((n,))
            for j in range(n):
                d, r = divmod(d, 2)
                arr[n - j - 1] = int(r)
            yield arr

    @staticmethod
    def get_visible_space(n):
        space = np.zeros((1 << n, n)).astype(np.byte)  # matrix of size 2^n x n
        for i in range(1 << n):
            d = i
            for j in range(n):
                d, r = divmod(d, 2)
                space[i, n - j - 1] = int(r)

        return space

    def Z(self): return partition(self)

    def partition(self):
        partition_ = np.exp(self.log_partition())
        return partition_

    def logZ(self): return self.log_partition()

    def log_partition(self):
        if "log_partition" in self.cache:
            return self.cache["log_partition"]

        free_energies = self.apply_to_visible_space(self.free_energy)
        logZ = special.logsumexp(free_energies)

        self.cache["log_partition"] = logZ
        return logZ

    def p(self, v): return self.probability(v)

    def probability(self, v):
        logZ = self.log_partition()
        return np.exp(self.free_energy(v) - logZ)

    def nll(self, data): return self.negative_log_likelihood(data)

    def negative_log_likelihood(self, data):
        if "nll" in self.cache:
            return self.cache["nll"]

        logZ = self.log_partition()
        total_free_energy = np.sum(np.apply_along_axis(self.free_energy,
                                                       1, data))

        self.cache["nll"] = len(data)*logZ - total_free_energy
        return self.cache["nll"]

    def overlap(self, target):
        if target is None:
            return None

        prob_vect = self.apply_to_visible_space(self.probability)
        prob_vect = np.sqrt(prob_vect)

        return np.dot(target, prob_vect)
