import numpy as np
from scipy import stats, special
from scipy.special import expit
from tqdm import tqdm_notebook, tqdm
import rbm_grad_updaters as rgu
import schedulers
from alias import alias, aliased


@aliased
class RBM:
    def __init__(self, num_visible=None, num_hidden=None, seed=1234,
                 weights=None, visible_bias=None, hidden_bias=None,
                 rand_state=None):
        if weights is not None:
            self.num_hidden, self.num_visible = weights.shape
        elif num_visible and num_hidden:
            self.num_visible = int(num_visible)
            self.num_hidden = int(num_hidden)
        else:
            raise ValueError("Not enough information given to build RBM. "
                             "Need either: the number of hidden and visible "
                             "units OR the weight matrix and bias vectors.")

        if rand_state:
            if isinstance(rand_state, tuple):  #
                self.rand_state = np.random.RandomState(seed)
                self.rand_state.set_state(rand_state)
            elif isinstance(rand_state, np.random.RandomState):
                self.rand_state = rand_state
            else:
                raise TypeError("rand_state must be either a tuple "
                                "or a np.random.RandomState object.")
        else:
            self.rand_state = np.random.RandomState(seed)

        self.cache = {}
        try:
            # Should move this somewhere else once we start dealing
            # with bigger datasets
            self.cache["visible_space"] = \
                self.get_visible_space(self.num_visible)
        except (MemoryError, ValueError):
            self.cache["visible_space"] = None
            # a visible space large enough to cause a MemoryError
            # will be too large/slow to deal with later on;
            # make sure log_every is set to 0

        if weights is not None:
            self.weights = weights
        else:
            self.weights = self.rand_state.randn(self.num_hidden,
                                                 self.num_visible)
            # set variance of Gaussian weights to 1/num_visible
            self.weights /= np.sqrt(self.num_visible)

        if visible_bias is not None:
            self.visible_bias = visible_bias
            if visible_bias.shape[0] != self.num_visible:
                raise ValueError("Given visible bias vector's shape does not "
                                 "match num_visible!")
        else:
            self.visible_bias = np.zeros(self.num_visible)

        if hidden_bias is not None:
            self.hidden_bias = hidden_bias
            if hidden_bias.shape[0] != self.num_hidden:
                raise ValueError("Given hidden bias vector's shape does not "
                                 "match num_hidden!")
        else:
            self.hidden_bias = np.zeros(self.num_hidden)

    def __repr__(self):
        return ("RBM(num_visible={}, num_hidden={})"
                .format(self.num_visible, self.num_hidden))

    def save(self, location):
        rng_name, keys, pos, has_gauss, cached_gaussian = \
            self.rand_state.get_state()
        np.savez_compressed(location,
                            weights=self.weights,
                            visible_bias=self.visible_bias,
                            hidden_bias=self.hidden_bias,
                            rng_name=rng_name,
                            keys=keys,
                            pos=pos,
                            has_gauss=has_gauss,
                            cached_gaussian=cached_gaussian)

    @classmethod
    def load(cls, location):
        rbm_params = np.load(location)
        rand_state = (rbm_params["rng_name"], rbm_params["keys"],
                      rbm_params["pos"], rbm_params["has_gauss"],
                      rbm_params["cached_gaussian"])

        return cls(rand_state=rand_state,
                   weights=rbm_params["weights"],
                   visible_bias=rbm_params["visible_bias"],
                   hidden_bias=rbm_params["hidden_bias"])

    def clear_cache(self, keep_visible_space=True):
        for key in list(self.cache.keys()):
            if key != "visible_space":
                del self.cache[key]

        if not keep_visible_space:
            del self.cache["visible_space"]

    @staticmethod
    def sigmoid(x): return expit(x)

    @staticmethod
    def ln1pexp(x):
        """Compute ln(1+exp(x)) quickly by using an approximation
        when x > 30: ln(1 + exp(x)) ~ ln(exp(x)) = x.

        Error -> 0 as x -> +infinity

        Maximum absolute error on order of 1e-14 for
        64 and 128 bit floats; precision loss of 32 bit floats
        overwhelms this systematic error.
        """
        a = x.copy()
        a[a <= 30] = np.log1p(np.exp(a[a <= 30]))
        return a

    def free_energy(self, v):
        if len(v.shape) == 2:
            free_energy_ = np.dot(v, self.visible_bias)
            free_energy_ += np.sum(
                self.ln1pexp(np.dot(self.weights, v.T)
                             + self.hidden_bias.reshape(-1, 1)).T,
                axis=1)
        else:
            free_energy_ = np.dot(self.visible_bias, v)
            free_energy_ += np.sum(self.ln1pexp(np.dot(self.weights, v)
                                                + self.hidden_bias))

        return free_energy_

    def prob_h_given_v(self, v):
        if len(v.shape) == 2:
            p = self.sigmoid(np.dot(self.weights, v.T)
                             + self.hidden_bias.reshape(-1, 1)).T
        else:
            p = self.sigmoid(np.dot(self.weights, v) + self.hidden_bias)
        return p

    def prob_v_given_h(self, h):
        if len(h.shape) == 2:
            p = self.sigmoid(np.dot(self.weights.T, h.T)
                             + self.visible_bias.reshape(-1, 1)).T
        else:
            p = self.sigmoid(np.dot(self.weights.T, h) + self.visible_bias)
        return p

    def sample_h_given_v(self, v, rs=None):
        rs = self.rand_state if rs is None else rs
        h = rs.binomial(1, self.prob_h_given_v(v)).astype(np.float32)
        return h

    def sample_v_given_h(self, h, rs=None):
        rs = self.rand_state if rs is None else rs
        v = rs.binomial(1, self.prob_v_given_h(h)).astype(np.float32)
        return v

    def gibbs_sampling(self, k, v0, rs=None):
        rs = self.rand_state if rs is None else rs
        h0 = self.sample_h_given_v(v0, rs=rs)

        v, h = v0, h0
        for _ in range(k):
            v = self.sample_v_given_h(h, rs=rs)
            h = self.sample_h_given_v(v, rs=rs)
        return v0, self.prob_h_given_v(v0), v, h, self.prob_h_given_v(v)

    def sample(self, k, n_samples=1, seed=None):
        """Draw samples from the RBM

        k -- number of Contrastive Divergence steps to perform
        n_samples -- number of samples to draw (default 1)
        seed -- the random seed (default: uses the RBMs internal PRNG)
        """
        rs = (np.random.RandomState(seed)
              if seed is not None
              else self.rand_state)
        v0 = rs.binomial(1, 0.5, size=(self.num_visible, n_samples))
        _, _, v, _, _ = self.gibbs_sampling(k, v0, rs=rs)
        return v

    def single_batch_gradients(self, batch, k, persistent=False, pbatch=None):
        """Computes gradients from a single batch

        batch -- the batch matrix (shape: (_, self.num_visible))
        k -- number of Contrastive Divergence steps to perform
        persistent -- whether to use Persistent Contrastive Divergence
                      (default False)
        pbatch -- the "batch" of persistent Markov chains (default None)
        """
        if len(batch) == 0:
            return (np.zeros_like(self.weights),
                    np.zeros_like(self.visible_bias),
                    np.zeros_like(self.hidden_bias))

        if not persistent:
            pbatch = batch
            # both positive and negative phases come from training data
            v0, ph0, vk, hk, phk = self.gibbs_sampling(k, pbatch)
        else:
            # Positive phase comes from training data
            v0, ph0, _, _, _ = self.gibbs_sampling(0, batch)
            # Negative phase comes from persistent Markov chain
            _, _, vk, hk, phk = self.gibbs_sampling(k, pbatch)
            pbatch = vk

        # Positive phase of the gradient
        w_grad = np.einsum("ij,ik->jk", ph0, v0)
        v_b_grad = np.einsum("ij->j", v0)
        h_b_grad = np.einsum("ij->j", ph0)

        # Negative phase of the gradient
        w_grad -= np.einsum("ij,ik->jk", phk, vk)
        v_b_grad -= np.einsum("ij->j", vk)
        h_b_grad -= np.einsum("ij->j", phk)

        w_grad /= float(len(batch))
        v_b_grad /= float(len(batch))
        h_b_grad /= float(len(batch))

        # Return negative gradients to match up nicely with the usual
        # parameter update rules, which *subtract* the gradient from
        # the parameters. This is in contrast with the RBM update
        # rules which ADD the gradients (scaled by the learning rate)
        # to the parameters .
        return -w_grad, -v_b_grad, -h_b_grad, pbatch

    def regularize_weight_gradients(self, w_grad, l1_reg, l2_reg):
        """Adds regularization terms to the weight gradient"""
        return (w_grad + (l2_reg * self.weights)
                + (l1_reg * np.sign(self.weights)))

    def train(self, data, target, epochs, batch_size,
              k=1, persistent=False, persist_from=0,
              lr=1e-3, momentum=0.9,
              method='sgd',
              l1_reg=0, l2_reg=0,
              log_every=10, progbar=False,
              **kwargs):
        """Train the RBM

        data -- the training data
        target -- the validation data (wavefunction values)
        epochs -- number of epochs to train for
        batch_size -- size of each mini-batch
        k -- number of Contrastive Divergence steps (default 1)
        persistent -- whether to use PCD as opposed to CD
                      (default False)
        persist_from -- use CD first then start using PCD after
                        this many epochs; ignored if `persistent`
                        is False (default 0)
        lr -- the learning rate; can be a float or a function that
              takes the epoch number and returns a float (default 1e-3)
        momentum -- the momentum parameter; can be a float or a
              function that takes the epoch number and returns
              a float; ignored if method is either "sgd" or "adam"
              (default 0.9)
        method -- the parameter update method; can be any of:
                  "adam", "nesterov", "momentum", or "sgd"
                  (default "sgd")
        l1_reg -- the l1 regularization parameter (default 0)
        l2_reg -- the l2 regularization parameter (default 0)
        log_every -- how often the validation statistics are recorded
                     in epochs (default 10)
        progbar -- whether to display a progress bar; can be a boolean
                   or "notebook" for displaying progress bars in a
                   jupyter notebook (default False)
        **kwargs -- extra keyword arguments passed to the parameter
                    update function; refer to `rbm_grad_updates.py`
                    for more info
        """
        nll_list, overlap_list = [], []
        disable_progbar = (progbar is False)
        prog_bar = tqdm_notebook if progbar == "notebook" else tqdm

        if not callable(lr):
            lr = schedulers.constant(lr)
        if not callable(momentum):
            momentum = schedulers.constant(momentum)

        updater, updater_data = rgu.get_updater(method,
                                                learning_rate=lr,
                                                momentum_param=momentum,
                                                **kwargs)

        pbatch = (self.rand_state.binomial(
                        1, 0.5, size=(batch_size, self.num_visible)
                    ).astype(np.float)
                  if persistent
                  else None)

        for ep in prog_bar(range(epochs+1), desc="Epochs ",
                           total=epochs, disable=disable_progbar):
            self.clear_cache()

            if log_every > 0 and ep % log_every == 0:
                if target is not None:
                    # don't compute overlap if target wavefunction
                    # data isn't provided
                    overlap = self.overlap(target)
                    overlap_list.append(overlap)
                nll = self.negative_log_likelihood(data)
                nll_list.append(nll)
                info_str = ("Epoch = {}".format(
                                str(ep).rjust(
                                        int(np.ceil(np.log10(epochs)+1)))))
                info_str += ("; NLL per training example = {: 12.8f}"
                             .format(nll / len(data)))

                if target is not None:
                    info_str += ("; Overlap = {: 12.8f}".format(overlap))

                tqdm.write(info_str)

            if ep == epochs:  # just wanted to log metrics one last time
                break

            self.rand_state.shuffle(data)
            batches = [data[batch_start:(batch_start + batch_size)]
                       for batch_start in range(0, len(data), batch_size)]

            for batch in prog_bar(batches, desc="Batches",
                                  leave=False, disable=disable_progbar):
                w_grad, v_b_grad, h_b_grad, pbatch = \
                    self.single_batch_gradients(
                        batch, k,
                        pbatch=(pbatch if ep >= persist_from else None),
                        persistent=(persistent and ep >= persist_from))

                grads = {
                    "weights": self.regularize_weight_gradients(w_grad,
                                                                l1_reg,
                                                                l2_reg),
                    "visible_bias": v_b_grad,
                    "hidden_bias": h_b_grad
                }

                updater_data = updater(self, ep, grads, updater_data)

        if target is not None:
            return nll_list, overlap_list
        else:
            return nll_list

    @staticmethod
    def get_visible_space(n: int):
        """Produces a matrix of size 2^n by n containing the entire
        visible space
        """
        space = np.zeros((1 << n, n)).astype(np.byte)
        for i in range(1 << n):
            d = i
            for j in range(n):
                d, r = divmod(d, 2)
                space[i, n - j - 1] = int(r)

        return space

    @staticmethod
    def generate_visible_space(n):
        for i in range(1 << n):
            d = i
            arr = np.zeros((n,))
            for j in range(n):
                d, r = divmod(d, 2)
                arr[n - j - 1] = int(r)
            yield arr

    def apply_to_visible_space(self, fn, direct=False):
        """Applies a function to every element of the visible space

        fn -- function to apply; must be able to take as input a
              single vector of the visible space
        direct -- if True, passes the visible space matrix (if available)
                  directly to fn
        """
        if self.cache["visible_space"] is not None:
            return np.apply_along_axis(fn, 1, self.cache["visible_space"])
        else:
            output = np.zeros((1 << self.num_visible, 1))
            visible_space = self.generate_visible_space(self.num_visible)
            for i, v in enumerate(visible_space):
                output[i] = fn(v)
            return output

    @alias("Z")
    def partition(self):
        """Computes the value of the partition function"""
        partition_ = np.exp(self.log_partition())
        return partition_

    @alias("logZ")
    def log_partition(self):
        """Computes the logarithm of the partition function"""
        if "log_partition" in self.cache:
            return self.cache["log_partition"]

        free_energies = self.apply_to_visible_space(self.free_energy,
                                                    direct=True)
        logZ = special.logsumexp(free_energies)

        self.cache["log_partition"] = logZ
        return logZ

    @alias("p")
    def probability(self, v):
        """Evaluates the probability of the given vector(s) of visible
        units; NOT RECOMMENDED FOR RBMS WITH A LARGE # OF VISIBLE UNITS
        """
        logZ = self.log_partition()
        return np.exp(self.free_energy(v) - logZ)

    def unnormalized_probability(self, v):
        """Evaluates the unnormalized probability of the given
        vector(s) of visible units
        """
        return np.exp(self.free_energy(v))

    @alias("nll")
    def negative_log_likelihood(self, data):
        """Computes the Negative Log Likelihood of the RBM
        over the given data
        """
        if "nll" in self.cache:
            return self.cache["nll"]

        logZ = self.log_partition()
        total_free_energy = np.sum(self.free_energy(data))

        self.cache["nll"] = len(data)*logZ - total_free_energy
        return self.cache["nll"]

    def overlap(self, target):
        if target is None:
            return None
        prob_vect = self.apply_to_visible_space(self.probability, direct=True)
        return np.dot(target, np.sqrt(prob_vect))

    def compute_numerical_kl(self, target):
        KL = stats.entropy(np.power(target, 2),
                           self.apply_to_visible_space(self.probability,
                                                       direct=True))
        return KL

    def test_gradient(self, target, param, alg_grad, eps=1e-8):
        print("Numerical\t Exact\t\t Abs. Diff.\tSame Sign")
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

            same_sign = np.sign(num_grad) == np.sign(alg_grad[i])

            print("{: 10.8f}\t{: 10.8f}\t{: 10.8f}\t{}"
                  .format(num_grad, alg_grad[i],
                          abs(num_grad - alg_grad[i]),
                          same_sign))

    def test_gradients(self, data, target, k, eps=1e-8):
        w_grad, v_b_grad, h_b_grad, _ = self.single_batch_gradients(data, k)

        print("Testing visible bias...")
        self.test_gradient(target, self.visible_bias, v_b_grad, eps=eps)
        print("\nTesting hidden bias...")
        self.test_gradient(target, self.hidden_bias, h_b_grad, eps=eps)
        print("\nTesting weights...")
        self.test_gradient(target, self.weights.flat, w_grad.flat, eps=eps)
