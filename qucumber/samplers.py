from collections import Counter, OrderedDict
from operator import itemgetter

import torch

__all__ = [
    "Sampler",
    "TractableSampler",
    "DataSampler"
]


class Sampler:
    """Abstract Sampler Class"""

    def probability_ratio(self, a, b):
        r"""Computes the ratio of the probabilities of ``a`` and ``b``:

        .. math::
            \frac{p(a)}{p(b)}

        :param a: The batch of samples whose probabilities will be in the
                  numerator.
        :type a: torch.Tensor
        :param b: The batch of samples whose probabilities will be in the
                  denominator. Must be the same shape as ``a``.
        :type b: torch.Tensor

        :return: The elementwise probability ratios of the inputs.
        :rtype: torch.Tensor
        """
        pass

    def log_probability_ratio(self, a, b):
        r"""Computes the (natural) logarithm of the ratio of the probabilities
        of ``a`` and ``b``:

        .. math::
            \log\left(\frac{p(a)}{p(b)}\right) = \log(p(a)) - \log(p(b))

        :param a: The batch of samples whose probabilities will be in the
                  numerator.
        :type a: torch.Tensor
        :param b: The batch of samples whose probabilities will be in the
                  denominator. Must be the same shape as ``a``.
        :type b: torch.Tensor

        :return: The elementwise logarithms of the probability ratios of the
                 inputs.
        :rtype: torch.Tensor
        """
        pass

    def sample(self, num_samples, **kwargs):
        """Generate samples from the sampler.

        :param num_samples: The number of samples to generate.
        :type num_samples: int
        :param \**kwargs: Keyword arguments for the Sampler.

        :returns: Samples drawn from the Sampler
        :rtype: torch.Tensor
        """
        pass


class TractableSampler(Sampler):
    """Abstract Class for Tractable Samplers (ie. Samplers whose probability
    densities can be computed easily).
    """

    def probability(self, samples):
        r"""Computes the probabilities of the given samples.

        :param a: A batch of samples.
        :type a: torch.Tensor

        :return: The probabilities of the samples.
        :rtype: torch.Tensor
        """
        pass

    def log_probability(self, samples):
        r"""Computes the (natural) logarithm of the probabilities of the
        given samples.

        :param a: A batch of samples.
        :type a: torch.Tensor

        :return: The log-probabilities of the samples.
        :rtype: torch.Tensor
        """
        pass

    def probability_ratio(self, a, b):
        return self.probability(a).div(self.probability(b))

    def log_probability_ratio(self, a, b):
        return self.log_probability(a).sub(self.log_probability(b))


class DataSampler(TractableSampler):
    """Concrete TractableSampler Class which samples from the given dataset

    :param data: The dataset to sample from
    :type data: torch.Tensor
    """
    def __init__(self, data):
        freq = Counter()
        data = torch.tensor(data)

        self.device = data.device
        self.dtype = data.dtype
        self.sample_size = data.size()[-1]

        for row in data:
            freq.update({
                tuple(row.numpy()): 1
            })
        total = float(sum(freq.values()))
        freq = sorted([(k, v) for k, v in freq.items()], key=itemgetter(1))
        self.probs = OrderedDict([(k, v/total) for k, v in freq])

        self.cdf = OrderedDict()
        for i, (ki, pi) in enumerate(self.probs.items()):
            cumulative_prob = 0.0
            for j, (kj, pj) in enumerate(self.probs.items()):
                cumulative_prob += pj
                if i == j:
                    break
            self.cdf[ki] = cumulative_prob

    def sample(self, num_samples, dtype=torch.float):
        unif = torch.rand(num_samples, device=self.device, dtype=dtype)
        samples = torch.zeros(num_samples, self.sample_size,
                              device=self.device, dtype=self.dtype)

        for i in range(num_samples):
            for k, p in self.cdf.items():
                if unif[i] < p:
                    samples[i] = torch.tensor(k, device=self.device,
                                              dtype=self.dtype)
                    break

        return samples

    def probability(self, samples):
        sample_probs = torch.zeros(samples.size()[0],
                                   device=samples.device,
                                   dtype=samples.dtype)

        for i, sample in enumerate(samples):
            key = tuple(sample.numpy())
            sample_probs[i] = self.probs.get(key, 0.0)

        return sample_probs

    def log_probability(self, samples):
        return self.probability(samples).log()
