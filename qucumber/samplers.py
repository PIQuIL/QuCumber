__all__ = [
    "Sampler",
    "TractableSampler"
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
