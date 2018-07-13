import torch
from torch.distributions.utils import log_sum_exp

__all__ = [
    "Observable",
    "TFIMChainEnergy",
    "TFIMChainMagnetization"
]


def to_pm1(samples):
    """Converts a tensor of spins from the :math:`\sigma_i = 0, 1` convention
    to the :math:`\sigma_i = -1, +1` convention.

    :param samples: A tensor of spins to convert.
                    Must be using the :math:`\sigma_i = 0, 1` convention.
    :type samples: |Tensor|
    """
    return samples.mul(2.).sub(1.)


def to_01(samples):
    """Converts a tensor of spins from the :math:`\sigma_i = -1, +1` convention
    to the :math:`\sigma_i = 0, 1` convention.

    :param samples: A tensor of spins to convert.
                    Must be using the :math:`\sigma_i = -1, +1` convention.
    :type samples: |Tensor|
    """
    return samples.add(1.).div(2.)


class Observable:
    """Base class for observables."""

    def apply(self, samples, sampler):
        """Computes the value of the observable, row-wise, on a batch of
        samples. Must be implemented by any subclasses.

        :param samples: A batch of samples to calculate the observable on.
        :type samples: |Tensor|
        :param sampler: The sampler that drew the samples.
        """
        pass

    def sample(self, sampler, num_samples, **kwargs):
        """Draws samples of the *observable* using the given sampler.

        :param sampler: The sampler to draw samples from.
        :param num_samples: The number of samples to draw.
        :type num_samples: :class:`int`
        :param \**kwargs: Keyword arguments to pass to the sampler's `sample`
                          function.
        """
        return self.apply(sampler.sample(num_samples, **kwargs), sampler)

    def expected_value(self, sampler, num_samples, batch_size=0, **kwargs):
        """Estimates the expected value of the observable over the distribution
        defined by the sampler.

        In order to avoid running out of memory, the expected value computation
        can be performed in batches.

        :param sampler: The sampler to draw samples from.
        :param num_samples: The number of samples to draw.
        :type num_samples: int
        :param batch_size: The size of the batches; if 0, will use one
                           batch containing all drawn samples.
        :param \**kwargs: Keyword arguments to pass to the sampler's `sample`
                          function.
        :returns: The estimated expected value of the observable.
        :rtype: float
        """
        stats = self.statistics(sampler, num_samples, batch_size, **kwargs)
        return stats["mean"]

    def variance(self, sampler, num_samples, batch_size=0, **kwargs):
        """Estimates the variance (using the sample variance) of the observable
        over the distribution defined by the sampler.

        In order to avoid running out of memory, the variance computation
        can be performed in batches.

        :param sampler: The sampler to draw samples from.
        :param num_samples: The number of samples to draw.
        :type num_samples: int
        :param batch_size: The size of the batches; if 0, will use one
                           batch containing all drawn samples.
        :param \**kwargs: Keyword arguments to pass to the sampler's `sample`
                          function.
        :returns: The estimated variance of the observable.
        :rtype: float
        """
        stats = self.statistics(sampler, num_samples, batch_size, **kwargs)
        return stats["variance"]

    def statistics(self, sampler, num_samples, batch_size, **kwargs):
        """Estimates both the expected value and variance of the observable
        over the distribution defined by the sampler.

        In order to avoid running out of memory, the computations can be
        performed in batches using the pairwise algorithm detailed in
        `Chan et al. (1979)`_.

        :param sampler: The sampler to draw samples from.
        :param num_samples: The number of samples to draw.
        :type num_samples: int
        :param batch_size: The size of the batches; if 0, will only use one
                           batch containing all drawn samples.
        :param \**kwargs: Keyword arguments to pass to the sampler's `sample`
                          function.
        :returns: A dictionary containing both the (estimated) expected value
                  and variance of the observable.
        :rtype: dict(str, float)

        .. _Chan et al. \(1979\):
            http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf
        """
        batch_size = num_samples if batch_size <= 0 else batch_size
        num_reg_batches, rem = divmod(num_samples, batch_size)
        batches = [batch_size] * num_reg_batches
        if rem != 0:
            batches.append(rem)

        def update_statistics(avg_a, var_a, len_a, avg_b, var_b, len_b):
            if len_a == len_b == 0:
                return 0.0, 0.0, 0

            new_len = len_a + len_b
            new_mean = ((avg_a * len_a) + (avg_b * len_b)) / new_len

            delta = avg_b - avg_a
            scaled_var_a = var_a * (len_a - 1)
            scaled_var_b = var_b * (len_b - 1)

            new_var = scaled_var_a + scaled_var_b
            new_var += ((delta ** 2) * len_a * len_b / float(new_len))
            new_var /= float(new_len - 1)

            return new_mean, new_var, new_len

        running_mean = 0.0
        running_var = 0.0
        running_length = 0

        for batch_size in batches:
            samples = self.sample(sampler, batch_size, **kwargs)
            batch_mean = samples.mean().item()
            batch_var = samples.var().item()

            running_mean, running_var, running_length = \
                update_statistics(running_mean, running_var, running_length,
                                  batch_mean, batch_var, batch_size)

        return {
            "mean": running_mean,
            "variance": running_var
        }


class TFIMChainEnergy(Observable):
    """Observable defining the energy of a Transverse Field Ising Model (TFIM)
    spin chain with nearest neighbour interactions, and :math:`J=1`.

    :param h: The strength of the tranverse field
    :type h: float
    :param density: Whether to compute the energy per spin site.
    :type density: bool
    :param periodic_bcs: If `True` use periodic boundary conditions,
                         otherwise use open boundary conditions.
    :type periodic_bcs: bool
    """

    def __init__(self, h, density=True, periodic_bcs=False):
        super(TFIMChainEnergy, self).__init__()
        self.h = h
        self.density = density
        self.periodic_bcs = periodic_bcs

    def __repr__(self):
        return (f"TFIMChainEnergy(h={self.h}, density={self.density},"
                f"periodic_bcs={self.periodic_bcs})")

    @staticmethod
    def _flip_spin(i, s):
        s[:, i] *= -1.0

    def apply(self, samples, sampler):
        """Computes the energy of each sample given a batch of
        samples.

        :param samples: A batch of samples to calculate the observable on.
                        Must be using the :math:`\sigma_i = 0, 1` convention.
        :type samples: |Tensor|
        :param sampler: The sampler that drew the samples. Must implement
                        the function :func:`effective_energy`, giving the
                        log probability of its inputs (up to an additive
                        constant).
        """
        samples = to_pm1(samples)
        log_psis = sampler.effective_energy(to_01(samples)).div(2.)

        shape = log_psis.shape + (samples.shape[-1],)
        log_flipped_psis = torch.zeros(*shape,
                                       dtype=torch.double,
                                       device=sampler.device)

        for i in range(samples.shape[-1]):  # sum over spin sites
            self._flip_spin(i, samples)  # flip the spin at site i
            log_flipped_psis[:, i] = sampler.effective_energy(
                to_01(samples)
            ).div(2.)
            self._flip_spin(i, samples)  # flip it back

        log_flipped_psis = log_sum_exp(
            log_flipped_psis, keepdim=True).squeeze()

        if self.periodic_bcs:
            perm_indices = list(range(sampler.shape[-1]))
            perm_indices = perm_indices[1:] + [0]
            interaction_terms = ((samples * samples[:, perm_indices])
                                 .sum(1))
        else:
            interaction_terms = ((samples[:, :-1] * samples[:, 1:])
                                 .sum(1))      # sum over spin sites

        transverse_field_terms = (log_flipped_psis
                                  .sub(log_psis)
                                  .exp())  # convert to ratio of probabilities

        energy = (transverse_field_terms
                  .mul(self.h)
                  .add(interaction_terms)
                  .mul(-1.))

        if self.density:
            return energy.div(samples.shape[-1])
        else:
            return energy


class TFIMChainMagnetization(Observable):
    """Observable defining the magnetization of a Transverse Field Ising Model
    (TFIM) spin chain.
    """

    def __init__(self):
        super(TFIMChainMagnetization, self).__init__()

    def __repr__(self):
        return "TFIMChainMagnetization()"

    def apply(self, samples, sampler=None):
        """Computes the magnetization of each sample given a batch of samples.

        :param samples: A batch of samples to calculate the observable on.
                        Must be using the :math:`\sigma_i = 0, 1` convention.
        :type samples: |Tensor|
        :param sampler: The sampler that drew the samples. Will be ignored.
        """
        return (to_pm1(samples)
                .mean(1)
                .abs())
