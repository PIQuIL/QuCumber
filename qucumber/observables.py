import torch
from torch.distributions.utils import log_sum_exp

__all__ = [
    "Observable",
    "TFIMChainEnergy",
    "TFIMChainMagnetization"
]


def to_pm1(samples):
    return samples.mul(2.).sub(1.)


def to_01(samples):
    return samples.add(1.).div(2.)


class Observable:
    def __init__(self):
        pass

    def apply(self, samples, sampler):
        pass

    def sample(self, sampler, num_samples, **kwargs):
        return self.apply(sampler.sample(num_samples, **kwargs), sampler)

    def expected_value(self, sampler, num_samples, batch_size=0, **kwargs):
        stats = self.statistics(sampler, num_samples, batch_size, **kwargs)
        return stats["mean"]

    def variance(self, sampler, num_samples, batch_size=0, **kwargs):
        stats = self.statistics(sampler, num_samples, batch_size, **kwargs)
        return stats["variance"]

    def statistics(self, sampler, num_samples, batch_size, **kwargs):
        batch_size = num_samples if batch_size <= 0 else batch_size
        num_reg_batches, rem = divmod(num_samples, batch_size)
        batches = [batch_size] * num_reg_batches
        if rem != 0:
            batches.append(rem)

        def update_statistics(avg_a, var_a, len_a, avg_b, var_b, len_b):
            if len_a == len_b == 0:
                return 0.0

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
    def __init__(self, h, density=True, boundary_conditions="open"):
        super(TFIMChainEnergy, self).__init__()
        self.h = h
        self.density = density
        self.boundary_conditions = boundary_conditions

    @staticmethod
    def _flip_spin(i, s):
        s[:, i] *= -1.0

    def apply(self, samples, sampler):
        samples = to_pm1(samples)
        log_psis = sampler.free_energy(to_01(samples)).div(2.)

        shape = log_psis.shape + (samples.shape[-1],)
        log_flipped_psis = torch.zeros(*shape,
                                       dtype=torch.double,
                                       device=sampler.device)

        for i in range(samples.shape[-1]):  # sum over spin sites
            self._flip_spin(i, samples)  # flip the spin at site i
            log_flipped_psis[:, i] = sampler.free_energy(
                to_01(samples)
            ).div(2.)
            self._flip_spin(i, samples)  # flip it back

        log_flipped_psis = log_sum_exp(
            log_flipped_psis, keepdim=True).squeeze()

        if self.boundary_conditions == "periodic":
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
    def __init__(self):
        super(TFIMChainMagnetization, self).__init__()

    def apply(self, samples, sampler=None):
        return (to_pm1(samples)
                .mean(1)
                .abs())
