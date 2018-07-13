import re

import torch
from torch.distributions.utils import log_sum_exp

__all__ = [
    "Observable",
    "TFIMChainEnergy",
    "TFIMChainMagnetization"
]


def format_alias(s):
    alias = s.strip(' _')
    if " " not in alias:
        # cf. https://stackoverflow.com/a/1176023
        alias = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', alias)
        alias = re.sub('([a-z0-9])([A-Z])', r'\1_\2', alias)
    else:
        alias = format_alias(alias.strip(' _')
                                  .replace(' ', '_'))

    return (alias.lower()
                 .strip(' _')
                 .replace('__', '_')
                 .replace(' ', ''))


def to_pm1(samples):
    return samples.mul(2.).sub(1.)


def to_01(samples):
    return samples.add(1.).div(2.)


class Observable:
    def __init__(self, name=None, variance_name=None, **kwargs):
        self.name = name
        self.mean_name = name if name else "mean"

        if variance_name:  # alias the variance function
            # if someone manages to put in a mangled enough string to
            # break this...they brought it on themselves
            variance_alias = format_alias(variance_name)
            setattr(self, variance_alias, self.variance)
            self.variance_name = variance_name
        else:
            self.variance_name = "variance"

    def apply(self, samples, sampler):
        pass

    def sample(self, sampler, num_samples, observable=None, **kwargs):
        if observable is None:
            observable = self.name
        return self.apply(sampler.sample(num_samples, sampler=sampler, observable=observable, **kwargs), sampler)

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
    def __init__(self, h, density=True, name="Energy",
                 variance_name="Heat Capacity", **kwargs):
        super(TFIMChainEnergy, self).__init__(name=name,
                                              variance_name=variance_name)
        self.h = h
        self.density = density

    @staticmethod
    def _flip_spin(i, s):
        s[:, i] *= -1.0

    def apply(self, samples, sampler):
        samples = to_pm1(samples)
        log_psis = sampler.rbm_module.effective_energy(to_01(samples)).div(2.)

        shape = log_psis.shape + (samples.shape[-1],)
        log_flipped_psis = torch.zeros(*shape,
                                       dtype=torch.double,
                                       device=sampler.rbm_module.device)

        for i in range(samples.shape[-1]):  # sum over spin sites
            self._flip_spin(i, samples)  # flip the spin at site i
            log_flipped_psis[:, i] = sampler.rbm_module.effective_energy(
                to_01(samples)
            ).div(2.)
            self._flip_spin(i, samples)  # flip it back

        log_flipped_psis = log_sum_exp(
            log_flipped_psis, keepdim=True).squeeze()

        interaction_terms = ((samples[:, :-1] * samples[:,1:]).sum(1) +
                            samples[:,0] * samples[:,samples.shape[-1]-1])
                            # sum over spin sites
        
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
    def __init__(self, name="Magnetization",
                 variance_name="Susceptibility", **kwargs):
        super(TFIMChainMagnetization, self).__init__(
            name=name, variance_name=variance_name)

    def apply(self, samples, sampler=None):
        return (to_pm1(samples)
                .mean(1)
                .abs())
