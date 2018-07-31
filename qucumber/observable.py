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

import torch
from torch.distributions.utils import log_sum_exp

__all__ = [
    "Observable"
]


def to_pm1(samples):
    """Converts a tensor of spins from the :math:`\sigma_i = 0, 1` convention
    to the :math:`\sigma_i = -1, +1` convention.

    :param samples: A tensor of spins to convert.
                    Must be using the :math:`\sigma_i = 0, 1` convention.
    :type samples: torch.Tensor
    """
    return samples.mul(2.).sub(1.)


def to_01(samples):
    """Converts a tensor of spins from the :math:`\sigma_i = -1, +1` convention
    to the :math:`\sigma_i = 0, 1` convention.

    :param samples: A tensor of spins to convert.
                    Must be using the :math:`\sigma_i = -1, +1` convention.
    :type samples: torch.Tensor
    """
    return samples.add(1.).div(2.)


class Observable:
    """Base class for observables."""

    def apply(self, samples, sampler):
        """Computes the value of the observable, row-wise, on a batch of
        samples. Must be implemented by any subclasses.

        :param samples: A batch of samples to calculate the observable on.
        :type samples: torch.Tensor
        :param sampler: The sampler that drew the samples.
        :type sampler: qucumber.samplers.Sampler
        """
        pass

    def sample(self, sampler, num_samples, **kwargs):
        """Draws samples of the *observable* using the given sampler.

        :param sampler: The sampler to draw samples from.
        :type sampler: qucumber.samplers.Sampler
        :param num_samples: The number of samples to draw.
        :type num_samples: int
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
        :type sampler: qucumber.samplers.Sampler
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
        :type sampler: qucumber.samplers.Sampler
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
        :type sampler: qucumber.samplers.Sampler
        :param num_samples: The number of samples to draw.
        :type num_samples: int
        :param batch_size: The size of the batches; if 0, will only use one
                           batch containing all drawn samples.
        :param \**kwargs: Keyword arguments to pass to the sampler's `sample`
                          function.
        :returns: A dictionary containing both the (estimated) expected value
                  (key: "mean") and variance (key: "variance") of the
                  observable.
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
