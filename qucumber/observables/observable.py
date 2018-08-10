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

import numpy as np

__all__ = ["Observable"]


class Observable:
    """Base class for observables."""

    def apply(self, samples, rbm):
        """Computes the value of the observable, row-wise, on a batch of
        samples. Must be implemented by any subclasses.

        :param samples: A batch of samples to calculate the observable on.
        :type samples: torch.Tensor
        :param rbm: The RBM that drew the samples.
        :type rbm: qucumber.rbm.BinomialRBM
        """
        pass

    def sample(self, rbm, num_samples, **kwargs):
        """Draws samples of the *observable* using the given RBM.

        :param rbm: The RBM to draw samples from.
        :type rbm: qucumber.rbm.BinomialRBM
        :param num_samples: The number of samples to draw.
        :type num_samples: int
        :param \**kwargs: Keyword arguments to pass to the RBM's `sample`
                          function.
        """
        return self.apply(rbm.sample(num_samples, **kwargs), rbm)

    def expected_value(self, rbm, num_samples, batch_size=0, **kwargs):
        """Estimates the expected value of the observable over the distribution
        defined by the RBM.

        In order to avoid running out of memory, the expected value computation
        can be performed in batches.

        :param rbm: The RBM to draw samples from.
        :type rbm: qucumber.rbm.BinomialRBM
        :param num_samples: The number of samples to draw.
        :type num_samples: int
        :param batch_size: The size of the batches; if 0, will use one
                           batch containing all drawn samples.
        :param \**kwargs: Keyword arguments to pass to the RBM's `sample`
                          function.
        :returns: The estimated expected value of the observable.
        :rtype: float
        """
        stats = self.statistics(rbm, num_samples, batch_size, **kwargs)
        return stats["mean"]

    def variance(self, rbm, num_samples, batch_size=0, **kwargs):
        """Estimates the variance (using the sample variance) of the observable
        over the distribution defined by the RBM.

        In order to avoid running out of memory, the variance computation
        can be performed in batches.

        :param rbm: The RBM to draw samples from.
        :type rbm: qucumber.rbm.BinomialRBM
        :param num_samples: The number of samples to draw.
        :type num_samples: int
        :param batch_size: The size of the batches; if 0, will use one
                           batch containing all drawn samples.
        :param \**kwargs: Keyword arguments to pass to the RBM's `sample`
                          function.
        :returns: The estimated variance of the observable.
        :rtype: float
        """
        stats = self.statistics(rbm, num_samples, batch_size, **kwargs)
        return stats["variance"]

    def std_error(self, rbm, num_samples, batch_size=0, **kwargs):
        stats = self.statistics(rbm, num_samples, batch_size, **kwargs)
        return stats["std_error"]

    def statistics(self, rbm, num_samples, num_chains=0, burn_in=1000, steps=1):
        """Estimates both the expected value and variance of the observable
        over the distribution defined by the RBM.

        :param rbm: The RBM to draw samples from.
        :type rbm: qucumber.rbm.BinomialRBM
        :param num_samples: The number of samples to draw. The actual number of
                            samples drawn may be slightly higher if
                            `num_samples % num_chains != 0`.
        :type num_samples: int
        :param num_chains: The number of Markov chains to run in parallel;
                           if 0, will use a number of chains equal to
                           `num_samples`.
        :type num_chains: int
        :param burn_in: The number of Gibbs Steps to perform before recording
                        any samples.
        :type burn_in: int
        :param steps: The number of Gibbs Steps to take between each sample.
        :type steps: int
        :returns: A dictionary containing both the (estimated) expected value
                  (key: "mean") and variance (key: "variance") of the
                  observable.
        :rtype: dict(str, float)
        """

        running_sum = 0.0
        running_sum_of_squares = 0.0

        chains = None
        num_chains = num_chains if num_chains != 0 else num_samples
        num_time_steps = int(np.ceil(num_samples / num_chains))
        for i in range(num_time_steps):
            num_gibbs_steps = burn_in if i == 0 else steps

            chains = rbm.sample(num_chains, k=num_gibbs_steps, initial_state=chains)

            samples = self.apply(chains, rbm).data

            running_sum += samples.sum().item()
            running_sum_of_squares += samples.pow(2).sum().item()

        N = float(num_time_steps * num_chains)  # total number of samples
        mean = running_sum / N

        variance = running_sum_of_squares - ((running_sum ** 2) / N)
        variance /= N - 1

        std_error = np.sqrt(variance / N)

        return {"mean": mean, "variance": variance, "std_error": std_error}
