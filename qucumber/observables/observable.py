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
import numpy as np

__all__ = [
    "Observable"
]


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

    def statistics(self, rbm, num_samples, num_chains, burn_in, steps,
                   **kwargs):
        """Estimates both the expected value and variance of the observable
        over the distribution defined by the RBM.

        In order to avoid running out of memory, the computations can be
        performed in batches using the pairwise algorithm detailed in
        `Chan et al. (1979)`_.

        :param rbm: The RBM to draw samples from.
        :type rbm: qucumber.rbm.BinomialRBM
        :param num_samples: The number of samples to draw.
        :type num_samples: int
        :param batch_size: The size of the batches; if 0, will only use one
                           batch containing all drawn samples.
        :param \**kwargs: Keyword arguments to pass to the RBM's `sample`
                          function.
        :returns: A dictionary containing both the (estimated) expected value
                  (key: "mean") and variance (key: "variance") of the
                  observable.
        :rtype: dict(str, float)

        .. _Chan et al. \(1979\):
            http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf
        """

        running_sums = torch.tensor([0.0]*num_chains,
                                    dtype=torch.double,
                                    device=rbm.device)
        running_sums_of_squares = torch.tensor([0.0]*num_chains,
                                               dtype=torch.double,
                                               device=rbm.device)
        running_length = 0

        chains = None
        for i in range(int(np.ceil(num_samples / num_chains))):
            num_gibbs_steps = burn_in if i == 0 else steps
            chains = rbm.sample(num_chains,
                                k=num_gibbs_steps,
                                initial_state=chains)

            samples = self.apply(chains, rbm)

            torch.add(running_sums, samples, out=running_sums)
            torch.add(running_sums_of_squares, samples.pow(2),
                      out=running_sums_of_squares)

            running_length += num_chains

        means = running_sums / float(running_length)
        variances = (running_sums_of_squares - running_sums.pow(2)) / float(running_length)

        return {
            "mean": running_mean,
            "variance": running_var
        }
