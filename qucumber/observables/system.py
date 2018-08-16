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


class System:
    """A class representing a physical system.

    It keeps track of multiple observables which it can evaluate simultaneously.

    :param \*observables:
    """

    def __init__(self, *observables):
        self.observables = {obs.name: obs for obs in observables}

    def measure(self, nn_state, num_samples, num_chains=0, burn_in=1000, steps=1):
        """Estimates the expected value, variance, and the standard error of the
        observables over the distribution defined by `nn_state`.

        :param nn_state: The Wavefunction to draw samples from.
        :type nn_state: qucumber.nn_states.Wavefunction
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
        :returns: A dictionary containing the (estimated) expected value
                  (key: "mean"), variance (key: "variance"), and standard error
                  (key: "std_error") of the observable.
        :rtype: dict(str, float)
        """
        running_sums = {name: 0.0 for name in self.observables.keys()}
        running_sums_of_squares = {name: 0.0 for name in self.observables.keys()}

        chains = None
        num_chains = num_chains if num_chains != 0 else num_samples
        num_time_steps = int(np.ceil(num_samples / num_chains))
        for i in range(num_time_steps):
            num_gibbs_steps = burn_in if i == 0 else steps

            chains = nn_state.sample(
                num_chains, k=num_gibbs_steps, initial_state=chains, overwrite=True
            )

            for obs_name, obs in self.observables.items():
                obs_samples = obs.apply(nn_state, chains).data

                running_sums[obs_name] += obs_samples.sum().item()
                running_sums_of_squares[obs_name] += obs_samples.pow(2).sum().item()

        N = float(num_time_steps * num_chains)  # total number of samples

        statistics = {}

        for obs_name in self.observables.keys():
            mean = running_sums[obs_name] / N

            variance = running_sums_of_squares[obs_name] - (
                (running_sums[obs_name] ** 2) / N
            )
            variance /= N - 1

            std_error = np.sqrt(variance / N)
            statistics[obs_name] = {
                "mean": mean,
                "variance": variance,
                "std_error": std_error,
            }

        return statistics

    def measure_samples(self, nn_state, samples):
        """Estimates the expected value, variance, and the standard error of the
        observables using the given samples.

        :param nn_state: The Wavefunction that drew the samples.
        :type nn_state: qucumber.nn_states.Wavefunction
        :param samples: A batch of sample states to calculate the observable on.
        :type samples: torch.Tensor
        """
        statistics = {}

        for obs_name, obs in self.observables.items():
            obs_samples = obs.apply(nn_state, samples)

            mean = obs_samples.mean().item()
            variance = obs_samples.var().item()
            std_error = np.sqrt(variance / len(obs_samples))

            statistics[obs_name] = {
                "mean": mean,
                "variance": variance,
                "std_error": std_error,
            }

        return statistics
