# Copyright 2019 PIQuIL - All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np

from .utils import _update_statistics


class System:
    r"""A class representing a physical system.

    It keeps track of multiple observables which it can evaluate simultaneously.

    :param \*observables: The Observables to evaluate.
    """

    def __init__(self, *observables):
        self.observables = {obs.name: obs for obs in observables}

    def statistics(
        self,
        nn_state,
        num_samples,
        num_chains=0,
        burn_in=1000,
        steps=1,
        initial_state=None,
        overwrite=False,
    ):
        """Estimates the expected value, variance, and the standard error of the
        observables over the distribution defined by `nn_state`.

        :param nn_state: The NeuralState to draw samples from.
        :type nn_state: qucumber.nn_states.NeuralStateBase
        :param num_samples: The number of samples to draw. The actual number of
                            samples drawn may be slightly higher if
                            `num_samples % num_chains != 0`.
        :type num_samples: int
        :param num_chains: The number of Markov chains to run in parallel;
                           if 0 or greater than `num_samples`, will use a
                           number of chains equal to `num_samples`. This is not
                           recommended in the case where a `num_samples` is
                           large, as this may use up all the available memory.
        :type num_chains: int
        :param burn_in: The number of Gibbs Steps to perform before recording
                        any samples.
        :type burn_in: int
        :param steps: The number of Gibbs Steps to take between each sample.
        :type steps: int
        :param initial_state: The initial state of the Markov Chain. If given,
                              `num_chains` will be ignored.
        :type initial_state: torch.Tensor
        :param overwrite: Whether to overwrite the `initial_state` tensor, if
                          provided, with the updated state of the Markov chain.
        :type overwrite: bool

        :returns: A dictionary of dictionaries. At the top level, the keys
                  will be the names of the observables this object is keeping
                  track of. The values will be dictionaries containing the
                  (estimated) expected value (key: "mean"), variance (key:
                  "variance"), and standard error (key: "std_error") of the
                  corresponding observable. Also outputs the total
                  number of drawn samples (key: "num_samples").
        :rtype: dict(str, dict(str, float))
        """
        means = {name: 0.0 for name in self.observables.keys()}
        variances = {name: 0.0 for name in self.observables.keys()}
        total_samples = 0

        if initial_state is not None:
            chains = initial_state if overwrite else initial_state.clone()
            num_chains = len(initial_state)
        else:
            chains = None
            num_chains = (
                min(num_chains, num_samples) if num_chains != 0 else num_samples
            )

        num_time_steps = int(np.ceil(num_samples / num_chains))
        for i in range(num_time_steps):
            num_gibbs_steps = burn_in if i == 0 else steps

            chains = nn_state.sample(
                num_samples=num_chains,
                k=num_gibbs_steps,
                initial_state=chains,
                overwrite=True,
            )

            for obs_name, obs in self.observables.items():
                obs_stats = obs.statistics_from_samples(nn_state, chains)

                means[obs_name], variances[obs_name], _ = _update_statistics(
                    means[obs_name],
                    variances[obs_name],
                    total_samples,
                    obs_stats["mean"],
                    obs_stats["variance"],
                    num_chains,
                )

            total_samples += num_chains

        statistics = {
            obs_name: {
                "mean": means[obs_name],
                "variance": variances[obs_name],
                "std_error": np.sqrt(variances[obs_name] / total_samples),
                "num_samples": total_samples,
            }
            for obs_name in self.observables.keys()
        }

        return statistics

    def statistics_from_samples(self, nn_state, samples):
        """Estimates the expected value, variance, and the standard error of the
        observables using the given samples.

        :param nn_state: The NeuralState that drew the samples.
        :type nn_state: qucumber.nn_states.NeuralStateBase
        :param samples: A batch of sample states to calculate the observable on.
        :type samples: torch.Tensor

        :returns: A dictionary of dictionaries. At the top level, the keys
                  will be the names of the observables this object is keeping
                  track of. The values will be dictionaries containing the
                  (estimated) expected value (key: "mean"), variance (key:
                  "variance"), and standard error (key: "std_error") of the
                  corresponding observable. Also outputs the total number of
                  drawn samples (key: "num_samples").
        :rtype: dict(str, dict(str, float))
        """
        statistics = {
            obs_name: obs.statistics_from_samples(nn_state, samples)
            for obs_name, obs in self.observables.items()
        }
        return statistics
