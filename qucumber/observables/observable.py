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

import abc
import numpy as np


class Observable(abc.ABC):
    """Base class for observables."""

    _name = None
    _symbol = None

    @property
    def name(self):
        """The name of the Observable."""
        return self._name

    @name.setter
    def name(self, new_name):
        self._name = new_name

    @property
    def symbol(self):
        """The algebraic symbol representing the Observable."""
        return self._symbol

    @symbol.setter
    def symbol(self, new_symbol):
        self._symbol = new_symbol

    def __str__(self):
        return self.symbol

    def __repr__(self):
        return self.name

    def __add__(self, other):
        return SumObservable(self, other)

    def __mul__(self, other):
        return ProdObservable(self, other)

    def __radd__(self, other):
        return SumObservable(self, other, right=True)

    def __rmul__(self, other):
        return ProdObservable(self, other)

    @abc.abstractmethod
    def apply(self, nn_state, samples):
        """Computes the value of the observable, row-wise, on a batch of
        samples. Must be implemented by any subclasses.

        :param nn_state: The Wavefunction that drew the samples.
        :type nn_state: qucumber.nn_states.Wavefunction
        :param samples: A batch of sample states to calculate the observable on.
        :type samples: torch.Tensor
        """
        raise NotImplementedError

    def sample(self, nn_state, k, num_samples=1, initial_state=None, overwrite=False):
        """Draws samples of the *observable* using the given Wavefunction.

        :param nn_state: The Wavefunction to draw samples from.
        :type nn_state: qucumber.nn_states.Wavefunction
        :param k: The number of Gibbs Steps to perform before drawing a sample.
        :type k: int
        :param num_samples: The number of samples to draw.
        :type num_samples: int
        :param initial_state: The initial state of the Markov Chain. If given,
                              `num_samples` will be ignored.
        :type initial_state: torch.Tensor
        :param overwrite: Whether to overwrite the initial_state tensor, if it is provided.
        :type overwrite: bool
        """
        return self.apply(
            nn_state.sample(
                k=k,
                num_samples=num_samples,
                initial_state=initial_state,
                overwrite=overwrite,
            ),
            nn_state,
        )

    def statistics(self, nn_state, num_samples, num_chains=0, burn_in=1000, steps=1):
        """Estimates the expected value, variance, and the standard error of the
        observable over the distribution defined by the Wavefunction.

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
        running_sum = 0.0
        running_sum_of_squares = 0.0

        chains = None
        num_chains = num_chains if num_chains != 0 else num_samples
        num_time_steps = int(np.ceil(num_samples / num_chains))
        for i in range(num_time_steps):
            num_gibbs_steps = burn_in if i == 0 else steps

            chains = nn_state.sample(
                num_chains, k=num_gibbs_steps, initial_state=chains, overwrite=True
            )

            samples = self.apply(nn_state, chains).data

            running_sum += samples.sum().item()
            running_sum_of_squares += samples.pow(2).sum().item()

        N = float(num_time_steps * num_chains)  # total number of samples
        mean = running_sum / N

        variance = running_sum_of_squares - ((running_sum ** 2) / N)
        variance /= N - 1

        std_error = np.sqrt(variance / N)

        return {"mean": mean, "variance": variance, "std_error": std_error}

    def statistics_from_samples(self, nn_state, samples):
        """Estimates the expected value, variance, and the standard error of the
        observable using the given samples.

        :param nn_state: The Wavefunction that drew the samples.
        :type nn_state: qucumber.nn_states.Wavefunction
        :param samples: A batch of sample states to calculate the observable on.
        :type samples: torch.Tensor
        """
        obs_samples = self.apply(nn_state, samples)

        mean = obs_samples.mean().item()
        variance = obs_samples.var().item()
        std_error = np.sqrt(variance / len(obs_samples))

        return {"mean": mean, "variance": variance, "std_error": std_error}


class SumObservable(Observable):
    def __init__(self, o1, o2, right=False, name=None, symbol=None):
        if not isinstance(o1, (float, int, Observable)):
            raise TypeError("o1 does not have the right type!")
        if not isinstance(o2, (float, int, Observable)):
            raise TypeError("o2 does not have the right type!")

        self.left = o1 if not right else o2
        self.right = o2 if not right else o1  # swap order if right == True

        if symbol is None:
            self.symbol = "(" + str(self.left) + " + " + str(self.right) + ")"
        else:
            self.symbol = symbol
        if name is None:
            self.name = "(" + repr(self.left) + " + " + repr(self.right) + ")"
        else:
            self.name = name

    def apply(self, samples, rbm):
        result = 0.
        if isinstance(self.left, (float, int)):
            result += self.left
        if isinstance(self.right, (float, int)):
            result += self.right

        if isinstance(self.left, Observable):
            result = result + self.left.apply(samples, rbm)
        if isinstance(self.right, Observable):
            result = result + self.right.apply(samples, rbm)

        return result


class ProdObservable(Observable):
    def __init__(self, o1, o2, name=None, symbol=None):
        if not isinstance(o1, (float, int, Observable)):
            raise TypeError("o1 does not have the right type!")
        if not isinstance(o2, (float, int, Observable)):
            raise TypeError("o2 does not have the right type!")

        # assign scalar value to self.left and the observable to self.right
        if isinstance(o1, (float, int)) and isinstance(o2, Observable):
            self.left = o1
            self.right = o2
        elif isinstance(o2, (float, int)) and isinstance(o1, Observable):
            self.left = o2
            self.right = o1
        else:
            raise ValueError("Exactly one of o1, o2 must be an Observable!")

        if symbol is None:
            self.symbol = "(" + str(self.left) + " * " + str(self.right) + ")"
        else:
            self.symbol = symbol
        if name is None:
            self.name = "(" + repr(self.left) + " * " + repr(self.right) + ")"
        else:
            self.name = name

    def apply(self, samples, rbm):
        return self.left * self.right.apply(samples, rbm)
