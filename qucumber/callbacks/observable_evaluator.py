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


from .callback import Callback
from qucumber.observables import System


class ObservableEvaluator(Callback):
    """Evaluate and hold on to the results of the given observable(s).

    This Callback is called at the end of each epoch.

    .. note::
        Since Callbacks are given to :func:`fit<qucumber.nn_states.Wavefunction.fit>`
        as a list, they will be called in a deterministic order. It is
        therefore recommended that instances of
        :class:`ObservableEvaluator<ObservableEvaluator>` be among the first callbacks in
        the list passed to :func:`fit<qucumber.rbm.nn_states.Wavefunction.fit>`,
        as one would often use it in conjunction with other callbacks like
        :class:`EarlyStopping<EarlyStopping>` which may depend on
        :class:`ObservableEvaluator<ObservableEvaluator>` having been called.

    :param period: Frequency with which the callback evaluates the given
                   observables(s).
    :type period: int
    :param observables: A list of Observables. Observable statistics are
                        evaluated by sampling the Wavefunction. Note that
                        observables that have the same name will conflict,
                        and precedence will be given to the right-most
                        observable argument.
    :type observables: list(qucumber.observables.Observable)
    :param verbose: Whether to print metrics to stdout.
    :type verbose: bool
    :param \**sampling_kwargs: Keyword arguments to be passed to `Observable.statistics`.
                               Ex. `num_samples`, `num_chains`, `burn_in`, `steps`.
    """

    def __init__(self, period, observables, verbose=False, **sampling_kwargs):
        self.period = period
        self.past_values = []
        self.system = System(*observables)
        self.last = {}
        self.verbose = verbose
        self.sampling_kwargs = sampling_kwargs

    def __len__(self):
        """Return the number of timesteps that observables have been evaluated for.

        :rtype: int
        """
        return len(self.past_values)

    def __getattr__(self, observable):
        """Return a list of all recorded statistics of the given observable.

        The list will have the form: [(epoch#, observable_value)].

        :param observable: The observable to retrieve.
        :type observable: str

        :returns: The past values of the observable.
        :rtype: list[tuple(int, dict)] or list[tuple(int, float)]
        """
        try:
            return [(epoch, values[observable]) for epoch, values in self.past_values]
        except KeyError:
            raise AttributeError

    def names(self):
        """The names of the tracked observables.

        :rtype: list[str]
        """
        return list(self.system.observables.keys())

    def clear_history(self):
        """Delete all statistics the instance is currently storing."""
        self.past_values = []
        self.last = {}

    def get_value(self, name, index=None):
        """Retrieve the statistics of the desired observable from the given timestep.

        :param name: The name of the observable to retrieve.
        :type name: str
        :param index: The index/timestep from which to retrieve the observable.
                      Negative indices are supported. If None, will just get
                      the most recent value.
        :type index: int or None
        """
        index = index if index is not None else -1
        return self.past_values[index][-1][name]

    def on_epoch_end(self, nn_state, epoch):
        if epoch % self.period == 0:
            obs_vals = self.system.statistics(nn_state, **self.sampling_kwargs)

            self.last = obs_vals.copy()
            self.past_values.append((epoch, obs_vals))

            if self.verbose is True:
                print("Epoch: {}\n".format(epoch), end="", flush=True)
                partially_formatted = {
                    k: "\t".join("{}: {:.6f}".format(s, sv) for s, sv in stats.items())
                    for k, stats in self.last.items()
                }
                print(
                    "\n".join(
                        "  {}:\n    {}".format(k, stats)
                        for k, stats in partially_formatted.items()
                    )
                )
