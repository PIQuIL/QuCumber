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


import csv

import numpy as np

from .callback import CallbackBase


class MetricEvaluator(CallbackBase):
    r"""Evaluate and hold on to the results of the given metric(s).

    This callback is called at the end of each epoch.

    .. note::
        Since callbacks are given to :func:`fit<qucumber.nn_states.NeuralStateBase.fit>`
        as a list, they will be called in a deterministic order. It is
        therefore recommended that instances of
        :class:`MetricEvaluator<MetricEvaluator>` be among the first callbacks in
        the list passed to :func:`fit<qucumber.nn_states.NeuralStateBase.fit>`,
        as one would often use it in conjunction with other callbacks like
        :class:`EarlyStopping<EarlyStopping>` which may depend on
        :class:`MetricEvaluator<MetricEvaluator>` having been called.

    :param period: Frequency with which the callback evaluates the given
                   metric(s).
    :type period: int
    :param metrics: A dictionary of callables where the keys are the names of
                    the metrics and the callables take the NeuralState being trained
                    as their positional argument, along with some keyword
                    arguments. The metrics are evaluated and put into an internal
                    dictionary structure resembling the structure of `metrics`.
    :type metrics: dict(str, callable)
    :param verbose: Whether to print metrics to stdout.
    :type verbose: bool
    :param log: A filepath to log metric values to in CSV format.
    :type log: str
    :param \**metric_kwargs: Keyword arguments to be passed to `metrics`.
    """

    def __init__(self, period, metrics, verbose=False, log=None, **metric_kwargs):
        self.period = period
        self.metrics = metrics
        self.metric_kwargs = metric_kwargs
        self.past_values = []
        self.last = {}
        self.verbose = verbose
        self.log = log

        self.csv_fields = ["epoch"] + list(self.metrics.keys())
        if self.log is not None:
            with open(self.log, "a") as log_file:
                writer = csv.DictWriter(log_file, fieldnames=self.csv_fields)
                writer.writeheader()

    def __len__(self):
        """Return the number of timesteps that metrics have been evaluated for.

        :rtype: int
        """
        return len(self.past_values)

    def __getattr__(self, metric):
        """Return an array of all recorded values of the given metric.

        :param metric: The metric to retrieve.
        :type metric: str

        :returns: The past values of the metric.
        :rtype: numpy.ndarray
        """
        try:
            return np.array([values[metric] for _, values in self.past_values])
        except KeyError:
            raise AttributeError

    def __getitem__(self, metric):
        """Alias for :func:`__getattr__<qucumber.callbacks.MetricEvaluator.__getattr__>`
        to enable subscripting."""
        return self.__getattr__(metric)

    @property
    def epochs(self):
        """Return a list of all epochs that have been recorded.

        :rtype: numpy.ndarray
        """
        return np.array([epoch for epoch, _ in self.past_values])

    @property
    def names(self):
        """The names of the tracked metrics.

        :rtype: list[str]
        """
        return list(self.metrics.keys())

    def clear_history(self):
        """Delete all metric values the instance is currently storing."""
        self.past_values = []
        self.last = {}

    def get_value(self, name, index=None):
        """Retrieve the value of the desired metric from the given timestep.

        :param name: The name of the metric to retrieve.
        :type name: str
        :param index: The index/timestep from which to retrieve the metric.
                      Negative indices are supported. If None, will just get
                      the most recent value.
        :type index: int or None
        """
        index = index if index is not None else -1
        return self.past_values[index][-1][name]

    def on_epoch_end(self, nn_state, epoch):
        if epoch % self.period == 0:
            metric_vals_for_epoch = {}
            for metric_name, metric_fn in self.metrics.items():
                val = metric_fn(nn_state, **self.metric_kwargs)
                metric_vals_for_epoch[metric_name] = val

            self.last = metric_vals_for_epoch.copy()
            self.past_values.append((epoch, metric_vals_for_epoch))

            if self.verbose is True:
                print(f"Epoch: {epoch}\t", end="", flush=True)
                print("\t".join(f"{k} = {v:.6f}" for k, v in self.last.items()))

            if self.log is not None:
                with open(self.log, "a") as log_file:
                    writer = csv.DictWriter(log_file, fieldnames=self.csv_fields)
                    writer.writerow(dict(epoch=epoch, **self.last))
