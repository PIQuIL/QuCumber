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

#from .callback import Callback
from callback import Callback

class MetricEvaluator(Callback):
    """Evaluate and hold on to the results of the given metric(s).

    This Callback is called at the end of each epoch.

    .. note::
        Since Callbacks are given to :func:`fit<qucumber.rbm.BinomialRBM.fit>`
        as a list, they will be called in a deterministic order. It is
        therefore recommended that instances of
        :class:`MetricEvaluator<MetricEvaluator>` be the first callbacks in
        the list passed to :func:`fit<qucumber.rbm.BinomialRBM.fit>`,
        as one would often use it in conjunction with other callbacks like
        :class:`EarlyStopping<EarlyStopping>` which may depend on
        :class:`MetricEvaluator<MetricEvaluator>` having been called.

    :param period: Frequency with which the callback evaluates the given
                   metric(s).
    :type period: int
    :param metrics: A dictionary of callables where the keys are the names of
                    the metrics and the callables take the RBM being trained
                    as their positional argument, along with some keyword
                    arguments. The metrics are evaluated and put into a
                    dictionary structure resembling the structure of `metrics`.
                    If one of the callables returns a dictionary,
                    the keys of that dictionary will be suitably modified
                    and will be merged with the metric dictionary.
    :type metrics: dict(str, callable)
    :param verbose: Whether to print metrics to stdout
    :type verbose: bool
    :param \**metric_kwargs: Keyword arguments to be passed to `metrics`.
    """
    def __init__(self, period, metrics, verbose=False, **metric_kwargs):
        self.period = period
        self.metrics = metrics
        self.metric_values = []
        self.last = {}
        self.verbose = verbose
        self.metric_kwargs = metric_kwargs

    def __len__(self):
        """Return the number of timesteps that metrics have been evaluated for.
        """
        return len(self.metric_values)

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
        return self.metric_values[index][-1][name]

    def on_epoch_end(self, rbm, epoch):
        if epoch % self.period == 0:
            metric_vals_for_epoch = {}
            for metric_name, metric_fn in self.metrics.items():
                val = metric_fn(rbm, **self.metric_kwargs)
                if isinstance(val, dict):
                    for k, v in val.items():
                        key = metric_name + "_" + k
                        metric_vals_for_epoch[key] = v
                else:
                    metric_vals_for_epoch[metric_name] = val

            self.last = metric_vals_for_epoch.copy()
            self.metric_values.append((epoch, metric_vals_for_epoch))

            if self.verbose is True:
                print("Epoch: {}\t".format(epoch), end='', flush=True)
                print("\t".join("{} = {:.6f}".format(k, v)
                                for k, v in self.last.items()))
