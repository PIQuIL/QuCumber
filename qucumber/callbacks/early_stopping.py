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


class EarlyStopping(Callback):
    r"""Stop training once the model stops improving.
    The specific criterion for stopping is:

    .. math:: \left\vert\frac{M_{t-p} - M_t}{M_{t-p}}\right\vert < \epsilon

    where :math:`M_t` is the metric value at the current evaluation
    (time :math:`t`), :math:`p` is the "patience" parameter, and
    :math:`\epsilon` is the tolerance.

    This Callback is called at the end of each epoch.

    :param period: Frequency with which the callback checks whether training
                   has converged (in epochs).
    :type period: int
    :param tolerance: The maximum relative change required to consider training
                      as having converged.
    :type tolerance: float
    :param patience: How many intervals to wait before claiming the training
                     has converged.
    :type patience: int
    :param metric_callback: An instance of
        :class:`MetricEvaluator<MetricEvaluator>` which computes the metric
        that we want to check for convergence.
    :type metric_callback: :class:`MetricEvaluator<MetricEvaluator>`
    :param metric_name: The name of the metric stored in `metric_callback`.
    :type metric_name: str
    """
    def __init__(self, period, tolerance, patience,
                 metric_callback, metric_name):
        self.period = period
        self.tolerance = tolerance
        self.patience = int(patience)
        self.metric_callback = metric_callback
        self.metric_name = metric_name
        self.last_epoch = None

    def on_epoch_end(self, rbm, epoch):
        if epoch % self.period == 0:
            past_metric_values = self.metric_callback.metric_values

            if len(self.past_metric_values) >= self.patience:
                change_in_metric = (
                    past_metric_values[-self.patience][-1][self.metric_name]
                    - past_metric_values[-1][-1][self.metric_name])

                relative_change = (
                    change_in_metric
                    / past_metric_values[-self.patience][-1][self.metric_name])

                if abs(relative_change) < self.tolerance:
                    rbm.stop_training = True
                    self.last_epoch = epoch
