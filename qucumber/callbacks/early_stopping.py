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


from .callback import CallbackBase


class EarlyStopping(CallbackBase):
    r"""Stop training once the model stops improving.
    The specific criterion for stopping is:

    .. math:: \left\vert\frac{M_{t-p} - M_t}{M_{t-p}}\right\vert < \epsilon

    where :math:`M_t` is the metric value at the current evaluation
    (time :math:`t`), :math:`p` is the "patience" parameter, and
    :math:`\epsilon` is the tolerance.

    This callback is called at the end of each epoch.

    :param period: Frequency with which the callback checks whether training
                   has converged (in epochs).
    :type period: int
    :param tolerance: The maximum relative change required to consider training
                      as having converged.
    :type tolerance: float
    :param patience: How many intervals to wait before claiming the training
                     has converged.
    :type patience: int
    :param evaluator_callback: An instance of
        :class:`MetricEvaluator<MetricEvaluator>` or
        :class:`ObservableEvaluator<ObservableEvaluator>` which computes the metric
        that we want to check for convergence.
    :type evaluator_callback: :class:`MetricEvaluator<MetricEvaluator>` or
                              :class:`ObservableEvaluator<ObservableEvaluator>`
    :param quantity_name: The name of the metric/observable stored in `evaluator_callback`.
    :type quantity_name: str
    """

    def __init__(self, period, tolerance, patience, evaluator_callback, quantity_name):
        self.period = period
        self.tolerance = tolerance
        self.patience = int(patience)
        self.evaluator_callback = evaluator_callback
        self.value_getter = self.evaluator_callback.get_value
        self.quantity_name = quantity_name
        self.last_epoch = None

    def on_epoch_end(self, nn_state, epoch):
        if epoch % self.period == 0:
            if len(self.evaluator_callback) >= self.patience:
                change_in_metric = self.value_getter(
                    self.quantity_name, -self.patience
                ) - self.value_getter(self.quantity_name)

                relative_change = change_in_metric / self.value_getter(
                    self.quantity_name, -self.patience
                )

                if abs(relative_change) < self.tolerance:
                    nn_state.stop_training = True
                    self.last_epoch = epoch
