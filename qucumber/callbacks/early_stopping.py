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


from .callback import CallbackBase
from .metric_evaluator import MetricEvaluator
from .observable_evaluator import ObservableEvaluator


class EarlyStopping(CallbackBase):
    r"""Stop training once the model stops improving.

    There are three different stopping criteria available:

    `relative`, which computes the relative change between the two model
    evaluation steps:

    .. math:: \left\vert\frac{M_{t-p} - M_t}{M_{t-p}}\right\vert < \epsilon

    `absolute` computes the absolute change:

    .. math:: \left\vert M_{t-p} - M_t\right\vert < \epsilon

    `variance` computes the absolute change, but scales the change by the
    standard deviation of the quantity of interest, such that the tolerance,
    `\epsilon` can now be interpreted as the "number of standard deviations":

    .. math:: \left\vert\frac{M_{t-p} - M_t}{\sigma_{t-p}}\right\vert < \epsilon

    where :math:`M_t` is the metric value at the current evaluation
    (time :math:`t`), :math:`p` is the "patience" parameter,
    :math:`\sigma_t` is the standard deviation of the metric, and
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
    :param criterion: The stopping criterion to use. Must be one of the following:
                      `relative`, `absolute`, `variance`.
    :type criterion: str
    """

    def __init__(
        self,
        period,
        tolerance,
        patience,
        evaluator_callback,
        quantity_name,
        criterion="relative",
    ):
        self.period = period
        self.tolerance = tolerance
        self.patience = int(patience)
        self.quantity_name = quantity_name

        if isinstance(evaluator_callback, MetricEvaluator):
            self.evaluator_callback = evaluator_callback
            self.value_getter = self.evaluator_callback.get_value

            if criterion == self._convergence_criteria[2]:
                raise TypeError(
                    "Can't use a variance based convergence criterion "
                    "with MetricEvaluator!"
                )
        elif isinstance(evaluator_callback, ObservableEvaluator):
            self.evaluator_callback = evaluator_callback
            self.value_getter = lambda *args: self.evaluator_callback.get_value(*args)[
                "mean"
            ]
            self.variance_getter = lambda *args: self.evaluator_callback.get_value(
                *args
            )["variance"]
        else:
            raise TypeError(
                "evaluator_callback must be an instance of "
                "either MetricEvaluator or ObservableEvaluator!"
            )

        convergence_criteria = {
            "relative": self._relative_change,
            "absolute": self._absolute_change,
            "variance": self._variance_scaled_abs_change,
        }

        try:
            self.criterion = criterion.strip().lower()
        except KeyError:
            raise ValueError(
                "criterion must be one of " + ", ".join(convergence_criteria.keys())
            )

        self.deviation = convergence_criteria[self.criterion]
        self.last_epoch = None

    def _change_in_metric(self):
        return self.value_getter(
            self.quantity_name, -self.patience
        ) - self.value_getter(self.quantity_name)

    def _relative_change(self):
        relative_change = self._change_in_metric() / self.value_getter(
            self.quantity_name, -self.patience
        )
        return abs(relative_change)

    def _absolute_change(self):
        return abs(self._change_in_metric())

    def _variance_scaled_abs_change(self):
        return abs(self._change_in_metric()) / np.sqrt(
            self.variance_getter(self.quantity_name, -self.patience)
        )

    def on_epoch_end(self, nn_state, epoch):
        if epoch % self.period == 0:
            if len(self.evaluator_callback) >= self.patience:
                if self.deviation() < self.tolerance:
                    nn_state.stop_training = True
                    self.last_epoch = epoch
