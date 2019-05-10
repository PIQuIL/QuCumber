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


from operator import itemgetter

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import ticker

from .callback import CallbackBase


class LivePlotting(CallbackBase):
    """Plots metrics/observables.

    This callback is called at the end of each epoch.

    :param period: Frequency with which the callback updates the plots
                   (in epochs).
    :type period: int
    :param evaluator_callback: An instance of
        :class:`MetricEvaluator<MetricEvaluator>` or
        :class:`ObservableEvaluator<ObservableEvaluator>`
        which computes the metric/observable that we want to plot.
    :type evaluator_callback: :class:`MetricEvaluator<MetricEvaluator>` or
                           :class:`ObservableEvaluator<ObservableEvaluator>`
    :param quantity_name: The name of the metric/observable stored in `evaluator_callback`.
    :type quantity_name: str
    :param error_name: The name of the error stored in `evaluator_callback`.
    :type error_name: str
    """

    def __init__(
        self,
        period,
        evaluator_callback,
        quantity_name,
        error_name=None,
        total_epochs=None,
        smooth=True,
    ):
        self.period = period
        self.evaluator_callback = evaluator_callback
        self.quantity_name = quantity_name
        self.error_name = error_name
        self.last_epoch = 0
        self.total_epochs = total_epochs
        self.smooth = smooth

    def on_train_start(self, nn_state):
        self.fig, self.ax = plt.subplots()

        if self.total_epochs:
            self.ax.set_xlim(0, self.total_epochs)

        self.ax.grid()
        self.ax.xaxis.set_major_locator(ticker.MultipleLocator(min(self.period, 5.0)))
        self.fig.show()
        self.fig.canvas.draw()

    def on_epoch_end(self, nn_state, epoch):
        if epoch % self.period == 0:
            self.last_epoch = epoch

            epochs = np.array(
                list(map(itemgetter(0), self.evaluator_callback.past_values))
            )

            past_values = np.array(
                list(
                    map(
                        itemgetter(self.quantity_name),
                        map(itemgetter(1), self.evaluator_callback.past_values),
                    )
                )
            )

            self.ax.clear()
            p = self.ax.plot(epochs, past_values)

            if self.error_name is not None:
                std_error = np.array(
                    list(
                        map(
                            itemgetter(self.error_name),
                            map(itemgetter(1), self.evaluator_callback.past_values),
                        )
                    )
                )

                lower = past_values - std_error
                upper = past_values + std_error

                self.ax.fill_between(
                    epochs, lower, upper, color=p[0].get_color(), alpha=0.4
                )

            y_avg = np.max(np.abs(past_values))
            y_log_avg = np.log10(y_avg) if y_avg != 0 else -1.0
            y_tick_exp = int(np.sign(y_log_avg) * np.ceil(np.abs(y_log_avg)))
            y_tick_interval = (10 ** y_tick_exp) / 2.0
            self.ax.yaxis.set_major_locator(ticker.MultipleLocator(y_tick_interval))

            self.ax.set_xlabel("Epochs")
            self.ax.set_ylabel(self.quantity_name)
            self.ax.grid()
            self.fig.canvas.draw()

    def on_train_end(self, nn_state):
        self.on_epoch_end(nn_state, self.last_epoch)
