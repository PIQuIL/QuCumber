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

from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

from .callback import Callback


class LivePlotting(Callback):
    def __init__(self, period, metric_evaluator, metric_name,
                 error_name=None, total_epochs=None, smooth=True):
        self.period = period
        self.metric_evaluator = metric_evaluator
        self.metric_name = metric_name
        self.error_name = error_name
        self.last_epoch = 0
        self.total_epochs = total_epochs
        self.smooth = smooth

    def on_train_start(self, rbm):
        self.fig, self.ax = plt.subplots()

        if self.total_epochs:
            self.ax.set_xlim(0, self.total_epochs)

        self.ax.grid()
        self.ax.xaxis.set_major_locator(
            ticker.MultipleLocator(min(self.period, 5.))
        )
        self.fig.show()
        self.fig.canvas.draw()

    def on_epoch_end(self, rbm, epoch):
        if epoch % self.period == 0:
            self.last_epoch = epoch

            epochs = np.array(list(
                map(itemgetter(0), self.metric_evaluator.metric_values)
            ))

            metric_values = np.array(list(
                map(itemgetter(self.metric_name),
                    map(itemgetter(1), self.metric_evaluator.metric_values))
            ))

            self.ax.clear()
            p = self.ax.plot(epochs, metric_values)

            if self.error_name is not None:
                std_error = np.array(list(
                    map(itemgetter(self.error_name),
                        map(itemgetter(1),
                            self.metric_evaluator.metric_values))
                ))

                lower = metric_values - std_error
                upper = metric_values + std_error

                self.ax.fill_between(epochs, lower, upper,
                                     color=p[0].get_color(),
                                     alpha=0.4)

            y_avg = np.max(np.abs(metric_values))
            y_log_avg = np.log10(y_avg) if y_avg != 0 else -1.
            y_tick_exp = int(np.sign(y_log_avg) * np.ceil(np.abs(y_log_avg)))
            y_tick_interval = (10 ** y_tick_exp) / 2.
            self.ax.yaxis.set_major_locator(
                ticker.MultipleLocator(y_tick_interval)
            )

            self.ax.set_xlabel("Epochs")
            self.ax.set_ylabel(self.metric_name)
            self.ax.grid()
            self.fig.canvas.draw()

    def on_train_end(self, rbm):
        self.on_epoch_end(rbm, self.last_epoch)
