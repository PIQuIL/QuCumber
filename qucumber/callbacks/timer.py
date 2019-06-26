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


import time

from .callback import CallbackBase


class Timer(CallbackBase):
    """Callback which records the training time.

    This callback is always called at the start and end of training. It will
    run at the end of an epoch or batch if the given model's `stop_training`
    property is set to True.

    :param verbose: Whether to print the elapsed time at the end of training.
    :type verbose: bool
    """

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.already_notified = False

    def on_train_start(self, nn_state):
        self.start_time = time.time()

    def on_batch_end(self, nn_state, epoch, batch):
        if nn_state.stop_training:
            if self.verbose and not self.already_notified:
                print(f"Training terminated at epoch: {epoch}, batch: {batch}")
                self.already_notified = True

    def on_epoch_end(self, nn_state, epoch):
        if nn_state.stop_training:
            if self.verbose and not self.already_notified:
                print(f"Training terminated at epoch: {epoch}")
                self.already_notified = True

    def on_train_end(self, nn_state):
        self.calculate_elapsed_time()

    def calculate_elapsed_time(self):
        self.end_time = time.time()
        self.training_time = self.end_time - self.start_time
        if self.verbose:
            print(
                "Total time elapsed during training: {:6.3f} s".format(
                    self.training_time
                )
            )
