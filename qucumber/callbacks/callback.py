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


class CallbackBase:
    """Base class for callbacks."""

    def on_train_start(self, nn_state):
        """Called at the start of the training cycle.

        :param nn_state: The WaveFunction being trained.
        :type nn_state: qucumber.nn_states.WaveFunctionBase
        """
        pass

    def on_train_end(self, nn_state):
        """Called at the end of the training cycle.

        :param nn_state: The WaveFunction being trained.
        :type nn_state: qucumber.nn_states.WaveFunctionBase
        """
        pass

    def on_epoch_start(self, nn_state, epoch):
        """Called at the start of each epoch.

        :param nn_state: The WaveFunction being trained.
        :type nn_state: qucumber.nn_states.WaveFunctionBase
        :param epoch: The current epoch.
        :type epoch: int
        """
        pass

    def on_epoch_end(self, nn_state, epoch):
        """Called at the end of each epoch.

        :param nn_state: The WaveFunction being trained.
        :type nn_state: qucumber.nn_states.WaveFunctionBase
        :param epoch: The current epoch.
        :type epoch: int
        """
        pass

    def on_batch_start(self, nn_state, epoch, batch):
        """Called at the start of each batch.

        :param nn_state: The WaveFunction being trained.
        :type nn_state: qucumber.nn_states.WaveFunctionBase
        :param epoch: The current epoch.
        :type epoch: int
        :param batch: The current batch index.
        :type batch: int
        """
        pass

    def on_batch_end(self, nn_state, epoch, batch):
        """Called at the end of each batch.

        :param nn_state: The WaveFunction being trained.
        :type nn_state: qucumber.nn_states.WaveFunctionBase
        :param epoch: The current epoch.
        :type epoch: int
        :param batch: The current batch index.
        :type batch: int
        """
        pass
