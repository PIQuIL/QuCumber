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


import os.path
from pathlib import Path

import torch

from .callback import CallbackBase


class ModelSaver(CallbackBase):
    """Callback which allows model parameters (along with some metadata)
    to be saved to disk at regular intervals.

    This callback is called at the end of each epoch. If `save_initial` is
    `True`, will also be called at the start of the training cycle.

    :param period: Frequency of model saving (in epochs).
    :type period: int
    :param folder_path: The directory in which to save the files
    :type folder_path: str
    :param file_name: The name of the output files. Should be a format string
                      with one blank, which will be filled with either the
                      epoch number or the word "initial".
    :type file_name: str
    :param save_initial: Whether to save the initial parameters (and metadata).
    :type save_initial: bool
    :param metadata: The metadata to save to disk with the model parameters
                     Can be either a function or a dictionary. In the case of a
                     function, it must take 2 arguments the RBM being trained,
                     and the current epoch number, and then return a dictionary
                     containing the metadata to be saved.
    :type metadata: callable or dict or None
    :param metadata_only: Whether to save *only* the metadata to disk.
    :type metadata_only: bool
    """

    def __init__(
        self,
        period,
        folder_path,
        file_name,
        save_initial=True,
        metadata=None,
        metadata_only=False,
    ):
        self.folder_path = folder_path
        self.period = period
        self.file_name = file_name
        self.save_initial = save_initial
        self.metadata = metadata
        self.metadata_only = metadata_only

        self.path = Path(folder_path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.path = self.path.resolve()

    def _save(self, nn_state, epoch, save_path):
        if callable(self.metadata):
            metadata = self.metadata(nn_state, epoch)
        elif isinstance(self.metadata, dict):
            metadata = self.metadata
        elif self.metadata is None:
            metadata = {}

        if self.metadata_only:
            torch.save(metadata, save_path)
        else:
            nn_state.save(save_path, metadata)

    def on_train_start(self, nn_state):
        if self.save_initial:
            save_path = os.path.join(self.path, self.file_name.format("initial"))
            self._save(nn_state, 0, save_path)

    def on_epoch_end(self, nn_state, epoch):
        if epoch % self.period == 0:
            save_path = os.path.join(self.path, self.file_name.format(epoch))
            self._save(nn_state, epoch, save_path)
