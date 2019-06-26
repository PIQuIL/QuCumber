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


from collections.abc import MutableSequence

from .callback import CallbackBase


class CallbackList(CallbackBase, MutableSequence):
    def __init__(self, callbacks):
        super().__init__()
        self.callbacks = list(callbacks)

    def __len__(self):
        return len(self.callbacks)

    def __getitem__(self, key):
        return self.callbacks[key]

    def __setitem__(self, key, value):
        if isinstance(value, CallbackBase):
            self.callbacks[key] = value
        else:
            raise TypeError(
                "value must be an instance of qucumber.callbacks.CallbackBase"
            )

    def __delitem__(self, index):
        del self.callbacks[index]

    def __iter__(self):
        return iter(self.callbacks)

    def __add__(self, other):
        return CallbackList(self.callbacks + other.callbacks)

    def insert(self, index, value):
        if isinstance(value, CallbackBase):
            self.callbacks.insert(index, value)
        else:
            raise TypeError(
                "value must be an instance of qucumber.callbacks.CallbackBase"
            )

    def on_train_start(self, rbm):
        for cb in self.callbacks:
            cb.on_train_start(rbm)

    def on_train_end(self, rbm):
        for cb in self.callbacks:
            cb.on_train_end(rbm)

    def on_epoch_start(self, rbm, epoch):
        for cb in self.callbacks:
            cb.on_epoch_start(rbm, epoch)

    def on_epoch_end(self, rbm, epoch):
        for cb in self.callbacks:
            cb.on_epoch_end(rbm, epoch)

    def on_batch_start(self, rbm, epoch, batch):
        for cb in self.callbacks:
            cb.on_batch_start(rbm, epoch, batch)

    def on_batch_end(self, rbm, epoch, batch):
        for cb in self.callbacks:
            cb.on_batch_end(rbm, epoch, batch)
