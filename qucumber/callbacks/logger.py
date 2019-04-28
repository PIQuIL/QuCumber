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


class Logger(CallbackBase):
    r"""Callback which logs output at regular intervals.

    This callback is called at the end of each epoch.

    :param period: Logging frequency (in epochs).
    :type period: int
    :param logger_fn: The function used for logging. Must take 1 string as
                      an argument. Defaults to the standard `print` function.
    :type logger_fn: callable
    :param msg_gen: A callable which generates the string to be logged.
                    Must take 2 positional arguments: the RBM being trained and
                    the current epoch. It must also be able to take some
                    keyword arguments.
    :type msg_gen: callable
    :param \**kwargs: Keyword arguments which will be passed to `msg_gen`.
    """

    def __init__(self, period, logger_fn=print, msg_gen=None, **msg_gen_kwargs):
        self.period = period
        self.logger_fn = logger_fn
        self.msg_gen = msg_gen if callable(msg_gen) else self._default_msg_gen
        self.msg_gen_kwargs = msg_gen_kwargs

    @staticmethod
    def _default_msg_gen(nn_state, epoch, **kwargs):
        return "Epoch " + str(epoch) + ": " + str(kwargs)

    def on_epoch_end(self, nn_state, epoch):
        if epoch % self.period == 0:
            self.logger_fn(self.msg_gen(nn_state, epoch, **self.msg_gen_kwargs))
