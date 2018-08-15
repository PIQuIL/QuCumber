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

from inspect import signature

from .callback import Callback


class LambdaCallback(Callback):
    """Class for creating simple callbacks.

    This callback is constructed using the passed functions that will be called
    at the appropriate time.

    :param on_train_start: A function to be called at the start of the training
        cycle. Must follow the same signature as
        :func:`Callback.on_train_start<Callback.on_train_start>`.
    :type on_train_start: callable or None

    :param on_train_end: A function to be called at the end of the training
        cycle. Must follow the same signature as
        :func:`Callback.on_train_end<Callback.on_train_end>`.
    :type on_train_end: callable or None

    :param on_epoch_start: A function to be called at the start of every epoch.
        Must follow the same signature as
        :func:`Callback.on_epoch_start<Callback.on_epoch_start>`.
    :type on_epoch_start: callable or None

    :param on_epoch_end: A function to be called at the end of every epoch.
        Must follow the same signature as
        :func:`Callback.on_epoch_end<Callback.on_epoch_end>`.
    :type on_epoch_end: callable or None

    :param on_batch_start: A function to be called at the start of every batch.
        Must follow the same signature as
        :func:`Callback.on_batch_start<Callback.on_batch_start>`.
    :type on_batch_start: callable or None

    :param on_batch_end: A function to be called at the end of every batch.
        Must follow the same signature as
        :func:`Callback.on_batch_end<Callback.on_batch_end>`.
    :type on_batch_end: callable or None
    """

    @staticmethod
    def _validate_function(fn, num_params, name):
        if callable(fn) and len(signature(fn).parameters) == num_params:
            return fn
        elif fn is None:
            return lambda *args: None
        else:
            raise TypeError(
                "{} must be either None ".format(name)
                + "or a function with {} arguments.".format(num_params)
            )

    def __init__(
        self,
        on_train_start=None,
        on_train_end=None,
        on_epoch_start=None,
        on_epoch_end=None,
        on_batch_start=None,
        on_batch_end=None,
    ):
        super(LambdaCallback, self).__init__()
        self.on_train_start = self._validate_function(
            on_train_start, 1, "on_train_start"
        )
        self.on_train_end = self._validate_function(on_train_end, 1, "on_train_end")

        self.on_epoch_start = self._validate_function(
            on_epoch_start, 2, "on_epoch_start"
        )
        self.on_epoch_end = self._validate_function(on_epoch_end, 2, "on_epoch_end")

        self.on_batch_start = self._validate_function(
            on_batch_start, 3, "on_batch_start"
        )
        self.on_batch_end = self._validate_function(on_batch_end, 3, "on_batch_end")
