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

import os.path
from pathlib import Path
from inspect import signature

import torch
import numpy as np


__all__ = [
    "Callback",
    "CallbackList",
    "ModelSaver",
    "Logger",
    "EarlyStopping",
    "VarianceBasedEarlyStopping",
    "MetricEvaluator"
]


class Callback:
    """Base class for callbacks."""

    def on_train_start(self, rbm):
        """Called at the start of the training cycle.

        :param rbm: The RBM being trained.
        :type rbm: BinomialRBM
        """
        pass

    def on_train_end(self, rbm):
        """Called at the end of the training cycle.

        :param rbm: The RBM being trained.
        :type rbm: BinomialRBM
        """
        pass

    def on_epoch_start(self, rbm, epoch):
        """Called at the start of each epoch.

        :param rbm: The RBM being trained.
        :type rbm: BinomialRBM
        :param epoch: The current epoch.
        :type epoch: int
        """
        pass

    def on_epoch_end(self, rbm, epoch):
        """Called at the end of each epoch.

        :param rbm: The RBM being trained.
        :type rbm: BinomialRBM
        :param epoch: The current epoch.
        :type epoch: int
        """
        pass

    def on_batch_start(self, rbm, epoch, batch):
        """Called at the start of each batch.

        :param rbm: The RBM being trained.
        :type rbm: BinomialRBM
        :param epoch: The current epoch.
        :type epoch: int
        :param batch: The current batch index.
        :type batch: int
        """
        pass

    def on_batch_end(self, rbm, epoch, batch):
        """Called at the end of each batch.

        :param rbm: The RBM being trained.
        :type rbm: BinomialRBM
        :param epoch: The current epoch.
        :type epoch: int
        :param batch: The current batch index.
        :type batch: int
        """
        pass


class CallbackList(Callback):
    def __init__(self, callbacks):
        super(CallbackList, self).__init__()
        self.callbacks = list(callbacks)

    def __len__(self):
        return len(self.callbacks)

    def __getitem__(self, key):
        return self.callbacks[key]

    def __setitem__(self, key, value):
        self.callbacks[key] = value

    def __iter__(self):
        return iter(self.callbacks)

    def __add__(self, other):
        return CallbackList(self.callbacks + other.callbacks)

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


class LambdaCallback(Callback):
    """Class for creating simple callbacks.

    This callback is constructed using the passed functions that will be called
    at the appropriate time.

    :param on_train_start: A function called at the start of the training
        cycle. Must follow the same signature as
        :func:`Callback.on_train_start<Callback.on_train_start>`.
    :type on_train_start: callable or None

    :param on_train_end: A function called at the end of the training cycle.
        Must follow the same signature as
        :func:`Callback.on_train_end<Callback.on_train_end>`.
    :type on_train_end: callable or None

    :param on_epoch_start: A function called at the start of every epoch.
        Must follow the same signature as
        :func:`Callback.on_epoch_start<Callback.on_epoch_start>`.
    :type on_epoch_start: callable or None

    :param on_epoch_end: A function called at the end of every epoch.
        Must follow the same signature as
        :func:`Callback.on_epoch_end<Callback.on_epoch_end>`.
    :type on_epoch_end: callable or None

    :param on_batch_start: A function called at the start of every batch.
        Must follow the same signature as
        :func:`Callback.on_batch_start<Callback.on_batch_start>`.
    :type on_batch_start: callable or None

    :param on_batch_end: A function called at the end of every batch.
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
            raise TypeError(f"{name} must be either None "
                            f"or a function with {num_params} arguments.")

    def __init__(self,
                 on_train_start=None, on_train_end=None,
                 on_epoch_start=None, on_epoch_end=None,
                 on_batch_start=None, on_batch_end=None):
        super(LambdaCallback, self).__init__()
        self.on_train_start = self._validate_function(on_train_start, 1,
                                                      "on_train_start")
        self.on_train_end = self._validate_function(on_train_end, 1,
                                                    "on_train_end")

        self.on_epoch_start = self._validate_function(on_epoch_start, 2,
                                                      "on_epoch_start")
        self.on_epoch_end = self._validate_function(on_epoch_end, 2,
                                                    "on_epoch_end")

        self.on_batch_start = self._validate_function(on_batch_start, 3,
                                                      "on_batch_start")
        self.on_batch_end = self._validate_function(on_batch_end, 3,
                                                    "on_batch_end")


class ModelSaver(Callback):
    """Callback which allows model parameters (along with some metadata)
    to be saved to disk at regular intervals.

    This Callback is called at the end of each epoch. If `save_initial` is
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

    def __init__(self, period, folder_path, file_name,
                 save_initial=True,
                 metadata=None, metadata_only=False):
        self.folder_path = folder_path
        self.period = period
        self.file_name = file_name
        self.save_initial = save_initial
        self.metadata = metadata
        self.metadata_only = metadata_only

        self.path = Path(folder_path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.path = self.path.resolve()

    def _save(self, rbm, epoch, save_path):
        if callable(self.metadata):
            metadata = self.metadata(rbm, epoch)
        elif isinstance(self.metadata, dict):
            metadata = self.metadata
        elif self.metadata is None:
            metadata = {}

        if self.metadata_only:
            torch.save(metadata, save_path)
        else:
            rbm.save(save_path, metadata)

    def on_train_start(self, rbm):
        if self.save_initial:
            save_path = os.path.join(self.path,
                                     self.file_name.format("initial"))
            self._save(rbm, 0, save_path)

    def on_epoch_end(self, rbm, epoch):
        if epoch % self.period == 0:
            save_path = os.path.join(self.path, self.file_name.format(epoch))
            self._save(rbm, epoch, save_path)


class Logger(Callback):
    """Callback which logs output at regular intervals.

    This Callback is called at the end of each epoch.

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
    def __init__(self, period, logger_fn=print,
                 msg_gen=None, **msg_gen_kwargs):
        self.period = period
        self.logger_fn = logger_fn
        self.msg_gen = msg_gen if callable(msg_gen) else self._default_msg_gen
        self.msg_gen_kwargs = msg_gen_kwargs

    @staticmethod
    def _default_msg_gen(rbm, epoch, **kwargs):
        return "Epoch " + str(epoch) + ": " + str(kwargs)

    def on_epoch_end(self, rbm, epoch):
        if epoch % self.period == 0:
            self.logger_fn(self.msg_gen(rbm, epoch, **self.msg_gen_kwargs))


class EarlyStopping(Callback):
    r"""Stop training once the model stops improving.
    The specific criterion for stopping is:

    .. math:: \left\vert\frac{M_{t-p} - M_t}{M_{t-p}}\right\vert < \epsilon

    where :math:`M_t` is the metric value at the current evaluation
    (time :math:`t`), :math:`p` is the "patience" parameter, and
    :math:`\epsilon` is the tolerance.

    This Callback is called at the end of each epoch.

    :param period: Frequency with which the callback checks whether training
                   has converged (in epochs).
    :type period: int
    :param tolerance: The maximum relative change required to consider training
                      as having converged.
    :type tolerance: float
    :param patience: How many intervals to wait before claiming the training
                     has converged.
    :type patience: int
    :param metric_callback: An instance of
        :class:`MetricEvaluator<MetricEvaluator>` which computes the metric
        that we want to check for convergence.
    :type metric_callback: :class:`MetricEvaluator<MetricEvaluator>`
    :param metric_name: The name of the metric stored in `metric_callback`.
    :type metric_name: str
    """
    def __init__(self, period, tolerance, patience,
                 metric_callback, metric_name):
        self.period = period
        self.tolerance = tolerance
        self.patience = int(patience)
        self.metric_callback = metric_callback
        self.metric_name = metric_name
        self.last_epoch = None

    def on_epoch_end(self, rbm, epoch):
        if epoch % self.period == 0:
            past_metric_values = self.metric_callback.metric_values

            if len(self.past_metric_values) >= self.patience:
                change_in_metric = (
                    past_metric_values[-self.patience][-1][self.metric_name]
                    - past_metric_values[-1][-1][self.metric_name])

                relative_change = (
                    change_in_metric
                    / past_metric_values[-self.patience][-1][self.metric_name])

                if abs(relative_change) < self.tolerance:
                    rbm.stop_training = True
                    self.last_epoch = epoch


class VarianceBasedEarlyStopping(Callback):
    r"""Stop training once the model stops improving. This is a variation
    on the :class:`EarlyStopping<EarlyStopping>` class which takes the variance
    of the metric into account.
    The specific criterion for stopping is:

    .. math:: \left\vert\frac{M_{t-p} - M_t}{\sigma_{t-p}}\right\vert < \kappa

    where :math:`M_t` is the metric value at the current evaluation
    (time :math:`t`), :math:`p` is the "patience" parameter,
    :math:`\sigma_t` is the variance of the metric, and
    :math:`\kappa` is the tolerance.

    This Callback is called at the end of each epoch.

    :param period: Frequency with which the callback checks whether training
                   has converged (in epochs).
    :type period: int
    :param tolerance: The maximum (standardized) change required to consider
                      training as having converged.
    :type tolerance: float
    :param patience: How many intervals to wait before claiming the training
                     has converged.
    :type patience: int
    :param metric_callback: An instance of
        :class:`MetricEvaluator<MetricEvaluator>` which computes the metric
        that we want to check for convergence.
    :type metric_callback: :class:`MetricEvaluator<MetricEvaluator>`
    :param metric_name: The name of the metric stored in `metric_callback`.
    :type metric_name: str
    :param variance_name: The name of the variance stored in `metric_callback`.
    :type variance_name: str
    """
    def __init__(self, period, tolerance, patience,
                 metric_callback, metric_name, variance_name):
        self.period = period
        self.tolerance = tolerance
        self.patience = int(patience)
        self.metric_callback = metric_callback
        self.value_getter = self.metric_callback.get_value
        self.metric_name = metric_name
        self.variance_name = variance_name
        self.last_epoch = None

    def on_epoch_end(self, rbm, epoch):
        if epoch % self.period == 0:
            if len(self.metric_callback) >= self.patience:
                change_in_metric = (
                    self.value_getter(self.metric_name, -self.patience)
                    - self.value_getter(self.metric_name))

                std_dev = np.sqrt(
                    self.value_getter(self.variance_name, -self.patience))

                if abs(change_in_metric) < (std_dev * self.tolerance):
                    rbm.stop_training = True
                    self.last_epoch = epoch


class MetricEvaluator(Callback):
    """Evaluate and hold on to the results of the given metric(s).

    This Callback is called at the end of each epoch.

    .. note::
        Since Callbacks are given to :func:`fit<qucumber.rbm.BinomialRBM.fit>`
        as a list, they will be called in a deterministic order. It is
        therefore recommended that instances of
        :class:`MetricEvaluator<MetricEvaluator>` be the first callbacks in
        the list passed to :func:`fit<qucumber.rbm.BinomialRBM.fit>`,
        as one would often use it in conjunction with other callbacks like
        :class:`EarlyStopping<EarlyStopping>` which may depend on
        :class:`MetricEvaluator<MetricEvaluator>` having been called.

    :param period: Frequency with which the callback evaluates the given
                   metric(s).
    :type period: int
    :param metrics: A dictionary of callables where the keys are the names of
                    the metrics and the callables take the RBM being trained
                    as their positional argument, along with some keyword
                    arguments. The metrics are evaluated and put into a
                    dictionary structure resembling the structure of `metrics`.
                    If one of the callables returns a dictionary,
                    the keys of that dictionary will be suitably modified
                    and will be merged with the metric dictionary.
    :type metrics: dict(str, callable)
    :param \**metric_kwargs: Keyword arguments to be passed to `metrics`.
    """
    def __init__(self, period, metrics, **metric_kwargs):
        self.period = period
        self.metrics = metrics
        self.metric_values = []
        self.last = {}
        self.metric_kwargs = metric_kwargs

    def __len__(self):
        """Return the number of timesteps that metrics have been evaluated for.
        """
        return len(self.metric_values)

    def get_value(self, name, index=None):
        """Retrieve the value of the desired metric from the given timestep.

        :param name: The name of the metric to retrieve.
        :type name: str
        :param index: The index/timestep from which to retrieve the metric.
                      Negative indices are supported. If None, will just get
                      the most recent value.
        :type index: int or None
        """
        index = index if index is not None else -1
        return self.metric_values[index][-1][name]

    def on_epoch_end(self, rbm, epoch):
        if epoch % self.period == 0:
            metric_vals_for_epoch = {}
            for metric_name, metric_fn in self.metrics.items():
                val = metric_fn(rbm, **self.metric_kwargs)
                if isinstance(val, dict):
                    for k, v in val.items():
                        key = metric_name + "_" + k
                        metric_vals_for_epoch[key] = v
                else:
                    metric_vals_for_epoch[metric_name] = val
            self.last = metric_vals_for_epoch.copy()
            self.metric_values.append((epoch, metric_vals_for_epoch))
            print('Epoch = %d\t' % epoch,end='',flush=True)
            for metric in self.metrics.keys():
                print(metric + " = %.6f\t" % self.last[metric].item(),end='',flush=True)
            print()



