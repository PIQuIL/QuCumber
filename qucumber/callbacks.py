import os.path
from pathlib import Path

import torch
import numpy as np

__all__ = [
    "ModelSaver",
    "Logger",
    "EarlyStopping",
    "VarianceBasedEarlyStopping",
    "ComputeMetrics"
]


class Callback:
    def on_train_start(self, rbm):
        """Called at the start of the training cycle"""
        pass

    def on_train_end(self, rbm):
        """Called at the end of the training cycle"""
        pass

    def on_epoch_start(self, rbm, epoch):
        """Called at the start of each epoch"""
        pass

    def on_epoch_end(self, rbm, epoch):
        """Called at the end of each epoch"""
        pass

    def on_batch_start(self, rbm, epoch, batch):
        """Called at the start of each batch"""
        pass

    def on_batch_end(self, rbm, epoch, batch):
        """Called at the end of each batch"""
        pass


class ModelSaver(Callback):
    def __init__(self, period, folder_path, file_name,
                 metadata=None, metadata_only=False):

        self.folder_path = folder_path
        self.period = period
        self.file_name = file_name
        self.metadata = metadata
        self.metadata_only = metadata_only

        self.path = Path(folder_path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.path = self.path.resolve()

    def on_epoch_end(self, rbm, epoch):
        if epoch % self.period == 0:
            save_path = os.path.join(self.path, self.file_name.format(epoch))

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


class Logger(Callback):
    def __init__(self, period, logger_fn, msg_gen, **msg_gen_kwargs):
        self.period = period
        self.logger_fn = logger_fn
        self.msg_gen = msg_gen
        self.msg_gen_kwargs = msg_gen_kwargs

    def on_epoch_end(self, rbm, epoch):
        if epoch % self.period == 0:
            self.logger_fn(self.msg_gen(rbm, epoch, **self.msg_gen_kwargs))


class EarlyStopping(Callback):
    def __init__(self, period, tolerance, patience,
                 metric, **metric_kwargs):
        self.period = period
        self.tolerance = tolerance
        self.patience = int(patience)
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.past_metric_values = []
        self.last_epoch = None

    def on_epoch_end(self, rbm, epoch):
        if epoch % self.period == 0:
            self.past_metric_values.append(
                self.metric(rbm, **self.metric_kwargs))

            if len(self.past_metric_values) >= self.patience:
                change_in_metric = (self.past_metric_values[-self.patience]
                                    - self.past_metric_values[-1])
                relative_change = (change_in_metric
                                   / self.past_metric_values[-self.patience])
                if abs(relative_change) < self.tolerance:
                    rbm.stop_training = True
                    self.last_epoch = epoch


class VarianceBasedEarlyStopping(Callback):
    def __init__(self, period, tolerance, patience,
                 metric_callback, metric_name, variance_name):
        self.period = period
        self.tolerance = tolerance
        self.patience = int(patience)
        self.metric_callback = metric_callback
        self.metric_name = metric_name
        self.variance_name = variance_name
        self.last_epoch = None

    def on_epoch_end(self, rbm, epoch):
        if epoch % self.period == 0:
            past_metric_values = self.metric_callback.metric_values

            if len(past_metric_values) >= self.patience:
                change_in_metric = (
                    past_metric_values[-self.patience][-1][self.metric_name]
                    - past_metric_values[-1][-1][self.metric_name])

                std_dev = np.sqrt(
                    past_metric_values[-self.patience][-1][self.variance_name])

                if abs(change_in_metric) < (std_dev * self.tolerance):
                    rbm.stop_training = True
                    self.last_epoch = epoch


class ComputeMetrics(Callback):
    def __init__(self, period, metrics, **metric_kwargs):
        self.period = period
        self.metrics = metrics
        self.metric_values = []
        self.last = {}
        self.metric_kwargs = metric_kwargs

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
