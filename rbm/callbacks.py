import os.path
from pathlib import Path

__all__ = [
    "ModelSaver",
    "Logger",
    "EarlyStopping",
    "ComputeMetrics"
]


class ModelSaver:
    def __init__(self, period, folder_path, rbm_name, **metadata):
        self.folder_path = folder_path
        self.period = period
        self.rbm_name = rbm_name
        self.metadata = metadata

        self.path = Path(folder_path, rbm_name)
        self.path.mkdir(parents=True, exist_ok=True)
        self.path = self.path.resolve()

    def __call__(self, rbm, epoch):
        if epoch % self.period == 0:
            rbm.save(os.path.join(self.path, "epoch{}".format(epoch)),
                     **{k: v(rbm) for k, v in self.metadata.items()})


class Logger:
    def __init__(self, period, logger_fn, msg_gen, **msg_gen_kwargs):
        self.period = period
        self.logger_fn = logger_fn
        self.msg_gen = msg_gen
        self.msg_gen_kwargs = msg_gen_kwargs

    def __call__(self, rbm, epoch):
        if epoch % self.period == 0:
            self.logger_fn(self.msg_gen(rbm, epoch, **self.msg_gen_kwargs))


class EarlyStopping:
    def __init__(self, period, tolerance, patience,
                 metric, **metric_kwargs):
        self.period = period
        self.tolerance = tolerance
        self.patience = int(patience)
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.past_metric_values = []

    def __call__(self, rbm, epoch):
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


class ComputeMetrics:
    def __init__(self, period, metrics, **metric_kwargs):
        self.period = period
        self.metrics = metrics
        self.metric_values = []
        self.metric_kwargs = metric_kwargs

    def __call__(self, rbm, epoch):
        if epoch % self.period == 0:
            metric_values_ep = {k: fn(rbm, **self.metric_kwargs)
                                for k, fn in self.metrics.items()}
            self.metric_values.append((epoch, metric_values_ep))
