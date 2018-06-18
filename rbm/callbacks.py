import os.path
from pathlib import Path

__all__ = [
    "ModelSaver",
    "Logger",
    "EarlyStopping"
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
            rbm.save(os.path.join(self.path,
                                  "epoch{}".format(epoch)),
                     {k: v(rbm) for k, v in self.metadata.items()})


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
                 metric, higher_is_better=False,
                 **metric_kwargs):
        self.period = period
        self.tolerance = tolerance
        self.patience = int(patience)
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.higher_is_better = higher_is_better
        self.past_metric_values = []

    def __call__(self, rbm, epoch):
        if epoch % self.period == 0:
            self.past_metric_values.append(
                self.metric(rbm, **self.metric_kwargs))

            if len(self.past_metric_values) >= self.patience:
                change_in_metric = (self.past_metric_values[-self.patience]
                                    - self.past_metric_values[-1])
                if self.higher_is_better:
                    # flip sign if we want to maximize the given metric
                    change_in_metric *= -1.0

                if change_in_metric < self.tolerance:
                    rbm.stop_training = True
