.. include:: macros.hrst

Callbacks
=========

.. autoclass:: qucumber.callbacks.CallbackBase
    :members:

.. autoclass:: qucumber.callbacks.LambdaCallback

.. autoclass:: qucumber.callbacks.ModelSaver

.. autoclass:: qucumber.callbacks.Logger

.. autoclass:: qucumber.callbacks.EarlyStopping

.. autoclass:: qucumber.callbacks.VarianceBasedEarlyStopping

.. autoclass:: qucumber.callbacks.MetricEvaluator
    :members: get_value, __len__, __getattr__, __getitem__, epochs, names, clear_history

.. autoclass:: qucumber.callbacks.ObservableEvaluator
    :members: get_value, __len__, __getattr__, __getitem__, epochs, names, clear_history

.. autoclass:: qucumber.callbacks.LivePlotting

.. autoclass:: qucumber.callbacks.Timer
