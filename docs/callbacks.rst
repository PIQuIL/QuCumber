.. include:: macros.hrst

Callbacks
=========

Base Callback
-------------
.. autoclass:: qucumber.callbacks.CallbackBase
    :members:

LambdaCallback
--------------
.. autoclass:: qucumber.callbacks.LambdaCallback

ModelSaver
----------
.. autoclass:: qucumber.callbacks.ModelSaver

Logger
------
.. autoclass:: qucumber.callbacks.Logger

EarlyStopping
-------------
.. autoclass:: qucumber.callbacks.EarlyStopping

VarianceBasedEarlyStopping
--------------------------
.. autoclass:: qucumber.callbacks.VarianceBasedEarlyStopping

MetricEvaluator
---------------
.. autoclass:: qucumber.callbacks.MetricEvaluator
    :members: get_value, __len__, __getattr__, __getitem__, epochs, names, clear_history

ObservableEvaluator
-------------------
.. autoclass:: qucumber.callbacks.ObservableEvaluator
    :members: get_value, __len__, __getattr__, __getitem__, epochs, names, clear_history

LivePlotting
------------
.. autoclass:: qucumber.callbacks.LivePlotting

Timer
-----
.. autoclass:: qucumber.callbacks.Timer
