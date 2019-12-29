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


import warnings
from functools import wraps


class auto_unsqueeze_args:
    def __init__(self, *arg_indices):
        self.arg_indices = list(arg_indices)

        if len(self.arg_indices) == 0:
            self.arg_indices.append(1)

    def __call__(self, f):
        @wraps(f)
        def wrapped_f(*args, **kwargs):
            args = list(args)
            unsqueeze = False

            for a in self.arg_indices:
                if args[a].dim() < 2:
                    unsqueeze = True
                    args[a] = args[a].unsqueeze(0)

            if unsqueeze:  # remove superfluous axis, if it exists
                return f(*args, **kwargs).squeeze_(0)
            else:
                return f(*args, **kwargs)

        return wrapped_f


# based on code found on StackOverflow:
# https://stackoverflow.com/questions/49802412/how-to-implement-deprecation-in-python-with-argument-alias
class deprecated_kwarg:
    def __init__(self, **aliases):
        self.aliases = aliases

    def rename(self, function, kwargs):
        for alias, true_name in self.aliases.items():
            if alias in kwargs:
                if true_name in kwargs:
                    raise TypeError(
                        f"{function} received both {alias} and {true_name}!"
                    )

                warnings.warn(
                    f"The argument {alias} is deprecated for {function}; use {true_name} instead."
                )

                kwargs[true_name] = kwargs.pop(alias)

        return kwargs

    def __call__(self, f):
        @wraps(f)
        def wrapped_f(*args, **kwargs):
            kwargs = self.rename(f.__name__, kwargs)
            return f(*args, **kwargs)

        return wrapped_f
