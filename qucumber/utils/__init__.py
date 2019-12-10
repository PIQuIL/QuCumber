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


from functools import wraps


class auto_unsqueeze_arg:
    def __init__(self, *arg_indices):
        self.arg_indices = arg_indices

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
