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

import torch

from .__version__ import __version__


def _warn_on_missing_gpu(gpu):
    if gpu and not torch.cuda.is_available():
        warnings.warn("Could not find GPU: will continue with CPU.", ResourceWarning)


def set_random_seed(seed, cpu=True, gpu=False, quiet=False):
    if gpu and torch.cuda.is_available():
        if not quiet:
            warnings.warn(
                "GPU random seeds are not completely deterministic. "
                "Proceed with caution."
            )
        torch.cuda.manual_seed(seed)

    if cpu:
        torch.manual_seed(seed)
