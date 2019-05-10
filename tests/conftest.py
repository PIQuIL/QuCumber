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


import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--nll",
        action="store_true",
        dest="nll",
        default=False,
        help=(
            "Run Negative Log Likelihood gradient tests. "
            "Note that they are non-deterministic and may fail."
        ),
    )


def pytest_collection_modifyitems(config, items):
    # run NLL tests only if the option is given
    if config.getoption("--nll"):
        return
    else:
        skip_nll = pytest.mark.skip(
            reason="doesn't give consistent results; add --nll option to run anyway"
        )
        for item in items:
            if "nll" in item.keywords:
                item.add_marker(skip_nll)
