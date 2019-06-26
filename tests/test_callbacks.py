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

from qucumber.callbacks import LambdaCallback


callback_stages = (
    "on_train_start",
    "on_train_end",
    "on_epoch_start",
    "on_epoch_end",
    "on_batch_start",
    "on_batch_end",
)


@pytest.mark.parametrize("stage", callback_stages)
def test_lambda_callback_value_error_num_args(stage):
    msg = f"LambdaCallback should fail if {stage} gets wrong # of arguments."

    with pytest.raises(ValueError):
        kwargs = {stage: lambda nn_state, epoch, batch, extra: "foobar"}
        LambdaCallback(**kwargs)
        pytest.fail(msg)


@pytest.mark.parametrize("stage", callback_stages)
def test_lambda_callback_type_error(stage):
    msg = f"LambdaCallback should fail if {stage} isn't a function or None."

    with pytest.raises(TypeError):
        LambdaCallback(**{stage: "foobar"})
        pytest.fail(msg)
