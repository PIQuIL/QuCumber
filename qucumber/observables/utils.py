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


def to_pm1(samples):
    r"""Converts a tensor of spins from the :math:`\sigma_i = 0, 1` convention
    to the :math:`\sigma_i = -1, +1` convention.
    :param samples: A tensor of spins to convert.
                    Must be using the :math:`\sigma_i = 0, 1` convention.
    :type samples: torch.Tensor
    """
    return samples.mul(2.0).sub(1.0)


def to_01(samples):
    r"""Converts a tensor of spins from the :math:`\sigma_i = -1, +1` convention
    to the :math:`\sigma_i = 0, 1` convention.
    :param samples: A tensor of spins to convert.
                    Must be using the :math:`\sigma_i = -1, +1` convention.
    :type samples: torch.Tensor
    """
    return samples.add(1.0).div(2.0)


def _update_statistics(avg_a, var_a, len_a, avg_b, var_b, len_b):
    if len_a == len_b == 0:
        return 0.0, 0.0, 0

    new_len = len_a + len_b
    new_mean = ((avg_a * len_a) + (avg_b * len_b)) / float(new_len)

    delta = avg_b - avg_a
    scaled_var_a = var_a * (len_a - 1)
    scaled_var_b = var_b * (len_b - 1)

    new_var = scaled_var_a + scaled_var_b
    new_var += (delta ** 2) * len_a * len_b / float(new_len)
    new_var /= float(new_len - 1)

    return new_mean, new_var, new_len
