# Copyright 2018 PIQuIL - All Rights Reserved

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


def to_pm1(samples):
    """Converts a tensor of spins from the :math:`\sigma_i = 0, 1` convention
    to the :math:`\sigma_i = -1, +1` convention.
    :param samples: A tensor of spins to convert.
                    Must be using the :math:`\sigma_i = 0, 1` convention.
    :type samples: torch.Tensor
    """
    return samples.mul(2.).sub(1.)


def to_01(samples):
    """Converts a tensor of spins from the :math:`\sigma_i = -1, +1` convention
    to the :math:`\sigma_i = 0, 1` convention.
    :param samples: A tensor of spins to convert.
                    Must be using the :math:`\sigma_i = -1, +1` convention.
    :type samples: torch.Tensor
    """
    return samples.add(1.).div(2.)
