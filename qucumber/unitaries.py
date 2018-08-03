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

import torch
import numpy as np


def create_dict(name=None, unitary=None):
    dictionary = {
        'X': (1./np.sqrt(2))*torch.tensor(
                [[[1., 1.], [1., -1.]], [[0., 0.], [0., 0.]]],
                dtype=torch.double),
        'Y': (1./np.sqrt(2))*torch.tensor(
                [[[1., 0.], [1., 0.]], [[0., -1.], [0., 1.]]],
                dtype=torch.double),
        'Z': torch.tensor([[[1., 0.], [0., 1.]], [[0., 0.], [0., 0.]]],
                          dtype=torch.double)
    }

    if (name is not None) and (unitary is not None):
        dictionary[name] = unitary

    return dictionary
