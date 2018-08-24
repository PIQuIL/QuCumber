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

from qucumber.utils import unitaries


def test_default_unitary_dict():
    unitary_dict = unitaries.create_dict()

    msg = "Default Unitary dictionary has the wrong keys!"
    assert set(["X", "Y", "Z"]) == set(unitary_dict.keys()), msg


def test_adding_unitaries():
    unitary_dict = unitaries.create_dict(
        A=(
            0.5
            * torch.tensor([[[1, 1], [1, 1]], [[1, -1], [-1, 1]]], dtype=torch.double)
        )
    )

    msg = "Unitary dictionary has the wrong keys!"
    assert set(["X", "Y", "Z", "A"]) == set(unitary_dict.keys()), msg
