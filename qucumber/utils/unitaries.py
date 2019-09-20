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


import torch
import numpy as np


def create_dict(**kwargs):
    r"""A function that creates a dictionary of unitary operators.

    By default, the dictionary contains the unitaries which perform a change of
    basis from the computational basis (Pauli-Z) to one of the other Pauli
    bases. The default keys (`X`, `Y`, and `Z`) denote the target basis.

    :param \**kwargs: Keyword arguments of any unitary operators to add to the
                      resulting dictionary. The given operators will overwrite
                      the default matrices if they share the same key.

    :returns: A dictionary of unitaries.
    :rtype: dict
    """
    dictionary = {
        "X": torch.tensor(
            [[[1.0, 1.0], [1.0, -1.0]], [[0.0, 0.0], [0.0, 0.0]]], dtype=torch.double
        )
        / np.sqrt(2),
        "Y": torch.tensor(
            [[[1.0, 0.0], [1.0, 0.0]], [[0.0, -1.0], [0.0, 1.0]]], dtype=torch.double
        )
        / np.sqrt(2),
        "Z": torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]], dtype=torch.double
        ),
    }

    dictionary.update(
        {
            name: (
                matrix.clone().detach()
                if isinstance(matrix, torch.Tensor)
                else torch.tensor(matrix)
            ).to(dtype=torch.double)
            for name, matrix in kwargs.items()
        }
    )

    return dictionary
