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


import numpy as np
import torch


def load_data(tr_samples_path, tr_psi_path=None, tr_bases_path=None, bases_path=None):
    r"""Load the data required for training.

    :param tr_samples_path: The path to the training data.
    :type tr_samples_path: str
    :param tr_psi_path: The path to the target/true wavefunction.
    :type tr_psi_path: str
    :param tr_bases_path: The path to the basis data.
    :type tr_bases_path: str
    :param bases_path: The path to a file containing all possible bases used in
                       the tr_bases_path file.
    :type bases_path: str

    :returns: A list of all input parameters.
    :rtype: list
    """
    data = []
    data.append(
        torch.tensor(np.loadtxt(tr_samples_path, dtype="float32"), dtype=torch.double)
    )

    if tr_psi_path is not None:
        target_psi_data = np.loadtxt(tr_psi_path, dtype="float32")
        target_psi = torch.zeros(2, len(target_psi_data), dtype=torch.double)
        target_psi[0] = torch.tensor(target_psi_data[:, 0], dtype=torch.double)
        target_psi[1] = torch.tensor(target_psi_data[:, 1], dtype=torch.double)
        data.append(target_psi)

    if tr_bases_path is not None:
        data.append(np.loadtxt(tr_bases_path, dtype=str))

    if bases_path is not None:
        bases_data = np.loadtxt(bases_path, dtype=str)
        bases = []
        for i in range(len(bases_data)):
            tmp = ""
            for j in range(len(bases_data[i])):
                if bases_data[i][j] != " ":
                    tmp += bases_data[i][j]
            bases.append(tmp)
        data.append(bases)
    return data


def extract_refbasis_samples(train_samples, train_bases):
    r"""Extract the reference basis samples from the data.

    :param train_samples: The training samples.
    :type train_samples: torch.Tensor
    :param train_bases: The bases of the training samples.
    :type train_bases: np.array(dtype=str)

    :returns: The samples in the data that are only in the reference basis.
    :rtype: torch.Tensor
    """
    idx = (
        torch.tensor((train_bases == "Z").astype(np.uint8), dtype=torch.uint8)
        .all(dim=1)
        .to(train_samples.device)
    )
    z_samples = train_samples[idx]
    return z_samples
