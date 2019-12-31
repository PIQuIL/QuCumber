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

import qucumber.utils.cplx as cplx


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
        data.append(np.loadtxt(bases_path, dtype=str, ndmin=1))
    return data


def load_data_DM(
    tr_samples_path,
    tr_mtx_real_path=None,
    tr_mtx_imag_path=None,
    tr_bases_path=None,
    bases_path=None,
):
    r"""Load the data required for training.

    :param tr_samples_path: The path to the training data.
    :type tr_samples_path: str
    :param tr_mtx_real_path: The path to the real part of the density matrix
    :type tr_mtx_real_path: str
    :param tr_mtx_imag_path: The path to the imaginary part of the density matrix
    :type tr_mtx_imag_path: str
    :param tr_bases_path: The path to the basis data.
    :type tr_bases_path: str
    :param bases_path: The path to a file containing all possible bases used in
                       the tr_bases_path file.
    :type bases_path: str

    :returns: A list of all input parameters, with the real and imaginary parts
              of the target density matrix (if provided) combined into one complex matrix.
    :rtype: list
    """
    data = []
    data.append(
        torch.tensor(np.loadtxt(tr_samples_path, dtype="float32"), dtype=torch.double)
    )

    if tr_mtx_real_path is not None:
        mtx_real = torch.tensor(
            np.loadtxt(tr_mtx_real_path, dtype="float32"), dtype=torch.double
        )

    if tr_mtx_imag_path is not None:
        mtx_imag = torch.tensor(
            np.loadtxt(tr_mtx_imag_path, dtype="float32"), dtype=torch.double
        )

    if tr_mtx_real_path is not None or tr_mtx_imag_path is not None:
        if tr_mtx_real_path is None or tr_mtx_imag_path is None:
            raise ValueError("Must provide a real and imaginary part of target matrix!")
        else:
            data.append(cplx.make_complex(mtx_real, mtx_imag))

    if tr_bases_path is not None:
        data.append(np.loadtxt(tr_bases_path, dtype=str))

    if bases_path is not None:
        data.append(np.loadtxt(bases_path, dtype=str, ndmin=1))

    return data


def extract_refbasis_samples(train_samples, train_bases):
    r"""Extract the reference basis samples from the data.

    :param train_samples: The training samples.
    :type train_samples: torch.Tensor
    :param train_bases: The bases of the training samples.
    :type train_bases: numpy.ndarray

    :returns: The samples in the data that are only in the reference basis.
    :rtype: torch.Tensor
    """
    torch_ver = int(torch.__version__[:3].replace(".", ""))
    dtype = torch.bool if torch_ver >= 12 else torch.uint8

    idx = (
        torch.tensor((train_bases == "Z").astype(np.uint8))
        .all(dim=1)
        .to(device=train_samples.device, dtype=dtype)
    )
    z_samples = train_samples[idx]
    return z_samples
