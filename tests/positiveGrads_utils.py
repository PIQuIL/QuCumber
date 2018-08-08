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

import pickle

import torch

import numpy as np


def generate_visible_space(num_visible, device="cpu"):
    """Generates all possible visible states.

    :returns: A tensor of all possible spin configurations.
    :rtype: torch.Tensor
    """
    space = torch.zeros((2**num_visible, num_visible),
                        device=device, dtype=torch.double)
    for i in range(1 << num_visible):
        d = i
        for j in range(num_visible):
            d, r = divmod(d, 2)
            space[i, num_visible - j - 1] = int(r)

    return space


def partition(nn_state, visible_space):
    """The natural logarithm of the partition function of the RBM.

    :param visible_space: A rank 2 tensor of the entire visible space.
    :type visible_space: torch.Tensor

    :returns: The natural log of the partition function.
    :rtype: torch.Tensor
    """
    return torch.tensor(
        nn_state.rbm_am.compute_partition_function(visible_space),
        dtype=torch.double, device=nn_state.device
    )


def probability(nn_state, v, Z):
    """Evaluates the probability of the given vector(s) of visible
    units; NOT RECOMMENDED FOR RBMS WITH A LARGE # OF VISIBLE UNITS

    :param v: The visible states.
    :type v: torch.Tensor
    :param Z: The partition function.
    :type Z: float

    :returns: The probability of the given vector(s) of visible units.
    :rtype: torch.Tensor
    """
    return nn_state.psi(v)[0]**2 / Z


def compute_numerical_kl(nn_state, target_psi, vis, Z):
    KL = 0.0
    for i in range(len(vis)):
        KL += ((target_psi[i, 0])**2)*((target_psi[i, 0])**2).log()
        KL -= (((target_psi[i, 0])**2)
               * (probability(nn_state, vis[i], Z)).log().item())
    return KL


def compute_numerical_NLL(nn_state, data, Z):
    NLL = 0
    batch_size = len(data)

    for i in range(batch_size):
        NLL -= (probability(nn_state, data[i], Z).log().item()
                / float(batch_size))

    return NLL


def algorithmic_gradKL(nn_state, target_psi, vis):
    Z = partition(nn_state, vis)
    grad_KL = torch.zeros(nn_state.rbm_am.num_pars,
                          dtype=torch.double, device=nn_state.device)
    for i in range(len(vis)):
        grad_KL += ((target_psi[i, 0])**2)*nn_state.gradient(vis[i])
        grad_KL -= probability(nn_state, vis[i], Z)*nn_state.gradient(vis[i])
    return grad_KL


def algorithmic_gradNLL(qr, data, k):
    #qr.nn_state.set_visible_layer(data)
    return qr.compute_batch_gradients(k, data, data)


def numeric_gradKL(nn_state, target_psi, param, vis, eps):
    num_gradKL = []
    for i in range(len(param)):
        param[i] += eps

        Z = partition(nn_state, vis)
        KL_p = compute_numerical_kl(nn_state, target_psi, vis, Z)

        param[i] -= 2*eps

        Z = partition(nn_state, vis)
        KL_m = compute_numerical_kl(nn_state, target_psi, vis, Z)

        param[i] += eps

        num_gradKL.append((KL_p - KL_m) / (2*eps))

    return torch.stack(num_gradKL)


def numeric_gradNLL(nn_state, param, data, vis, eps):
    num_gradNLL = []
    for i in range(len(param)):
        param[i] += eps

        Z = partition(nn_state, vis)
        NLL_p = compute_numerical_NLL(nn_state, data, Z)

        param[i] -= 2*eps

        Z = partition(nn_state, vis)
        NLL_m = compute_numerical_NLL(nn_state, data, Z)

        param[i] += eps

        num_gradNLL.append((NLL_p - NLL_m) / (2*eps))

    return torch.tensor(np.array(num_gradNLL), dtype = torch.double)
