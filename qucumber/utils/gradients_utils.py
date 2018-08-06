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


# based on code found in PyTorch
def vector_to_grads(vec, parameters):
    r"""Convert one vector to the parameters

    :param vec: a single vector represents the parameters of a model.
    :type vec:  torch.Tensor
    :param parameters: an iterator of Tensors that are the parameters of a
                       model.
    :type parameters: list[torch.Tensor]
    """
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError('expected torch.Tensor, but got: {}'
                        .format(torch.typename(vec)))
    # Flag for the device where the parameter is located
    param_device = None

    # Pointer for slicing the vector for each parameter gradient
    pointer = 0
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        # The length of the parameter
        num_param = param.numel()

        # Slice the vector, reshape it, and replace the gradient data of
        # the parameter
        param.grad = vec[pointer:pointer + num_param].view(param.size()).data

        # Increment the pointer
        pointer += num_param


def _check_param_device(param, old_param_device):
    r"""This helper function is to check if the parameters are located
    in the same device. Currently, the conversion between model parameters
    and single vector form is not supported for multiple allocations,
    e.g. parameters in different GPUs, or mixture of CPU/GPU.

    :param param: a Tensor of a parameter of a model
    :type param: torch.Tensor
    :param old_param_device: the device where the first parameter of a model
                             is allocated.
    :type old_param_device: int

    :returns: old_param_device or -1
    :rtype: int
    """

    # Meet the first parameter
    if old_param_device is None:
        old_param_device = param.get_device() if param.is_cuda else -1
    else:
        different_devices = False
        if param.is_cuda:  # Check if in same GPU
            different_devices = (param.get_device() != old_param_device)
        else:  # Check if in CPU
            different_devices = (old_param_device != -1)

        if different_devices:
            raise TypeError('Found two parameters on different devices, '
                            'this is currently not supported.')
    return old_param_device
