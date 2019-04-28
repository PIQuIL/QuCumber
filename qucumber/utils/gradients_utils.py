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
from torch.nn.utils.convert_parameters import _check_param_device


# based on code found in PyTorch
def vector_to_grads(vec, parameters):
    r"""Convert one vector to the parameters.

    :param vec: a single vector represents the parameters of a model.
    :type vec:  torch.Tensor
    :param parameters: an iterator of Tensors that are the parameters of a
                       model.
    :type parameters: list[torch.Tensor]
    """
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError(
            "expected torch.Tensor, but got: {}".format(torch.typename(vec))
        )
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
        param.grad = vec[pointer : pointer + num_param].view(param.size()).data

        # Increment the pointer
        pointer += num_param
