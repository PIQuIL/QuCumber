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


I = torch.Tensor([0, 1])  # noqa: E741


def make_complex(x, y=None):
    """A function that creates a torch compatible complex tensor.

    .. note:: `x` and `y` must have the same shape.

    :param x: The real part or a complex numpy array. If a numpy array,
              will ignore `y`.
    :type x: torch.Tensor or numpy.ndarray
    :param y: The imaginary part. Can be `None`, in which case, the resulting
              complex tensor will have imaginary part equal to zero.
    :type y: torch.Tensor

    :returns: The tensor :math:`[x,y] = x + iy`.
    :rtype: torch.Tensor
    """
    if isinstance(x, np.ndarray):
        return make_complex(torch.tensor(x.real), torch.tensor(x.imag)).contiguous()

    if y is None:
        y = torch.zeros_like(x)
    return torch.cat((x.unsqueeze(0), y.unsqueeze(0)), dim=0)


def numpy(x):
    """Converts a complex torch tensor into a numpy array


    :param x: The tensor to convert.
    :type x: torch.Tensor

    :returns: A complex numpy array containing the data from `x`.
    :rtype: numpy.ndarray
    """
    return real(x).detach().cpu().numpy() + 1j * imag(x).detach().cpu().numpy()


def real(x):
    """Returns the real part of a complex tensor.

    :param x: The complex tensor
    :type x: torch.Tensor

    :returns: The real part of `x`; will have one less dimension than `x`.
    :rtype: torch.Tensor
    """
    return x[0, ...]


def imag(x):
    """Returns the imaginary part of a complex tensor.

    :param x: The complex tensor
    :type x: torch.Tensor

    :returns: The imaginary part of `x`; will have one less dimension than `x`.
    :rtype: torch.Tensor
    """
    return x[1, ...]


def scalar_mult(x, y, out=None):
    """A function that computes the product between complex matrices and scalars,
    complex vectors and scalars or two complex scalars.

    :param x: A complex scalar, vector or matrix.
    :type x: torch.Tensor
    :param y: A complex scalar, vector or matrix.
    :type y: torch.Tensor

    :returns: The product between `x` and `y`.
              Either overwrites `out`, or returns a new tensor.
    :rtype: torch.Tensor
    """
    y = y.to(x)
    if out is None:
        out = torch.zeros(2, *((real(x) * real(y)).shape)).to(x)
    else:
        if out is x or out is y:
            raise RuntimeError("Can't overwrite an argument!")

    torch.mul(real(x), real(y), out=real(out)).sub_(torch.mul(imag(x), imag(y)))
    torch.mul(real(x), imag(y), out=imag(out)).add_(torch.mul(imag(x), real(y)))

    return out


def matmul(x, y):
    """A function that computes complex matrix-matrix and matrix-vector products.

    .. note:: If one wishes to do matrix-vector products, the vector must be
              the second argument (y).

    :param x: A complex matrix.
    :type x: torch.Tensor
    :param y: A complex vector or matrix.
    :type y: torch.Tensor

    :returns: The product between x and y.
    :rtype: torch.Tensor
    """
    y = y.to(x)
    re = torch.matmul(real(x), real(y)).sub_(torch.matmul(imag(x), imag(y)))
    im = torch.matmul(real(x), imag(y)).add_(torch.matmul(imag(x), real(y)))

    return make_complex(re, im)


def inner_prod(x, y):
    """A function that returns the inner product of two complex vectors,
    x and y (<x|y>).

    :param x: A complex vector.
    :type x: torch.Tensor
    :param y: A complex vector.
    :type y: torch.Tensor

    :raises ValueError: If x and y are not complex vectors with their first
                        dimensions being 2, then the function will not execute.

    :returns: The inner product, :math:`\\langle x\\vert y\\rangle`.
    :rtype: torch.Tensor
    """
    y = y.to(x)

    if x.dim() == 2 and y.dim() == 2:
        return make_complex(
            torch.dot(real(x), real(y)) + torch.dot(imag(x), imag(y)),
            torch.dot(real(x), imag(y)) - torch.dot(imag(x), real(y)),
        )
    elif x.dim() == 1 and y.dim() == 1:
        return make_complex(
            (real(x) * real(y)) + (imag(x) * imag(y)),
            (real(x) * imag(y)) - (imag(x) * real(y)),
        )
    else:
        raise ValueError("Unsupported input shapes!")


def outer_prod(x, y):
    """A function that returns the outer product of two complex vectors, x
    and y.

    :param x: A complex vector.
    :type x: torch.Tensor
    :param y: A complex vector.
    :type y: torch.Tensor

    :raises ValueError:	If x and y are not complex vectors with their first
                        dimensions being 2, then an error will be raised.

    :returns: The outer product between x and y,
        :math:`\\vert x \\rangle\\langle y\\vert`.
    :rtype: torch.Tensor
    """
    if x.dim() != 2 or y.dim() != 2:
        raise ValueError("An input is not of the right dimension.")

    z = torch.zeros(2, x.size()[1], y.size()[1], dtype=x.dtype, device=x.device)
    z[0] = torch.ger(real(x), real(y)) - torch.ger(imag(x), -imag(y))
    z[1] = torch.ger(real(x), -imag(y)) + torch.ger(imag(x), real(y))

    return z


def einsum(equation, a, b, real_part=True, imag_part=True):
    """Complex-aware version of `einsum`. See the torch documentation for more
    details.

    :param equation: The index equation. Passed directly to `torch.einsum`.
    :type equation: str
    :param a: A complex tensor.
    :type a: torch.Tensor
    :param b: A complex tensor.
    :type b: torch.Tensor
    :param real_part: Whether to compute and return the real part of the result.
    :type real_part: bool
    :param imag_part: Whether to compute and return the imaginary part of the result.
    :type imag_part: bool

    :returns: The Einstein summation of the input tensors performed according to
              the given equation. If both `real_part` and `imag_part` are true,
              the result will be a complex tensor, otherwise a real tensor.
    :rtype: torch.Tensor
    """
    if real_part:
        r = torch.einsum(equation, real(a), real(b)).sub_(
            torch.einsum(equation, imag(a), imag(b))
        )
    if imag_part:
        i = torch.einsum(equation, real(a), imag(b)).add_(
            torch.einsum(equation, imag(a), real(b))
        )

    if real_part and imag_part:
        return make_complex(r, i)
    elif real_part:
        return r
    elif imag_part:
        return i
    else:
        return None


def conjugate(x):
    """Returns the conjugate transpose of the argument.

    In the case of a scalar or vector, only the complex conjugate is taken.
    In the case of a rank-2 or higher tensor, the complex conjugate is taken,
    then the first two indices of the tensor are swapped.

    :param x: A complex tensor.
    :type x: torch.Tensor

    :returns: The conjugate of x.
    :rtype: torch.Tensor
    """
    if x.dim() < 3:
        return conj(x)
    else:
        return make_complex(
            torch.transpose(real(x), 0, 1), -torch.transpose(imag(x), 0, 1)
        )


def conj(x):
    """Returns the element-wise complex conjugate of the argument.

    :param x: A complex tensor.
    :type x: torch.Tensor

    :returns: The complex conjugate of x.
    :rtype: torch.Tensor
    """
    return make_complex(real(x), -imag(x))


def elementwise_mult(x, y):
    """Alias for :func:`scalar_mult`."""
    return scalar_mult(x, y)


def elementwise_division(x, y):
    """Element-wise division of x by y.

    :param x: A complex tensor.
    :type x: torch.Tensor
    :param y: A complex tensor.
    :type y: torch.Tensor

    :rtype: torch.Tensor
    """
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape!")

    y_star = conj(y)

    sqrd_abs_y = absolute_value(y).pow_(2)

    return elementwise_mult(x, y_star).div_(sqrd_abs_y)


def absolute_value(x):
    """Returns the complex absolute value elementwise.

    :param x: A complex tensor.
    :type x: torch.Tensor

    :returns: A real tensor.
    :rtype: torch.Tensor
    """
    x_star = conj(x)
    return real(elementwise_mult(x, x_star)).sqrt_()


def kronecker_prod(x, y):
    """Returns the tensor / Kronecker product of 2 complex matrices, x and y.

    :param x: A complex matrix.
    :type x: torch.Tensor
    :param y: A complex matrix.
    :type y: torch.Tensor

    :raises ValueError: If x and y do not have 3 dimensions or their first
                        dimension is not 2, the function cannot execute.

    :returns: The Kronecker product of x and y, :math:`x \\otimes y`.
    :rtype: torch.Tensor
    """
    if not (x.dim() == y.dim() == 3):
        raise ValueError("Inputs must be complex matrices!")

    return einsum("ab,cd->acbd", x, y).reshape(
        2, x.shape[1] * y.shape[1], x.shape[2] * y.shape[2]
    )


def sigmoid(x, y):
    r"""Returns the sigmoid function of a complex number. Acts elementwise.

    :param x: The real part of the complex number
    :type x: torch.Tensor
    :param y: The imaginary part of the complex number
    :type y: torch.Tensor
    :returns: The complex sigmoid of :math:`x + iy`
    :rtype: torch.Tensor
    """
    z = (x.cpu().numpy()) + 1j * (y.cpu().numpy())

    out = np.exp(z) / (1 + np.exp(z))
    out = torch.tensor([np.real(out), np.imag(out)]).to(x)

    return out


def scalar_divide(x, y):
    """Divides `x` by `y`.
    If `x` and `y` have the same shape, then acts elementwise.
    If `y` is a complex scalar, then performs a scalar division.

    :param x: The numerator (a complex tensor).
    :type x: torch.Tensor
    :param y: The denominator (a complex tensor).
    :type y: torch.Tensor

    :returns: x / y
    :rtype: torch.Tensor
    """
    return scalar_mult(x, inverse(y))


def inverse(z):
    """Returns the multiplicative inverse of `z`. Acts elementwise.

    :param z: The complex tensor.
    :type z: torch.Tensor

    :returns: 1 / z
    :rtype: torch.Tensor
    """
    z_star = conj(z)
    denominator = real(scalar_mult(z, z_star))

    return z_star / denominator


def norm_sqr(x):
    """Returns the squared norm of the argument.

    :param x: A complex scalar.
    :type x: torch.Tensor

    :returns: :math:`|x|^2`.
    :rtype: torch.Tensor
    """
    return real(inner_prod(x, x))


def norm(x):
    """Returns the norm of the argument.

    :param x: A complex scalar.
    :type x: torch.Tensor

    :returns: :math:`|x|`.
    :rtype: torch.Tensor
    """
    return norm_sqr(x).sqrt_()
