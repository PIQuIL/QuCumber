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


import unittest

import numpy as np
import torch

from qucumber.utils import cplx


class TestCplx(unittest.TestCase):
    def assertTensorsEqual(self, a, b, msg=None):
        self.assertTrue(torch.equal(a, b), msg=msg)

    def assertTensorsAlmostEqual(self, a, b, tol=1e-7, msg=None):
        self.assertTrue(((a - b).abs() <= tol).all(), msg=msg)

    def test_make_complex_vector(self):
        x = torch.tensor([1, 2, 3, 4])
        y = torch.tensor([5, 6, 7, 8])
        z = cplx.make_complex(x, y)

        expect = torch.tensor([1 + 5j, 2 + 6j, 3 + 7j, 4 + 8j])

        self.assertTensorsEqual(expect, z, msg="Make Complex Vector failed!")

    def test_make_complex_vector_with_zero_imaginary_part(self):
        x = torch.tensor([1, 2, 3, 4])
        z = cplx.make_complex(x)

        expect = torch.tensor([1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j])

        self.assertTensorsEqual(
            expect, z, msg="Making a complex vector with zero imaginary part failed!"
        )

    def test_make_complex_matrix(self):
        x = torch.tensor([[1, 2], [3, 4]])
        y = torch.tensor([[5, 6], [7, 8]])
        z = cplx.make_complex(x, y)

        expect = torch.tensor([[1 + 5j, 2 + 6j], [3 + 7j, 4 + 8j]])

        self.assertTensorsEqual(expect, z, msg="Make Complex Matrix failed!")

    def test_real_part_of_vector(self):
        x = torch.tensor([1, 2])
        y = torch.tensor([5, 6])
        z = cplx.make_complex(x, y)

        self.assertTensorsEqual(
            x, cplx.real(z).to(x), msg="Real part of vector failed!"
        )

    def test_imag_part_of_vector(self):
        x = torch.tensor([1, 2])
        y = torch.tensor([5, 6])
        z = cplx.make_complex(x, y)

        self.assertTensorsEqual(
            y, cplx.imag(z).to(y), msg="Imaginary part of vector failed!"
        )

    def test_real_part_of_matrix(self):
        x = torch.tensor([[1, 2], [3, 4]])
        y = torch.tensor([[5, 6], [7, 8]])
        z = cplx.make_complex(x, y)

        self.assertTensorsEqual(
            x, cplx.real(z).to(x), msg="Real part of matrix failed!"
        )

    def test_imag_part_of_matrix(self):
        x = torch.tensor([[1, 2], [3, 4]])
        y = torch.tensor([[5, 6], [7, 8]])
        z = cplx.make_complex(x, y)

        self.assertTensorsEqual(
            y, cplx.imag(z).to(y), msg="Imaginary part of matrix failed!"
        )

    def test_real_part_of_tensor(self):
        x = torch.randn(3, 3, 3)
        y = torch.randn(3, 3, 3)
        z = cplx.make_complex(x, y)

        self.assertTensorsEqual(
            x, cplx.real(z), msg="Real part of rank-3 tensor failed!"
        )

    def test_imag_part_of_tensor(self):
        x = torch.randn(3, 3, 3)
        y = torch.randn(3, 3, 3)
        z = cplx.make_complex(x, y)

        self.assertTensorsEqual(
            y, cplx.imag(z), msg="Imaginary part of rank-3 tensor failed!"
        )

    def test_bad_complex_matrix(self):
        with self.assertRaises(RuntimeError):
            x = torch.tensor([[1, 2, 3]])
            y = torch.tensor([[4, 5, 6, 7]])
            return cplx.make_complex(x, y)

    def test_elementwise_mult(self):
        z1 = torch.tensor([2 + 6j, 3 + 7j, 5 + 2j])
        z2 = torch.tensor([1 + 3j, 2 + 4j, 2 + 8j])

        expect = torch.tensor([-16 + 12j, -22 + 26j, -6 + 44j])

        self.assertTensorsEqual(
            cplx.elementwise_mult(z1, z2),
            expect,
            msg="Elementwise multiplication failed!",
        )

    def test_elementwise_div(self):
        z1 = torch.tensor([2 + 6j, 3 + 7j, 5 + 2j])
        z2 = torch.tensor([1 + 3j, 2 + 4j, 2 + 8j])

        expect = torch.tensor([2 + 0j, (17 / 10) + (1j / 10), (13 / 34) + (-9j / 17)],)

        self.assertTensorsAlmostEqual(
            cplx.elementwise_division(z1, z2),
            expect,
            msg="Elementwise division failed!",
        )

    def test_elementwise_div_fail(self):
        with self.assertRaises(ValueError):
            z1 = torch.tensor([[2, 3], [6, 7]], dtype=torch.double)
            z2 = torch.tensor([[1, 2, 2], [3, 4, 8]], dtype=torch.double)
            return cplx.elementwise_division(z1, z2)

    def test_scalar_vector_mult(self):
        scalar = torch.tensor([2 + 3j])
        vector = torch.tensor([1 + 3j, 2 + 4j])

        expect = torch.tensor([-7 + 9j, -8 + 14j])

        self.assertTensorsEqual(
            cplx.scalar_mult(scalar, vector),
            expect,
            msg="Scalar * Vector multiplication failed!",
        )

    def test_scalar_matrix_mult(self):
        scalar = torch.tensor([2 + 3j])
        matrix = torch.tensor([[1 + 5j, 2 + 6j], [3 + 7j, 4 + 8j]])

        expect = torch.tensor([[-13 + 13j, -14 + 18j], [-15 + 23j, -16 + 28j]])

        self.assertTensorsEqual(
            cplx.scalar_mult(scalar, matrix),
            expect,
            msg="Scalar * Matrix multiplication failed!",
        )

    def test_scalar_mult_overwrite(self):
        scalar = torch.tensor([2 + 3j])
        vector = torch.tensor([1 + 3j, 2 + 4j])

        expect = torch.tensor([-7 + 9j, -8 + 14j])

        out = torch.zeros_like(vector)
        cplx.scalar_mult(scalar, vector, out=out)

        self.assertTensorsEqual(
            out,
            expect,
            msg="Scalar * Vector multiplication with 'out' parameter failed!",
        )

    def test_scalar_mult_overwrite_fail(self):
        scalar = torch.tensor([2 + 3j])
        vector = torch.tensor([1 + 3j, 2 + 4j])

        with self.assertRaises(RuntimeError):
            cplx.scalar_mult(scalar, vector, out=vector)

        with self.assertRaises(RuntimeError):
            cplx.scalar_mult(scalar, vector, out=scalar)

    def test_matrix_vector_matmul(self):
        matrix = torch.tensor([[1 + 5j, 2 + 6j], [3 + 7j, 4 + 8j]])
        vector = torch.tensor([1 + 3j, 2 + 4j])

        expect = torch.tensor([-34 + 28j, -42 + 48j])

        self.assertTensorsEqual(
            cplx.matmul(matrix, vector),
            expect,
            msg="Matrix * Vector multiplication failed!",
        )

    def test_matrix_matrix_matmul(self):
        matrix1 = torch.tensor([[1 + 5j, 2 + 6j], [3 + 7j, 4 + 8j]])
        matrix2 = torch.tensor([[1 + 0j, 0 + 6j], [3 + 0j, 0 + 8j]])
        expect = torch.tensor([[7 + 23j, -78 + 22j], [15 + 31j, -106 + 50j]])
        self.assertTensorsEqual(
            cplx.matmul(matrix1, matrix2),
            expect,
            msg="Matrix * Matrix multiplication failed!",
        )

    def test_scalar_inner_prod(self):
        scalar = torch.tensor(1 + 2j)
        expect = torch.tensor(5 + 0j)
        self.assertTensorsEqual(
            cplx.inner_prod(scalar, scalar), expect, msg="Scalar inner product failed!"
        )

    def test_vector_inner_prod(self):
        vector = torch.tensor([1 + 3j, 2 + 4j])
        expect = torch.tensor(30 + 0j)
        self.assertTensorsEqual(
            cplx.inner_prod(vector, vector), expect, msg="Vector inner product failed!"
        )

    def test_outer_prod(self):
        vector = torch.tensor([1 + 3j, 2 + 4j])
        expect = torch.tensor([[10 + 0j, 14 + 2j], [14 - 2j, 20 + 0j]])
        self.assertTensorsEqual(
            cplx.outer_prod(vector, vector), expect, msg="Outer product failed!"
        )

    def test_outer_prod_error_small(self):
        # take outer prod of 2 scalars, instead of vectors
        scalar = torch.tensor(1 + 2j)
        with self.assertRaises(ValueError):
            cplx.outer_prod(scalar, scalar)

    def test_outer_prod_error_large(self):
        # take outer prod of 2 matrices, instead of vectors
        matrix = torch.tensor([[1 + 5j, 2 + 6j], [3 + 7j, 4 + 8j]])
        with self.assertRaises(ValueError):
            cplx.outer_prod(matrix, matrix)

    def test_conjugate(self):
        vector = torch.tensor([1 + 3j, 2 + 4j])

        expect = torch.tensor([1 - 3j, 2 - 4j])

        self.assertTensorsEqual(
            cplx.conjugate(vector), expect, msg="Vector conjugate failed!"
        )

    def test_matrix_conjugate(self):
        matrix = torch.tensor([[1 + 5j, 2 + 6j], [3 + 7j, 4 + 8j]])
        expect = torch.tensor([[1 - 5j, 3 - 7j], [2 - 6j, 4 - 8j]])
        self.assertTensorsEqual(
            cplx.conjugate(matrix), expect, msg="Matrix conjugate failed!"
        )

    def test_kronecker_prod(self):
        matrix = torch.tensor([[1 + 5j, 2 + 6j], [3 + 7j, 4 + 8j]])

        expect = torch.tensor(
            [
                [-24 + 10j, -28 + 16j, -28 + 16j, -32 + 24j],
                [-32 + 22j, -36 + 28j, -36 + 32j, -40 + 40j],
                [-32 + 22j, -36 + 32j, -36 + 28j, -40 + 40j],
                [-40 + 42j, -44 + 52j, -44 + 52j, -48 + 64j],
            ],
        )

        self.assertTensorsEqual(
            cplx.kronecker_prod(matrix, matrix), expect, msg="Kronecker product failed!"
        )

    def test_kronecker_prod_error_small(self):
        # take KronProd of 2 rank 2 tensors, instead of rank 3
        tensor = torch.tensor([1 + 3j, 2 + 4j])
        with self.assertRaises(ValueError):
            cplx.kronecker_prod(tensor, tensor)

    def test_kronecker_prod_error_large(self):
        # take KronProd of 2 rank 4 tensors, instead of rank 3
        tensor = torch.arange(16, dtype=torch.double).reshape(2, 2, 2, 2)
        with self.assertRaises(ValueError):
            cplx.kronecker_prod(tensor, tensor)

    def test_vector_scalar_divide(self):
        scalar = torch.tensor(1 + 2j)
        vector = torch.tensor([1 + 3j, 2 + 4j])
        expect = torch.tensor([1.4 + 0.2j, 2.0 + 0.0j])

        self.assertTensorsAlmostEqual(
            cplx.scalar_divide(vector, scalar),
            expect,
            msg="Vector / Scalar divide failed!",
        )

    def test_matrix_scalar_divide(self):
        scalar = torch.tensor(1 + 2j)
        matrix = torch.tensor([[1 + 5j, 2 + 6j], [3 + 7j, 4 + 8j]])

        expect = torch.tensor([[2.2 + 0.6j, 2.8 + 0.4j], [3.4 + 0.2j, 4.0 + 0.0j]])

        self.assertTensorsAlmostEqual(
            cplx.scalar_divide(matrix, scalar),
            expect,
            msg="Matrix / Scalar divide failed!",
        )

    def test_norm_sqr(self):
        scalar = torch.tensor(3 + 4j)
        expect = torch.tensor(25)

        self.assertTensorsEqual(
            cplx.norm_sqr(scalar).to(expect), expect, msg="Norm failed!"
        )

    def test_norm(self):
        scalar = torch.tensor(3 + 4j)
        expect = torch.tensor(5)

        self.assertTensorsEqual(
            cplx.norm(scalar).to(expect), expect, msg="Norm failed!"
        )

    def test_absolute_value(self):
        tensor = torch.tensor(
            [[5 + 2j, 5 - 2j, -5 + 2j, -5 - 2j], [3 - 7j, 6 + 8j, -9 + 0j, 1 + 4j]]
        )

        expect = torch.tensor([[np.sqrt(29)] * 4, [np.sqrt(58), 10, 9, np.sqrt(17)]])

        self.assertTensorsAlmostEqual(
            cplx.absolute_value(tensor), expect, msg="Absolute Value failed!"
        )


if __name__ == "__main__":
    unittest.main()
