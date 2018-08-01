import unittest
import cplx

import torch

class TestCplx(unittest.TestCase):

    def test_make_complex(self):
        x = torch.tensor([[1,2],[3,4]])        
        y = torch.tensor([[5,6],[7,8]])
        result = torch.sum(cplx.make_complex(x,y) - 
                 torch.tensor([ [[1,2],[3,4]], [[5,6],[7,8]] ],
                 dtype = torch.double)).item()
        self.assertEqual(result, 0.0)

    def test_scalar_mult(self):
        scalar = torch.tensor([2,3], dtype = torch.double)
        
        matrix = torch.tensor([ [[1,2],[3,4]], [[5,6],[7,8]] ], 
                 dtype = torch.double)
        vector = torch.tensor([[1,2],[3,4]], dtype = torch.double)
        
        resultMS = torch.sum(cplx.scalar_mult(scalar, matrix) - 
                   torch.tensor([ [[-13,-14],[-15,-16]], [[13,18],[23,28]] ],
                   dtype = torch.double)).item()
                           
        resultVS = torch.sum(cplx.scalar_mult(scalar, vector) - 
                   torch.tensor([[-7,-8],[9,14]], dtype = torch.double)).item()

        self.assertEqual(resultMS, 0.0)
        self.assertEqual(resultVS, 0.0)
    
    def test_matmul(self):
        matrix1 = torch.tensor([ [[1,2],[3,4]], [[5,6],[7,8]] ], 
                 dtype = torch.double)
    
        matrix2 = torch.tensor([ [[1,0],[3,0]], [[0,6],[0,8]] ], 
                 dtype = torch.double)
    
        vector = torch.tensor([[1,2],[3,4]], dtype = torch.double)
        
        resultMM = torch.sum(cplx.matmul(matrix1, matrix2) - 
                   torch.tensor([ [[7,-78],[15,-106]], [[23,22],[31,50]] ], 
                   dtype = torch.double)).item()

        resultMV = torch.sum(cplx.matmul(matrix1, vector) - 
                   torch.tensor([[-34,-42],[28,48]], dtype = torch.double)).item()

        self.assertEqual(resultMM, 0.0)
        self.assertEqual(resultMV, 0.0)
    
    def test_inner_prod(self):
        scalar = torch.tensor([1,2], dtype = torch.double)
        vector = torch.tensor([[1,2],[3,4]], dtype = torch.double)
        
        result1 = torch.sum(cplx.inner_prod(scalar, scalar) - 
                  torch.tensor([5,0], dtype = torch.double)).item()

        result2 = torch.sum(cplx.inner_prod(vector, vector) - 
                  torch.tensor([30,0], dtype = torch.double)).item()

        self.assertEqual(result1, 0.0)
        self.assertEqual(result2, 0.0)

    def test_outer_prod(self):
        vector = torch.tensor([[1,2],[3,4]], dtype = torch.double)

        result = torch.sum(cplx.outer_prod(vector, vector) - 
                 torch.tensor([ [[10,14],[14,20]], [[0,2],[-2,0]] ], 
                 dtype = torch.double)).item()

        self.assertEqual(result, 0.0)

    def test_conjugate(self):
        vector = torch.tensor([[1,2],[3,4]], dtype = torch.double) 
        
        matrix = torch.tensor([ [[1,2],[3,4]], [[5,6],[7,8]] ], 
                 dtype = torch.double)

        resultV = torch.sum(cplx.conjugate(vector) - 
                  torch.tensor([[1,2],[-3,-4]], dtype = torch.double)).item()
    
        resultM = torch.sum(cplx.conjugate(matrix) -
                  torch.tensor([ [[1,3],[2,4]], [[-5,-7],[-6,-8]] ],
                  dtype = torch.double)).item()
        
        self.assertEqual(resultV, 0.0)
        self.assertEqual(resultM, 0.0)

    def test_kronecker_prod(self):
        matrix = torch.tensor([ [[1,2],[3,4]], [[5,6],[7,8]] ], 
                 dtype = torch.double)

        result = torch.sum(cplx.kronecker_prod(matrix, matrix) - 
                 torch.tensor([ [[-24,-28,-28,-32],[-32,-36,-36,-40],
                                 [-32,-36,-36,-40],[-40,-44,-44,-48]],
                                [[10,16,16,24],[22,28,32,40],[22,32,28,40],
                                 [42,52,52,64]] ], dtype = torch.double)).item()

        self.assertEqual(result, 0.0)

    def test_scalar_divide(self):
        scalar = torch.tensor([1,2], dtype = torch.double)
        vector = torch.tensor([[1,2],[3,4]], dtype = torch.double) 
        matrix = torch.tensor([ [[1,2],[3,4]], [[5,6],[7,8]] ], 
                 dtype = torch.double)
        
        resultM = torch.sum(cplx.scalar_divide(matrix,scalar) - 
                  torch.tensor([ [[2.2,2.8],[3.4,4.0]], [[0.6,0.4],[0.2,0.0]] ],
                  dtype = torch.double)).item()

        resultV = torch.sum(cplx.scalar_divide(vector,scalar) - 
                  torch.tensor([[1.4,2.0],[0.2,0.0]], dtype = torch.double)).item()

        self.assertEqual(resultV, 0.0)
        self.assertEqual(resultM, 0.0)

    def test_norm(self):
        scalar = torch.tensor([3,4], dtype = torch.double)
    
        result = torch.sum(cplx.norm(scalar) - 
                 torch.tensor(25, dtype = torch.double)).item()

        self.assertEqual(result, 0.0)

if __name__ == '__main__':
    unittest.main()
    
