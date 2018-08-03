import unittest
import sys
sys.path.append("../utils/")
import cplx

import torch

class TestCplx(unittest.TestCase):

    def test_make_complex(self):
        x = torch.tensor([[1,2],[3,4]])        
        y = torch.tensor([[5,6],[7,8]])

        expect = torch.tensor([ [[1,2],[3,4]], [[5,6],[7,8]] ], 
                              dtype = torch.double)
        result = 0.0
        for i in range(2):
            for j in range(list(x.size())[0]):
                for k in range(list(x.size())[1]):

                    result += (cplx.make_complex(x,y)[i][j][k] - 
                               expect[i][j][k]).item() 

        self.assertEqual(result, 0.0)

    def test_scalar_mult(self):
        scalar = torch.tensor([2,3], dtype = torch.double)
        matrix = torch.tensor([ [[1,2],[3,4]], [[5,6],[7,8]] ], 
                 dtype = torch.double)
        vector = torch.tensor([[1,2],[3,4]], dtype = torch.double)
        
        expectMS = torch.tensor([ [[-13,-14],[-15,-16]], [[13,18],[23,28]] ],
                   dtype = torch.double)  
        expectVS = torch.tensor([[-7,-8],[9,14]], dtype = torch.double)

        resultMS = 0.0
        resultVS = 0.0       
        for i in range(2):
            for j in range(list(matrix.size())[1]):
                for k in range(list(matrix.size())[2]): 
                    resultMS += (cplx.scalar_mult(scalar, matrix)[i][j][k] -
                                 expectMS[i][j][k]).item()
        
        for i in range(2):
            for j in range(list(vector.size())[1]):
                resultVS += (cplx.scalar_mult(scalar, vector)[i][j] - 
                             expectVS[i][j]).item()

        self.assertEqual(resultMS, 0.0)
        self.assertEqual(resultVS, 0.0)
    
    def test_matmul(self):
        matrix1 = torch.tensor([ [[1,2],[3,4]], [[5,6],[7,8]] ], 
                               dtype = torch.double)
        matrix2 = torch.tensor([ [[1,0],[3,0]], [[0,6],[0,8]] ], 
                               dtype = torch.double)
        vector = torch.tensor([[1,2],[3,4]], dtype = torch.double)
       
        expectMM = torch.tensor([ [[7,-78],[15,-106]], [[23,22],[31,50]] ], 
                                dtype = torch.double)
        expectMV = torch.tensor([[-34,-42],[28,48]], dtype = torch.double)

        resultMM = 0.0
        resultMV = 0.0
        for i in range(2):
            for j in range(list(matrix1.size())[1]):
                for k in range(list(matrix2.size())[2]):
                    resultMM += (cplx.matmul(matrix1, matrix2)[i][j][k] - 
                               expectMM[i][j][k]).item() 

        for i in range(2):
            for j in range(list(vector.size())[1]):
                resultMV += (cplx.matmul(matrix1, vector)[i][j] - 
                             expectMV[i][j]).item()

        self.assertEqual(resultMM, 0.0)
        self.assertEqual(resultMV, 0.0)
    
    def test_inner_prod(self):
        scalar = torch.tensor([1,2], dtype = torch.double)
        vector = torch.tensor([[1,2],[3,4]], dtype = torch.double)
       
        expectS = torch.tensor([5,0], dtype = torch.double)
        expectV = torch.tensor([30,0], dtype = torch.double)
     
        result1 = 0.0
        result2 = 0.0
        for i in range(2):
            result1 += (cplx.inner_prod(scalar, scalar)[i] - 
                        expectS[i]).item()

            result2 += (cplx.inner_prod(vector, vector)[i] - 
                        expectV[i]).item()

        self.assertEqual(result1, 0.0)
        self.assertEqual(result2, 0.0)

    def test_outer_prod(self):
        vector = torch.tensor([[1,2],[3,4]], dtype = torch.double)

        expect = torch.tensor([ [[10,14],[14,20]], [[0,2],[-2,0]] ], 
                              dtype = torch.double)
        
        result = 0.0
        for i in range(2):
            for j in range(list(vector.size())[1]):
                for k in range(list(vector.size())[1]):
                    result += (cplx.outer_prod(vector, vector)[i][j][k] - 
                               expect[i][j][k]).item()

        self.assertEqual(result, 0.0)

    def test_conjugate(self):
        vector = torch.tensor([[1,2],[3,4]], dtype = torch.double) 
        matrix = torch.tensor([ [[1,2],[3,4]], [[5,6],[7,8]] ], 
                              dtype = torch.double)
        
        expectV = torch.tensor([[1,2],[-3,-4]], dtype = torch.double)
        expectM = torch.tensor([ [[1,3],[2,4]], [[-5,-7],[-6,-8]] ],
                               dtype = torch.double)

        resultV = 0.0
        resultM = 0.0
        for i in range(2):
            for j in range(list(vector.size())[1]):
                resultV += (cplx.conjugate(vector)[i][j] - expectV[i][j]).item()

        for i in range(2):
            for j in range(list(matrix.size())[1]):
                for k in range(list(matrix.size())[2]):
                    resultM += (cplx.conjugate(matrix)[i][j][k] -
                                expectM[i][j][k])
        
        self.assertEqual(resultV, 0.0)
        self.assertEqual(resultM, 0.0)

    def test_kronecker_prod(self):
        matrix = torch.tensor([ [[1,2],[3,4]], [[5,6],[7,8]] ], 
                 dtype = torch.double)

        expect = torch.tensor([ [[-24,-28,-28,-32],[-32,-36,-36,-40],
                                 [-32,-36,-36,-40],[-40,-44,-44,-48]],
                                [[10,16,16,24],[22,28,32,40],[22,32,28,40],
                                 [42,52,52,64]] ], dtype = torch.double)

        result = 0.0
        for i in range(2):
            for j in range(list(expect.size())[1]):
                for k in range(list(expect.size())[2]):
                    result = (cplx.kronecker_prod(matrix, matrix)[i][j][k] - 
                              expect[i][j][k]).item()

        self.assertEqual(result, 0.0)

    def test_scalar_divide(self):
        scalar = torch.tensor([1,2], dtype = torch.double)
        vector = torch.tensor([[1,2],[3,4]], dtype = torch.double) 
        matrix = torch.tensor([ [[1,2],[3,4]], [[5,6],[7,8]] ], 
                              dtype = torch.double)
       
        expectM = torch.tensor([ [[2.2,2.8],[3.4,4.0]], [[0.6,0.4],[0.2,0.0]] ],
                               dtype = torch.double)
        expectV = torch.tensor([[1.4,2.0],[0.2,0.0]], dtype = torch.double)
        
        resultM = 0.0
        resultV = 0.0
        for i in range(2):
            for j in range(list(matrix.size())[1]):
                for k in range(list(matrix.size())[2]):
                    resultM = (cplx.scalar_divide(matrix,scalar)[i][j][k] - 
                               expectM[i][j][k]).item()

        for i in range(2):
            for j in range(list(vector.size())[1]):
                resultV = (cplx.scalar_divide(vector,scalar)[i][j] - 
                           expectV[i][j]).item()

        self.assertEqual(resultV, 0.0)
        self.assertEqual(resultM, 0.0)

    def test_norm(self):
        scalar = torch.tensor([3,4], dtype = torch.double)
   
        result = torch.sum(cplx.norm(scalar) - 
                 torch.tensor(25, dtype = torch.double)).item()

        self.assertEqual(result, 0.0)

if __name__ == '__main__':
    unittest.main()
    
