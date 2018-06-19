import torch

'''
A class that allows torch to handle complex algebra.
-----------------------------------------------------------------------------------------
SYNTAX / ORDERING OF INDICES
matrices / tensors: m[2][i][j] >>> 2 = real and imaginary part
                               >>> i = number of rows in the real and imaginary parts
                               >>> j = number of columns in the real and imaginary parts

vectors: v[2][i]               >>> 2 = real and imaginary part
                               >>> i = nmber of rows in the real and imaginary parts

scalars: s[2]                  >>> 2 = real and imaginary part
-----------------------------------------------------------------------------------------
'''


def cplx_make_complex_vector(x, y):
    '''A function that takes two vector (a REAL (x) and IMAGINARY part (y)) and returns the combined
 complex tensor.
        :param: x: Expl
        :sadasd: x: string'''
    if x.size()[0] != y.size()[0]:
        raise ValueError(
            'Real and imaginary parts do not have the same dimension.')

    z = torch.zeros(2, x.size()[0], dtype=torch.double)
    z[0] = x
    z[1] = y

    return z


def cplx_make_complex_matrix(x, y):
    '''A function that takes two tensors (a REAL (x) and IMAGINARY part (y)) and returns the combine
 complex tensor.'''
    if x.size()[0] != y.size()[0] or x.size()[1] != y.size()[1]:
        raise ValueError(
            'Real and imaginary parts do not have the same dimension.')

    z = torch.zeros(2, x.size()[0], x.size()[1], dtype=torch.double)
    z[0] = x
    z[1] = y

    return z


def cplx_scalar_mult(x, y):
    '''A function that does complex scalar multiplication between two complex scalars, x and y.'''
    if list(x.size())[0] < 2 or list(y.size())[0] < 2:
        raise ValueError('An input is not of the right dimension.')

    z = torch.zeros(2, dtype=torch.double)
    z[0] = x[0]*y[0] - x[1]*y[1]
    z[1] = x[0]*y[1] + x[1]*y[0]

    return z


def cplx_VS_mult(x, y):
    '''A function that returns x*y, where x is a complex scalar and y is a complex vector.'''
    if list(x.size())[0] != 2 or list(y.size())[0] != 2:
        raise ValueError('An input is not of the right dimension.')

    z = torch.zeros_like(y)
    z[0] = x[0]*y[0] - x[1]*y[1]
    z[1] = x[0]*y[1] + x[1]*y[0]

    return z


def cplx_MS_mult(x, y):
    '''A function that takes a given input complex matrix (y) and multiplies it by a complex scalar (x).'''
    if len(list(y.size())) != 3 or list(y.size())[0] != 2 or list(x.size())[0] != 2:
        raise ValueError('An input is not of the right dimension.')
    z = torch.zeros_like(y)
    z[0] = x[0]*y[0] - x[1]*y[1]
    z[1] = x[0]*y[1] + x[1]*y[0]

    return z


def cplx_MV_mult(x, y):
    '''A function that returns x*y, where x is a complex tensor and y is a complex vector.'''
    if len(list(x.size())) != 3 or len(list(y.size())) != 2 or list(x.size())[0] != 2 or list(y.size())[0] != 2:
        raise ValueError('An input is not of the right dimension.')

    z = torch.zeros(2, x.size()[1], dtype=torch.double)
    z[0] = torch.mv(x[0], y[0]) - torch.mv(x[1], y[1])
    z[1] = torch.mv(x[0], y[1]) + torch.mv(x[1], y[0])

    return z


def cplx_MM_mult(x, y):
    '''A function that returns x*y, where x and y are complex tensors.'''
    if len(list(x.size())) != 3 or len(list(y.size())) != 3 or list(x.size())[0] != 2 or list(y.size())[0] != 2:
        raise ValueError('An input is not of the right dimension.')

    z = torch.zeros(2, x.size()[1], y.size()[2], dtype=torch.double)
    z[0] = torch.matmul(x[0], y[0]) - torch.matmul(x[1], y[1])
    z[1] = torch.matmul(x[0], y[1]) + torch.matmul(x[1], y[0])

    return z


def cplx_add(x, y):
    '''A function that adds two complex tensors, vectors, or scalars, x and y.'''
    if list(x.size())[0] != 2 or list(y.size())[0] != 2:
        raise ValueError('An input is not of the right dimension.')

    z = torch.zeros_like(x)
    z[0] = x[0] + y[0]
    z[1] = x[1] + y[1]

    return z


def cplx_inner(x, y):
    '''A function that returns the inner product of two complex vectors, x and y >>> <x|y>.'''
    if list(x.size())[0] != 2 or list(y.size())[0] != 2:
        raise ValueError('An input is not of the right dimension.')

    z = torch.zeros(2, dtype=torch.double)
    z[0] = torch.dot(x[0], y[0]) - torch.dot(-x[1], y[1])
    z[1] = torch.dot(x[0], y[1]) + torch.dot(-x[1], y[0])

    return z


def cplx_outer(x, y):
    '''A function that returns the outer product of two complex vectors, x and y.'''
    if len(list(x.size())) < 2 or len(list(y.size())) < 2:
        raise ValueError('An input is not of the right dimension.')

    z = torch.zeros(2, x.size()[1], y.size()[1], dtype=torch.double)
    z[0] = torch.ger(x[0], y[0]) - torch.ger(x[1], -y[1])
    z[1] = torch.ger(x[0], -y[1]) + torch.ger(x[1], y[0])

    return z


def cplx_compT_matrix(x):
    '''A function that returns the complex transpose of a complex tensor, x.'''
    if len(list(x.size())) != 3 or list(x.size())[0] != 2:
        raise ValueError('An input is not of the right dimension.')

    z = torch.zeros(2, x.size()[2], x.size()[1], dtype=torch.double)
    z[0] = torch.transpose(x[0], 0, 1)
    z[1] = -torch.transpose(x[1], 0, 1)

    return z


def cplx_comp_conj_vector(x):
    '''A function that returns the complex conjugate of a complex vector, x.'''
    if len(list(x.size())) != 2 or list(x.size())[0] != 2:
        raise ValueError('An input is not of the right dimension.')

    z = torch.zeros(2, x.size()[1], dtype=torch.double)
    z[0] = x[0]
    z[1] = -x[1]

    return z


def cplx_kronecker(x, y):
    '''A function that returns the tensor / kronecker product of 2 comlex tensors, x and y.'''
    if len(list(x.size())) != 3 or len(list(y.size())) != 3:
        raise ValueError('An input is not of the right dimension.')

    z = torch.zeros(2, x.size()[1]*y.size()[1],
                    x.size()[2]*y.size()[2], dtype=torch.double)

    row_count = 0

    for i in range(x.size()[1]):
        for k in range(y.size()[1]):
            column_count = 0
            for j in range(x.size()[2]):
                for l in range(y.size()[2]):

                    z[0][row_count][column_count] = x[0][i][j] * \
                        y[0][k][l] - x[1][i][j]*y[1][k][l]
                    z[1][row_count][column_count] = x[0][i][j] * \
                        y[1][k][l] + x[1][i][j]*y[0][k][l]

                    column_count += 1

            row_count += 1

    return z

def cplx_VS_divide(x,y):
    '''Computes x/y, where x is a complex vector and y is a complex scalar.'''
    if list(x.size())[0] != 2 or list(y.size())[0] != 2:
        raise ValueError('An input is not of the right dimension.')

    y_star = torch.zeros_like(y)
    y_star[0] = y[0]
    y_star[1] = -y[1]

    numerator = cplx_VS_mult(y_star, x)
    denominator = cplx_scalar_mult(y, y_star)[0]

    return numerator / denominator

def cplx_MS_divide(x,y):
    '''Computes x/y, where x is a complex tensor and y is a complex scalar.'''
    if list(x.size())[0] != 2 or list(y.size())[0] != 2:
        raise ValueError('An input is not of the right dimension.')

    y_star = torch.zeros_like(y)
    y_star[0] = y[0]
    y_star[1] = -y[1]

    numerator = cplx_MS_mult(y_star, x)
    denominator = cplx_scalar_mult(y, y_star)[0]

    return numerator / denominator

def cplx_norm(x):
    '''A function that returns |<x|x>|^2. Argument must be <x|x> (i.e. a scalar).'''
    if list(x.size())[0] != 2:
        raise ValueError('An input is not of the right dimension.')

    x_conj = torch.zeros_like(x)
    x_conj[0] = x[0]
    x_conj[1] = -x[1]

    return cplx_scalar_mult(x_conj, x)[0]
