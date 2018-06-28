import torch

"""
A class that allows torch to handle complex algebra.
---------
SYNTAX / ORDERING OF INDICES
matrices / tensors: m[2][i][j] >>> 2 = real and imaginary part
							   >>> i = number of rows in the real and imaginary parts
							   >>> j = number of columns in the real and imaginary parts

vectors: v[2][i]               >>> 2 = real and imaginary part
							   >>> i = nmber of rows in the real and imaginary parts

scalars: s[2]                  >>> 2 = real and imaginary part
---------
"""

def make_complex_vector(x, y):
	"""A function that takes two vector (a REAL (x) and IMAGINARY part (y)) and returns the combined complex tensor.
	
	
	# Arguments
	
	x: torch.doubleTensor
		The real part of your vector.
	y: torch.doubleTensor
		The imaginary part of your vector.

	
	# Raises
	
	ValueError
		This function will not execute if x and y do not have the same dimension.


	# Returns
	
	z: torch.doubleTensor
		The full vector with the real and imaginary parts seperated as previously mentioned.
	"""	
	if x.size()[0] != y.size()[0]:
		raise ValueError('Real and imaginary parts do not have the same dimension.')

	z = torch.zeros(2, x.size()[0], dtype=torch.double)
	z[0] = x
	z[1] = y

	return z

def make_complex_matrix(x, y):
	"""A function that takes two tensors (a REAL (x) and IMAGINARY part (y)) and returns the combine complex tensor.


	# Arguments
	
	x: torch.doubleTensor
		The real part of your matrix.
	y: torch.doubleTensor
		The imaginary part of your matrix.

	# Raises
	
	ValueError
		This function will not execute if x and y do not have the same dimension.


	# Returns
	
	z: torch.doubleTensor
		The full vector with the real and imaginary parts seperated as previously mentioned.

	"""
	if x.size()[0] != y.size()[0] or x.size()[1] != y.size()[1]:
		raise ValueError(
			'Real and imaginary parts do not have the same dimension.')

	z = torch.zeros(2, x.size()[0], x.size()[1], dtype=torch.double)
	z[0] = x
	z[1] = y

	return z

def scalar_mult(x, y):
	"""A function that does complex scalar multiplication between two complex scalars, x and y.


	# Arguments
	
	x: torch.doubleTensor
		A complex scalar. (x[0], x[1]) = (real part, imaginary part).
	y: torch.doubleTensor
 		A complex scalar. (x[0], x[1]) = (real part, imaginary part).


	# Raises
	
	ValueError
		If x or y do not have 2 entries (one for the real and imaginary parts each), then this function will not execute.
	

	# Returns
	
	z: torch.doubleTensor
		The product of x and y.
	"""
	if list(x.size())[0] < 2 or list(y.size())[0] < 2:
		raise ValueError('An input is not of the right dimension.')

	z = torch.zeros(2, dtype=torch.double)
	z[0] = x[0]*y[0] - x[1]*y[1]
	z[1] = x[0]*y[1] + x[1]*y[0]

	return z

def VS_mult(x, y):
	"""A function that returns x*y, where x is a complex scalar and y is a complex vector.
	

	# Arguments
	
	x: torch.doubleTensor
		A complex scalar.
	y: torch.doubleTensor
		A complex vector.


	# Raises
	
	ValueError
		If x and y do not have 2 entries (one for the real and imaginary parts each), then this function will not execute.


	# Returns
	
	z: torch.doubleTensor
		The vector scalar product of x and y.
	"""
	if list(x.size())[0] != 2 or list(y.size())[0] != 2:
		raise ValueError('An input is not of the right dimension.')

	z = torch.zeros_like(y)
	z[0] = x[0]*y[0] - x[1]*y[1]
	z[1] = x[0]*y[1] + x[1]*y[0]

	return z


def MS_mult(x, y):
	"""A function that takes a given input complex matrix (y) and multiplies it by a complex scalar (x).
	
	
	# Arguments
	
	x: torch.doubleTensor
		A complex scalar.
	y: torch.doubleTensor
		A complex matrix.

	
	# Raises
	
	ValueError
		If y is not a complex tensor (i.e. has 3 dimensions) and if its first dimension is not 2, OR if x's dimension is not 2, the function will not execute.


	# Returns
	
	z: torch.doubleTensor
		The matrix-scalar product - x*y.

	"""
	if len(list(y.size())) != 3 or list(y.size())[0] != 2 or list(x.size())[0] != 2:
		raise ValueError('An input is not of the right dimension.')
	z = torch.zeros_like(y)
	z[0] = x[0]*y[0] - x[1]*y[1]
	z[1] = x[0]*y[1] + x[1]*y[0]

	return z

def MV_mult(x, y):
	"""A function that returns x*y, where x is a complex tensor and y is a complex vector.
	

	# Arguments
	
	x: torch.doubleTensor
		A complex matrix.
	y: torch.doubleTensor
		A complex vector.


	# Raises
	
	ValueError 
		If y is not a complex vector (i.e. has 2 dimensions) and if its first dimension is not 2, OR if x is not a complex matrix (i.e. has 3 dimensions) and if its first dimention is not 2, then the function will not execute.


	# Returns
	
	z: torch.doubleTensor
		The matrix-vector product, xy.
	"""
	if len(list(x.size())) != 3 or len(list(y.size())) != 2 or list(x.size())[0] != 2 or list(y.size())[0] != 2:
		raise ValueError('An input is not of the right dimension.')

	z = torch.zeros(2, x.size()[1], dtype=torch.double)
	z[0] = torch.mv(x[0], y[0]) - torch.mv(x[1], y[1])
	z[1] = torch.mv(x[0], y[1]) + torch.mv(x[1], y[0])

	return z


def MM_mult(x, y):
	"""A function that returns x*y, where x and y are complex tensors.
	

	# Arguments
	
	x: torch.doubleTensor
		A complex matrix.
	y: torch.doubleTensor
		A complex matrix.


	# Raises
	
	ValueError
		If x and y do not have 3 dimensions or their first dimension is not 2, the function cannot execute.


	# Returns
	
	z: torch.doubleTensor
		The matrix-matrix product, xy.
	"""
	if len(list(x.size())) != 3 or len(list(y.size())) != 3 or list(x.size())[0] != 2 or list(y.size())[0] != 2:
		raise ValueError('An input is not of the right dimension.')

	z = torch.zeros(2, x.size()[1], y.size()[2], dtype=torch.double)
	z[0] = torch.matmul(x[0], y[0]) - torch.matmul(x[1], y[1])
	z[1] = torch.matmul(x[0], y[1]) + torch.matmul(x[1], y[0])

	return z

def inner_prod(x, y):
	"""A function that returns the inner product of two complex vectors, x and y >>> <x|y>.


	# Arguments
	
	x: torch.doubleTensor
		A complex vector.
	y: torch.doubleTensor
		A complex vector.

	
	# Raises
	
	ValueError
		If x and y are not complex vectors with their first dimensions being 2, then the function will not execute.

	
	# Returns
	
	z: torch.doubleTensor
		The inner product, <x|y>
	"""
	if list(x.size())[0] != 2 or list(y.size())[0] != 2:
		raise ValueError('An input is not of the right dimension.')

	z = torch.zeros(2, dtype=torch.double)
	z[0] = torch.dot(x[0], y[0]) - torch.dot(-x[1], y[1])
	z[1] = torch.dot(x[0], y[1]) + torch.dot(-x[1], y[0])

	return z

def outer_prod(x, y):
	"""A function that returns the outer product of two complex vectors, x and y.
	

	# Arguments
	
	x: torch.doubleTensor
		A complex vector.
	y: torch.doubleTensor
		A complex vector.

	
	# Raises
	
	ValueError
		If x and y are not complex vectors with their first dimensions being 2, then the function will not execute.

	
	# Returns
	
	z: torch.doubleTensor
		The outer product between x and y, |x><y|.
	"""
	if len(list(x.size())) < 2 or len(list(y.size())) < 2:
		raise ValueError('An input is not of the right dimension.')

	z = torch.zeros(2, x.size()[1], y.size()[1], dtype=torch.double)
	z[0] = torch.ger(x[0], y[0]) - torch.ger(x[1], -y[1])
	z[1] = torch.ger(x[0], -y[1]) + torch.ger(x[1], y[0])

	return z

def compT_matrix(x):
	"""A function that returns the complex transpose of a complex tensor, x.
	

	# Arguments
	
	x: torch.doubleTensor
		A complex matrix


	# Raises
	
	ValueError
		If x does not have 3 dimensions and its first dimension isn't 2, the function cannot execute.


	# Returns
	
	z: torch.doubleTensor
		The complex transpose of x.
	"""
	if len(list(x.size())) != 3 or list(x.size())[0] != 2:
		raise ValueError('An input is not of the right dimension.')

	z = torch.zeros(2, x.size()[2], x.size()[1], dtype=torch.double)
	z[0] = torch.transpose(x[0], 0, 1)
	z[1] = -torch.transpose(x[1], 0, 1)

	return z

def compT_vector(x):
	"""A function that returns the complex conjugate of a complex vector, x.


	# Arguments
	
	x: torch.doubleTensor
		A complex vector.


	# Raises
	
	ValueError
		If x does not have 2 dimensions and its first dimension isn't 2, the function cannot execute.


	# Returns
	
	z: torch.doubleTensor
		The complex transpose of x.

	"""
	if len(list(x.size())) != 2 or list(x.size())[0] != 2:
		raise ValueError('An input is not of the right dimension.')

	z = torch.zeros(2, x.size()[1], dtype=torch.double)
	z[0] = x[0]
	z[1] = -x[1]

	return z

def kronecker_prod(x, y):
	"""A function that returns the tensor / kronecker product of 2 comlex tensors, x and y.


	# Arguments
	
	x: torch.doubleTensor
		A complex matrix.
	y: torch.doubleTensor
		A complex matrix.


	# Raises
	
	ValueError
		If x and y do not have 3 dimensions or their first dimension is not 2, the function cannot execute.


	# Returns
	
	z: torch.doubleTensor
		The tensorproduct of x and y.
	"""
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

def VS_divide(x,y):
	"""Computes x/y, where x is a complex vector and y is a complex scalar.

	
	# Arguments
	
	x: torch.doubleTensor
		A complex vector.
	y: torch.doubleTensor
		A complex scalar.


	# Raises
	
	ValueError
		If x and y do not have 2 entries (one for the real and imaginary parts each), then this function will not execute.


	# Returns
	
	z: torch.doubleTensor
		z = x/y.
	"""
	if list(x.size())[0] != 2 or list(y.size())[0] != 2:
		raise ValueError('An input is not of the right dimension.')

	y_star = torch.zeros_like(y)
	y_star[0] = y[0]
	y_star[1] = -y[1]

	numerator = VS_mult(y_star, x)
	denominator = scalar_mult(y, y_star)[0]

	return numerator / denominator

def MS_divide(x,y):
	"""Computes x/y, where x is a complex tensor and y is a complex scalar.


	# Arguments
	
	x: torch.doubleTensor
		A complex matrix.
	y: torch.doubleTensor
		A complex scalar.

	
	# Raises
	
	ValueError
		If x is not a complex tensor (i.e. has 3 dimensions) and if its first dimension is not 2, OR if y's dimension is not 2, the function will not execute.


	# Returns
	
	z: torch.doubleTensor
		z = x/y
	"""
	if list(x.size())[0] != 2 or list(y.size())[0] != 2:
		raise ValueError('An input is not of the right dimension.')

	y_star = torch.zeros_like(y)
	y_star[0] = y[0]
	y_star[1] = -y[1]

	numerator = MS_mult(y_star, x)
	denominator = scalar_mult(y, y_star)[0]

	return numerator / denominator

def norm(x):
	"""A function that returns |<x|x>|^2. Argument must be <x|x> (i.e. a scalar).


	# Arguments
		
	x: torch.doubleTensor
		A complex scalar.


	# Raises
	
	ValueError
		If x is not a complex scalar, this function will not execute.


	# Returns
	
	z: torch.doubleTensor
		The norm of x.
	"""
	if list(x.size())[0] != 2:
		raise ValueError('An input is not of the right dimension.')

	x_conj = torch.zeros_like(x)
	x_conj[0] = x[0]
	x_conj[1] = -x[1]

	return scalar_mult(x_conj, x)[0]
