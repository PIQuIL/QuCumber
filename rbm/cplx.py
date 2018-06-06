import torch

def make_cplx(x,y):
	'''A function that takes two tensors (a real (x) and imaginary part (y)) and returns the combined complex tensor.'''
	if x.size()[0] != y.size()[0] or x.size()[1] != y.size()[1]:
		raise ValueError('Input tensors do not have the same dimension.')
	
	z = torch.zeros(2, x.size()[0], x.size()[1])
	z[0] = x
	z[1] = y

	return z

def cplx_SS(x, y):
	'''A function that does complex scalar multiplication.'''
	if list(x.size())[0] < 2 or list(y.size())[0] < 2:
		raise ValueError('An input is not of the right dimension.')

	z = torch.zeros(2)	
	z[0] = x[0]*y[0] - x[1]*y[1]
	z[1] = x[0]*y[1] + x[1]*y[0]

	return z

def cplx_VS(x, y):
	'''A function that returns x*y, where x is a complex scalar and y is a complex vector.'''
	if list(x.size())[0] < 2 or list(y.size())[0] < 2:
		raise ValueError('An input is not of the right dimension.')	

	z = torch_zeros_like(y)
	z[0] = x[0]*y[0] - x[1]*y[1]
	z[1] = x[0]*y[1] + x[1]*y[0]

	return z

def cplx_divideVS(x, y):
	'''A function that returns x/y, where x is a complex scalar and y is a complex vector.'''
	if list(x.size())[0] < 2 or list(y.size())[0] < 2:
		raise ValueError('An input is not of the right dimension.')

	x_star = torch.zeros_like(x)
	x_star[0] = x[0]
	x_star[1] = -x[1]

	denominator = cplx_SS(x, x_star)[0] # should only contain a real part
	numerator   = cplx_VS(x_star, y)

	return numerator / denominator


def cplx_MS(x, y):
	'''A function that takes a given input complex matrix (y) and multiplies it by a complex scalar (x).'''i
	if len(list(y.size())) < 3:
		raise ValueError('An input is not of the right dimension.')
	
	z = torch.zeros_like(y)
	z[0] = x[0]*y[0] - x[1]*y[1]
	z[1] = x[0]*y[1] + x[1]*y[0]

	return z

def cplx_MV(x, y):
	'''A function that returns x*y, where x is a complex tensor and y is a complex vector.''' 
	if len(list(x.size())) < 3 or len(list(y.size())) < 2:
		raise ValueError('An input is not of the right dimension.')

	z = torch.zeros(2, x.size()[1])
	z[0] = torch.mv(x[0],y[0]) - torch.mv(x[1],y[1])
	z[1] = torch.mv(x[0],y[1]) + torch.mv(x[1],y[0])

	return z

def cplx_MM(x, y):
	'''A function that returns x*y, where x and y are complex tensors.'''
	if len(list(x.size())) < 3 or len(list(y.size())) < 3:
		raise ValueError('An input is not of the right dimension.')

	z = torch.zeros(2, x.size()[1], y.size()[2])
	z[0] = torch.matmul(x[0],y[0]) - torch.matmul(x[1], y[1])
	z[1] = torch.matmul(x[0],y[1]) + torch.matmul(x[1], y[0])

	return z

def cplx_ADD(x, y):
	'''A function that adds two complex tensors or vectors, x and y.'''
	if len(list(x.size())) < 2 or len(list(y.size())) < 2:
		raise ValueError('An input is not of the right dimension.')

	z = torch.zeros_like(x)
	z[0] = x[0] + y[0]
	z[1] = x[1] + y[1]

	return z
	
def cplx_DOT(x,y):
	'''A function that returns the dot product of two complex vectors, x and y.'''	
	if len(list(x.size())) < 2 or len(list(y.size())) < 2:
		raise ValueError('An input is not of the right dimension.')
	
	z = torch.zeros(2)
	z[0] = torch.dot(x[0], y[0]) - torch.dot(x[1], y[1])
	z[1] = torch.dot(x[0], y[1]) + torch.dot(x[1], y[0])
	
	return z

def cplx_OUTER(x,y):
	'''A function that returns the outer product of two complex vectors, x and y.'''
	if len(list(x.size())) < 2 or len(list(y.size())) < 2:
		raise ValueError('An input is not of the right dimension.')

	z = torch.zeros(2, x.size()[1], y.size()[1])
	z[0] = torch.ger(x[0], y[0]) - torch.ger(x[1], y[1])
	z[1] = torch.ger(x[0], y[1]) + torch.ger(x[1], y[0])

	return z

def cplx_TRANSPOSE(x):
	'''A function that returns the complex transpose of a complex tensor or vector, x.'''
	if len(list(x.size())) < 3:
		raise ValueError('An input is not of the right dimension.')

	z = torch.zeros(2, x.size()[2], x.size()[1])
	z[0] = torch.transpose(x[0], 0, 1) 
	z[1] = -torch.transpose(x[1], 0, 1) 

	return z
