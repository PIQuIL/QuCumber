import torch
import numpy as np

def create_dict(name=None, unitary=None):
    dictionary = {'X' : (1./np.sqrt(2))*torch.tensor([ [[1., 1.],[1., -1.]], [[0., 0.],[0., 0.]] ], dtype = torch.double), 
                  'Y' : (1./np.sqrt(2))*torch.tensor([ [[1., 0.],[1., 0.]], [[0.,-1.],[0., 1.]] ], dtype = torch.double), 
                  'Z' : torch.tensor([ [[1., 0.],[0., 1.]], [[0., 0.],[0., 0.]] ], dtype = torch.double)}

    if name != None and unitary != None:
        dictionary[name] = unitary

    return dictionary
