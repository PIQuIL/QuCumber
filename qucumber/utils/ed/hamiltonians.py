import numpy as np
import argparse
import math as m
from pauli import *

###########################################################

# Transverse-field Ising model

def TransverseFieldIsing(N,hx,J=1.0,PBC=False):
    
    D = 1<<N    # dimension of Hilbert space
    ''' Return the full Hamiltonian '''
    H = np.zeros((D,D))	
    for i in range(N-1):
        H += -J*sigmaZsigmaZ(N,i,i+1)
    for i in range(N):
        H += -hx*sigmaX(N,i)
    
    if (PBC):
        H += -J*sigmaZsigmaZ(N,0,N-1)
    
    return H


###########################################################

# Heisenberg model (with a transverse field)

def Heisenberg(N,hx=0.0,PBC=False):
    
    D = 1<<N    # dimension of Hilbert space
    ''' Return the full Hamiltonian '''
    H = np.zeros((D,D),dtype=complex)	
    for i in range(N-1):
        H += sigmaXsigmaX(N,i,i+1)	
        H += sigmaYsigmaY(N,i,i+1)	
        H += sigmaZsigmaZ(N,i,i+1)	
    for i in range(N):
        H += hx*sigmaX(N,i)
    #if (PBC):
    #    H += sigmaXsigmaX(N,0,N-1)
    #    H += sigmaYsigmaY(N,0,N-1)
    #    H += sigmaZsigmaZ(N,0,N-1)

    return H


###########################################################

# XY model 

def XY(N,PBC=False,hx=None):

    D = 1<<N    # dimension of Hilbert space
    ''' Return the full Hamiltonian '''
    H = np.zeros((D,D))	
    for i in range(N-1):
        H += sigmaXsigmaX(N,i,i+1)	
        H += sigmaYsigmaY(N,i,i+1)	
       
    if (hx is not None):
        for i in range(N):
            H += hx*sigmaX(N,i)
    
    if (PBC):
        H += sigmaXsigmaX(N,0,N-1)
        H += sigmaYsigmaY(N,0,N-1)

    return Ham

