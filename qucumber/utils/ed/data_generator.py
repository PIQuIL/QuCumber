import numpy as np
import sys
import cmath 
import math as m
from functools import reduce
from pauli import *

local_rotation = {
        "X": rotationX,
        "Y": rotationY,
        "Z": I
        }

def UnitaryRotation(N,basis):
    OpList = []
    for k in range(N):
        for key in local_rotation.keys():
            if (basis[k] == key):
                OpList.append(local_rotation[key])
    return reduce(np.kron,OpList)

def RotateWavefunction(N,psi,basis):
    psi_b=0
    U = UnitaryRotation(N,basis)
    psi_b = U.dot(psi)
    return psi_b


def GenerateBasisSet(name_code,N):

    basis_set = []
    tmp = []
    for j in range(N):
        tmp.append('Z')
    basis_set.append(tmp)
    if (name_code == 'x'):
        for j in range(N):
            tmp = []
            for j2 in range(N):
                if (j==j2):
                    tmp.append('X')
                else:
                    tmp.append('Z')
            basis_set.append(tmp)
    if (name_code == 'xy1' or name_code == 'xy2nn'):
        for b in ['X','Y']:
            for j in range(N):
                tmp = []
                for j2 in range(N):
                    if (j==j2):
                        tmp.append(b)
                    else:
                        tmp.append('Z')
                basis_set.append(tmp)
     
    if (name_code == 'xy2nn'):
        for b1 in ['X','Y']:
            for b2 in ['X','Y']:
                for i in range(N-1):
                    tmp = []
                    for j in range(N-1):
                        if (i==j):
                            tmp.append(b1)
                            tmp.append(b2)
                        else:
                            tmp.append('Z')
                    basis_set.append(tmp)   
    return basis_set


# Generate the training dataset
def GenerateSamples(N,psi,Nsamples,fout=None):

    D = 1<<N
    config = np.zeros((D,N))
    samples = np.zeros((Nsamples,N))
    psi2 = np.zeros((D),dtype='float32')  
    
    # Build all spin states and psi^2
    for i in range(D):
        state = (bin(i)[2:].zfill(N)).split()
        for j in range(N):
            config[i,j] = int(state[0][j])
        psi2[i] = (psi[i]*np.conj(psi[i])).real
    config_index = range(D)
    # Generate the trainset
    index_samples = np.random.choice(config_index,
                                     Nsamples,
                                     p=psi2)
    for i in range(Nsamples):
        samples[i] = config[index_samples[i]]
        for j in range(N):
            if (fout is not None):
                fout.write("%d " % config[index_samples[i]][j])
        if (fout is not None):
            fout.write("\n")
    return samples

# Build the basis set
def GenerateBases(N,Nsamples,basis,fout=None):
    bases = []
    for i in range(Nsamples):
        tmp = []
        for j in range(N):
            tmp.append(basis[j])
            if fout is not None:
                fout.write('%c ' % basis[j])
        if fout is not None:
            fout.write('\n')
        bases.append(tmp)
    
    
    return bases
#def GenerateDataset(N,Nsamples
