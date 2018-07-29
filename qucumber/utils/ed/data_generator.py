import numpy as np
import sys
import random
import cmath 
import math as m
from functools import reduce
from pauli import *
from scipy import special
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
    tmp=0
    for i in range(N):
        if basis[i] != 'Z':
            tmp = 1
            break
    if tmp == 0:
        psi_b = psi
    else:
        U = UnitaryRotation(N,basis)
        psi_b = U.dot(psi)
    return psi_b


def GenerateDataChosenBases(N,Nsamples,psi,bases_set,fout=None):
    train_bases = []
    train_samples = []
    for b in range(len(bases_set)):
        psi_b = RotateWavefunction(N,psi,bases_set[b])
        train_bases += GenerateBases(N,Nsamples,bases_set[b],fout)
        train_samples += GenerateSamples(N,psi_b,Nsamples,fout)
            
    
    return np.asarray(train_samples,dtype=float),train_bases













def GenerateDataRandomBases(N,q,Nsamples,psi,fout=None):

    samples_per_basis = 10
    num_bases_q = []
    prob_q = []
    tot = 0
    for k in range(q+1):
        num_bases_q.append(special.binom(N,k)*pow(2,k))
        tot += num_bases_q[k]
    prob_q = np.asarray(num_bases_q,dtype=float)/float(tot)
    
    number_of_rotations = np.random.choice(range(q+1),
                                     Nsamples,
                                     p=prob_q)

    bases = []
    samples = np.zeros((Nsamples,N))
    for x in range(int(Nsamples/samples_per_basis)):
        num_nontrivial_U = number_of_rotations[x]
        bases.append(draw_random_basis(N,num_nontrivial_U))
        RotateWavefunction(N,psi,bases[x])
        #print(num_nontrivial_U,basis)
        samples_tmp = GenerateSamples(N,psi,samples_per_basis,fout)
        if fout is not None:
            for i in range(Nsamples):
                for j in range(N):
                    fout.write('%c ' % bases[x][j])
                fout.write('\n')
        for y in range(samples_per_basis):
            samples[x*samples_per_basis+y] = samples_tmp[y]

    return samples,bases

def GenerateDataSet(N,Nsamples,psi,name_code = None,fout=None):
    if name_code is None:
        train_samples = GenerateSamples(N,psi,Nsamples)
        train_bases = None
    
    elif name_code == 'random':
        train_samples = 0.0
        train_bases = 0.0
    
    else:
        bases_set = CreateBasisSet(name_code,N)
        train_samples,train_bases = GenerateDataChosenBases(N,Nsamples,psi,bases_set)
    
    return {'samples':np.asarray(train_samples,dtype=float),'bases':train_bases}



def CreateBasisSet(name_code,N):

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
    #samples = np.zeros((Nsamples,N))
    samples = []
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
        samples.append(config[index_samples[i]])
        #samples[i] = config[index_samples[i]]
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


def draw_random_basis(N,q):

    basis = []
    for i in range(N):
        basis.append('Z')
    j=0
    while(j<q):
        site = random.randint(0,N-1)
        if (basis[site] == 'Z'):
            if (random.random() > 0.5):
                basis[site] = 'X'
            else:
                basis[site] = 'Y'
            j += 1
    return basis



