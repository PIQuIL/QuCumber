import os

import numpy as np


# Write the wavefunction on a file
def WriteWavefunction(psi,fout):
    # Each row is made out of two numbers:
    # Real    Imaginary
    # ...     ....
    for i in range(len(psi)):
        fout.write("%.10f " % psi[i].real)
        fout.write("%.10f\n" % psi[i].imag)
    fout.write("\n")
 
# Write a basis on a file
def WriteBases(N,basis,fout):
    # Example:
    # Z Z Z X Z Y Z X .. 
    for n in range(len(basis)):
        for i in range(N):
            fout.write('%c ' % basis[n][i])
        fout.write('\n')
    

# Build the density matrix of a state Psi
def BuildDensityMatrix(N,Psi):
    
    D = 1 << N
    rho = np.zeros((D,D),dtype=complex)

    for i in range(D):
        for j in range(D):
            rho[i,j] = Psi[i]*np.conjugate(Psi[j])
    
    return rho

# Compute the reduced density matrix

def ComputeReducedDensityMatrix(N,rho,l):

    n = N-l
    d = 2**l

    bra0 = np.asarray([1,0])
    bra1 = np.asarray([0,1])
    ket0 = np.asarray([[1],[0]])
    ket1 = np.asarray([[0],[1]])
    
    rhoA = np.zeros((d,d),dtype=complex)

    states = np.zeros((n))
    
    for i in range(1<<n):
        st = bin(i)[2:].zfill(n)
        state = st.split()
        
        OpList = []
        
        for j in range(n):
            if (int(state[0][j]) == 0):
                OpList.append(ket0)
            else:
                OpList.append(ket1)
         
        basisState = reduce(np.kron,OpList)

        ketMat = np.kron(basisState,np.identity(2**(N-n)))
        braMat = np.transpose(ketMat)
        rhoA += np.dot(braMat,np.dot(rho,ketMat))

    return rhoA
