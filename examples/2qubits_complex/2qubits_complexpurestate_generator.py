import numpy as np
import sys
import random
import cmath 
import math as m
import argparse

# Local unitaries
H = (1./m.sqrt(2))*np.asarray([[1.,1.],[1.,-1.]])   # Base X
K = (1./m.sqrt(2))*np.asarray([[1.,-1j],[1,1j]])    # Base Y
I = np.asarray([[1.,0.],[0.,1.]])   # Identity


# Print the wavefunction on screen
def printWF(psi,N):

    for i in range(len(psi)):
        state = bin(i)[2:].zfill(N)
        
        print ("Psi("),
        print ("%s" % state.split()[0][0]),
        for j in range(1,N):
            print (", %s" % state.split()[0][j]),
        print (") = %f  + i %f" %
                (psi[i].real,psi[i].imag))

# Write the wavefunction on a file
def writeWF(psi,fout):
    # Each row is made out of two numbers:
    # Real    Imaginary
    # ...     ....
    for i in range(len(psi)):
        fout.write("%.10f " % psi[i].real)
        fout.write("%.10f\n" % psi[i].imag)
    fout.write("\n")

# Write the unitary matrices  on a file
def writeUnitary(U,fout):
    
    # Write the real part of the matrix
    for i in range(len(U)):
        for j in range(len(U[i])):
            fout.write("%.5f " % U[i][j].real)
        fout.write("\n") 
    fout.write("\n")
    
    # Write the imaginary part of the matrix
    for i in range(len(U)):
        for j in range(len(U[i])):
            fout.write("%.5f " % U[i][j].imag)
        fout.write("\n")
    fout.write("\n")

# Build unitary matrix rotating to X in site b
def build_X_rotation(N,b):

    OpList = []
    for k in range(N):
        if (k==b):
            OpList.append(H)
        else:
            OpList.append(I)

    return reduce(np.kron,OpList)

# Build unitary matrix rotating to Y in site b
def build_Y_rotation(N,b):

    OpList = []
    for k in range(N):
        if (k==b):
            OpList.append(K)
        else:
            OpList.append(I)

    return reduce(np.kron,OpList)

# Build a dataset given the psi
def build_dataset(N,psi,Nsamples,fout):

    states = np.zeros((1<<N,N))
    prob = np.zeros((1<<N))
    
    for i in range(1<<N):
        st = bin(i)[2:].zfill(N)
        state = st.split()
        for j in range(N):
            states[i,j] = int(state[0][j])
        
        prob[i] = abs(psi[i])**2
    print prob
    state_index = range(1<<N)
    index_samples = np.random.choice(state_index,
                                     Nsamples,
                                     p=prob)
    for i in range(Nsamples):
        for j in range(N):
            fout.write("%d " % states[index_samples[i]][j])
        fout.write("\n")
        #print states[index_samples[i]]
    #print 

# Build the basis set
def build_basisset(N,Nsamples,fout,basis):

    for i in range(Nsamples):
        for j in range(N):
            fout.write('%c ' % basis[j])
        fout.write('\n')


def RBM_energy(sigma,W,b,c):

    e = 0.

    for j in range(len(b)):
        e += b[j]*sigma[j]

    for i in range(len(c)):
        tmp=0.
        for j in range(len(b)):
            tmp += W[i][j] * sigma[j]
        e += m.log(1.+m.exp(tmp+c[i]))

    return -e


def main():

    N = 2
    Nsamples = 100 
    D = 1<<N
    psi_fout =     open("2qubits_psi.txt",'w')
    basis_fout =   open("2qubits_train_bases.txt",'w')
    data_fout =    open("2qubits_train_samples.txt",'w')
    #unitary_fout = open("2qubits_unitaries.txt",'w')

    random.seed(1234)

    Norm = 0.0
    Psi = np.zeros((D),dtype=complex)

    # GENERATE WAVEFUNCTION ON THE sZ BASIS
    #for i in range(D):
   
    #    ## RANDOM REAL COEFFICIENTS
    #    #Psi[i] = random.uniform(-1,1)
    #    
    #    # RANDOM COMPLEX COEFFICIENTS
    #    Psi[i] = random.uniform(0,1)
    #    phi = random.uniform(0,2*m.pi)
    #    Psi[i] *= cmath.exp(1j*phi)
    #    #Psi[i] = cmath.exp(1j*phi)    
    #    Norm += Psi[i]*np.conjugate(Psi[i])
    #Psi /= m.sqrt(Norm)

    ## RBM-like state
    #b_lambda = np.asarray([0.1,-0.74])
    #c_lambda = np.asarray([-0.4,-1.2])
    #W_lambda = np.asarray([[-0.8,0.3],[0.5,-0.2]])
    #b_mu = np.asarray([0.31,-0.4])
    #c_mu = np.asarray([-0.45,-0.2])
    #W_mu = np.asarray([[-1.0,0.1],[-0.5,0.48]])
    #
    #Norm = 0.0
    #sigma = np.zeros((N))
    #for i in range(1<<N):
    #    st = bin(i)[2:].zfill(N)
    #    state = st.split()
    #    for j in range(N):
    #        sigma[j] = int(state[0][j])
    #    Psi[i] = cmath.exp(-0.5*(RBM_energy(sigma,W_lambda,b_lambda,c_lambda)+1j*RBM_energy(sigma,W_mu,b_mu,c_mu)))
    #    Norm += Psi[i]*np.conjugate(Psi[i])
    #Psi /= m.sqrt(Norm)

    print Psi

    ## Hand-crafted state
    Psi[0] = 0.5
    Psi[1] = -0.5
    Psi[2] = 1j*0.5
    Psi[3] = -1j*0.5
    
    basis = []

    # Reference basis
    for i in range(N):
        basis.append("Z")

    # Save psi in sZ basis
    writeWF(Psi,psi_fout)
    
    # Build data
    build_basisset(N,Nsamples,basis_fout,basis)
    build_dataset(N,Psi,Nsamples,data_fout)

    # BASES X 
    for j in range(N):
        basis = []
        for i in range(N):
            if (i==j):
                basis.append('X')
            else:
                basis.append('Z')
        
        U = build_X_rotation(N,j)
        psi_b = U.dot(Psi)
        #writeUnitary(U,unitary_fout)
        writeWF(psi_b,psi_fout)
        build_basisset(N,Nsamples,basis_fout,basis)
        build_dataset(N,psi_b,Nsamples,data_fout)
    
    # BASES Y 
    for j in range(N):
        basis = []
        for i in range(N):
            if (i==j):
                basis.append('Y')
            else:
                basis.append('Z')
        
        U = build_Y_rotation(N,j)
        psi_b = U.dot(Psi)
        #writeUnitary(U,unitary_fout)
        writeWF(psi_b,psi_fout)
        build_basisset(N,Nsamples,basis_fout,basis)
        build_dataset(N,psi_b,Nsamples,data_fout)
   
    basis_fout.close()
    psi_fout.close()
    #unitary_fout.close()
    data_fout.close()
 
if __name__ == "__main__":
    main()
	

