import numpy as np
import sys
import cmath 
import math as m
from functools import reduce

###########################################################
# LIST OF 1-QUBIT OPERATORS

# Identity Matrix
I = np.asarray([[1.,0.],[0.,1.]])

# Pauli X Matrix
X = np.asarray([[0.,1.],[1.,0.]])

# Pauli Y Matrix
Y = np.asarray([[0.,-1j],[1j,0.]])

# Pauli Z Matrix
Z = np.asarray([[1.,0.],[0.,-1.]])

# Rotation into the Pauli X basis 
rotationX = 1./(m.sqrt(2))*np.asarray([[1.,1.],[1.,-1.]])

# Rotation into the Pauli Y basis
rotationY= 1./(m.sqrt(2))*np.asarray([[1.,-1j],[1.,1j]])

# Rotation of theta around Z
def R_Z(theta):
    return np.asarray([[1.,0.],[0.,cmath.exp(-1j*theta)]])


###########################################################
# LIST OF 2-QUBITS OPERATORS

# Controlled-NOT
CX = np.asarray([[1.,0.,0.,0.],
                 [0.,1.,0.,0.],
                 [0.,0.,0.,1.],
                 [0.,0.,1.,0.]])

###########################################################
# N-QUBITS OPERATORS FUNCTIONS

# Pauli X on site i of N qubits
def sigmaX(N,i):
    ''' Return the many-body operator
        I x I x .. x Sx x I x .. x I
        with Sx acting on qubit i '''
    op = 1.0
    OpList = []
    for k in range(N):
        if (k == i):
            OpList.append(X)
            op = np.kron(op,X)
        else:
            OpList.append(I)
            op = np.kron(op,I)
    #print(out
    #return op
    return reduce(np.kron,OpList)

# Pauli Y on site i of N qubits
def sigmaY(N,i):
    ''' Return the many-body operator
        I x I x .. x Sy x I x .. x I
        with Sx acting on qubit i '''
    OpList = []
    for k in range(N):
        if (k == i):
            OpList.append(Y)
        else:
            OpList.append(I)

    return reduce(np.kron,OpList)

# Pauli Z on site i of N qubits
def sigmaZ(N,i):
    ''' Return the many-body operator
        I x I x .. x Sz x I x .. x I
        with Sx acting on qubit i '''
    OpList = []
    for k in range(N):
        if (k == i):
            OpList.append(Z)
        else:
            OpList.append(I)

    return reduce(np.kron,OpList)

# Pauli X on site i and site j of N qubits
def sigmaXsigmaX(N,i,j):
    ''' Return the many-body operator
        I x .. x Sx x I x Sx x I
        with Sx acting on qubit i and j '''
    op = 1.0
    OpList = []
    for k in range(N):
        if (k == i):
            op = np.kron(op,X)
            OpList.append(X)
        elif (k == j):
            op = np.kron(op,X)
            OpList.append(X)
        else:
            op = np.kron(op,I)
            OpList.append(I)
    return op
    #return reduce(np.kron,OpList)

# Pauli Y on site i and site j of N qubits
def sigmaYsigmaY(N,i,j):
    ''' Return the many-body operator
        I x .. x Sy x I x Sy x I
        with Sx acting on qubit i and j '''
    op = 1.0
    OpList = []
    for k in range(N):
        if (k == i):
            op = np.kron(op,Y)
            OpList.append(Y)
        elif (k == j):
            op = np.kron(op,Y)
            OpList.append(Y)
        else:
            op = np.kron(op,I)
            OpList.append(I)

    return op
    #return reduce(np.kron,OpList)

# Pauli Z on site i and site j of N qubits
def sigmaZsigmaZ(N,i,j):
    ''' Return the many-body operator
        I x .. x Sz x I x Sz x I
        with Sx acting on qubit i and j '''
    OpList = []
    for k in range(N):
        if (k == i):
            OpList.append(Z)
        elif (k == j):
            OpList.append(Z)
        else:
            OpList.append(I)

    return reduce(np.kron,OpList)


# Controlled-NOT on site i of N qubits
def ControlledNot(N,i):
    OpList = []
    for k in range(N-1):
        if (k==i):
            OpList.append(CX)
        else:
            OpList.append(I)

    return reduce(np.kron,OpList)

# Z rotation of theta on site i of N qubits
def getR_Z(N,i,theta):
    
    OpList = []
    for k in range(N):
        if (k==i):
            OpList.append(R_Z(theta))
        else:
            OpList.append(I)

    return reduce(np.kron,OpList)

