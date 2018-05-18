import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from RBM_helper import spin_config, spin_list, overlapp_fct, RBM

# ------------------------------------------------------------------------------
# GPU and CPU tested on
# python 2.7.15
# torch 0.4.0
# numpy 1.14.2
# ------------------------------------------------------------------------------
# CPU tested on
# python 3.6.4
# torch 0.3.1.post2
# numpy 1.13.3

batch_size = 64
epochs = 20
all_spins = spin_list(10)
gpu = False

filename = 'training_data.txt'
with open(filename, 'r') as fobj:
	data = torch.FloatTensor([[int(num) for num in line.split()] for line in fobj])

filename = 'target_psi.txt'
with open(filename, 'r') as fobj:
	psi = torch.FloatTensor([float(line.split()[0]) for line in fobj])

vis = len(data[0]) #input dimension

rbm = RBM(n_vis = vis, n_hin = 10, k=10, gpu = gpu)
if gpu:
    rbm = rbm.cuda()
    all_spins = all_spins.cuda()
    psi = psi.cuda()

#DUMMY TRAINING SET
# ------------------------------------------------------------------------------
#define a simple training set and check if rbm.draw() returns this after training.
dummy_training = False
if dummy_training:
    data = torch.FloatTensor([[0]*10]*1000) #torch.FloatTensor([[1,0,1,0,1,0,1,0,1,0], [0]*10, [1]*10]*1000)
    test = Variable(torch.FloatTensor([1,1,1,1,1,0,0,0,0,0]))
    psi = Variable(torch.FloatTensor([1/np.sqrt(1)]))
# ------------------------------------------------------------------------------

train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                           shuffle=True)

for epoch in range(epochs):
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                               shuffle=True)
    print(epoch)
    momentum = 1 - 0.1*(epochs-epoch)/epochs #starts at 0.9 and goes up to 1
    lr = (0.1*np.exp(-epoch/epochs*10))+0.0001
    rbm.train(train_loader, lr = lr, momentum = momentum)
    if epoch%1 == 0:
        a = 0
        for i in range(len(psi)):
            a += psi[i]*torch.sqrt(rbm.probability_of_v(all_spins, all_spins[i]))
        print('OVERLAPP:', a.data[0], 'next')


#TRAINING WITH BUILT IN SGD WORKS VIA FREE ENERGY GAP
#--------------------------------------------------------------------------------
#train_op = optim.SGD(rbm.parameters(), lr = 0.1, momentum = 0.95)
#for epoch in range(1000):
#    loss_ = []
#    for _, data in enumerate(train_loader):
#        data = Variable(data.view(-1,vis)) # convert tensor to node in computational graph
#        #sample_data = data.bernoulli()
#        v,v1 = rbm(data) # returns batch before and after Gibbs sampling steps
#        loss = (rbm.free_energy(v).mean() - rbm.free_energy(v1).mean()) # calc difference in free energy before and after Gibbs sampling
#        # .mean() for averaging over whole batch
#        # KLL =~ F(spins) - log (Z), by looking at the difference between F before and after the iteration, one gets rid of Z
#        loss_.append(loss.data[0])
#        train_op.zero_grad() # reset gradient to zero (pytorch normally accumulates them to use it e.g. for RNN)
#        loss.backward() # calc gradient
#        train_op.step() # make one step of SGD
#    print('OVERLAPP:', overlapp_fct(all_spins, all_spins, psi))
#    print( np.mean(loss_))
