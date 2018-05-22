import sys
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from RBM_helper import spin_config, spin_list, overlapp_fct, RBM
import gzip
import pickle

"""
To run this file call it like that:

python RBM_pytorch_benchmark_runner.py data/Ising2d_L4.pkl.gz 32 100 100 10 0.1 0.9 0

or in ipython

%run RBM_pytorch_benchmark_runner.py data/Ising2d_L4.pkl.gz 32 100 100 10 0.1 0.9 0

Order of the parameters:

filename    data/path   batch_size  epochs  hidden_units    learning_rate   momentum    gpu_usage(0= False / 1 = True)
"""

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

if len(sys.argv) != 9:
    sys.exit('arguments missing! 1. filname, 2. batch_size, 3. epochs, 4. hidden_units, 5. k, 6. learning rate, 7. momentum, 8. gpu (0 / 1)')
filename = sys.argv[1]
batch_size = int(sys.argv[2])
epochs = int(sys.argv[3])
hidden_units = int(sys.argv[4])
k = int(sys.argv[5])
lr = float(sys.argv[6])
momentum = float(sys.argv[7])
gpu = int(sys.argv[8])


#filename = 'data/Ising2d_L4.pkl.gz'
with gzip.open(filename, 'rb') as f:
    if sys.version_info[0] < 3:
        data = pickle.load(f) # python 2
    else:
        data = pickle.load(f, encoding = 'latin1') #python3


vis = len(data[0]) #input dimension

rbm = RBM(n_vis = vis, n_hin = hidden_units, k=k, gpu = gpu)
if gpu:
    rbm = rbm.cuda()
    all_spins = all_spins.cuda()

train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                           shuffle=True)

for epoch in range(epochs):
    rbm.train(train_loader, lr = lr, momentum = momentum)


