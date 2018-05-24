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
import click
print("Imported all libraries")

"""
EXAMPLE HOW To run this file:

In Ipython

%run Pytorch_final_with_click.py run_training -e 10 -n 10 -b 32 -k 10 -lr 0.1 -m 0.9 --seed 1111 --gpu False --train-path data/Ising2d_L16.pkl.gz

Parameters:

-e epochs  -n hidden_units  -b batch size   -k number of gibbs samplings    -l learning_rate   -m momentum  -seed random seed  -gpu gpu_usage 
"""
def load_train(L):
    test = np.load("/home/mbeach/configs%s.npy" %(L))
    test = test.reshape(test.shape[0], L*L)
    return test.astype('float32')

def load_train_pickle(train_path):
    with gzip.open(train_path, 'rb') as f:
        if sys.version_info[0] < 3:
            data = pickle.load(f) # python 2
        else:
            data = pickle.load(f, encoding = 'latin1') #python3
    return data


@click.group(context_settings={"help_option_names": ['-h', '--help']})
def cli():
    """Simple tool for training an RBM"""
    pass


@cli.command("run_training")
@click.option('--train-path', default='data/Ising2d_L4.pkl.gz',
              show_default=True, type=click.Path(exists=True),
              help="path to the training data")
@click.option('-n', '--num_hidden', default=10, type=int,
              help=("number of hidden units in the RBM; defaults to "
                    "number of visible units"))
@click.option('-size', '--size', default=12, type=int,
              help=("system size"))
@click.option('-e', '--epochs', default=1000, show_default=True, type=int)
@click.option('-b', '--batch_size', default=32, show_default=True, type=int)
@click.option('-k', default=1, show_default=True, type=int,
              help="number of Contrastive Divergence steps")
@click.option('-lr', '--learning_rate', default=1e-3,
              show_default=True, type=float)
@click.option('-m', '--momentum', default=0.5, show_default=True, type=float,
              help=("value of the momentum parameter; ignored if "
                    "using SGD or Adam optimization"))
@click.option('--seed', default=1234, show_default=True, type=int,
              help="random seed to initialize the RBM with")
@click.option('--gpu', default=False, show_default=True, type=bool,
              help="boolean that makes RBM run on GPU")
def run_training(train_path, num_hidden, size, epochs, batch_size, k, learning_rate, momentum, seed, gpu):
    torch.manual_seed(seed)
    if gpu:
        torch.cuda.manual_seed(seed)

#    data = load_train(L)
    data = load_train_pickle(train_path)
    print("Loaded all datasets")
    vis = len(data[0]) #input dimension

    rbm = RBM(n_vis = vis, n_hin = num_hidden, k=k, gpu = gpu)
    if gpu:
        rbm = rbm.cuda()
        data = torch.FloatTensor(data)
        data = Variable(data)
        data = data.cuda()
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                               shuffle=True)
    print("Begining training ... ")
    for epoch in range(epochs):
        print(epoch)
        rbm.train(train_loader, lr = learning_rate, momentum = momentum)
    print("Done everything")

if __name__ == '__main__':
    cli()


