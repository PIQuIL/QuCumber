from rbm_complex import RBM
import click
import gzip
import pickle
import csv
import numpy as np
import torch
import cplx

@click.group(context_settings={"help_option_names": ['-h', '--help']})
def cli():
    """Simple tool for training an RBM"""
    pass

@cli.command("train")
@click.option('--train-path', default='../benchmarks/data/2qubits_complex/2qubits_train_samples.txt',
              show_default=True, type=click.Path(exists=True),
              help="path to the training data")
@click.option('--basis-path', default='../benchmarks/data/2qubits_complex/2qubits_train_bases.txt',
              show_default=True, type=click.Path(exists=True),
              help="path to the basis data")
@click.option('-n', '--num-hidden-amp', default=None, type=int,
              help=("number of hidden units in the amp RBM; defaults to "
                    "number of visible units"))
@click.option('-n', '--num-hidden-phase', default=None, type=int,
              help=("number of hidden units in the phase RBM; defaults to "
                    "number of visible units"))
@click.option('-e', '--epochs', default=100, show_default=True, type=int)
@click.option('-b', '--batch-size', default=100, show_default=True, type=int)
@click.option('-k', default=1, show_default=True, type=int,
              help="number of Contrastive Divergence steps")
@click.option('-l', '--learning-rate', default=1e-3,
              show_default=True, type=float)
@click.option('-m', '--momentum', default=0.5, show_default=True, type=float,
              help=("value of the momentum parameter; ignored if "
                    "using SGD or Adam optimization"))
@click.option('--L1', default=0, show_default=True, type=float,
              help="L1 regularization parameter")
@click.option('--L2', default=0, show_default=True, type=float,
              help="L2 regularization parameter")
@click.option('--log-every', default=10, show_default=True, type=int,
              help=("how often the validation statistics are recorded, "
                    "in epochs; 0 means no logging"))
@click.option('--seed', default=1234, show_default=True, type=int,
              help="random seed to initialize the RBM with")

def train(train_path, basis_path, num_hidden_amp, num_hidden_phase, epochs, batch_size,
          k, learning_rate, momentum, L1, L2,
          seed, log_every):
    """Train an RBM"""

    data = np.loadtxt(train_path, dtype= 'float32')

    train_set = torch.tensor(data, dtype = torch.double)
    basis_set = np.loadtxt(basis_path, dtype = str)

    f = open('unitary_library.txt')
    num_rows = len(f.readlines())
    f.close()

    '''A function that pytrochifies the unitary matrix given its name. It must be in the unitary_library!'''
    unitary_dictionary = {}
    with open('unitary_library.txt', 'r') as f:
        for i, line in enumerate(f):
            if i % 5 == 0:
                a = torch.tensor(np.genfromtxt('unitary_library.txt', delimiter='\t', skip_header = i+1, 
                                 skip_footer = num_rows - i - 3), dtype = torch.double)
                b = torch.tensor(np.genfromtxt('unitary_library.txt', delimiter='\t', skip_header = i+3, 
                                 skip_footer = num_rows - i - 5), dtype = torch.double)
                character = line.strip('\n')
                unitary_dictionary[character] = cplx.make_complex_matrix(a,b)
    f.close()

    basis_list = ['Z' 'Z', 'X' 'Z', 'Z' 'X', 'Y' 'Z', 'Z' 'Y']
    psi_file = np.loadtxt('../benchmarks/data/2qubits_complex/2qubits_psi.txt')
    psi_dictionary = {}

    for i in range(len(basis_list)): # 5 possible wavefunctions: ZZ, XZ, ZX, YZ, ZY
        psi      = torch.zeros(2, 2**train_set.shape[-1], dtype = torch.double)
        psi_real = torch.tensor(psi_file[i*4:(i*4+4),0], dtype = torch.double)
        psi_imag = torch.tensor(psi_file[i*4:(i*4+4),1], dtype = torch.double)
        psi[0]   = psi_real
        psi[1]   = psi_imag
        psi_dictionary[basis_list[i]] = psi
        
    num_hidden_amp   = train_set.shape[-1] if num_hidden_amp is None else num_hidden_amp
    num_hidden_phase = train_set.shape[-1] if num_hidden_phase is None else num_hidden_phase

    rbm = RBM(unitaries=unitary_dictionary, psi_dictionary=psi_dictionary, num_visible=train_set.shape[-1],
              num_hidden_amp=num_hidden_amp,
              num_hidden_phase=num_hidden_phase,
              seed=seed)

    rbm.train(train_set, basis_set, epochs,
              batch_size, k=k,
              lr=learning_rate,
              momentum=momentum,
              L1_reg=L1, L2_reg=L2,
              log_every=log_every,
              )

if __name__ == '__main__':
    cli()
