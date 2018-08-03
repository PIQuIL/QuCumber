#from qucumber.rbm import BinomialRBM
import sys
sys.path.append('../')
import torch
import numpy as np
from positive_wavefunction import PositiveWavefunction
from complex_wavefunction import ComplexWavefunction
from quantum_reconstruction import QuantumReconstruction
import importlib.util
import click
import pickle
import test_grads_positive as test_positive
import test_grads_complex as test_complex

@click.group(context_settings={"help_option_names": ['-h', '--help']})
def cli():
    """Simple tool for training an RBM"""
    pass

@cli.command("grad_positive")
@click.option('-k', default=100, show_default=True, type=int,
        help="number of Contrastive Divergence steps")
@click.option('-nc', '--num-chains', default=100,
        show_default=True, type=int)
@click.option('--seed', default=1234, show_default=True, type=int,
        help="random seed to initialize the RBM with")

def grad_positive(k,num_chains,seed):
    with open('data_test.pkl', 'rb') as fin:
        test_data = pickle.load(fin)

    train_samples = torch.tensor(test_data['tfim1d']['train_samples'],dtype = torch.double)
    target_psi=torch.tensor(test_data['tfim1d']['target_psi'],dtype = torch.double)
    nh = train_samples.shape[-1]
    eps = 1.e-6

    nn_state = PositiveWavefunction(num_visible=train_samples.shape[-1],num_hidden=nh, seed=seed)
    qr = QuantumReconstruction(nn_state)
    vis = test_positive.generate_visible_space(train_samples.shape[-1])
    test_positive.run(qr,target_psi,train_samples, vis, eps,k)


@cli.command("grad_complex")
@click.option('-k', default=100, show_default=True, type=int,
        help="number of Contrastive Divergence steps")
@click.option('-nc', '--num-chains', default=100,
        show_default=True, type=int)
@click.option('--seed', default=1234, show_default=True, type=int,
        help="random seed to initialize the RBM with")

def grad_complex(k,num_chains,seed):
    
    with open('data_test.pkl', 'rb') as fin:
        test_data = pickle.load(fin)
    train_bases = test_data['2qubits']['train_bases']
    train_samples = torch.tensor(test_data['2qubits']['train_samples'],dtype = torch.double)
    bases_data = test_data['2qubits']['bases'] 
    target_psi_tmp=torch.tensor(test_data['2qubits']['target_psi'],dtype = torch.double)
    nh = train_samples.shape[-1]
    bases = test_complex.transform_bases(bases_data)
    unitary_dict = test_complex.unitaries.create_dict()
    psi_dict = test_complex.load_target_psi(bases,target_psi_tmp) 
    vis = test_positive.generate_visible_space(train_samples.shape[-1]) 
    nn_state = ComplexWavefunction(num_visible=train_samples.shape[-1],
                                   num_hidden=nh)
    qr = QuantumReconstruction(nn_state)
    eps= 1.e-6
    test_complex.run(qr,psi_dict,train_samples,train_bases,unitary_dict,bases,vis,eps,k) 

if __name__ == '__main__':
    cli()
