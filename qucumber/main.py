import utils.unitaries as unitaries
from positive_wavefunction import PositiveWavefunction
from complex_wavefunction import ComplexWavefunction
from quantum_reconstruction import QuantumReconstruction
from callbacks import MetricEvaluator
import click
import numpy as np
import torch
import utils
import pickle
import sys
sys.path.append("utils/")
import training_statistics as ts
from data import load_data,extract_refbasis_samples

@click.group(context_settings={"help_option_names": ['-h', '--help']})
def cli():
    """Simple tool for training an RBM"""
    pass

def load_params(param_file):
    with open(param_file, 'rb+') as f:
        param_dict = torch.load(f)
    return param_dict

# REAL POSITIVE WAVEFUNCTION
@cli.command("train_real")
@click.option('--tr-samples-path', default='../examples/01_Ising/tfim1d_N10_train_samples.txt',
              show_default=True, type=click.Path(exists=True),
              help="path to the training data")
@click.option('--target-psi-path', default='../examples/01_Ising/tfim1d_N10_psi.txt',
              show_default=True, type=click.Path(exists=True),
              help=("path to the file containing the true wavefunctions "
                    "in each basis."))

@click.option('-nh', '--num-hidden', default=None, type=int,
              help=("number of hidden units in the RBM; defaults to "
                    "number of visible units"))
@click.option('-e', '--epochs', default=1000, show_default=True, type=int)
@click.option('-bs', '--batch-size', default=100,
              show_default=True, type=int)
@click.option('-nc', '--num-chains', default=100,
              show_default=True, type=int)
@click.option('-k', default=10, show_default=True, type=int,
              help="number of Contrastive Divergence steps")
@click.option('-lr', '--learning-rate', default=1e-1,
              show_default=True, type=float)
@click.option('--seed', default=1234, show_default=True, type=int,
              help="random seed to initialize the RBM with")
@click.option('--no-prog', is_flag=True)
def train_real(tr_samples_path,target_psi_path,num_hidden, epochs, batch_size,
               num_chains, k, learning_rate, seed, no_prog):
    """Train an RBM without any phase."""
    train_samples,target_psi = load_data(tr_samples_path,target_psi_path) 
    num_visible = train_samples.shape[-1]
    num_hidden = train_samples.shape[-1] if num_hidden is None else num_hidden
    nn_state = PositiveWavefunction(num_visible=train_samples.shape[-1],
                      num_hidden=num_hidden, seed=seed)
    qr = QuantumReconstruction(nn_state)
    if (num_visible <20):
        nn_state.space = nn_state.generate_Hilbert_space(num_visible)
        callbacks = [MetricEvaluator(10,{'Fidelity':ts.fidelity,'KL':ts.KL},target_psi=target_psi)] 
    qr.fit(train_samples, epochs, batch_size, num_chains, k,
            learning_rate, progbar=no_prog,callbacks = callbacks)

    print('\nFinished training. Saving results...')
    nn_state.save('saved_params.pkl')
    print('Done.')
    

# COMPLEX WAVEFUNCTION
@cli.command("train_complex")
@click.option('--tr-samples-path', default='../examples/02_qubits/qubits_train_samples.txt',
              show_default=True, type=click.Path(exists=True),
              help="path to the training data")
@click.option('--tr-bases-path', default='../examples/02_qubits/qubits_train_bases.txt',
              show_default=True, type=click.Path(exists=True),
              help="path to the training bases")
@click.option('--target-psi-path', default='../examples/02_qubits/qubits_psi.txt',
              show_default=True, type=click.Path(exists=True),
              help=("path to the file containing the target wavefunctions "))
@click.option('--bases-path', default='../examples/02_qubits/qubits_bases.txt',
              show_default=True, type=click.Path(exists=True),
              help="path to the set of  bases")
@click.option('-nh', '--num-hidden', default=2, type=int,
              help=("number of hidden units; defaults to "
                    "number of visible units"))
@click.option('-bs', '--batch-size', default=100,
              show_default=True, type=int)
@click.option('-nc', '--num-chains', default=100,
              show_default=True, type=int)
@click.option('-e', '--epochs', default=1000, show_default=True, type=int)
@click.option('-k', default=10, show_default=True, type=int,
              help="number of Contrastive Divergence steps")
@click.option('-lr', '--learning-rate', default=1e-1,
              show_default=True, type=float)
@click.option('--log-every', default=0, show_default=True, type=int,
              help=("how often the validation statistics are recorded, "
                    "in epochs; 0 means no logging"))
@click.option('--seed', default=1234, show_default=True, type=int,
              help="random seed to initialize the RBM with")
@click.option('--test-grads', is_flag=True)
@click.option('--no-prog', is_flag=True)

def train_complex(tr_samples_path,tr_bases_path,target_psi_path,bases_path,num_hidden,epochs, batch_size, num_chains, k, learning_rate,
                  log_every, seed, test_grads, no_prog):
    
    """Train an RBM with a phase."""

    train_samples,target_psi,train_bases,bases = load_data(tr_samples_path,target_psi_path,tr_bases_path,bases_path)
    num_bases = len(bases)
    unitary_dict = unitaries.create_dict()
    
    num_visible      = train_samples.shape[-1]
    num_hidden   = (train_set.shape[-1]
                        if num_hidden is None
                        else num_hidden)

    nn_state = ComplexWavefunction(num_visible=num_visible,
                               num_hidden=num_hidden)
   
    
    z_samples = extract_refbasis_samples(train_samples,train_bases)

    if (num_visible <20):
        nn_state.space = nn_state.generate_Hilbert_space(num_visible)
        callbacks = [MetricEvaluator(1,{'Fidelity':ts.fidelity,'KL':ts.KL},target_psi=target_psi,bases=bases)]

    qr = QuantumReconstruction(nn_state)
    qr.fit(train_samples, epochs, batch_size, num_chains, k,
            learning_rate,callbacks = callbacks,input_bases = train_bases,progbar=(no_prog),z_samples=z_samples)



if __name__ == '__main__':
    cli()
