import utils.unitaries as unitaries
from positive_wavefunction import PositiveWavefunction
from complex_wavefunction import ComplexWavefunction
from quantum_reconstruction import QuantumReconstruction
import click
import numpy as np
import torch
import utils


@click.group(context_settings={"help_option_names": ['-h', '--help']})
def cli():
    """Simple tool for training an RBM"""
    pass

def load_params(param_file):
    with open(param_file, 'rb+') as f:
        param_dict = torch.load(f)
    return param_dict

# REAL PURE WAVEFUNCTION
@cli.command("train_real")
@click.option('--train-path', default='../examples/quantum_ising/tfim1d_N10_train_samples.txt',
              show_default=True, type=click.Path(exists=True),
              help="path to the training data")
@click.option('--true-psi-path', default='../examples/quantum_ising/tfim1d_N10_psi.txt',
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
@click.option('--no-prog', is_flag=False)
def train_real(train_path, true_psi_path, num_hidden, epochs, batch_size,
               num_chains, k, learning_rate, seed, no_prog):
    """Train an RBM without any phase."""
    train_set = torch.tensor(np.loadtxt(train_path, dtype='float32'),dtype=torch.double)

    num_hidden = 10#train_set.shape[-1] if num_hidden is None else num_hidden

    psi = load_target_psi(train_set.shape[-1],true_psi_path)#path_to_target_psi) 

    nn_state = PositiveWavefunction(num_visible=train_set.shape[-1],
                      num_hidden=num_hidden, seed=seed)
    print(nn_state.rbm_am.weights)
    print(nn_state.rbm_am.visible_bias)
    print(nn_state.rbm_am.hidden_bias)

#    nn_state.save('train_benchmark_params_real.pkl')
    
    #nn_state.fit(train_set, epochs, batch_size, num_chains, k=k,
            #lr=learning_rate, progbar=(not no_prog))
    qr = QuantumReconstruction(nn_state)
    
    qr.fit(train_set, epochs, batch_size, num_chains, k=k,
            lr=learning_rate, progbar=(not no_prog),target_psi=psi)

    
    print('Finished training. Saving results...')
    nn_state.save('saved_params_real.pkl')
#    print(nn_state.rbm_am.weights)
    print('Done.')

# COMPLEX PURE WAVEFUNCTION
@cli.command("train_complex")
@click.option('--train-path', default='../tools/benchmarks/data/2qubits_complex/2qubits_train_samples.txt',
              show_default=True, type=click.Path(exists=True),
              help="path to the training data")
@click.option('--basis-path', default='../tools/benchmarks/data/2qubits_complex/2qubits_train_bases.txt',
              show_default=True, type=click.Path(exists=True),
              help="path to the basis data")
@click.option('--true-psi-path', default='../tools/benchmarks/data/2qubits_complex/2qubits_psi.txt',
              show_default=True, type=click.Path(exists=True),
              help=("path to the file containing the true wavefunctions "
                    "in each basis."))
@click.option('-nh', '--num-hidden', default=2, type=int,
              help=("number of hidden units; defaults to "
                    "number of visible units"))
@click.option('-bs', '--batch-size', default=100,
              show_default=True, type=int)
@click.option('-nc', '--num-chains', default=100,
              show_default=True, type=int)
@click.option('-e', '--epochs', default=100, show_default=True, type=int)
@click.option('-k', default=10, show_default=True, type=int,
              help="number of Contrastive Divergence steps")
@click.option('-lr', '--learning-rate', default=1e-3,
              show_default=True, type=float)
@click.option('--log-every', default=0, show_default=True, type=int,
              help=("how often the validation statistics are recorded, "
                    "in epochs; 0 means no logging"))
@click.option('--seed', default=1234, show_default=True, type=int,
              help="random seed to initialize the RBM with")
@click.option('--test-grads', is_flag=True)
@click.option('--no-prog', is_flag=False)
@click.option('--true-psi-path', default='../tools/benchmarks/data/2qubits_complex/2qubits_psi.txt',
              show_default=True, type=click.Path(exists=True),
              help=("path to the file containing the true wavefunctions "
                    "in each basis."))

def train_complex(train_path, basis_path, true_psi_path, num_hidden,
                  epochs, batch_size, num_chains, k, learning_rate,
                  log_every, seed, test_grads, no_prog):
    
    """Train an RBM with a phase."""

    train_data = torch.tensor(np.loadtxt(train_path, dtype='float32'),dtype=torch.double)
    train_bases = np.loadtxt(basis_path, dtype=str)
    unitary_dict = unitaries.create_dict()
    
    num_visible      = train_data.shape[-1]
    num_hidden   = (train_set.shape[-1]
                        if num_hidden is None
                        else num_hidden)
    psi = load_target_psi(num_visible,true_psi_path)#path_to_target_psi) 

    nn_state = ComplexWavefunction(num_visible=num_visible,
                               num_hidden=num_hidden)
   
    qr = QuantumReconstruction(nn_state)

    qr.fit(train_data, epochs, batch_size, num_chains, k,
            learning_rate,train_bases, progbar=(not no_prog),target_psi=psi)
    print(nn_state.rbm_am.weights)
    print(nn_state.rbm_ph.weights)



def load_target_psi(N,path_to_target_psi):
    psi_data = np.loadtxt(path_to_target_psi)
    D = 2**N#int(len(psi_data)/float(len(bases)))
    psi=torch.zeros(2,D, dtype=torch.double)
    if (len(psi_data.shape)<2):
        psi[0] = torch.tensor(psi_data,dtype=torch.double)
        psi[1] = torch.zeros(D,dtype=torch.double)
    else:
        psi_real = torch.tensor(psi_data[0:D,0],dtype=torch.double)
        psi_imag = torch.tensor(psi_data[0:D,1],dtype=torch.double)
        psi[0]   = psi_real
        psi[1]   = psi_imag
        
    return psi 

if __name__ == '__main__':
    cli()
