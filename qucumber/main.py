import utils.unitaries as unitaries
from positive_wavefunction import PositiveWavefunction
from complex_wavefunction import ComplexWavefunction
from quantum_reconstruction import QuantumReconstruction
import click
import numpy as np
import torch
import utils
import pickle
import sys
sys.path.append("utils/")
import training_statistics as ts
sys.path.append("../examples/observables/")


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
def train_real(num_hidden, epochs, batch_size,
               num_chains, k, learning_rate, seed, no_prog):
    """Train an RBM without any phase."""
    with open('tests/data_test.pkl', 'rb') as fin:
        test_data = pickle.load(fin)
   
    train_samples = torch.tensor(test_data['tfim1d']['train_samples'],dtype = torch.double)
    target_psi=torch.tensor(test_data['tfim1d']['target_psi'],dtype = torch.double)
    #target_psi = load_target_psi(train_samples.shape[-1],target_psi_tmp)
    num_hidden = train_samples.shape[-1] if num_hidden is None else num_hidden
    nn_state = PositiveWavefunction(num_visible=train_samples.shape[-1],
                      num_hidden=num_hidden, seed=seed)

    train_stats = ts.TrainingStatistics(train_samples.shape[-1])
    train_stats.load(target_psi=target_psi)
    qr = QuantumReconstruction(nn_state)
    qr.fit(train_samples, epochs, batch_size, num_chains, k,
            learning_rate, progbar=no_prog,observer = train_stats)

    print('\nFinished training. Saving results...')
    nn_state.save('saved_params.pkl')
    print('Done.')
    
    
    #tfim = TFIM.TransverseFieldIsingChain(1.0,1000)
    #
    #Energy = tfim.Energy(nn_state,n_eq=1000) 
    #print(Energy)

# COMPLEX WAVEFUNCTION
@cli.command("train_complex")
#@click.option('--train-path', default='../tools/benchmarks/data/2qubits_complex/2qubits_train_samples.txt',
#              show_default=True, type=click.Path(exists=True),
#              help="path to the training data")
#@click.option('--basis-path', default='../tools/benchmarks/data/2qubits_complex/2qubits_train_bases.txt',
#              show_default=True, type=click.Path(exists=True),
#              help="path to the basis data")
#@click.option('--true-psi-path', default='../tools/benchmarks/data/2qubits_complex/2qubits_psi.txt',
#              show_default=True, type=click.Path(exists=True),
#              help=("path to the file containing the true wavefunctions "
#                    "in each basis."))
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
#@click.option('--true-psi-path', default='../tools/benchmarks/data/2qubits_complex/2qubits_psi.txt',
 #             show_default=True, type=click.Path(exists=True),
#              help=("path to the file containing the true wavefunctions "
#                    "in each basis."))

def train_complex(num_hidden,
                  epochs, batch_size, num_chains, k, learning_rate,
                  log_every, seed, test_grads, no_prog):
    
    """Train an RBM with a phase."""

    with open('tests/data_test.pkl', 'rb') as fin:
        test_data = pickle.load(fin)

    train_bases = test_data['2qubits']['train_bases']
    train_samples = torch.tensor(test_data['2qubits']['train_samples'],dtype = torch.double)
    bases = test_data['2qubits']['bases'] 
    target_psi_dict=torch.tensor(test_data['2qubits']['target_psi'],dtype = torch.double)
    num_bases = len(bases)
    unitary_dict = unitaries.create_dict()
    
    num_visible      = train_samples.shape[-1]
    num_hidden   = (train_set.shape[-1]
                        if num_hidden is None
                        else num_hidden)
    #psi = load_target_psi(num_visible,true_psi_path)#path_to_target_psi) 

    nn_state = ComplexWavefunction(num_visible=num_visible,
                               num_hidden=num_hidden)
   
    train_stats = ts.TrainingStatistics(train_samples.shape[-1],frequency=1)
    train_stats.load(bases = bases,target_psi_dict = target_psi_dict)
    
    qr = QuantumReconstruction(nn_state)
    ##data = {"samples":train_data,"bases":train_bases}
    ##qr.fit(data,epochs, batch_size, num_chains, k,learning_rate,progbar=(not no_prog),target_psi=psi)
    qr.fit(train_samples, epochs, batch_size, num_chains, k,
            learning_rate,observer=train_stats,input_bases = train_bases,progbar=(no_prog))



@cli.command("simulate")
@click.option('--param-path', default='saved_params.pkl',
              show_default=True, type=click.Path(exists=True),
              help="path to the saved weights and biases")
@click.option('--num-samples', default=1000, show_default=True, type=int,
              help="number of samples to be generated.")
@click.option('-k', default=100, show_default=True, type=int,
              help="number of Contrastive Divergence steps.")
def generate(param_path, num_samples, k):
    """Generate new data from trained RBM parameters."""
    
     

    params = load_params(param_path)
    num_visible = params['rbm_am']['visible_bias'].size()[0]
    num_hidden  = params['rbm_am']['hidden_bias'].size()[0]
    seed =1234

    tfim = TFIM.TransverseFieldIsingChain(1.0,1000)
    
    nn_state = PositiveWavefunction(num_visible=num_visible,
                      num_hidden=num_hidden, seed=seed)

    nn_state.load(param_path)
    with open('tests/data_test.pkl', 'rb') as fin:
        test_data = pickle.load(fin)
    target_psi=torch.tensor(test_data['tfim1d']['target_psi'],dtype = torch.double)

    train_stats = ts.TrainingStatistics(num_visible)
    train_stats.load(target_psi=target_psi)
    train_stats.scan(0,nn_state)  



    Energy = tfim.Energy(nn_state,n_eq=1000) 
    print(Energy)
    #new_data = (rbm.sample(num_samples, k)).data.numpy()

    #np.savetxt('generated_samples.txt', new_data, fmt='%d')

#def load_target_psi(N,psi_data):#path_to_target_psi):
#    #psi_data = np.loadtxt(path_to_target_psi)
#    D = 2**N#int(len(psi_data)/float(len(bases)))
#    psi=torch.zeros(2,D, dtype=torch.double)
#    if (len(psi_data.shape)<2):
#        psi[0] = torch.tensor(psi_data,dtype=torch.double)
#        psi[1] = torch.zeros(D,dtype=torch.double)
#    else:
#        psi_real = torch.tensor(psi_data[0:D,0],dtype=torch.double)
#        psi_imag = torch.tensor(psi_data[0:D,1],dtype=torch.double)
#        psi[0]   = psi_real
#        psi[1]   = psi_imag
#        
#    return psi 

if __name__ == '__main__':
    cli()
