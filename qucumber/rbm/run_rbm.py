from rbm import RBM_Module, ComplexRBM, BinomialRBM
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

def load_train(L):
	data = np.load("/home/data/critical-2d-ising/L={}/q=2/configs.npy"	
				   .format(L))
	data = data.reshape(data.shape[0], L*L)
	return data.astype('float32')

@cli.command("train_real")
@click.option('--train-path', default='../data/Ising2d_L4.pkl.gz',
			  show_default=True, type=click.Path(exists=True),
			  help="path to the training data")
@click.option('-n', '--num-hidden', default=None, type=int,
			  help=("number of hidden units in the RBM; defaults to "
					"number of visible units"))
@click.option('-e', '--epochs', default=1000, show_default=True, type=int)
@click.option('-b', '--batch-size', default=100, show_default=True, type=int)
@click.option('-k', default=1, show_default=True, type=int,
			  help="number of Contrastive Divergence steps")
@click.option('-l', '--learning-rate', default=1e-3,
			  show_default=True, type=float)
@click.option('-m', '--momentum', default=0.5, show_default=True, type=float,
			  help=("value of the momentum parameter; ignored if "
					"using SGD or Adam optimization"))
@click.option('--l1', default=0, show_default=True, type=float,
			  help="L1 regularization parameter")
@click.option('--l2', default=0, show_default=True, type=float,
			  help="L2 regularization parameter")
@click.option('--log-every', default=0, show_default=True, type=int,
			  help=("how often the validation statistics are recorded, "
					"in epochs; 0 means no logging"))
@click.option('--seed', default=1234, show_default=True, type=int,
			  help="random seed to initialize the RBM with")
@click.option('--no-prog', is_flag=True)

def train_real(train_path, save, num_hidden, epochs, batch_size,
		  k, learning_rate, momentum, l1, l2,
		  seed, log_every, no_prog):
	"""Train an RBM without any phase."""
	with gzip.open(train_path) as f:
		train_set = pickle.load(f, encoding='bytes')

	num_hidden = train_set.shape[-1] if num_hidden is None else num_hidden

	rbm = BinomialRBM(num_visible=train_set.shape[-1],
			  num_hidden=num_hidden,
			  seed=seed)

	rbm.fit(train_set, epochs,
			  batch_size, k=k,
			  lr=learning_rate,
			  momentum=momentum,
			  l1_reg=l1, l2_reg=l2,
			  log_every=log_every,
			  progbar=(not no_prog))

@cli.command("train_complex")
@click.option('--train-path', default='../cpp/data/2qubits_train_samples.txt',
			  show_default=True, type=click.Path(exists=True),
			  help="path to the training data")
@click.option('--basis-path', default='../cpp/data/2qubits_train_bases.txt',
			  show_default=True, type=click.Path(exists=True),
			  help="path to the basis data")
@click.option('--true-psi-path', default='../cpp/data/2qubits_psi.txt',
			  show_default=True, type=click.Path(exists=True),
			  help="path to the file containing the true wavefunctions in each basis.")
@click.option('-nha', '--num-hidden-amp', default=None, type=int,
			  help=("number of hidden units in the amp RBM; defaults to "
					"number of visible units"))
@click.option('-nhp', '--num-hidden-phase', default=None, type=int,
			  help=("number of hidden units in the phase RBM; defaults to "
					"number of visible units"))
@click.option('-e', '--epochs', default=100, show_default=True, type=int)
@click.option('-b', '--batch-size', default=100, show_default=True, type=int)
@click.option('-k', default=1, show_default=True, type=int,
			  help="number of Contrastive Divergence steps")
@click.option('-l', '--learning-rate', default=1e-3,
			  show_default=True, type=float)
@click.option('--l1', default=0, show_default=True, type=float,
			  help="l1 regularization parameter")
@click.option('--l2', default=0, show_default=True, type=float,
			  help="l2 regularization parameter")
@click.option('--log-every', default=10, show_default=True, type=int,
			  help=("how often the validation statistics are recorded, "
					"in epochs; 0 means no logging"))
@click.option('--seed', default=1234, show_default=True, type=int,
			  help="random seed to initialize the RBM with")

def train_complex(train_path, basis_path, true_psi_path, num_hidden_amp, num_hidden_phase, epochs, batch_size, k, learning_rate, l1, l2, seed, log_every):
	"""Train an RBM with a phase."""

	data = np.loadtxt(train_path, dtype= 'float32')

	train_set = torch.tensor(data, dtype = torch.double)
	basis_set = np.loadtxt(basis_path, dtype = str)

	f = open('unitary_library.txt')
	num_rows = len(f.readlines())
	f.close()

	# Dictionary for unitaries
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

	# Dictionary for true wavefunctions
	basis_list = ['Z' 'Z', 'X' 'Z', 'Z' 'X', 'Y' 'Z', 'Z' 'Y']
	psi_file = np.loadtxt(true_psi_path)
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

	rbm = ComplexRBM(unitaries=unitary_dictionary, psi_dictionary=psi_dictionary,
					 num_visible=train_set.shape[-1], 
					 num_hidden_amp=num_hidden_amp,
			  		 num_hidden_phase=num_hidden_phase,
			  		 seed=seed)

	rbm.fit(train_set, basis_set, epochs, batch_size, k=k, lr=learning_rate, l1_reg=l1, l2_reg=l2, log_every=log_every)


if __name__ == '__main__':
	cli()
