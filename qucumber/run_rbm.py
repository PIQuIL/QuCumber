import click
import numpy as np
import torch

from rbm import BinomialRBM, ComplexRBM

from . import unitaries


@click.group(context_settings={"help_option_names": ['-h', '--help']})
def cli():
    """Simple tool for training an RBM"""
    pass


def load_params(param_file):
    with open(param_file, 'rb+') as f:
        param_dict = torch.load(f)
    return param_dict


@cli.command("train_real")
@click.option('--train-path', default='tfim1d_N10_train_samples.txt',
              show_default=True, type=click.Path(exists=True),
              help="path to the training data")
@click.option('--true-psi-path', default='tfim1d_N10_psi.txt',
              show_default=True, type=click.Path(exists=True),
              help=("path to the file containing the true wavefunctions "
                    "in each basis."))
@click.option('-n', '--num-hidden', default=None, type=int,
              help=("number of hidden units in the RBM; defaults to "
                    "number of visible units"))
@click.option('-e', '--epochs', default=1000, show_default=True, type=int)
@click.option('-pb', '--pos-batch-size', default=100,
              show_default=True, type=int)
@click.option('-nb', '--neg-batch-size', default=200,
              show_default=True, type=int)
@click.option('-k', default=1, show_default=True, type=int,
              help="number of Contrastive Divergence steps")
@click.option('-lr', '--learning-rate', default=1e-3,
              show_default=True, type=float)
@click.option('--seed', default=1234, show_default=True, type=int,
              help="random seed to initialize the RBM with")
@click.option('--save-psi', is_flag=True)
@click.option('--no-prog', is_flag=True)
def train_real(train_path, true_psi_path, num_hidden, epochs, pos_batch_size,
               neg_batch_size, k, learning_rate, seed, save_psi, no_prog):
    """Train an RBM without any phase."""

    train_set = np.loadtxt(train_path, dtype='float32')

    num_hidden = train_set.shape[-1] if num_hidden is None else num_hidden

    rbm = BinomialRBM(num_visible=train_set.shape[-1],
                      num_hidden=num_hidden, save_psi=save_psi, seed=seed)

    rbm.fit(train_set, epochs, pos_batch_size, neg_batch_size, k=k,
            lr=learning_rate, progbar=(not no_prog))

    print('Finished training. Saving results...')

    if save_psi:
        vis = rbm.rbm_module.generate_visible_space()
        Z = rbm.rbm_module.partition(vis)
        rbm.save(location='saved_params.pkl',
                 metadata={
                    "RBMpsi": rbm.rbm_module.probability(vis, Z).sqrt().data
                 })
    else:
        rbm.save(location='saved_params.pkl')

    print('Done.')


@cli.command("generate")
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

    num_visible = params['rbm_module.visible_bias'].size()[0]
    num_hidden = params['rbm_module.hidden_bias'].size()[0]

    rbm = BinomialRBM(num_visible=num_visible, num_hidden=num_hidden,
                      save_psi=False, seed=None)

    rbm.load(param_path)
    new_data = (rbm.sample(num_samples, k)).data.numpy()

    np.savetxt('generated_samples.txt', new_data, fmt='%d')


# NOTE: TRAIN_COMPLEX IS CURRENTLY IN DEVELOPMENT. COULD BE UNSTABLE IF USED.
@cli.command("train_complex")
@click.option('--train-path', default='2qubits_train_samples.txt',
              show_default=True, type=click.Path(exists=True),
              help="path to the training data")
@click.option('--basis-path', default='2qubits_train_bases.txt',
              show_default=True, type=click.Path(exists=True),
              help="path to the basis data")
@click.option('--true-psi-path', default='2qubits_psi.txt',
              show_default=True, type=click.Path(exists=True),
              help=("path to the file containing the true wavefunctions "
                    "in each basis."))
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
@click.option('-lr', '--learning-rate', default=1e-3,
              show_default=True, type=float)
@click.option('--log-every', default=0, show_default=True, type=int,
              help=("how often the validation statistics are recorded, "
                    "in epochs; 0 means no logging"))
@click.option('--seed', default=1234, show_default=True, type=int,
              help="random seed to initialize the RBM with")
@click.option('--test-grads', is_flag=True)
@click.option('--no-prog', is_flag=True)
def train_complex(train_path, basis_path, true_psi_path, num_hidden_amp,
                  num_hidden_phase, epochs, batch_size, k, learning_rate,
                  log_every, seed, test_grads, no_prog):
    """Train an RBM with a phase."""

    train_set = np.loadtxt(train_path, dtype='float32')
    basis_set = np.loadtxt(basis_path, dtype=str)

    unitary_dict = unitaries.create_dict()

    full_unitary_file = np.loadtxt('2qubits_unitaries.txt')
    full_unitary_dictionary = {}

    # Dictionary for true wavefunctions
    basis_list = ['Z' 'Z', 'X' 'Z', 'Z' 'X', 'Y' 'Z', 'Z' 'Y']
    psi_file   = np.loadtxt(true_psi_path)
    psi_dictionary = {}

    for i in range(len(basis_list)):
        # 5 possible wavefunctions: ZZ, XZ, ZX, YZ, ZY
        psi      = torch.zeros(2, 2**train_set.shape[-1], dtype=torch.double)
        psi_real = torch.tensor(psi_file[i*4:(i*4+4),0], dtype=torch.double)
        psi_imag = torch.tensor(psi_file[i*4:(i*4+4),1], dtype=torch.double)
        psi[0]   = psi_real
        psi[1]   = psi_imag
        psi_dictionary[basis_list[i]] = psi

        full_unitary      = torch.zeros(2, 2**train_set.shape[-1],
                                        2**train_set.shape[-1],
                                        dtype=torch.double)
        full_unitary_real = torch.tensor(full_unitary_file[i*8:(i*8+4)],
                                         dtype=torch.double)
        full_unitary_imag = torch.tensor(full_unitary_file[(i*8+4):(i*8+8)],
                                         dtype=torch.double)
        full_unitary[0]   = full_unitary_real
        full_unitary[1]   = full_unitary_imag
        full_unitary_dictionary[basis_list[i]] = full_unitary

    num_hidden_amp   = (train_set.shape[-1]
                        if num_hidden_amp is None
                        else num_hidden_amp)
    num_hidden_phase = (train_set.shape[-1]
                        if num_hidden_phase is None
                        else num_hidden_phase)

    rbm = ComplexRBM(full_unitaries=full_unitary_dictionary,
                     psi_dictionary=psi_dictionary,
                     num_visible=train_set.shape[-1],
                     num_hidden_amp=num_hidden_amp,
                     num_hidden_phase=num_hidden_phase,
                     test_grads=test_grads)

    rbm.fit(train_set, basis_set, unitary_dict, epochs, batch_size,
            k=k, lr=learning_rate, log_every=log_every, progbar=(not no_prog))


if __name__ == '__main__':
    cli()
