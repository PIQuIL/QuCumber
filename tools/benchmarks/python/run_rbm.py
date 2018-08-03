import numpy as np
import matplotlib.pyplot as plt
from rbm import RBM
import click
import gzip
import pickle


@click.group(context_settings={"help_option_names": ['-h', '--help']})
def cli():
    """Simple tool for training an RBM"""
    pass

# @click.option('--target-path', type=click.Path(exists=True),
#               help="path to the wavefunction data")


@cli.command("train")
@click.option('--train-path', default='../data/Ising2d_L4.pkl.gz',
              show_default=True, type=click.Path(exists=True),
              help="path to the training data")
@click.option('--save', type=click.Path(),
              help="where to save the trained RBM parameters (if at all)")
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
@click.option('-p', '--persistent', is_flag=True,
              help="use Persistent Contrastive Divergence (PCD)")
@click.option('--persist-from', default=0, show_default=True, type=int,
              help=("if PCD flag is given, use vanilla CD until the given "
                    "epoch, then switch to PCD"))
@click.option('--plot', is_flag=True)
@click.option('--no-prog', is_flag=True)
@click.option('--method', default='momentum', show_default=True,
              type=click.Choice(["nesterov", "momentum", "sgd", "adam"]),
              help="the optimization method to use")
def train(train_path, save, num_hidden, epochs, batch_size,
          k, persistent, persist_from, learning_rate, momentum, l1, l2,
          method, seed, log_every, plot, no_prog):
    """Train an RBM"""
    # train_set = np.loadtxt(train_path)
    # target_psi = np.loadtxt(target_path) if target_path is not None else None
    with gzip.open(train_path) as f:
        train_set = pickle.load(f, encoding='bytes')

    num_hidden = train_set.shape[-1] if num_hidden is None else num_hidden

    rbm = RBM(num_visible=train_set.shape[-1],
              num_hidden=num_hidden,
              seed=seed)

    # learning_rate = schedulers.bounded_exponential_decay(0.1, 1e-6, epochs)
    # momentum = schedulers.bounded_exponential_decay(0.5, 0.99, epochs)

    nll_list = rbm.train(train_set, None, epochs,
                         batch_size, k=k,
                         persistent=persistent,
                         persist_from=persist_from,
                         lr=learning_rate,
                         momentum=momentum,
                         l1_reg=l1, l2_reg=l2,
                         beta1=0.9, beta2=0.999, epsilon=1e-8,
                         method=method,
                         log_every=log_every,
                         progbar=(not no_prog))

    if save:
        rbm.save(save)

    if plot and nll_list:
        fig, ax1 = plt.subplots(figsize=(10, 10))
        ax1.plot(log_every * np.arange(len(nll_list)),
                 np.array(nll_list) / len(train_set), 'b')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("NLL per training example", color='b')
        ax1.tick_params('y', colors='b')
        ax1.set_xlim(0, epochs)

        if persistent and persist_from > 0:
            # mark starting point of PCD if enabled and not zero
            ax1.axvline(x=persist_from, linestyle=':', color='g')

        # ax2 = ax1.twinx()
        # ax2.plot(log_every * np.arange(len(overlap_list)),
        #          overlap_list, 'r')
        # ax2.set_ylabel('Overlap', color='r')
        # ax2.tick_params('y', colors='r')
        # ax2.axhline(y=1, xmin=0, xmax=len(overlap_list),
        #             linestyle=':', color='r')  # plot maximum overlap

        plt.show()


# @cli.command("test")
# @click.option('--train-path', default='../c++/training_data.txt',
#               show_default=True)
# @click.option('--target-path', default='../c++/target_psi.txt',
#               show_default=True)
# @click.option('-n', '--num-hidden', default=None, type=int,
#               help=("number of hidden units in the RBM; defaults to "
#                     "number of visible units"))
# @click.option('-k', default=1, show_default=True, type=int,
#               help="number of Contrastive Divergence steps")
# @click.option('-e', '--epsilon', default=1e-8, show_default=True, type=float)
# @click.option('--seed', default=1234, show_default=True, type=int,
#               help="random seed to initialize the RBM with")
# def test(train_path, target_path, num_hidden, k, epsilon, seed):
#     """Tests the RBM's gradient computations"""
#     train_set = np.loadtxt(train_path)
#     target_psi = np.loadtxt(target_path)

#     num_hidden = train_set.shape[-1] if num_hidden is None else num_hidden

#     rbm = RBM(num_visible=train_set.shape[-1],
#               num_hidden=num_hidden,
#               seed=seed)

#     rbm.test_gradients(train_set, target_psi, k, epsilon)


if __name__ == '__main__':
    cli()
