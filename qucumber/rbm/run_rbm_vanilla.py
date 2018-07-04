from rbm import RBM
import click
import gzip
import pickle
import csv
import numpy as np
import torch


@click.group(context_settings={"help_option_names": ['-h', '--help']})
def cli():
    """Simple tool for training an RBM"""
    pass


def load_train(L):
    data = np.load("/home/data/critical-2d-ising/L={}/q=2/configs.npy"
                   .format(L))
    data = data.reshape(data.shape[0], L*L)
    return data.astype('float32')


@cli.command("benchmark")
@click.option('-n', '--num-hidden', default=None, type=int,
              help=("number of hidden units in the RBM; defaults to "
                    "number of visible units"))
@click.option('-e', '--epochs', default=250, show_default=True, type=int)
@click.option('-b', '--batch-size', default=32, show_default=True, type=int)
@click.option('-k', multiple=True, type=int,
              help="number of Contrastive Divergence steps")
@click.option('-l', '--learning-rate', default=1e-3,
              show_default=True, type=float)
@click.option('-m', '--momentum', default=0.0, show_default=True, type=float,
              help=("value of the momentum parameter; ignored if "
                    "using SGD or Adam optimization"))
@click.option('-o', '--output-file', type=click.Path(),
              help="where to save the benchmarks csv file.")
def benchmark(num_hidden, epochs, batch_size,
              k, learning_rate, momentum, output_file):
    """Trains RBM on several datasets and measures the training time"""
    from os import listdir
    from os.path import isdir, join
    import time
    from random import shuffle
    from itertools import product

    dataset_sizes = [int(f.split('=')[-1])
                     for f in listdir('/home/data/critical-2d-ising/')
                     if isdir(join('/home/data/critical-2d-ising/', f))]

    runs = list(product(dataset_sizes, k))

    shuffle(runs)

    with open(output_file, 'a') as f:
        writer = csv.DictWriter(f, ['L', 'k', 'time'])
        for L, k_ in runs:
            print("L = {}; k = {}".format(L, k_))
            train_set = load_train(L)

            num_hidden = (train_set.shape[-1]
                          if num_hidden is None
                          else num_hidden)

            rbm = RBM(num_visible=train_set.shape[-1],
                      num_hidden=num_hidden)

            time_elapsed = -time.perf_counter()
            rbm.train(train_set, epochs,
                      batch_size, k=k_,
                      lr=learning_rate,
                      momentum=momentum,
                      initial_gaussian_noise=0,
                      log_every=0,
                      progbar=False)
            time_elapsed += time.perf_counter()
            torch.cuda.empty_cache()

            writer.writerow({'L': L,
                             'k': k_,
                             'time': time_elapsed})


@cli.command("train")
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
def train(train_path, save, num_hidden, epochs, batch_size,
          k, learning_rate, momentum, l1, l2,
          seed, log_every, no_prog):
    """Train an RBM"""
    with gzip.open(train_path) as f:
        train_set = pickle.load(f, encoding='bytes')

    num_hidden = train_set.shape[-1] if num_hidden is None else num_hidden

    rbm = RBM(num_visible=train_set.shape[-1],
              num_hidden=num_hidden,
              seed=seed)

    rbm.train(train_set, epochs,
              batch_size, k=k,
              lr=learning_rate,
              momentum=momentum,
              l1_reg=l1, l2_reg=l2,
              log_every=log_every,
              progbar=(not no_prog))


@cli.command("train-alt")
@click.option('-L', help="data set size")
@click.option('-n', '--num-hidden', default=None, type=int,
              help=("number of hidden units in the RBM; defaults to "
                    "number of visible units"))
@click.option('-e', '--epochs', default=1000, show_default=True, type=int)
@click.option('-b', '--batch-size', default=100, show_default=True, type=int)
@click.option('-k', default=1, show_default=True, type=int,
              help="number of Contrastive Divergence steps")
@click.option('-l', '--learning-rate', default=1e-3,
              show_default=True, type=float)
@click.option('-m', '--momentum', default=0.0, show_default=True, type=float,
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
def train_alt(L, save, num_hidden, epochs, batch_size,
              k, learning_rate, momentum, l1, l2,
              seed, log_every, no_prog):
    """Train an RBM"""
    train_set = load_train(L)

    num_hidden = train_set.shape[-1] if num_hidden is None else num_hidden

    rbm = RBM(num_visible=train_set.shape[-1],
              num_hidden=num_hidden,
              seed=seed)

    rbm.train(train_set, epochs,
              batch_size, k=k,
              lr=learning_rate,
              momentum=momentum,
              l1_reg=l1, l2_reg=l2,
              log_every=log_every,
              progbar=False)


if __name__ == '__main__':
    cli()
