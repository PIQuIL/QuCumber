from rbm import RBM
import click
import gzip
import pickle


@click.group(context_settings={"help_option_names": ['-h', '--help']})
def cli():
    """Simple tool for training an RBM"""
    pass


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

    rbm.train(train_set, None, epochs,
              batch_size, k=k,
              lr=learning_rate,
              momentum=momentum,
              l1_reg=l1, l2_reg=l2,
              log_every=log_every,
              progbar=(not no_prog))


if __name__ == '__main__':
    cli()
