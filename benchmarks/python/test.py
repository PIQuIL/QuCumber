import numpy as np
import matplotlib.pyplot as plt
from rbm import RBM
import click


@click.command()
@click.option('--train-path', default='../c++/training_data.txt')
@click.option('--target-path', default='../c++/target_psi.txt')
@click.option('-n', '--num-hidden', default=None, type=int)
@click.option('-e', '--epochs', default=1000, type=int)
@click.option('-b', '--batch-size', default=100, type=int)
@click.option('-k', default=1, type=int)
@click.option('-l', '--learning-rate', default=1e-3, type=float)
@click.option('--log-every', default=10, type=int)
@click.option('-p', '--plot', is_flag=True)
@click.option('--gpu', is_flag=True)
def main(train_path, target_path, num_hidden, epochs, batch_size,
         k, learning_rate, log_every, plot, gpu):
    train_set = np.loadtxt(train_path)
    target_psi = np.loadtxt(target_path)

    num_hidden = train_set.shape[-1] if num_hidden is None else num_hidden

    rbm = RBM(num_visible=train_set.shape[-1],
              num_hidden=num_hidden,
              seed=1234)

    nll_loop, overlap_loop = rbm.train(train_set, target_psi, epochs,
                                       batch_size, k, lr=learning_rate,
                                       log_every=log_every,
                                       notebook=False)

    if plot:
        fig, ax1 = plt.subplots(figsize=(10, 10))
        ax1.plot(np.arange(len(nll_loop)),
                 np.array(nll_loop) / len(train_set), 'b')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("NLL", color='b')
        ax1.tick_params('y', colors='b')
        ax1.set_xlim(0, len(nll_loop))
        ax1.axhline(y=4.7, xmin=0, xmax=len(nll_loop),
                    linestyle=':', color='b')

        ax2 = ax1.twinx()
        ax2.plot(np.arange(len(overlap_loop)), overlap_loop, 'r')
        ax2.set_ylabel('Overlap', color='r')
        ax2.tick_params('y', colors='r')
        ax2.axhline(y=0.999, xmin=0, xmax=len(overlap_loop),
                    linestyle=':', color='r')

        plt.show()


if __name__ == '__main__':
    main()
