import tensorflow as tf
import numpy as np
import click
import pickle
import gzip


def momentum_update(momentum, lr, old, new):
    return (momentum * old) - (lr * new / tf.to_float(tf.shape(new)[0]))


class RBM:
    def __init__(self, num_visible, num_hidden, k, lr, momentum):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.k = k
        self.lr = lr
        self.momentum = momentum

        self.input = tf.placeholder(tf.float32,
                                    [None, self.num_visible],
                                    name='input')

        self.weights = tf.Variable(
            tf.random_normal([self.num_visible, self.num_hidden],
                             mean=0.0, stddev=1./np.sqrt(self.num_visible)),
            name='weights',
            dtype=tf.float32)
        self.visible_bias = tf.Variable(tf.zeros([self.num_visible]),
                                        name='visible_bias',
                                        dtype=tf.float32)
        self.hidden_bias = tf.Variable(tf.zeros([self.num_hidden]),
                                       name='hidden_bias',
                                       dtype=tf.float32)

        self.weight_grad = tf.Variable(
            tf.zeros([self.num_visible, self.num_hidden]), dtype=tf.float32)
        self.visible_bias_grad = tf.Variable(
            tf.zeros([self.num_visible]), dtype=tf.float32)
        self.hidden_bias_grad = tf.Variable(
            tf.zeros([self.num_hidden]), dtype=tf.float32)

        self.param_updaters = None
        self.grad_updaters = None

        self.visible_space = None
        self.logZ = None
        self.nll_val = None
        self.data_len = None

        self._init_vars()
        init = tf.global_variables_initializer()

        self.sess = tf.Session()
        self.sess.run(init)

    def prob_h_given_v(self, v):
        return tf.nn.sigmoid(tf.matmul(v, self.weights)
                             + self.hidden_bias)

    def sample_h_given_v(self, v):
        bernoulli = tf.distributions.Bernoulli(probs=self.prob_h_given_v(v),
                                               dtype=tf.float32)
        return bernoulli.sample()

    def prob_v_given_h(self, h):
        return tf.nn.sigmoid(tf.matmul(h, self.weights, transpose_b=True)
                             + self.visible_bias)

    def sample_v_given_h(self, h):
        bernoulli = tf.distributions.Bernoulli(probs=self.prob_v_given_h(h),
                                               dtype=tf.float32)
        return bernoulli.sample()

    def gibbs_step(self, i, v0, h0):
        v = self.sample_v_given_h(h0)
        h = self.sample_h_given_v(v)
        return [i+1, v, h]

    def gibbs_sampling(self, v0):
        h0 = self.sample_h_given_v(v0)
        i0 = tf.constant(0)
        [_, v, h] = tf.while_loop(lambda i, v, h: tf.less(i, self.k),
                                  self.gibbs_step,
                                  [i0, v0, h0],
                                  shape_invariants=[
                                    i0.get_shape(),
                                    tf.TensorShape([None, self.num_visible]),
                                    tf.TensorShape([None, self.num_hidden])])

        return v0, h0, v, h, self.prob_h_given_v(v)

    def _init_vars(self):
        v0, h0, vk, hk, phk = self.gibbs_sampling(self.input)

        pos_weight_grad = tf.matmul(tf.transpose(v0), h0)
        neg_weight_grad = tf.matmul(tf.transpose(vk), phk)

        new_weight_grad = momentum_update(self.momentum,
                                          self.lr,
                                          self.weight_grad,
                                          neg_weight_grad - pos_weight_grad)
        new_vb_grad = momentum_update(self.momentum,
                                      self.lr,
                                      self.visible_bias_grad,
                                      tf.reduce_mean(vk - self.input, 0))
        new_hb_grad = momentum_update(self.momentum,
                                      self.lr,
                                      self.hidden_bias_grad,
                                      tf.reduce_mean(phk - h0, 0))

        weight_grad_updater = self.weight_grad.assign(new_weight_grad)
        vb_grad_updater = self.visible_bias_grad.assign(new_vb_grad)
        hb_grad_updater = self.hidden_bias_grad.assign(new_hb_grad)

        weight_updater = self.weights.assign_add(weight_grad_updater)
        visible_bias_updater = self.visible_bias.assign_add(vb_grad_updater)
        hidden_bias_updater = self.hidden_bias.assign_add(hb_grad_updater)

        self.param_updaters = [weight_updater,
                               visible_bias_updater,
                               hidden_bias_updater]
        self.grad_updaters = [weight_grad_updater,
                              vb_grad_updater,
                              hb_grad_updater]

        self.data_len = tf.convert_to_tensor(-float('inf'), dtype=tf.float32)
        self.visible_space = self.generate_visible_space()
        self.logZ = self.log_partition()
        self.nll_val = self.nll()

    def train(self, data, epochs, batch_size):
        for ep in range(epochs+1):
            np.random.shuffle(data)

            batches = [data[batch_start:(batch_start + batch_size)]
                       for batch_start in range(0, len(data), batch_size)]

            # if ep % 100 == 0:
            #     logZ = self.sess.run(self.logZ)
            #     nll = self.sess.run(self.nll_val,
            #                         feed_dict={
            #                             self.input: data,
            #                             self.logZ: logZ,
            #                             self.data_len: len(data)
            #                         })
            #     print(nll / len(data))

            if ep == epochs:
                break

            for batch in batches:
                self.sess.run(self.param_updaters + self.grad_updaters,
                              feed_dict={self.input: batch})

    def get_params(self):
        return (self.sess.run(self.weights),
                self.sess.run(self.visible_bias),
                self.sess.run(self.hidden_bias))

    def free_energy(self, v):
        visible_bias_term = tf.matmul(v,
                                      tf.reshape(self.visible_bias, [-1, 1]))
        hidden_bias_term = tf.reduce_sum(tf.nn.softplus(
            tf.matmul(v, self.weights) + self.hidden_bias
        ), axis=1, keepdims=True)
        return visible_bias_term + hidden_bias_term

    def generate_visible_space(self):
        space = np.zeros((1 << self.num_visible, self.num_visible))

        for i in range(1 << self.num_visible):
            d = i
            for j in range(self.num_visible):
                d, r = divmod(d, 2)
                space[i, self.num_visible - j - 1] = int(r)

        return tf.convert_to_tensor(space, dtype=tf.float32)

    def log_partition(self):
        free_energies = self.free_energy(self.visible_space)
        return tf.reduce_logsumexp(free_energies)

    def nll(self):
        total_free_energy = tf.reduce_sum(self.free_energy(self.input))
        return (self.data_len*self.logZ) - total_free_energy


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
def train(train_path, num_hidden, epochs, batch_size,
          k, learning_rate, momentum):
    """Train an RBM"""
    with gzip.open(train_path) as f:
        train_set = pickle.load(f, encoding='bytes')

    num_hidden = train_set.shape[-1] if num_hidden is None else num_hidden

    rbm = RBM(num_visible=train_set.shape[-1],
              num_hidden=num_hidden,
              k=k, lr=learning_rate,
              momentum=momentum)

    rbm.train(train_set, epochs, batch_size)


if __name__ == '__main__':
    cli()
