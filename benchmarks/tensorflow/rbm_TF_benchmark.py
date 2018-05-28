
# coding: utf-8

# In[19]:


import tensorflow as tf
import numpy as np
import click
import itertools as it


# In[2]:


class RBM:
    
    def __init__(self, training_data_file, num_hidden=10, num_visible=10, learning_rate=1., momentum=0.5, 
                 batch_size=32, num_gibbs_iters=1, _seed=1234):
              
        # number of hidden units and visible units
        self.training_data_file = training_data_file
        self.num_hidden = num_hidden
        self.num_visible = num_visible
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.batch_size = batch_size 
        self.num_gibbs_iters = num_gibbs_iters
        self._seed = _seed
        
        # initialize weights, visible bias and hidden bias
        self.weights = tf.Variable(tf.random_normal(shape=(self.num_visible, self.num_hidden), mean = 0.0, 
                                                    stddev = 0.1, dtype = tf.float32, seed = _seed, name = None), 
                                   tf.float32) # rows = num_hid, cols = num_vis
        
        self.hidden_bias = tf.Variable(tf.zeros(shape=(self.num_hidden,1)), tf.float32)
        self.visible_bias = tf.Variable(tf.zeros(shape=(self.num_visible,1)), tf.float32)  

    # prob of a hidden unit given a visible unit (can take batches of data)
    def prob_hidden_given_visible(self, visible_samples):
        arg = tf.matmul(visible_samples, self.weights) + tf.transpose(self.hidden_bias)
        return tf.nn.sigmoid(arg)
    
    # prob of a visible unit given a hidden unit (can take batches of data)
    def prob_visible_given_hidden(self, hidden_samples):
        arg = tf.matmul(hidden_samples, self.weights, transpose_b=True) + tf.transpose(self.visible_bias)
        return tf.nn.sigmoid(arg)

    # sample a hidden unit given a visible unit (can take batches of data)
    # see 'sample' function at end of class (samples a binary tensor)
    def sample_hidden_given_visible(self,visible_samples):
        b = tf.shape(visible_samples)[0] # the number of visible samples
        m = self.num_hidden 
        samples = self.sample(self.prob_hidden_given_visible(visible_samples), b, m)
        return samples

    # sample a visible unit given a hidden unit (can take batches of data)
    # see 'sample' function at end of class (samples a binary tensor)
    def sample_visible_given_hidden(self,hidden_samples):
        b = tf.shape(hidden_samples)[0] # the number of hidden samples
        n = self.num_visible
        samples = self.sample(self.prob_visible_given_hidden(hidden_samples), b, n)
        return samples

    # gibbs sampling for CD
    def gibbs_sampling(self, v0_samples): # initialize a hidden sample from a batch of training data
                                                     # v0 -- matrix of shape (batch_size, num_visible)
        h0_samples = self.sample_hidden_given_visible(v0_samples)
        v_samples, h_samples = v0_samples, h0_samples 

        for i in range(self.num_gibbs_iters):
            v_samples = self.sample_visible_given_hidden(h_samples)
            h_samples = self.sample_hidden_given_visible(v_samples)

        # spit out original visible and hidden samples, the reconstructed sample and the prob_h_given_v vector
        return v0_samples, h0_samples, v_samples, self.prob_hidden_given_visible(v_samples)
    
    # gradient of neg. log liklihood
    def grad_NLL_one_batch(self, batch):     

        # sample hidden and visible units via gibbs sampling
        v0, h0, vk, prob_h = self.gibbs_sampling(batch)

        # calculate gradients for a batch of training data
        w_grad  = tf.matmul(tf.transpose(v0), h0) - tf.matmul(tf.transpose(vk), prob_h)
        
        # when calculating the gradients for the biases, tf.reduce_mean will divide the
        # gradients by the batch size.
        vb_grad = tf.reshape(tf.reduce_mean(v0 - vk, 0), shape = (self.num_visible,1))
        hb_grad = tf.reshape(tf.reduce_mean(h0 - prob_h, 0), shape = (self.num_hidden,1))
        
        # divide the weight gradient matrix by the batch size
        w_grad  /= tf.to_float(tf.shape(batch)[0])
        
        return w_grad, vb_grad, hb_grad

    # for training function in main block of code. 
    # Just gets a batch from the training data
    def get_batch(self, index):
        batch_start = index*self.batch_size
        batch_end   = batch_start + self.batch_size 
        batch       = self.training_data_file[batch_start:batch_end,:]
        return batch
    
    # learning algorithm for 1 batch      
    def learn(self, batch):
        # calculate gradients for a batch
        weight_grad, visible_bias_grad, hidden_bias_grad = self.grad_NLL_one_batch(batch)
        
        # update weights and biases 
        update_weights        = tf.assign(self.weights, self.weights + self.learning_rate*weight_grad)
        update_visible_biases = tf.assign(self.visible_bias, self.visible_bias + self.learning_rate*visible_bias_grad)
        update_hidden_biases  = tf.assign(self.hidden_bias, self.hidden_bias + self.learning_rate*hidden_bias_grad)            
       
        return [update_weights, update_visible_biases, update_hidden_biases]

    # sample a binary tensor
    @staticmethod
    def sample(probs, m, n):
        return tf.where(
            tf.less(tf.random_uniform(shape=(m,n)), probs),
            tf.ones(shape=(m,n)),
            tf.zeros(shape=(m,n))
        )
    
# -----------------------------------------------------------------------------------------------------------------------#
# --------------------------------------------- End of RBM class --------------------------------------------------------#
# -----------------------------------------------------------------------------------------------------------------------#


# In[16]:


# must have this class to pass function arguments to functions that will
# be computed when the tf session starts
class Placeholders(object):
    pass

# shuffles the training data for SGD

def data_randomizer(data_file):
    shuffled_data = tf.random_shuffle(data_file)
    return shuffled_data

def load_training_pickle(file_path):
    with gzip.open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

placeholders = Placeholders()

@click.group(context_settings={"help_option_names": ['-h', '--help']})

def cli():
    pass

@cli.command('train')

@click.option('--training-data-path', default='data/Ising2d_L4.pkl.gz', show_default=True, type=click.Path(exists=True), 
              help='Path to the training data file.')

@click.option('-n', '--num_hidden', default=10, type=int,
              help=('Number of hidden units. Default = number of visible units.'))

@click.option('-e', '--num-epochs', default=1000, show_default=True, type=int)

@click.option('-b', '--batch-size', default=32, show_default=True, type=int, help='The size of mini batches.')

@click.option('-k', default=1, show_default=True, type=int, help='Number of CD steps.')

@click.option('-lr', '--learning-rate', default=1e-3, show_default=True, type=float)

@click.option('-m', '--momentum', default=0.5, show_default=True, type=float, help='Momentum parameter value.')

@click.option('--seed', default=1234, show_default=True, type=int, help='Random seed used to initialize the RBM.')


# In[18


# main training function that will call the learn function in the rbm class
def train(training_data_path, num_hidden, num_epochs, batch_size, k, learning_rate, momentum, seed):
    
    training_data = load_training_pickle(training_data_path)
    num_visible = len(training_data[0])
        
    # hard coded parameters
    num_batches     = int(training_data.shape[0]/batch_size)
    epoch_list      = []
    
    # intialize the rbm
    rbm = RBM(training_data, num_hidden = num_hidden, num_visible = num_visible,
              learning_rate = learning_rate, momentum = momentum, batch_size = batch_size, 
              num_gibbs_iters = k, _seed = seed)
    
    # placeholders for the visible samples (batches), the whole visible space, and the target psi
    placeholders.visible_samples = tf.placeholder(tf.float32, shape=(batch_size, num_visible))
    
    # a learning step to be ran once the session begins
    step = rbm.learn(visible_samples)
        
    with tf.Session() as sess:
       
        # initialize the batch counter and epoch counter
        batch_count = 0
        epoch       = 1
        
        # initialize global variables to begin the tf session
        init = tf.global_variables_initializer()
        sess.run(init)
        print 'Training is starting.'
        
        # total number of steps = num_epochs*num_batches
        for i in range(num_epochs*num_batches):

            if batch_size*batch_count + batch_size >= training_data.shape[0]:
                epoch_list.append(epoch)

                if epoch%20 == 0:
                    print 'Epoch: %d' % (epoch)

                epoch += 1
                
                # shuffle data again
                training_data = data_randomizer(training_data)
                batch_count = 0
            
            batch_count += 1
            new_batch = rbm.get_batch(batch_count)  
            sess.run(step, feed_dict = {placeholders.visible_samples: new_batch})
    
    print 'Done training.'

# In[ ]:

if __name__ == '__main__':
    cli()
