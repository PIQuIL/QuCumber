# Restrict Boltzmann Machine 
import tensorflow as tf 
import math
import timeit
import numpy as np 
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
from utils import tile_raster_images

class RBM(object):
	"""A Restricted Boltzmann Machines class"""
	def __init__(self, inp = None, n_visible = 784, n_hidden = 500, W = None, hbias = None, vbias = None):
		self.n_visible = n_visible
		self.n_hidden = n_hidden
		if inp is None:
			inp = tf.placeholder(dtype = tf.float32, shape=[None, self.n_visible])
		self.input = inp
		if W is None:
			low = -4.0 * math.sqrt(6. / (n_visible + n_hidden))
			high = 4.0 * math.sqrt(6. / (n_visible + n_hidden))
			W = tf.Variable(tf.random_uniform([self.n_visible,self.n_hidden], minval = low, maxval = high, dtype = tf.float32))
		self.W = W
		if hbias is None:
			hbias = tf.Variable(tf.zeros([n_hidden]), dtype = tf.float32)
		self.hbias = hbias
		if vbias is None:
			vbias = tf.Variable(tf.zeros([n_visible]), dtype = tf.float32)
		self.vbias = vbias
		self.params = [self.W, self.hbias, self.vbias]

	def propup(self, visible):
		"""This function propagates the visible units activation upwards to the hidden units"""
		return tf.nn.sigmoid(tf.matmul(visible,self.W) + self.hbias)


	def propdown(self, hidden):
		"""This function propagates the hidden units activation downwards to the visible units"""
		return tf.nn.sigmoid(tf.matmul(hidden, tf.transpose(self.W)) + self.vbias)

	def sample_prob(self, prob):
		'''Do sampling with the given probability'''
		return tf.nn.relu(tf.sign(prob - tf.random_uniform(tf.shape(prob))))
	
	def sample_h_given_v(self, v0_sample):
		''' This function infers state of hidden units given visible units'''
		# get a sample of the hiddens given their activation
		h1_mean = self.propup(v0_sample)
		h1_sample = self.sample_prob(h1_mean)
		return (h1_mean, h1_sample)

	def sample_v_given_h(self, h0_sample):
		''' This function infers state of visible units given hidden units '''
		# get a sample of the visible given their activation
		v1_mean = self.propdown(h0_sample)
		v1_sample = self.sample_prob(v1_mean)
		return (v1_mean, v1_sample)

		'''gibbs_vhv which performs a step of Gibbs sampling starting from the visible units. 
    	As we shall see, this will be useful for sampling from the RBM.
		gibbs_hvh which performs a step of Gibbs sampling starting from the hidden units. 
		This function will be useful for performing CD and PCD updates.'''

	def gibbs_hvh(self, h_sample):
		'''This function implements one step of Gibbs sampling,
            starting from the hidden state'''
		v1_mean, v1_sample = self.sample_v_given_h(h_sample)
		h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
		return (v1_mean, v1_sample, h1_mean, h1_sample)


	def gibbs_vhv(self, v_sample):
		''' This function implements one step of Gibbs sampling,
            starting from the visible state'''
		h1_mean, h1_sample = self.sample_h_given_v(v_sample)
		v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
		return (h1_mean, h1_sample, v1_mean, v1_sample)


	def free_energy(self, v_sample):
		'''function to compute the free energy which need for
    		computing the gradient of the parameters'''
		wx_b = tf.matmul(v_sample, self.W) + self.hbias
		vbias_term = tf.matmul(v_sample, tf.expand_dims(self.vbias, axis = 1))
		hidden_term = tf.reduce_sum(tf.log(1 + tf.exp(wx_b)), axis = 1)
		return -hidden_term - vbias_term

	# we then add a get_cost_updates method, whose purpose is to generate the
	# symbolic gradients forn CD-k and PCD-k updates

	def get_cost_updates(self, lr = 0.1, persistent = None, k = 1):
		'''This functions implements one step of CD-k or PCD-k
      :param lr: learning rate used to train the RBM
      :param persistent: None for CD. For PCD, shared variable containing 
      old state of Gibbs chain. This must be a shared variable of size(batch
      size, number of hidden units)
      :param k: number of Gibbs step to do in CD-k/PCD-k
    	Return a proxy for the cost and the updates dictionary. 
      The dictionary contains the update rules for weights and biases
      but also an update of the shared variable used to store the persistent
      chain, if one is used.'''
		
		#compute positive phase

		ph_mean, ph_sample = self.sample_h_given_v(self.input)

		#decide how to initialize persistent chain:
		# for CD, we use the newly generate hidden sample
		#forn PCD, we initialize from the old state of the chain
		if persistent is None:
			chain_start = ph_sample
		else:
			chain_start = persistent
		# perform actual negative phase
		# in order to implement CD-k/ PCD-k we need to scan over the
		# function that implements one gibbs step k times
		#print( tf.shape(chain_start))
		cond = lambda i, nv_mean, nv_sample, nh_mean, nh_sample: i < k
		body = lambda i, nv_mean, nv_sample, nh_mean, nh_sample: (i + 1, ) + self.gibbs_hvh(nh_sample)
		i, nv_mean, nv_sample, nh_mean, nh_sample = tf.while_loop(cond, body, loop_vars=[tf.constant(0), tf.zeros(tf.shape(self.input)), tf.zeros(tf.shape(self.input)), tf.zeros(tf.shape(chain_start)), chain_start])
		# determine gradients on RBM parameters

		# note that we only need the sample at the end of the chain
		chain_end = tf.stop_gradient(nv_sample)

		self.cost = tf.reduce_mean(self.free_energy(self.input)) - tf.reduce_mean(self.free_energy(chain_end))
		# We must not compute the gradient through the gibbs sampling
		#compute the gradients
		gparams = tf.gradients(ys = [self.cost], xs = self.params)
		new_params = []
		for gparam, param in zip(gparams, self.params):
			new_params.append(tf.assign(param, param - gparam * lr))

		if persistent is not None:
			new_persistent = [tf.assign(persistent, nh_sample)]
		else:
			new_persistent = []
		return new_params + new_persistent
		
	def get_reconstruction_cost(self):
		'''compute the cross-entropy of the original input and the reconstruction'''
		act_h = self.propup(self.input)
		act_v = self.propdown(act_h)

		cross_entropy = -tf.reduce_mean(tf.reduce_sum(self.input * tf.log(act_v) + (1.0 - self.input)* tf.log(1.0 - act_v), axis = 1))
		return cross_entropy
		"""
	def get_reconstruction_cost(self):
		'''Compute the cross-entropy of the original input and the reconstruction'''
		activation_h = self.propup(self.input)
		activation_v = self.propdown(activation_h)
		# Do this to not get Nan
		activation_v_clip = tf.clip_by_value(activation_v, clip_value_min=1e-30, clip_value_max=1.0)
		reduce_activation_v_clip = tf.clip_by_value(1.0 - activation_v, clip_value_min=1e-30, clip_value_max=1.0)
		cross_entropy = -tf.reduce_mean(tf.reduce_sum(self.input*(tf.log(activation_v_clip)) + 
                                    (1.0 - self.input)*(tf.log(reduce_activation_v_clip)), axis=1))
		return cross_entropy   
		"""


	def reconstruction(self, v):
		h = self.propup(v)
		return self.propdown(h)

			
def test_rbm():
	''' Demonstrate how to train and afterwards sample from it
	this is demonstrate on mnist
	:param leaning_rate: learning rate used for training the rbm
	:param training_epochs: number of epochs used for training
	:param dataset: path to the picked dataset
	:param batch_size: size of a batch used to train the RBM
	:param n_chains: number of parallel GIbbs chains to be used for sampling
	:param n_samples: number of samples to plot for each chain
	'''
	# import dataset
	mnist = input_data.read_data_sets("MNIST_data", one_hot = True)
	#parameters
	learning_rate = 0.1
	training_epochs = 1
	batch_size = 20
	display_step = 1
	n_chains = 20
	n_samples = 10
	#define input
	x = tf.placeholder(tf.float32, [None, 784])
	#network parameters
	n_visible = 784
	n_hidden = 500
	rbm = RBM(x, n_visible = n_visible, n_hidden = n_hidden)

	cost = rbm.get_reconstruction_cost()

	#create the persistent variable
	persistent_chain = tf.Variable(tf.zeros([batch_size, n_hidden]), dtype = tf.float32)
	cost_updates = rbm.get_cost_updates(lr = learning_rate, persistent = persistent_chain, k = 15)
	#initializing the variables
	init = tf.global_variables_initializer()

	#------------------
	#	Training RBM
	#------------------

	with tf.Session() as sess:
		start_time = timeit.default_timer()
		sess.run(init)

		total_batch = int(mnist.train.num_examples/batch_size)
		for epoch in range(training_epochs):
			c = 0.0
			#loop over all batches
			for i in range(total_batch):
				batch_xs, batch_ys = mnist.train.next_batch(batch_size)
				
				#run optimization op (backprop) and cost op (get loss value)
				_ = sess.run(cost_updates, feed_dict = {x: batch_xs})
				c += sess.run(cost, feed_dict = {x: batch_xs}) / total_batch
		#display logs per epoch step
			if epoch % display_step ==0:
				print("epoch", '%04d' % (epoch +1), "cost", "{:.4f}".format(c))

		#construct image from the weight matrix
		
		#image = plt.imshow(tile_raster_images(X = rbm.W, img_shape = (28, 28), tile_shape = (10,10), tile_spacing = (1,1)))
			plt.imsave("new_filters_at_{0}.png".format(epoch),tile_raster_images(X = sess.run(tf.transpose(rbm.W)), img_shape = (28, 28), tile_shape = (10,10), tile_spacing = (1,1)), cmap = 'gray')
			plt.show()

		end_time = timeit.default_timer()
		training_time = end_time - start_time
		print("Finished!")
		print("  The training ran for {0} minutes.".format(training_time/60,))


		#################################
		#     Sampling from the RBM     #
		#################################
		# Reconstruct the image by sampling
		print("...Sampling from the RBM")
		number_test_examples = mnist.test.num_examples
		#randomly select the n_chains examples

		test_indexs = np.random.randint(number_test_examples - n_chains)
		test_samples = mnist.test.images[test_indexs:test_indexs + n_chains]
		#create the persistent variable saving the visiable state
		persistent_v_chain = tf.Variable(tf.to_float(test_samples), dtype = tf.float32)
		# the step of gibbs
		step_every = 1000

		# implement the gibbs sampling
		cond = lambda j, h_mean, h_sample, v_mean, v_sample: j < step_every
		body = lambda j, h_mean, h_sample, v_mean, v_sample: (j+1, ) + rbm.gibbs_vhv(v_sample)
		j, h_mean, h_sample, v_mean, v_sample = tf.while_loop(cond, body, loop_vars=[tf.constant(0), tf.zeros([n_chains, n_hidden]), 
                                                            tf.zeros([n_chains, n_hidden]), tf.zeros(tf.shape(persistent_v_chain)), persistent_v_chain])
		# Update the persistent_v_chain
		new_persistent_v_chain = tf.assign(persistent_v_chain, v_sample)
		# Store the image by sampling
		image_data = np.zeros((29*(n_samples+1)+1, 29*(n_chains)-1),
                          dtype="uint8")
		# Add the original images
		image_data[0:28,:] = tile_raster_images(X=test_samples,
                                            img_shape=(28, 28),
                                            tile_shape=(1, n_chains),
                                            tile_spacing=(1, 1))
		# Initialize the variable
		sess.run(tf.variables_initializer(var_list=[persistent_v_chain]))
		# Do successive sampling
		for idx in range(1, n_samples+1):
			sample = sess.run(v_mean)
			sess.run(new_persistent_v_chain)
			print("...plotting sample", idx)
			image_data[idx*29:idx*29+28,:] = tile_raster_images(X=sample,
                                            img_shape=(28, 28),
                                            tile_shape=(1, n_chains),
                                            tile_spacing=(1, 1))
			#image = plt.imshow(image_data)
		plt.imsave("new_original_and_{0}samples.png".format(n_samples), image_data, cmap = 'gray')
		plt.show()


if __name__ == '__main__':
	test_rbm()
