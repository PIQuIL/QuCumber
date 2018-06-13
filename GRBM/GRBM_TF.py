# Gaussian Restrict Boltzmann Machine

import tensorflow as tf 
import math
import timeit
import numpy as np 
import matplotlib.pyplot as plt
from RBM_TF import RBM
from tensorflow.examples.tutorials.mnist import input_data
from utils import tile_raster_images

class GRBM(RBM):
	# Gaussian Restrict Boltzmann Machine
	def __init__(self, inp = None, n_visible = 784, n_hidden = 500, W = None, hbias = None, vbias = None, sigma = 1.0):
		"""
		self.n_visible = n_visible
		self.n_hidden = n_hidden
		if inp is None:
			inp = tf.placeholder(dtype = tf.float32, shape = [None, self.n_visible])
		self.input = inp
		if W is None:
			low = -4.0 * math.sqrt(6.0 / (n_visible + n_hidden))
			high = 4.0 * math.sqrt(6.0 / (n_visible + n_hidden))
			W = tf.Variable(tf.random_uniform([self.n_visible, self.n_hidden], minval = low, maxval = high, dtype = tf.float32))
		self.W = W
		if hbias is None:
			hbias = tf.Variable(tf.zeros([n_hidden]), dtype = tf.float32)
		self.hbias = hbias
		if vbias is None:
			vbias = tf.Variable(tf.zeros([n_visible]), dtype = tf.float32)
		self.vbias = vbias"""
		super(GRBM, self).__init__(inp, n_visible, n_hidden, W, hbias, vbias)
		self.sigma = sigma
		#self.params = [self.W, self.hbias, self.vbias]


	def propup(self, visible):
		#print(visible.shape)
		#print(self.W)
		# This function propagates the visible units activation upwards to the hidden unit
		return tf.nn.sigmoid(tf.matmul(visible, self.W) / (self.sigma **2) + self.hbias)	

	def propdown(self, hidden):
		# This function propagates the hidden units activaion downwards to the visible unit
		#dist = tf.contrib.distributions.Normal(loc = tf.matmul(hidden, tf.transpose(self.W)) + self.vbias, scale = self.sigma **2)
		return tf.matmul(hidden, tf.transpose(self.W)) + self.vbias
		#return dist.prob(tf.matmul(hidden, tf.transpose(self.W)) + self.vbias)

	def sample_bernoulli(self, prob):
		return tf.nn.relu(tf.sign(prob - tf.random_uniform(tf.shape(prob))))

	def sample_gaussian(self, x, sigma):
		return x + tf.random_normal(tf.shape(x), mean = 0.0, stddev = sigma, dtype=tf.float32)


	def sample_h_given_v(self, v0_sample):
		# This function infers state of hidden units given visible units
		# get a sample of the hiddens given their activation
		h1_mean = self.propup(v0_sample)
		h1_sample = self.sample_bernoulli(h1_mean)
		return (h1_mean, h1_sample)

	def sample_v_given_h(self, h0_sample):
		# This function infers state of visible units given hidden units
		# get a sample of the hiddens given their activation
		v1_mean = self.propdown(h0_sample)
		v1_sample = self.sample_gaussian(v1_mean, self.sigma)
		return (v1_mean, v1_sample)

		'''gibbs_vhv which performs a step of Gibbs sampling starting from the visible units. 
		As we shall see, this will be useful for sampling from the RBM.
		gibbs_hvh which performs a step of Gibbs sampling starting from the hidden units. 
		This function will be useful for performing CD and PCD updates.'''

	def gibbs_hvh(self, h_sample):
		# this function implements one step of Gibbs sampling starting from the hidden state
		v1_mean, v1_sample = self.sample_v_given_h(h_sample)
		h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
		return (v1_mean, v1_sample, h1_mean, h1_sample)

	def gibbs_vhv(self, v_sample):
		# this function implements one step of gibbs sampling starting from the visible state
		h1_mean, h1_sample = self.sample_h_given_v(v_sample)
		v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
		return (h1_mean, h1_sample, v1_mean, v1_sample)

	def free_energy(self, v_sample):
		#function to compute the free energy which need for computing the gradient
		wx_b = tf.matmul(v_sample, self.W) / self.sigma**2 + self.hbias
		vbias_term = tf.reduce_sum(0.5*((v_sample - self.vbias)/ self.sigma)**2, axis = 1)
		hidden_term = tf.reduce_sum(tf.log(1 + tf.exp(wx_b)),axis =1)
		return -hidden_term + vbias_term

	#we then add a train_ops method, whose purpose is to generate the sysbolic gradients from CD-k and PCD-k updates
	def train_ops(self, lr = 0.1, persistent = None, k = 1):
		''' This function implements one step of CD-k or PCD-k
		:param lr: leaning rate used to train the rbm
		:param persistent: none for Cd, for PCD, shared variable containing old state of
		Gibbs chain. This must be a shared variable of size(batch_size), number of hidden units)
		:param k : number if Gibbs step to do in CD-k, PCD-k
		Return a proxy for the cost and the updates dictionary.
		The dictionary contains the updates rules for the weights and biases
		but also an update of the shared variable used to store the persistent
		chain, if one is used'''

		# compute positive phase
		ph_mean, ph_sample = self.sample_h_given_v(self.input)
		#decide how to initialize persistent chain:
		#for cd, we use the newly generate hidden sample
		# for PCD, we initialize from the old state of the chain

		if persistent is None:
			chain_start = ph_sample
		else:
			chain_start = persistent

		#perform actual negative phase
		#in order to implement CD-k/ PCd-k we need to scan over the function that implement one gibbs step k times

		cond = lambda i, nv_mean, nv_sample, nh_mean, nh_sample: i < k
		body = lambda i, nv_mean, nv_sample, nh_mean, nh_sample: (i+1, ) + self.gibbs_hvh(nh_sample)
		
		i, nv_mean, nv_sample, nh_mean, nh_sample = tf.while_loop(cond, body, loop_vars = [tf.constant(0), tf.zeros(tf.shape(self.input)), tf.zeros(tf.shape(self.input)), tf.zeros(tf.shape(chain_start)), chain_start])
		# determine gradients on RBM parameters

		# note that we only need the sample at the end of the chain
		chain_end = tf.stop_gradient(nv_sample)

		self.cost = tf.reduce_mean(self.free_energy(self.input)) - tf.reduce_mean(self.free_energy(chain_end))

		#we must not compute the gradient through the gibbs sampling
		# compute the gradients
		gparams = tf.gradients(ys = [self.cost], xs = self.params)
		new_params = []
		for gparam, param in zip(gparams, self.params):
			new_params.append(tf.assign(param, param - gparam * lr))

		if persistent is not None:
			new_persistent = [tf.assign(persistent, nh_sample)]
		else:
			new_persistent = []
		return new_params + new_persistent


	def get_reconstruct_cost(self):
		#compute the cross-entropy of the original inout and the reconstruction
		act_h = self.propup(self.input)
		act_v = self.propdown(act_h)
		#print(act_h.shape)
		#print(act_v)
		#cross_entropy = -tf.reduce_mean(tf.reduce_sum(self.input * tf.log(act_v) + (1.0 - self.input) * tf.log(1.0 - act_v), axis = 1))
		cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.square(self.input - act_v), axis=1))
		return cross_entropy
	'''
	def reconstruction(self, inp):
		act_h = self.propup(inp)
		return self.propdown(act_h)
	'''
def test_rbm():
	''' Demonstrate how to train and afterwards sample from 
	:param leaning_rate:
	:param training_ecpochs
	:param dataset
	:param barch_size
	:param n_chains
	:param n_samples
	'''


	#import dataset
	mnist = input_data.read_data_sets("MNIST_data", one_hot = True)
	# parameters
	leaning_rate = 0.01
	training_epochs = 3
	batch_size = 50
	display_step = 1


	#define input
	x = tf.placeholder(tf.float32, [None, 784])
	#network parameters
	n_visible = 784
	n_hidden = 500
	grbm = GRBM(x, n_visible = n_visible, n_hidden = n_hidden)

	cost = grbm.get_reconstruct_cost()

	#create the persistent variable
	#persistent_chain = tf.Variable(tf.zeros([batch_size, n_hidden]), dtype = tf.float32)
	persistent_chain = None
	train = grbm.train_ops(lr = leaning_rate, persistent = persistent_chain, k = 1)
	#initializing the variables
	init = tf.global_variables_initializer()

	#################
	# training RBM
	#################

	with tf.Session() as sess:
		start_time = timeit.default_timer()
		sess.run(init)
		total_batch = int(mnist.train.num_examples / batch_size)
		for epoch in range(training_epochs):
			c = 0.0
			#print(sess.run(grbm.W))
			# loop over all batches
			for i in range(total_batch):
				batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			
				# run optimization op (batchprop) and cost op
				_ = sess.run(train, feed_dict = {x : batch_xs})
				c += sess.run(cost, feed_dict = {x : batch_xs})/ total_batch
			#display logs per epoch step
			if epoch % display_step == 0:
				print("epoch", '%04d'%(epoch +1), "cost","{:.4f}".format(c))
				#print(sess.run(grbm.W))

			#construct image from the weight matrix

			plt.imsave("new_filters_at_{0}.png".format(epoch),tile_raster_images(X = sess.run(tf.transpose(grbm.W)),
				 img_shape = (28, 28), tile_shape = (10,10), tile_spacing = (1,1)), cmap = 'gray')
			plt.show()
		end_time = timeit.default_timer()
		training_time = end_time - start_time
		print("TIME:{0} minutes".format(training_time/ 60,))

		"""
		#################################
		#     Sampling from the RBM     #
		#################################
		# Reconstruct the image by sampling
		print("...Sampling from the RBM")
		n_chains = 20
		n_batch = 10
		n_samples = n_batch *2
		number_test_examples = mnist.test.num_examples
		#randomly select the n_chains examples
		test_indexs = np.random.randint(number_test_examples - n_chains * n_batch)
		test_samples = mnist.test.images[test_indexs:test_indexs + n_chains * n_batch]
		#create the persistent variable saving the visiable state
		#persistent_v_chain = tf.Variable(tf.to_float(test_samples), dtype = tf.float32)
		# the step of gibbs
		#step_every = 1000
		'''
		# implement the gibbs sampling
		cond = lambda j, h_mean, h_sample, v_mean, v_sample: j < step_every
		body = lambda j, h_mean, h_sample, v_mean, v_sample: (j+1, ) + grbm.gibbs_vhv(v_sample)
		j, h_mean, h_sample, v_mean, v_sample = tf.while_loop(cond, body, loop_vars=[tf.constant(0), tf.zeros([n_chains, n_hidden]), 
                                                            tf.zeros([n_chains, n_hidden]), tf.zeros(tf.shape(persistent_v_chain)), persistent_v_chain])
		'''
		# Update the persistent_v_chain
		#new_persistent_v_chain = tf.assign(persistent_v_chain, v_sample)
		# Store the image by sampling
		image_data = np.zeros((29*(n_samples+1)+1, 29*(n_chains)-1),
                          dtype="uint8")
		# Initialize the variable
		#sess.run(tf.variables_initializer(var_list=[persistent_v_chain]))
		# Do successive sampling
		for idx in range(n_batch):
			#sample = sess.run(v_mean)
			#sess.run(new_persistent_v_chain)
			# Add the original images
			'''
			image_data[2*idx*29 : 2*idx *29 + 28,:] = tile_raster_images(X=test_samples[idx*n_batch, (idx+1)*n_chains],
                                            img_shape=(28, 28),
                                            tile_shape=(1, n_chains),
                                            tile_spacing=(1, 1))			
			'''
			sample = sess.run(grbm.reconstruction, feed_dict = {x : test_samples[idx*n_batch, (idx+1)*n_chains]})
			print("...plotting sample", idx)
			image_data[(2*idx +1)*29:(2 * idx +1)*29+28,:] = tile_raster_images(X=sample,
                                            img_shape=(28, 28),
                                            tile_shape=(1, n_chains),
                                            tile_spacing=(1, 1))
			#image = plt.imshow(image_data)
		plt.imsave("new_original_and_{0}samples.png".format(n_samples), image_data, cmap = 'gray')
		plt.show()
		"""
		# Randomly select the 'n_chains' examples

		n_chains = 20
		n_batch = 10
		n_samples = n_batch*2
		number_test_examples = mnist.test.num_examples
		test_indexs = np.random.randint(number_test_examples - n_chains*n_batch)
		test_samples = mnist.test.images[test_indexs:test_indexs+n_chains*n_batch]
		image_data = np.zeros((29*(n_samples+1)+1, 29*(n_chains)-1),
                          dtype="uint8")
		# Add the original images

		for i in range(n_batch):
			image_data[2*i*29:2*i*29+28,:] = tile_raster_images(X=test_samples[i*n_batch:(i+1)*n_chains],
                                            img_shape=(28, 28),
                                            tile_shape=(1, n_chains),
                                            tile_spacing=(1, 1))
            
			samples = sess.run(grbm.reconstruction(x), feed_dict={x:test_samples[i*n_batch:(i+1)*n_chains]})
#			x is the place holder for the input size 
			image_data[(2*i+1)*29:(2*i+1)*29+28,:] = tile_raster_images(X=samples,
                                            img_shape=(28, 28),
                                            tile_shape=(1, n_chains),
                                            tile_spacing=(1, 1))
        	
		image = plt.imsave("original_and_reconstruct.png",image_data, cmap = 'gray')
		plt.show()


if __name__ == '__main__':
	''' Demonstrate how to train and afterwards sample from 
	:param leaning_rate:
	:param training_ecpochs
	:param dataset
	:param barch_size
	:param n_chains
	:param n_samples
	'''


	#import dataset
	mnist = input_data.read_data_sets("MNIST_data", one_hot = True)
	# parameters
	leaning_rate = 0.01
	training_epochs = 1000
	batch_size = 50
	display_step = 1


	#define input
	x = tf.placeholder(tf.float32, [None, 784])
	#network parameters
	n_visible = 784
	n_hidden = 500
	grbm = GRBM(x, n_visible = n_visible, n_hidden = n_hidden)

	cost = grbm.get_reconstruct_cost()

	#create the persistent variable
	#persistent_chain = tf.Variable(tf.zeros([batch_size, n_hidden]), dtype = tf.float32)
	persistent_chain = None
	train = grbm.train_ops(lr = leaning_rate, persistent = persistent_chain, k = 1)
	#initializing the variables
	init = tf.global_variables_initializer()

	#################
	# training RBM
	#################

	with tf.Session() as sess:
		start_time = timeit.default_timer()
		sess.run(init)
		total_batch = int(mnist.train.num_examples / batch_size)
		for epoch in range(training_epochs):
			c = 0.0
			#print(sess.run(grbm.W))
			# loop over all batches
			for i in range(total_batch):
				batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			
				# run optimization op (batchprop) and cost op
				_ = sess.run(train, feed_dict = {x : batch_xs})
				c += sess.run(cost, feed_dict = {x : batch_xs})/ total_batch
			#display logs per epoch step
			if epoch % display_step == 0:
				print("epoch", '%04d'%(epoch +1), "cost","{:.4f}".format(c))
				#print(sess.run(grbm.W))

			#construct image from the weight matrix
			X = sess.run(tf.transpose(grbm.W))
			plt.imsave("new_filters_at_{0}.png".format(epoch),tile_raster_images(X = sess.run(tf.transpose(grbm.W)),
				 img_shape = (28, 28), tile_shape = (10,10), tile_spacing = (1,1)), cmap = 'gray')
			plt.show()
		end_time = timeit.default_timer()
		training_time = end_time - start_time
		print("TIME:{0} minutes".format(training_time/ 60,))

		"""
		#################################
		#     Sampling from the RBM     #
		#################################
		# Reconstruct the image by sampling
		print("...Sampling from the RBM")
		n_chains = 20
		n_batch = 10
		n_samples = n_batch *2
		number_test_examples = mnist.test.num_examples
		#randomly select the n_chains examples
		test_indexs = np.random.randint(number_test_examples - n_chains * n_batch)
		test_samples = mnist.test.images[test_indexs:test_indexs + n_chains * n_batch]
		#create the persistent variable saving the visiable state
		#persistent_v_chain = tf.Variable(tf.to_float(test_samples), dtype = tf.float32)
		# the step of gibbs
		#step_every = 1000
		'''
		# implement the gibbs sampling
		cond = lambda j, h_mean, h_sample, v_mean, v_sample: j < step_every
		body = lambda j, h_mean, h_sample, v_mean, v_sample: (j+1, ) + grbm.gibbs_vhv(v_sample)
		j, h_mean, h_sample, v_mean, v_sample = tf.while_loop(cond, body, loop_vars=[tf.constant(0), tf.zeros([n_chains, n_hidden]), 
                                                            tf.zeros([n_chains, n_hidden]), tf.zeros(tf.shape(persistent_v_chain)), persistent_v_chain])
		'''
		# Update the persistent_v_chain
		#new_persistent_v_chain = tf.assign(persistent_v_chain, v_sample)
		# Store the image by sampling
		image_data = np.zeros((29*(n_samples+1)+1, 29*(n_chains)-1),
                          dtype="uint8")
		# Initialize the variable
		#sess.run(tf.variables_initializer(var_list=[persistent_v_chain]))
		# Do successive sampling
		for idx in range(n_batch):
			#sample = sess.run(v_mean)
			#sess.run(new_persistent_v_chain)
			# Add the original images
			'''
			image_data[2*idx*29 : 2*idx *29 + 28,:] = tile_raster_images(X=test_samples[idx*n_batch, (idx+1)*n_chains],
                                            img_shape=(28, 28),
                                            tile_shape=(1, n_chains),
                                            tile_spacing=(1, 1))			
			'''
			sample = sess.run(grbm.reconstruction, feed_dict = {x : test_samples[idx*n_batch, (idx+1)*n_chains]})
			print("...plotting sample", idx)
			image_data[(2*idx +1)*29:(2 * idx +1)*29+28,:] = tile_raster_images(X=sample,
                                            img_shape=(28, 28),
                                            tile_shape=(1, n_chains),
                                            tile_spacing=(1, 1))
			#image = plt.imshow(image_data)
		plt.imsave("new_original_and_{0}samples.png".format(n_samples), image_data, cmap = 'gray')
		plt.show()
		"""
		# Randomly select the 'n_chains' examples

		n_chains = 20
		n_batch = 10
		n_samples = n_batch*2
		number_test_examples = mnist.test.num_examples
		test_indexs = np.random.randint(number_test_examples - n_chains*n_batch)
		test_samples = mnist.test.images[test_indexs:test_indexs+n_chains*n_batch]
		image_data = np.zeros((29*(n_samples+1)+1, 29*(n_chains)-1),
                          dtype="uint8")
		# Add the original images

		for i in range(n_batch):
			image_data[2*i*29:2*i*29+28,:] = tile_raster_images(X=test_samples[i*n_batch:(i+1)*n_chains],
                                            img_shape=(28, 28),
                                            tile_shape=(1, n_chains),
                                            tile_spacing=(1, 1))
            
			samples = sess.run(grbm.reconstruction(x), feed_dict={x:test_samples[i*n_batch:(i+1)*n_chains]})
			image_data[(2*i+1)*29:(2*i+1)*29+28,:] = tile_raster_images(X=samples,
                                            img_shape=(28, 28),
                                            tile_shape=(1, n_chains),
                                            tile_spacing=(1, 1))
        	
		image = plt.imsave("original_and_reconstruct.png",image_data, cmap = 'gray')
		plt.show()

