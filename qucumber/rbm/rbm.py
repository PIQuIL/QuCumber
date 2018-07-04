import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
import warnings
import cplx
from tqdm import tqdm, tqdm_notebook

class RBM_Module(nn.Module):

	def __init__(self, num_visible, num_hidden, zero_weights=False, gpu=True, seed=1234):
		super(RBM_Module, self).__init__()
		self.num_visible = int(num_visible)
		self.num_hidden  = int(num_hidden)

		if gpu and not torch.cuda.is_available():
			warnings.warn("Could not find GPU: will continue with CPU.",
						  ResourceWarning)

		self.gpu = gpu and torch.cuda.is_available()
		if self.gpu:
			torch.cuda.manual_seed(seed)
			self.device = torch.device('cuda')
		else:
			torch.manual_seed(seed)
			self.device = torch.device('cpu')

		if zero_weights:
			self.weights = nn.Parameter((torch.zeros(self.num_hidden, self.num_visible,
							 device=self.device, dtype=torch.double)), requires_grad=True)
		else:
			self.weights = nn.Parameter(
				(torch.randn(self.num_hidden, self.num_visible,
							 device=self.device, dtype=torch.double)
				 / np.sqrt(self.num_visible)),
				requires_grad=True)

		self.visible_bias = nn.Parameter(torch.zeros(self.num_visible,
													 device=self.device,
													 dtype=torch.double),
										 requires_grad=True)
		self.hidden_bias = nn.Parameter(torch.zeros(self.num_hidden,
													device=self.device,
													dtype=torch.double),
										requires_grad=True)

	def effective_energy(self, v):
		"""The effective energies of the given visible states.
		.. :math:`E_{\\lambda} = b^{\\lambda}v + \\sum_{i}\\log\\sum_{h_{i}^{\\lambda}}e^{h_{i}^{\\lambda}\\left(c_{i}^{\\lambda} + W_{i}^{\\lambda}v\\right)}`.

		:param v: The visible states.
		:type v: torch.doubleTensor

		:returns: The effective energies of the given visible states.
		:rtype: torch.doubleTensor
		"""
		if len(v.shape) < 2:
			v = v.view(1, -1)
		visible_bias_term = torch.mv(v, self.visible_bias)
		hidden_bias_term = F.softplus(
			F.linear(v, self.weights, self.hidden_bias)
		).sum(1)

		return visible_bias_term + hidden_bias_term

	def prob_v_given_h(self, h):
		"""Given a hidden unit configuration, compute the probability
		vector of the visible units being on.

		:param h: The hidden unit
		:type h: torch.doubleTensor

		:returns: The probability of visible units being active given the
				  hidden state.
		:rtype: torch.doubleTensor
		"""
		p = F.sigmoid(F.linear(h, self.weights.t(), self.visible_bias))
		return p

	def prob_h_given_v(self, v):
		"""Given a visible unit configuration, compute the probability
		vector of the hidden units being on.

		:param h: The hidden unit.
		:type h: torch.doubleTensor

		:returns: The probability of hidden units being active given the
				  visible state.
		:rtype: torch.doubleTensor
		"""
		p = F.sigmoid(F.linear(v, self.weights, self.hidden_bias))
		return p

	def sample_v_given_h(self, h):
		"""Sample/generate a visible state given a hidden state.

		:param h: The hidden state.
		:type h: torch.doubleTensor

		:returns: Tuple containing prob_v_given_h(h) and the sampled visible
				  state.
		:rtype: tuple(torch.doubleTensor, torch.doubleTensor)
		"""
		p = self.prob_v_given_h(h)
		v = p.bernoulli()
		return p, v

	def sample_h_given_v(self, v):
		"""Sample/generate a hidden state given a visible state.

		:param h: The visible state.
		:type h: torch.doubleTensor

		:returns: Tuple containing prob_h_given_v(v) and the sampled hidden
				  state.
		:rtype: tuple(torch.doubleTensor, torch.doubleTensor)
		"""
		p = self.prob_h_given_v(v)
		h = p.bernoulli()
		return p, h

	def gibbs_sampling(self, k, v0):
		"""Performs k steps of Block Gibbs sampling given an initial visible state v0.

		:param k: Number of Block Gibbs steps.
		:type k: int
		:param v0: The initial visible state.
		:type v0: torch.doubleTensor

		:returns: Tuple containing the initial visible state, v0,
				  the hidden state sampled from v0,
				  the visible state sampled after k steps,
				  the hidden state sampled after k steps and its corresponding
				  probability vector.
		:rtype: tuple(torch.doubleTensor, torch.doubleTensor,
					  torch.doubleTensor, torch.doubleTensor,
					  torch.doubleTensor)
		"""
		ph, h0 = self.sample_h_given_v(v0)
		v, h = v0, h0
		for _ in range(k):
			pv, v = self.sample_v_given_h(h)
			ph, h = self.sample_h_given_v(v)
		return v0, h0, v, h, ph

	def sample(self, num_samples, k):
		"""Samples from the RBM using k steps of Block Gibbs sampling.

		:param num_samples: The number of samples to be generated
		:type num_samples: int
		:param k: Number of Block Gibbs steps.
		:type k: int

		:returns:
		:rtype: torch.doubleTensor
		"""
		dist = torch.distributions.bernoulli.Bernoulli(probs=0.5)
		v0 = (dist.sample(torch.Size([num_samples, self.num_visible]))
				  .to(device=self.device, dtype=torch.double))
		_, _, v, _, _ = self.gibbs_sampling(k, v0)
		return v

	def unnormalized_probability(self, v):
		"""The unnormalized probabilities of the given visible states.
		.. :math: `p(v) = e^E_{\\lambda}(v)`

		:param v: The visible states.
		:type v: torch.doubleTensor

		:returns: The unnormalized probability of the given visible state(s).
		:rtype: torch.doubleTensor
		"""
		return self.effective_energy(v).exp()

	def probability_ratio(self, a, b):
		"""The probability ratio of two sets of visible states

		.. note:: `a` and `b` must be have the same shape

		:param a: The visible states for the numerator.
		:type a: torch.doubleTensor
		:param b: The visible states for the denominator.
		:type b: torch.doubleTensor

		:returns: The probability ratios of the given visible states
		:rtype: torch.doubleTensor
		"""
		prob_a = self.unnormalized_probability(a)
		prob_b = self.unnormalized_probability(b)

		return prob_a.div(prob_b)

	def log_probability_ratio(self, a, b):
		"""The natural logarithm of the probability ratio of
		two sets of visible states

		.. note:: `a` and `b` must be have the same shape

		:param a: The visible states for the numerator.
		:type a: torch.doubleTensor
		:param b: The visible states for the denominator.
		:type b: torch.doubleTensor

		:returns: The log-probability ratios of the given visible states
		:rtype: torch.doubleTensor
		"""
		log_prob_a = self.effective_energy(a)
		log_prob_b = self.effective_energy(b)

		return log_prob_a.sub(log_prob_b)

	def generate_visible_space(self):
		"""Generates all possible visible states.

		:returns: A tensor of all possible spin configurations.
		:rtype: torch.doubleTensor
		"""
		space = torch.zeros((1 << self.num_visible, self.num_visible),
							device=self.device, dtype=torch.double)
		for i in range(1 << self.num_visible):
			d = i
			for j in range(self.num_visible):
				d, r = divmod(d, 2)
				space[i, self.num_visible - j - 1] = int(r)

		return space

	def log_partition(self, visible_space):
		"""The natural logarithm of the partition function of the RBM.

		:param visible_space: A rank 2 tensor of the entire visible space.
		:type visible_space: torch.doubleTensor

		:returns: The natural log of the partition function.
		:rtype: torch.doubleTensor
		"""
		free_energies = self.effective_energy(visible_space)
		max_free_energy = free_energies.max()

		f_reduced = free_energies - max_free_energy
		logZ = max_free_energy + f_reduced.exp().sum().log()

		return logZ

	def partition(self, visible_space):
		"""The partition function of the RBM.

		:param visible_space: A rank 2 tensor of the entire visible space.
		:type visible_space: torch.doubleTensor

		:returns: The partition function.
		:rtype: torch.doubleTensor
		"""
		return self.log_partition(visible_space).exp()

	def nll(self, data, visible_space):
		"""The negative log likelihood of the given data.

		:param data: A rank 2 tensor of visible states.
		:type data: torch.doubleTensor
		:param visible_space: A rank 2 tensor of the entire visible space.
		:type visible_space: torch.doubleTensor

		:returns: The negative log likelihood of the given data.
		:rtype: torch.doubleTensor
		"""
		total_free_energy = self.effective_energy(data).sum()
		logZ = self.log_partition(visible_space)
		return (len(data)*logZ) - total_free_energy

class BinomialRBM:

	def __init__(self, num_visible, num_hidden, gpu=True, seed=1234):

		self.num_visible = int(num_visible)
		self.num_hidden  = int(num_hidden)
		self.rbm_module  = RBM_Module(num_visible, num_hidden, gpu=gpu, seed=seed)

	def regularize_weight_gradients(self, w_grad, l1_reg, l2_reg):
		"""Applies regularization to the given weight gradient

		:param w_grad: The entire weight gradient matrix,
					   :math:`\\nabla_{\\lambda_{weights}}NLL`.
		:type w_grad: torch.doubleTensor
		:param l1_reg: l1 regularization hyperparameter (default = 0.0).
		:type l1_reg: double
		:param l2_reg: l2 regularization hyperparameter (default = 0.0).
		:type l2_reg: double

		:returns:
		.. :math: `[\\nabla_{W_{\\lambda}}NLL]_{i,j} =
				   [\\nabla_{W_{\\lambda}}NLL]_{i,j}
					+ L_1sign([W_{\\lambda}]_{i,j})
					+ L_2\\vert[W_{\\lambda}]_{i,j}\\vert^2`.
		:rtype: torch.doubleTensor
		"""
		return (w_grad
				+ (l2_reg * self.rbm_module.weights)
				+ (l1_reg * self.rbm_module.weights.sign()))

	def compute_batch_gradients(self, k, batch,
								persistent=False,
								pbatch=None,
								l1_reg=0.0, l2_reg=0.0,
								stddev=0.0):
		"""This function will compute the gradients of a batch of the training
		data (data_file) given the basis measurements (chars_file).

		:param k: Number of contrastive divergence steps in training.
		:type k: int
		:param batch: Batch of the input data.
		:type batch: torch.doubleTensor
		:param L1_reg: L1 regularization hyperparameter (default = 0.0)
		:type L1_reg: double
		:param L2_reg: L2 regularization hyperparameter (default = 0.0)
		:type L2_reg: double
		:param stddev: Standard deviation of random noise that can be added to
					   the weights.	This is also a hyperparameter.
					   (default = 0.0)
		:type stddev: double

		:returns: Dictionary containing all the gradients.
		:rtype: dict
		"""
		if len(batch) == 0:
			return (torch.zeros_like(self.rbm_module.weights,
									 device=self.self.rbm_module.device,
									 dtype=torch.double),
					torch.zeros_like(self.rbm_module.visible_bias,
									 device=self.rbm_module.device,
									 dtype=torch.double),
					torch.zeros_like(self.rbm_module.hidden_bias,
									 device=self.rbm_module.device,
									 dtype=torch.double))

		if not persistent:
			v0, h0, vk, hk, phk = self.rbm_module.gibbs_sampling(k, batch)
		else:
			# Positive phase comes from training data
			v0, h0, _, _, _ = self.rbm_module.gibbs_sampling(0, batch)
			# Negative phase comes from Markov chains
			_, _, vk, hk, phk = self.rbm_module.gibbs_sampling(k, pbatch)

		w_grad  = torch.einsum("ij,ik->jk", (h0, v0))
		vb_grad = torch.einsum("ij->j", (v0,))
		hb_grad = torch.einsum("ij->j", (h0,))

		w_grad  -= torch.einsum("ij,ik->jk", (phk, vk))
		vb_grad -= torch.einsum("ij->j", (vk,))
		hb_grad -= torch.einsum("ij->j", (phk,))

		w_grad  /= float(len(batch))
		vb_grad /= float(len(batch))
		hb_grad /= float(len(batch))

		w_grad = self.regularize_weight_gradients(w_grad, l1_reg, l2_reg)

		if stddev != 0.0:
			w_grad += (stddev * torch.randn_like(w_grad, device=self.rbm_module.device))

		# Return negative gradients to match up nicely with the usual
		# parameter update rules, which *subtract* the gradient from
		# the parameters. This is in contrast with the RBM update
		# rules which ADD the gradients (scaled by the learning rate)
		# to the parameters.
		return (
			{"weights": -w_grad,
			 "visible_bias": -vb_grad,
			 "hidden_bias": -hb_grad},
			(vk if persistent else None)
		)

	def fit(self, data, epochs, batch_size,
			  k=10, persistent=False,
			  lr=1e-3, momentum=0.0,
			  l1_reg=0.0, l2_reg=0.0,
			  initial_gaussian_noise=0.0, gamma=0.55,
			  callbacks=[], progbar=False, starting_epoch=0):
		"""Execute the training of the RBM.

		:param data: The actual training data
		:type data: array_like of doubles
		:param epochs: The number of parameter (i.e. weights and biases)
					   updates (default = 100).
		:type epochs: int
		:param batch_size: The size of batches taken from the data
						   (default = 100).
		:type batch_size: int
		:param k: The number of contrastive divergence steps (default = 10).
		:type k: int
		:param persistent: Whether to use persistent contrastive divergence
						   (aka. Stochastic Maximum Likelihood)
						   (default = False).
		:type persistent: bool
		:param lr: Learning rate (default = 1.0e-3).
		:type lr: double
		:param momentum: Momentum hyperparameter (default = 0.0).
		:type momentum: double
		:param l1_reg: L1 regularization hyperparameter (default = 0.0).
		:type l1_reg: double
		:param l2_reg: L2 regularization hyperparameter (default = 0.0).
		:type l2_reg: double
		:param initial_gaussian_noise: Initial gaussian noise used to calculate
									   stddev of random noise added to weight
									   gradients (default = 0.01).
		:type initial_gaussian_noise: double
		:param gamma: Parameter used to calculate stddev (default = 0.55).
		:type gamma: double
		:param callbacks: A list of Callback functions to call at the beginning
						  of each epoch
		:type callbacks: [Callback]
		:param progbar: Whether or not to display a progress bar. If "notebook"
						is passed, will use a Jupyter notebook compatible
						progress bar.
		:type progbar: bool or "notebook"
		:param starting_epoch: Which epoch to start from. Doesn't actually
							   affect the model, exists simply for book-keeping
							   when restarting training.

		:returns: None
		"""

		disable_progbar = (progbar is False)
		progress_bar = tqdm_notebook if progbar == "notebook" else tqdm

		data = torch.tensor(data).to(device=self.rbm_module.device,
									 dtype=torch.double)
		optimizer = torch.optim.SGD([self.rbm_module.weights,
									 self.rbm_module.visible_bias,
									 self.rbm_module.hidden_bias],
									lr=lr)

		if persistent:
			dist = torch.distributions.bernoulli.Bernoulli(probs=0.5)
			pbatch = (dist.sample(torch.Size([batch_size, self.num_visible]))
						  .to(device=self.rbm_module.device, dtype=torch.double))
		else:
			pbatch = None

		for ep in progress_bar(range(starting_epoch, epochs + 1),
							   desc="Epochs ", total=epochs,
							   disable=disable_progbar):
			batches = DataLoader(data, batch_size=batch_size, shuffle=True)

			for cb in callbacks:
				cb(self, ep)

			if self.stop_training:  # check for stop_training signal
				break

			if ep == epochs:
				break

			stddev = torch.tensor(
				[initial_gaussian_noise / ((1 + ep) ** gamma)],
				dtype=torch.double, device=self.rbm_module.device).sqrt()

			for batch in progress_bar(batches, desc="Batches",
									  leave=False, disable=True):
				grads, pbatch = self.compute_batch_gradients(
					k, batch,
					pbatch=pbatch,
					persistent=persistent,
					l1_reg=l1_reg, l2_reg=l2_reg, stddev=stddev)

				optimizer.zero_grad()  # clear any cached gradients

				# assign all available gradients to the corresponding parameter
				for name in grads.keys():
					getattr(self, name).grad = grads[name]

				optimizer.step()  # tell the optimizer to apply the gradients

class ComplexRBM:

	def __init__(self, unitaries, psi_dictionary, num_visible,
				 num_hidden_amp, num_hidden_phase, gpu=True, seed=1234):
		self.num_visible      = int(num_visible)
		self.num_hidden_amp   = int(num_hidden_amp)
		self.num_hidden_phase = int(num_hidden_phase)
		self.unitaries        = unitaries
		self.psi_dictionary   = psi_dictionary
		self.rbm_amp          = RBM_Module(num_visible, num_hidden_amp, gpu=gpu, seed=seed)
		self.rbm_phase        = RBM_Module(num_visible, num_hidden_phase, zero_weights=True, gpu=gpu, seed=seed)
		self.device           = self.rbm_amp.device

	def basis_state_generator(self, s):
		"""Only works for binary visible units at the moment. Generates a vector
	    given a spin value (0 or 1).

		:param s: A spin's value (either 0 or 1).
		:type s: double

		:returns: If s = 0, this is the (1,0) state in the basis of the
				  measurement. If s = 1, this is the (0,1) state in the basis of
				  the measurement.
		:rtype: torch.doubleTensor	
		"""
		if s == 0.:
			return torch.tensor([[1., 0.],[0., 0.]], dtype = torch.double)
		if s == 1.:
			return torch.tensor([[0., 1.],[0., 0.]], dtype = torch.double) 

	def state_generator(self, num_non_trivial_unitaries):
		"""A function that returns all possible configurations of
		'num_non_trivial_unitaries' spins. Similar to generate_visible_space.

		:param num_non_trivial_unitaries: The number of sites measured in the
										  non-computational basis.
		:type num_non_trivial_unitaries: int
  
		:returns: An array of all possible spin configurations of
				  'num_non_trivial_unitaries' spins.
		:rtype: torch.doubleTensor	
		"""
		states = torch.zeros((2**num_non_trivial_unitaries, 
							  num_non_trivial_unitaries), device = self.device,
							 dtype=torch.double)
		for i in range(2**num_non_trivial_unitaries):
			temp = i
			for j in range(num_non_trivial_unitaries): 
				temp, remainder = divmod(temp, 2)
				states[i][num_non_trivial_unitaries - j - 1] = remainder
		return states

	def unnormalized_probability_amp(self, v):
		"""The effective energy of the phase RBM.

		:param v: Visible unit(s).
		:type v: torch.doubleTensor

		:returns: :math:`p_{\\lambda} = e^{E_{\\lambda}}`.
		:rtype: torch.doubleTensor
		""" 
		return self.rbm_amp.effective_energy(v).exp()

	def unnormalized_probability_phase(self, v):
		"""The effective energy of the phase RBM.

		:param v: Visible unit(s).
		:type v: torch.doubleTensor

		:returns: :math:`p_{\\mu} = e^{E_{\\mu}}`.
		:rtype: torch.doubleTensor
		""" 
		return self.rbm_phase.effective_energy(v).exp()

	def normalized_wavefunction(self, v):
		"""The RBM wavefunction.

		:param v: Visible unit(s).
		:type v: torch.doubleTensor
		
		:returns: :math:`\\psi_{\\lambda\\mu} = \\sqrt{\\frac{p_{\\lambda}}{Z_{\\lambda}}}e^{\\frac{i\\log(p_{\\mu})}{2}}`.
		:rtype: torch.doubleTensor
		""" 
		v_prime   = v.view(-1,self.num_visible)
		temp1     = (self.unnormalized_probability_amp(v_prime)).sqrt()
		temp2     = ((self.unnormalized_probability_phase(v_prime)).log())*0.5

		cos_angle = temp2.cos()
		sin_angle = temp2.sin()
		
		psi       = torch.zeros(2, v_prime.size()[0], dtype = torch.double)
		psi[0]    = temp1*cos_angle
		psi[1]    = temp1*sin_angle

		sqrt_Z    = (self.rbm_amp.partition(self.rbm_amp.generate_visible_space())).sqrt()

		return psi / sqrt_Z

	def unnormalized_wavefunction(self, v):
		"""The unnormalized RBM wavefunction.

		:param v: Visible unit(s).
		:type v: torch.doubleTensor
	
		:returns: :math:`\\tilde{\\psi}_{\\lambda\mu} =\\sqrt{p_{\\lambda}}e^{\\frac{i\\log(p_{\\mu})}{2}}`.
		:rtype: torch.doubleTensor
		""" 
		v_prime   = v.view(-1,self.num_visible)
		temp1     = (self.unnormalized_probability_amp(v_prime)).sqrt()
		temp2     = ((self.unnormalized_probability_phase(v_prime)).log())*0.5
		cos_angle = temp2.cos()
		sin_angle = temp2.sin()
		
		psi       = torch.zeros(2, v_prime.size()[0], dtype = torch.double)
		
		psi[0]    = temp1*cos_angle
		psi[1]    = temp1*sin_angle

		return psi

	def get_true_psi(self, basis):
		"""Picks out the true psi in the correct basis.
		
		:param basis: E.g. XZZZX.
		:type basis: str

		:returns: The true wavefunction in the basis.
		:rtype: torch.doubleTensor
		"""
		key = ''
		for i in range(len(basis)):
			key += basis[i]
		return self.psi_dictionary[key]

	def overlap(self, visible_space, basis):
		"""Computes the overlap between the RBM and true wavefunctions.
		
		:param visible_space: An array of all possible spin configurations.
		:type visible_space: torch.doubleTensor	
		:param basis: E.g. XZZZX.
		:type basis: str
		
		:returns: :math:`O = \\langle{\\psi_{true}}\\vert\\psi_{\\lambda\\mu}\\rangle`.
		:rtype: double		
		"""
		overlap_ = cplx.inner_prod(self.get_true_psi(basis),
						   self.normalized_wavefunction(visible_space))
		return overlap_

	def fidelity(self, visible_space, basis):
		"""Computed the fidelity of the RBM and true wavefunctions. 
	
		:param visible_space: An array of all possible spin configurations.
		:type visible_space: torch.doubleTensor	
		:param basis: E.g. XZZZX.
		:type basis: str
		
		:returns: :math:`F = |O|^2`.
		:rtype: double		
		"""
		return cplx.norm(self.overlap(visible_space, basis))

	def compute_batch_gradients(self, k, batch, chars_batch, l1_reg, l2_reg, 
								stddev=0.0):
		"""This function will compute the gradients of a batch of the training 
		data (data_file) given the basis measurements (chars_file).

		:param k: Number of contrastive divergence steps in amplitude training.
		:type k: int
		:param batch: Batch of the input data.
		:type batch: torch.doubleTensor
		:param chars_batch: Batch of bases that correspondingly indicates the 
							basis each site in the batch was measured in.
		:type chars_batch: array_like (str)
		:param l1_reg: L1 regularization hyperparameter (default = 0.0)
		:type l1_reg: double
		:param l2_reg: L2 regularization hyperparameter (default = 0.0)
		:type l2_reg: double
		:param stddev: Standard deviation of random noise that can be added to 
					   the weights.	This is also a hyperparamter. (default = 0.0)
		:type stddev: double

		:returns: Dictionary containing all the gradients (negative): Gradient 
				  of weights, visible bias and hidden bias for the amplitude, 
				  Gradients of weights, visible bias and hidden bias for the 
				  phase.
		:rtype: dict
		"""
		
		vis = self.rbm_amp.generate_visible_space()
		batch_size = len(batch)

		g_weights_amp   = torch.zeros_like(self.rbm_amp.weights)
		g_vb_amp        = torch.zeros_like(self.rbm_amp.visible_bias)
		g_hb_amp        = torch.zeros_like(self.rbm_amp.hidden_bias)

		g_weights_phase = torch.zeros_like(self.rbm_phase.weights)
		g_vb_phase      = torch.zeros_like(self.rbm_phase.visible_bias)
		g_hb_phase      = torch.zeros_like(self.rbm_phase.hidden_bias)

		[batch, h0_amp_batch, vk_amp_batch, hk_amp_batch, phk_amp_batch] = self.rbm_amp.gibbs_sampling(k, batch) 
		# Iterate through every data point in the batch.
		for row_count, v0 in enumerate(batch):

			# A counter for the number of non-trivial unitaries 
			# (non-computational basis) in the data point.
			num_non_trivial_unitaries = 0

			# tau_indices will contain the index numbers of spins not in the 
			# computational basis (Z). z_indices will contain the index numbers 
			# of spins in the computational basis.
			tau_indices = []
			z_indices   = []

			for j in range(self.num_visible):
				# Go through the unitaries (chars_batch[row_count]) of each site
				# in the data point, v0, and save inidices of non-trivial.
				if chars_batch[row_count][j] != 'Z':
					num_non_trivial_unitaries += 1
					tau_indices.append(j)
				else:
					z_indices.append(j)

			if num_non_trivial_unitaries == 0:
				# If there are no non-trivial unitaries for the data point v0, 
				# calculate the positive phase of regular (i.e. non-complex RBM)
				# gradient. Use the actual data point, v0.
				g_weights_amp -= torch.einsum("i,j->ij",(h0_amp_batch[row_count], v0)) / batch_size 
				g_vb_amp      -= v0 / batch_size
				g_hb_amp      -= h0_amp_batch[row_count] / batch_size

			else:
				# Compute the rotated gradients.
				[L_weights_amp, L_vb_amp, L_hb_amp, L_weights_phase, L_vb_phase, 
				L_hb_phase] = self.compute_rotated_grads(v0, 
														 chars_batch[row_count],
														 num_non_trivial_unitaries, 
														 z_indices, tau_indices) 


				# Gradents of amplitude parameters take the real part of the 
				# rotated gradients.
				g_weights_amp -= L_weights_amp[0] / batch_size
				g_vb_amp      -= L_vb_amp[0] / batch_size
				g_hb_amp      -= L_hb_amp[0] / batch_size
				
				# Gradents of phase parameters take the imaginary part of the 
				# rotated gradients.
				g_weights_phase += L_weights_phase[1] / batch_size
				g_vb_phase      += L_vb_phase[1] / batch_size
				g_hb_phase      += L_hb_phase[1] / batch_size
				
		# Block gibbs sampling for negative phase.	
		g_weights_amp += torch.einsum("ij,ik->jk",(phk_amp_batch, vk_amp_batch)) / batch_size 
		g_vb_amp      += torch.einsum("ij->j", (vk_amp_batch,)) / batch_size
		g_hb_amp      += torch.einsum("ij->j", (phk_amp_batch,)) / batch_size
	   
		# Perform weight regularization if l1_reg and/or l2_reg are not zero.
		if l1_reg != 0 or l2_reg != 0:
			g_weights_amp   = self.regularize_weight_gradients_amp(g_weights_amp, 
																   l1_reg, 
																   l2_reg)
			g_weights_phase = self.regularize_weight_gradients_phase(g_weights_phase,
																	 l1_reg, 
																	 l2_reg)

		# Add small random noise to weight gradients if stddev is not zero.
		if stddev != 0.0:
			g_weights_amp   += (stddev*torch.randn_like(g_weights_amp, 
														device = self.device))
			g_weights_phase += (stddev*torch.randn_like(g_weights_phase, 
														device = self.device))
		

		"""Return negative gradients to match up nicely with the usual
		parameter update rules, which *subtract* the gradient from
		the parameters. This is in contrast with the RBM update
		rules which ADD the gradients (scaled by the learning rate)
		to the parameters."""
 
		return { "rbm_amp":{"weights": g_weights_amp,
				"visible_bias": g_vb_amp,
				"hidden_bias": g_hb_amp},
				 "rbm_phase":{
				"weights": g_weights_phase,
				"visible_bias": g_vb_phase,
				"hidden_bias": g_hb_phase}
				}   

	def compute_rotated_grads(self, v0, characters, num_non_trivial_unitaries, 
							  z_indices, tau_indices): 
		"""Computes the rotated gradients.

		:param v0: A visible unit.
		:type v0: torch.doubleTensor
		:param characters: A string of characters corresponding to the basis 
						   that each site in v0 was measured in.
		:type characters: str
		:param num_non_trivial_unitaries: The number of sites in v0 that are not
										  measured in the computational basis.
		:type num_non_trivial_unitaries: int
		:param z_indices: A list of indices that correspond to sites of v0 that 
						  are measured in the computational basis.
		:type z_indices: list of ints
		:param tau_indices: A list of indices that correspond to sites of v0 
							that are not measured in the computational basis.	
		:type tau_indices: list of ints	

		:returns: Dictionary of the rotated gradients: L_weights_amp, L_vb_amp, 
				  L_hb_amp, L_weights_phase, L_vb_phase, L_hb_phase
		:rtype: dict
		"""
		"""Initialize the 'A' parameters (see alg 4.2)."""
		A_weights_amp = torch.zeros(2, self.rbm_amp.weights.size()[0], 
									self.rbm_amp.weights.size()[1], 
									device=self.device, dtype = torch.double)
		A_vb_amp      = torch.zeros(2, self.rbm_amp.visible_bias.size()[0], 
									device=self.device, dtype = torch.double)
		A_hb_amp      = torch.zeros(2, self.rbm_amp.hidden_bias.size()[0], 
									device=self.device, dtype = torch.double)
		
		A_weights_phase = torch.zeros(2, self.rbm_phase.weights.size()[0], 
									  self.rbm_phase.weights.size()[1], 
									  device=self.device, dtype = torch.double)
		A_vb_phase      = torch.zeros(2, self.rbm_phase.visible_bias.size()[0], 
									  device=self.device, dtype = torch.double)
		A_hb_phase      = torch.zeros(2, self.rbm_phase.hidden_bias.size()[0], 
									  device=self.device, dtype = torch.double)
		# 'B' will contain the coefficients of the rotated unnormalized wavefunction.
		B = torch.zeros(2, device = self.device, dtype = torch.double)

		w_grad_amp      = torch.zeros_like(self.rbm_amp.weights)
		vb_grad_amp     = torch.zeros_like(self.rbm_amp.visible_bias)
		hb_grad_amp     = torch.zeros_like(self.rbm_amp.hidden_bias)
	
		w_grad_phase    = torch.zeros_like(self.rbm_phase.weights)
		vb_grad_phase   = torch.zeros_like(self.rbm_phase.visible_bias)
		hb_grad_phase   = torch.zeros_like(self.rbm_phase.hidden_bias)

		zeros_for_w_amp    = torch.zeros_like(w_grad_amp)
		zeros_for_w_phase  = torch.zeros_like(w_grad_phase)
		zeros_for_vb       = torch.zeros_like(vb_grad_amp)
		zeros_for_hb_amp   = torch.zeros_like(hb_grad_amp) 
		zeros_for_hb_phase = torch.zeros_like(hb_grad_phase)

		# Loop over Hilbert space of the non trivial unitaries to build the state.
		for j in range(2**num_non_trivial_unitaries):
			s = self.state_generator(num_non_trivial_unitaries)[j]
			# Creates a matrix where the jth row is the desired state, |S>, a vector.
	
			# This is the sigma state.
			constructed_state = torch.zeros(self.num_visible, dtype = torch.double)
			
			U = torch.tensor([1., 0.], dtype = torch.double, device = self.device)
		
			# Populate the |sigma> state (aka constructed_state) accirdingly.
			for index in range(len(z_indices)):
				# These are the sites in the computational basis.
				constructed_state[z_indices[index]] = v0[z_indices[index]]
		
			for index in range(len(tau_indices)):
				# These are the sites that are NOT in the computational basis.
				constructed_state[tau_indices[index]] = s[index]
		
				aa = self.unitaries[characters[tau_indices[index]]]
				bb = self.basis_state_generator(v0[tau_indices[index]])
				cc = self.basis_state_generator(s[index])
			
				temp = cplx.inner_prod( cplx.MV_mult(cplx.compT_matrix(aa), bb), cc )
		
				U = cplx.scalar_mult(U, temp)
			
			# Positive phase gradients for phase and amp. Will be added into the
			# 'A' parameters.
			w_grad_amp  = torch.ger(F.sigmoid(F.linear(constructed_state, 
													   self.rbm_amp.weights, 
													   self.rbm_amp.hidden_bias)),
									constructed_state)
			vb_grad_amp = constructed_state
			hb_grad_amp = F.sigmoid(F.linear(constructed_state,
											 self.rbm_amp.weights,
											 self.rbm_amp.hidden_bias))

			w_grad_phase  = torch.ger(F.sigmoid(F.linear(constructed_state,
														 self.rbm_phase.weights,
														 self.rbm_phase.hidden_bias)),
									  constructed_state)
			vb_grad_phase = constructed_state
			hb_grad_phase = F.sigmoid(F.linear(constructed_state,
											   self.rbm_phase.weights,
											   self.rbm_phase.hidden_bias))

			"""
			In order to calculate the 'A' parameters below with my current
			complex library, I need to make the weights and biases complex.
			I fill the complex parts of the parameters with a tensor of zeros.
			"""
			temp_w_grad_amp  = cplx.make_complex_matrix(w_grad_amp, 
														zeros_for_w_amp)
			temp_vb_grad_amp = cplx.make_complex_vector(vb_grad_amp, 
														zeros_for_vb)
			temp_hb_grad_amp = cplx.make_complex_vector(hb_grad_amp, 
														zeros_for_hb_amp)
 
			temp_w_grad_phase  = cplx.make_complex_matrix(w_grad_phase, 
														  zeros_for_w_phase)
			temp_vb_grad_phase = cplx.make_complex_vector(vb_grad_phase,
														  zeros_for_vb)
			temp_hb_grad_phase = cplx.make_complex_vector(hb_grad_phase,
														  zeros_for_hb_phase)
		
			# Temp = U*psi(sigma)
			temp = cplx.scalar_mult(U, self.unnormalized_wavefunction(constructed_state))
			
			A_weights_amp += cplx.MS_mult(temp, temp_w_grad_amp)
			A_vb_amp      += cplx.VS_mult(temp, temp_vb_grad_amp)
			A_hb_amp      += cplx.VS_mult(temp, temp_hb_grad_amp)
		
			A_weights_phase += cplx.MS_mult(temp, temp_w_grad_phase)
			A_vb_phase      += cplx.VS_mult(temp, temp_vb_grad_phase)
			A_hb_phase      += cplx.VS_mult(temp, temp_hb_grad_phase)
		   
			# Rotated wavefunction.
			B += temp
		
		L_weights_amp = cplx.MS_divide(A_weights_amp, B)
		L_vb_amp      = cplx.VS_divide(A_vb_amp, B)
		L_hb_amp      = cplx.VS_divide(A_hb_amp, B)
		
		L_weights_phase = cplx.MS_divide(A_weights_phase, B)
		L_vb_phase      = cplx.VS_divide(A_vb_phase, B)
		L_hb_phase      = cplx.VS_divide(A_hb_phase, B)
	   
		return [L_weights_amp, L_vb_amp, L_hb_amp, L_weights_phase, L_vb_phase,
				L_hb_phase ]

	def fit(self, data, character_data, epochs, batch_size, k=1, lr=1e-3, 
			  l1_reg=0.0, l2_reg=0.0, initial_gaussian_noise=0.01, gamma=0.55, 
			  log_every=50, **kwargs):

		"""Execute the training of the RBM.

		:param data: The actual training data
		:type data: array_like of doubles 
		:param character_data: The corresponding bases that each site in the 
							   data has been measured in.
		:type character_data: array_like of str's
		:param epochs: The number of parameter (i.e. weights and biases) updates
					   (default = 100).
		:type epochs: int
		:param batch_size: The size of batches taken from the data (default = 100).
		:type batch_size: int
		:param k: The number of contrastive divergence steps (default = 1).
		:type k: int
		:param lr: Learning rate (default = 1.0e-3).
		:type lr: double		
		:param l1_reg: L1 regularization hyperparameter (default = 0.0).
		:type l1_reg: double
		:param l2_reg: L2 regularization hyperparameter (default = 0.0).
		:type l2_reg: double
		:param initial_gaussian_noise: Initial gaussian noise used to calculate 
									   stddev of random noise added to weight gradients (default = 0.01).
		:type initial_gaussian_noise: double	
		:param gamma: Parameter used to calculate stddev (default = 0.55).
		:type gamma: double
		:param log_every: Indicates how often (i.e. after how many epochs) to 
						  calculate convergence parameters (e.g. fidelity, 
						  energy, etc.).
		:type log_every: int

		:returns: Currently returns nothing. Just calculates fidelity. Will 
				  eventually need to return weights and biases (and save them). 
		"""
		# Make data file into a torch tensor.
		data = torch.tensor(data).to(device=self.device)

		# Use the Adam optmizer to update the weights and biases.
		optimizer = torch.optim.Adam([self.rbm_amp.weights,
									  self.rbm_amp.visible_bias,
									  self.rbm_amp.hidden_bias,
									  self.rbm_phase.weights,
									  self.rbm_phase.visible_bias,
									  self.rbm_phase.hidden_bias],
									  lr=lr)

		vis = self.rbm_amp.generate_visible_space()
		print ('Generated visible space. Ready to begin training.')

		# Empty lists to put calculated convergence quantities in.
		fidelity_list = []
		epoch_list = []

		for ep in range(0,epochs+1):
			# Shuffle the data to ensure that the batches taken from the data 
			# are random data points.
			random_permutation = torch.randperm(data.shape[0])

			shuffled_data           = data[random_permutation]   
			shuffled_character_data = character_data[random_permutation]

			# List of all the batches.
			batches = [shuffled_data[batch_start:(batch_start + batch_size)] 
					   for batch_start in range(0, len(data), batch_size)]

			# List of all the bases.
			char_batches = [shuffled_character_data[batch_start:(batch_start + batch_size)] 
							for batch_start in range(0, len(data), batch_size)]

			# Calculate convergence quantities every "log-every" steps.
			if ep % log_every == 0:
				fidelity_ = self.fidelity(vis, 'Z' 'Z')
				print ('Epoch = ',ep,'\nFidelity = ',fidelity_)
				fidelity_list.append(fidelity_)
				epoch_list.append(ep)
		   
			# Save fidelities at the end of training.
			if ep == epochs:
				print ('Finished training. Saving results...' )               
				fidelity_file = open('fidelity_file.txt', 'w')

				for i in range(len(fidelity_list)):
					fidelity_file.write('%.5f' % fidelity_list[i] + ' %d\n' % epoch_list[i])

				fidelity_file.close()
				print ('Done.')
				break
			
			stddev = torch.tensor(
				[initial_gaussian_noise / ((1 + ep) ** gamma)],
				dtype=torch.double, device=self.device).sqrt()

			# Loop through all of the batches and calculate the batch gradients.
			for batch_index in range(len(batches)):
				
				all_grads = self.compute_batch_gradients(k, batches[batch_index], 
													 char_batches[batch_index],
													 l1_reg, l2_reg,
													 stddev=stddev)

				# Clear any cached gradients.
				optimizer.zero_grad()  

				# Assign all available gradients to the corresponding parameter.
				for name, grads  in all_grads.items():
					selected_RBM = getattr(self, name)
					for param in grads.keys():
						getattr(selected_RBM, param).grad = grads[param]
			   
				# Tell the optimizer to apply the gradients and update the parameters.
				optimizer.step()  
