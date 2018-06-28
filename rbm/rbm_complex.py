import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
import warnings
import cplx
from matplotlib import pyplot as plt

class RBM(nn.Module):
	"""Class to build the Restricted Boltzmann Machine

	Parameters
	----------
	unitaries : dict
		Dictionary of unitary names (key) that correspond to (2x2) unitary matrices (value)
	psi_dictionary : dict
		Dictionary of true wavefunctions of the system (value) in a particular basis (key)
	num_visible : int
		Number of visible units (determined from training data)
	num_hidden_amp : int
		Number of hidden units to learn the amplitude (default = num_visible)
	num_hidden_phase : int
		Number of hidden units to learn the phase (default = num_visible)
	gpu : bool
		Should the GPU be used for the training (default = True)
	seed : int
		Fix the random number seed to make results reproducable (default = 1234)
	
	"""

	def __init__(self, unitaries, psi_dictionary, num_visible,
				 num_hidden_amp, num_hidden_phase, gpu=True, seed=1234):
		super(RBM, self).__init__()
		self.num_visible      = int(num_visible)
		self.num_hidden_amp   = int(num_hidden_amp)
		self.num_hidden_phase = int(num_hidden_phase)
		self.unitaries        = unitaries
		self.psi_dictionary   = psi_dictionary

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

		'''Initialize weights for amplitude as random numbers from a normal distribution, phase weights as zero.'''
		self.weights_amp = nn.Parameter(
			(torch.randn(self.num_hidden_amp, self.num_visible,
						 device=self.device, dtype=torch.double)
			 / np.sqrt(self.num_visible)),
			requires_grad=True)
		
		self.weights_phase = nn.Parameter(
			(torch.zeros(self.num_hidden_phase, self.num_visible,
						 device=self.device, dtype=torch.double)), requires_grad=True)

		'''Initialize all biases to zero.'''
		self.visible_bias_amp   = nn.Parameter(torch.zeros(self.num_visible,
													 device=self.device,
													 dtype=torch.double),
										 requires_grad=True)

		self.visible_bias_phase = nn.Parameter(torch.zeros(self.num_visible,
													 device=self.device,
													 dtype=torch.double),
										 requires_grad=True)

		self.hidden_bias_amp    = nn.Parameter(torch.zeros(self.num_hidden_amp,
													device=self.device,
													dtype=torch.double),
										requires_grad=True)

		self.hidden_bias_phase  = nn.Parameter(torch.zeros(self.num_hidden_phase,
													device=self.device,
													dtype=torch.double),
										requires_grad=True)
		
	def compute_batch_gradients(self, k, batch, chars_batch, l1_reg, l2_reg, stddev=0.0):
		'''This function will compute the gradients of a batch of the training 
		data (data_file) given the basis measurements (chars_file).

		Parameters
		----------
		k : int
			Number of contrastive divergence steps in amplitude training
		batch : array_like
			Batch of the input data
		chars_batch : array_like
			Batch of bases that correspondingly indicates the basis each site in the batch was measured in
		l1_reg : float
			L1 regularization hyperparameter (default = 0.0)
		l2_reg : float
			L2 regularization hyperparameter (default = 0.0)
		stddev : float
			standard deviation of random noise that can be added to the weights.
			This is also a hyperparamter. (default = 0.0)

		Returns 
		----------
		Gradients : dict
			Dictionary containing all the gradients (negative) in the following order:
			Gradient of weights, visible bias and hidden bias for the amplitude 
			Gradients of weights, visible bias and hidden bias for the phase.
		'''
		
		vis = self.generate_visible_space()
		batch_size = len(batch)

		g_weights_amp   = torch.zeros_like(self.weights_amp)
		g_vb_amp        = torch.zeros_like(self.visible_bias_amp)
		g_hb_amp        = torch.zeros_like(self.hidden_bias_amp)

		g_weights_phase = torch.zeros_like(self.weights_phase)
		g_vb_phase      = torch.zeros_like(self.visible_bias_phase)
		g_hb_phase      = torch.zeros_like(self.hidden_bias_phase)

		'''Iterate through every data point in the batch.'''
		for row_count, v0 in enumerate(batch):

			'''A counter for the number of non-trivial unitaries (non-computational basis) in the data point.'''
			num_non_trivial_unitaries = 0

			'''tau_indices will contain the index numbers of spins not in the   
			computational basis (Z). 
			z_indices will contain the index numbers of spins in the computational 
			basis.'''
			tau_indices = []
			z_indices   = []

			for j in range(self.num_visible):
				"""Go through the unitaries (chars_batch[row_count]) of each site in the data point, v0, and save inidices of non-trivial."""
				if chars_batch[row_count][j] != 'Z':
					num_non_trivial_unitaries += 1
					tau_indices.append(j)
				else:
					z_indices.append(j)

			if num_non_trivial_unitaries == 0:
				'''If there are no non-trivial unitaries for the data point v0, calculate the positive phase of regular (i.e. non-complex RBM) gradient. Use the actual data point, v0.'''
				g_weights_amp -= torch.ger(F.sigmoid(F.linear(v0, self.weights_amp, self.hidden_bias_amp)), v0) / batch_size 
				g_vb_amp      -= v0 / batch_size
				g_hb_amp      -= F.sigmoid(F.linear(v0, self.weights_amp, self.hidden_bias_amp)) / batch_size

			else:
				'''Compute the rotated gradients.''' 
				L_weights_amp, L_vb_amp, L_hb_amp, L_weights_phase, L_vb_phase, L_hb_phase = self.compute_rotated_grads(v0, chars_batch[row_count], 
																														num_non_trivial_unitaries,
																														z_indices, tau_indices) 


				'''Gradents of amplitude parameters take the real part of the rotated gradients.'''
				g_weights_amp -= L_weights_amp[0] / batch_size
				g_vb_amp      -= L_vb_amp[0] / batch_size
				g_hb_amp      -= L_hb_amp[0] / batch_size
				
				'''Gradents of phase parameters take the imaginary part of the rotated gradients.'''
				g_weights_phase += L_weights_phase[1] / batch_size
				g_vb_phase      += L_vb_phase[1] / batch_size
				g_hb_phase      += L_hb_phase[1] / batch_size

		batch, h0_amp_batch, vk_amp_batch, hk_amp_batch, phk_amp_batch = self.gibbs_sampling_amp(k, batch) 
		for i in range(batch_size):
			'''Negative phase of amplitude gradient. Phase parameters do not have a negative phase.'''
			g_weights_amp += torch.ger(F.sigmoid(F.linear(vk_amp_batch[i], self.weights_amp, self.hidden_bias_amp)), vk_amp_batch[i]) / batch_size 
			g_vb_amp      += vk_amp_batch[i] / batch_size
			g_hb_amp      += F.sigmoid(F.linear(vk_amp_batch[i], self.weights_amp, self.hidden_bias_amp)) / batch_size
			
	   
		'''Perform weight regularization if l1 and/or l2 are not zero.'''
	  
		if l1_reg != 0 or l2_reg != 0:
			g_weights_amp   = self.regularize_weight_gradients_amp(g_weights_amp, l1_reg, l2_reg)
			g_weights_phase = self.regularize_weight_gradients_phase(g_weights_phase, l1_reg, l2_reg)

		'''Add small random noise to weight gradients if stddev is not zero.'''
		if stddev != 0.0:
			g_weights_amp   += (stddev*torch.randn_like(g_weights_amp, device = self.device))
			g_weights_phase += (stddev*torch.randn_like(g_weights_phase, device = self.device))
		

		'''Return negative gradients to match up nicely with the usual
		parameter update rules, which *subtract* the gradient from
		the parameters. This is in contrast with the RBM update
		rules which ADD the gradients (scaled by the learning rate)
		to the parameters.'''
 
		return {"weights_amp": g_weights_amp,
				"visible_bias_amp": g_vb_amp,
				"hidden_bias_amp": g_hb_amp,
				"weights_phase": g_weights_phase,
				"visible_bias_phase": g_vb_phase,
				"hidden_bias_phase": g_hb_phase
				}   

	def compute_rotated_grads(self, v0, characters, num_non_trivial_unitaries, z_indices, tau_indices): 
		'''Computes the rotated gradients.

		
		Parameters
		----------
		v0 : torch.doubleTensor
			A visible unit.
		characters : str
			A string of characters corresponding to the basis that each site in v0 was measured in.
		num_non_trivial_unitaries : int
			The number of sites in v0 that are not measured in the computational basis.
		z_indices : list
			A list of indices that correspond to sites of v0 that are measured in the computational basis.
		tau_indicies : list
			A list of indices that correspond to sites of v0 that are not measured in the computational basis.
	

		Returns
		----------
		Returns a dictionary of the rotated gradients below.
		L_weights_amp : torch.doubleTensor
			The rotated gradients for the weights for the amplitude RBM.	
		L_vb_amp : torch.doubleTensor
			The rotated gradients for the visible biases for the amplitude RBM.
		L_hb_amp : torch.doubleTensor
			The rotated gradients for the hidden biases for the phase RBM. 
		L_weights_phase : torch.doubleTensor
			The rotated gradients for the weights for the phase RBM.
		L_vb_phase : torch.doubleTensor
			The rotated gradients for the visible biases for the phase RBM.
		L_hb_phase : torch.doubleTensor
			The rotated gradients for the hidden biases for the phase RBM.
		'''

		'''Initialize the 'A' parameters (see alg 4.2).'''
		A_weights_amp = torch.zeros(2, self.weights_amp.size()[0], self.weights_amp.size()[1], 
									device=self.device, dtype = torch.double)
		A_vb_amp      = torch.zeros(2, self.visible_bias_amp.size()[0], 
									device=self.device, dtype = torch.double)
		A_hb_amp      = torch.zeros(2, self.hidden_bias_amp.size()[0], 
									device=self.device, dtype = torch.double)
		
		A_weights_phase = torch.zeros(2, self.weights_phase.size()[0], self.weights_phase.size()[1], 
									  device=self.device, dtype = torch.double)
		A_vb_phase      = torch.zeros(2, self.visible_bias_phase.size()[0], 
									  device=self.device, dtype = torch.double)
		A_hb_phase      = torch.zeros(2, self.hidden_bias_phase.size()[0], 
									  device=self.device, dtype = torch.double)
		''' 'B' will contain the coefficients of the rotated unnormalized wavefunction. '''
		B = torch.zeros(2, device = self.device, dtype = torch.double)

		w_grad_amp      = torch.zeros_like(self.weights_amp)
		vb_grad_amp     = torch.zeros_like(self.visible_bias_amp)
		hb_grad_amp     = torch.zeros_like(self.hidden_bias_amp)
	
		w_grad_phase    = torch.zeros_like(self.weights_phase)
		vb_grad_phase   = torch.zeros_like(self.visible_bias_phase)
		hb_grad_phase   = torch.zeros_like(self.hidden_bias_phase)

		zeros_for_w  = torch.zeros_like(w_grad_amp)
		zeros_for_vb = torch.zeros_like(vb_grad_amp)
		zeros_for_hb = torch.zeros_like(hb_grad_amp) 
		'''NOTE! THIS WILL CURRENTLY ONLY WORK IF AND ONLY IF THE NUMBER OF HIDDEN UNITS FOR PHASE AND AMP ARE THE SAME!!!'''

		'''Loop over Hilbert space of the non trivial unitaries to build the state.'''
		for j in range(2**num_non_trivial_unitaries):
			s = self.state_generator(num_non_trivial_unitaries)[j]
			'''Creates a matrix where the jth row is the desired state, |S>, a vector.'''
	
			'''This is the sigma state.'''	
			constructed_state = torch.zeros(self.num_visible, dtype = torch.double)
			
			U = torch.tensor([1., 0.], dtype = torch.double, device = self.device)
		
			'''Populate the |sigma> state (aka constructed_state) accirdingly. '''
			for index in range(len(z_indices)):
				'''These are the sites in the computational basis.'''
				constructed_state[z_indices[index]] = v0[z_indices[index]]
		
			for index in range(len(tau_indices)):
				'''These are the sites that are NOT in the computational basis. '''
				constructed_state[tau_indices[index]] = s[index]
		
				aa = self.unitaries[characters[tau_indices[index]]]
				bb = self.basis_state_generator(v0[tau_indices[index]])
				cc = self.basis_state_generator(s[index])
			
				temp = cplx.inner_prod( cplx.MV_mult(cplx.compT_matrix(aa), bb), cc )
		
				U = cplx.scalar_mult(U, temp)
			
			'''Positive phase gradients for phase and amp. Will be added into the 'A' parameters.'''
			w_grad_amp  = torch.ger(F.sigmoid(F.linear(constructed_state, self.weights_amp, self.hidden_bias_amp)), constructed_state)
			vb_grad_amp = constructed_state
			hb_grad_amp = F.sigmoid(F.linear(constructed_state, self.weights_amp, self.hidden_bias_amp))

			w_grad_phase  = torch.ger(F.sigmoid(F.linear(constructed_state, self.weights_phase, self.hidden_bias_phase)), constructed_state)
			vb_grad_phase = constructed_state
			hb_grad_phase = F.sigmoid(F.linear(constructed_state, self.weights_phase, self.hidden_bias_phase))

			'''
			In order to calculate the 'A' parameters below with my current complex library, I need to make the weights and biases complex.
			I fill the complex parts of the parameters with a tensor of zeros.
			'''
			temp_w_grad_amp  = cplx.make_complex_matrix(w_grad_amp, zeros_for_w)
			temp_vb_grad_amp = cplx.make_complex_vector(vb_grad_amp, zeros_for_vb)
			temp_hb_grad_amp = cplx.make_complex_vector(hb_grad_amp, zeros_for_hb)
 
			temp_w_grad_phase  = cplx.make_complex_matrix(w_grad_phase, zeros_for_w)
			temp_vb_grad_phase = cplx.make_complex_vector(vb_grad_phase, zeros_for_vb)
			temp_hb_grad_phase = cplx.make_complex_vector(hb_grad_phase, zeros_for_hb)
		
			''' Temp = U*psi(sigma)'''
			temp = cplx.scalar_mult(U, self.unnormalized_wavefunction(constructed_state))
			
			A_weights_amp += cplx.MS_mult(temp, temp_w_grad_amp)
			A_vb_amp      += cplx.VS_mult(temp, temp_vb_grad_amp)
			A_hb_amp      += cplx.VS_mult(temp, temp_hb_grad_amp)
		
			A_weights_phase += cplx.MS_mult(temp, temp_w_grad_phase)
			A_vb_phase      += cplx.VS_mult(temp, temp_vb_grad_phase)
			A_hb_phase      += cplx.VS_mult(temp, temp_hb_grad_phase)
		   
			'''Rotated wavefunction.'''
			B += temp
		
		L_weights_amp = cplx.MS_divide(A_weights_amp, B)
		L_vb_amp      = cplx.VS_divide(A_vb_amp, B)
		L_hb_amp      = cplx.VS_divide(A_hb_amp, B)
		
		L_weights_phase = cplx.MS_divide(A_weights_phase, B)
		L_vb_phase      = cplx.VS_divide(A_vb_phase, B)
		L_hb_phase      = cplx.VS_divide(A_hb_phase, B)
	   
		return L_weights_amp, L_vb_amp, L_hb_amp, L_weights_phase, L_vb_phase, L_hb_phase 

	def train(self, data, character_data, epochs, batch_size,
			  k=1, lr=1e-3, momentum=0.0,
			  l1_reg=0.0, l2_reg=0.0,
			  initial_gaussian_noise=0.01, gamma=0.55,
			  log_every=50):

		'''This function will execute the training of the RBM.

		Parameters
		----------
		data : array_like
			The actual training data.
		character_data : array_like
			The corresponding bases that each site in the data has been measured in.
		epochs : int
			The number of parameter (i.e. weights and biases) updates (default = 100).
		batch_size : int
			The size of batches taken from the data (default = 100).
		k : int
			The number of contrastive divergence steps (default = 1).
		lr : float
			Learning rate (default = 1.0e-3)
		momentum : float
			Momentum hyperparameter (default = 0.0)
		l1_reg : float
			L1 regularization hyperparameter (default = 0.0)
		l2_reg : float
			L2 regularization hyperparameter (default = 0.0)
		initial_gaussian_noise : float
			Initial gaussian noise used to calculate stddev of random noise added to weight gradients (default = 0.01).
		gamma : float
			Parameter used to calculate stddev (default = 0.55).
		log_every : int
			Indicates how often (i.e. after how many epochs) to calculate convergence parameters (e.g. fidelity, energy, etc.).


		Returns 
		----------
		fidelity : float
			Overlap squared.
		'''
		
		'''Make data file into a torch tensor.'''
		data = torch.tensor(data).to(device=self.device)

		'''Use the Adam optmizer to update the weights and biases.'''
		optimizer = torch.optim.Adam([self.weights_amp,
									  self.visible_bias_amp,
									  self.hidden_bias_amp,
									  self.weights_phase,
									  self.visible_bias_phase,
									  self.hidden_bias_phase],
									  lr=lr)

		vis = self.generate_visible_space()
		print ('Generated visible space. Ready to begin training.')

		'''Empty lists to put calculated convergence quantities in.'''
		fidelity_list = []
		epoch_list = []

		for ep in range(0,epochs+1):
			'''Shuffle the data to ensure that the batches taken from the data are random data points.''' 
			random_permutation = torch.randperm(data.shape[0])

			shuffled_data           = data[random_permutation]   
			shuffled_character_data = character_data[random_permutation]

			'''List of all the batches.'''
			batches = [shuffled_data[batch_start:(batch_start + batch_size)] 
					   for batch_start in range(0, len(data), batch_size)]

			'''List of all the bases.'''
			char_batches = [shuffled_character_data[batch_start:(batch_start + batch_size)] 
							for batch_start in range(0, len(data), batch_size)]

			'''Calculate convergence quantities every "log-every" steps.'''
			if ep % log_every == 0:
				fidelity_ = self.fidelity(vis, 'Z' 'Z')
				print ('Epoch = ',ep,'\nFidelity = ',fidelity_)
				fidelity_list.append(fidelity_)
				epoch_list.append(ep)
		   
			'''Save fidelities at the end of training.''' 
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

			'''Loop through all of the batches and calculate the batch gradients.'''
			for batch_index in range(len(batches)):
				
				grads = self.compute_batch_gradients(k, batches[batch_index], char_batches[batch_index],
													 l1_reg, l2_reg,
													 stddev=stddev)
				'''
				For testing gradients. 
				self.test_gradients(vis, k, batches[batch_index], char_batches[batch_index],
													 l1_reg, l2_reg,
													 stddev=stddev)               
				'''

				'''Clear any cached gradients.'''
				optimizer.zero_grad()  

				'''Assign all available gradients to the corresponding parameter.'''
				for name in grads.keys():
					getattr(self, name).grad = grads[name]
			   
				'''Tell the optimizer to apply the gradients and update the parameters.''' 
				optimizer.step()  
			   
	def prob_v_given_h_amp(self, h):
		'''Given a hidden amplitude unit, whats the probability of a visible unit.
		
		Parameters
		----------
		h : torch.doubleTensor 
			The hidden unit.
		
		Returns 
		----------
		p : torch.doubleTensor 
			probability of a visible unit given the hidden unit, h
		'''
		p = F.sigmoid(F.linear(h, self.weights_amp.t(), self.visible_bias_amp))
		return p

	def prob_v_given_h_phase(self, h):
		'''Given a hidden phase unit, whats the probability of a visible unit.

		Parameters
		----------
		h : torch.doubleTensor
			The hidden unit. 

		Returns 
		----------
		p : torch.doubleTensor
			probability of a visible unit given the hidden unit, h
		'''
		p = F.sigmoid(F.linear(h, self.weights_phase.t(), self.visible_bias_phase))
		return 

	def prob_h_given_v_amp(self, v):
		'''Given a visible unit, whats the probability of a hidden amplitude unit.

		Parameters
		----------
		v : torch.doubleTensor
			The visible unit.

		Returns 
		----------
		p : torch.doubleTensor
			probability of a hidden unit given the visible unit, v
		'''
		p = F.sigmoid(F.linear(v, self.weights_amp, self.hidden_bias_amp))
		return p

	def prob_h_given_v_phase(self, v):
		'''Given a visible unit, whats the probability of a hidden phase unit.

		Parameters
		----------
		v : torch.doubleTensor
			The visible unit. 

		Returns 
		----------
		p : torch.doubleTensor
			probability of a hidden unit given the visible unit, v
		'''
		p = F.sigmoid(F.linear(v, self.weights_phase, self.hidden_bias_phase))
		return p

	def sample_v_given_h_amp(self, h):
		'''Sample/generate a visible unit given a hidden amplitude unit.

		Parameters
		----------
		h : torch.doubleTensor
			The hidden unit.            
  
		Returns 
		----------
		p : torch.doubleTensor
			see prob_v_given_h_amp
		v : torch.doubleTensor
			The sampled visible unit.
		'''
		p = self.prob_v_given_h_amp(h)
		v = p.bernoulli()
		return p, v

	def sample_h_given_v_amp(self, v):
		'''Sample/generate a hidden amplitude unit given a visible unit.

		Parameters
		----------
		v : torch.doubleTensor
			The visible unit. 
			  
		Returns 
		----------
		p : torch.doubleTensor
			see prob_h_given_v_amp
		h : torch.doubleTensor
			The sampled hidden unit.
		'''
		p = self.prob_h_given_v_amp(v)
		h = p.bernoulli()
		return p, h

	def sample_v_given_h_phase(self, h):
		'''Sample/generate a visible unit given a hidden phase unit.

		Parameters
		----------
		h : torch.doubleTensor
			The hidden unit. 

		Returns 
		----------
		p : torch.doubleTensor
			see prob_v_given_h_phase
		v : torch.doubleTensor
			The sampled visible unit.
		'''
		p = self.prob_v_given_h_phase(h)
		v = p.bernoulli()
		return p, v

	def sample_h_given_v_phase(self, v):
		'''Sample/generate a hidden phase unit given a visible unit.

		Parameters
		----------
		v : torch.doubleTensor
			A data point.
			  
		Returns 
		----------
		p : torch.doubleTensor
			see prob_h_given_v_phase
		h : torch.doubleTensor
			The sampled hidden unit.
		'''
		p = self.prob_h_given_v_phase(v)
		h = p.bernoulli()
		return p, h

	def gibbs_sampling_amp(self, k, v0):
		'''Contrastive divergence/gibbs sampling algorithm for generating samples from the RBM.

		Parameters
		----------
		k : int
			Number of contrastive divergence iterations.
		v0 : torch.doubleTensor
			A visible unit from the data
		
		Returns 
		----------
		v0 : torch.doubleTensor
		h0 : torch.doubleTensor
			The hidden unit sampled from v0.
		v : torch.doubleTensor
			The sampled visible unit after k steps.
		h : torch.doubleTensor
			The sampled hidden unit after k steps.
		ph : torch.doubleTensor
			See sample_h_given_v_amp
		'''
		ph, h0 = self.sample_h_given_v_amp(v0)
		v, h = v0, h0
		for _ in range(k):
			pv, v = self.sample_v_given_h_amp(h)
			ph, h = self.sample_h_given_v_amp(v)
		return v0, h0, v, h, ph

	def gibbs_sampling_phase(self, k, v0_phase):
		'''Contrastive divergence/gibbs sampling algorithm for generating samples from the RBM.

		Parameters
		----------
		k : int
			Number of contrastive divergence iterations.
		v0 : torch.doubleTensor
			A visible unit from the data
		
		Returns 
		----------
		v0 : torch.doubleTensor
		h0 : torch.doubleTensor
			The hidden unit sampled from v0.
		v : torch.doubleTensor
			The sampled visible unit after k steps.
		h : torch.doubleTensor
			The sampled hidden unit after k steps.
		ph : torch.doubleTensor
			See sample_h_given_v_phase
		'''
		ph, h0 = self.sample_h_given_v_phase(v0)
		v, h   = v0, h0
		for _ in range(k):
			pv, v = self.sample_v_given_h_phase(h)
			ph, h = self.sample_h_given_v_phase(v)
		return v0, h0, v, h, ph

	def regularize_weight_gradients_amp(self, w_grad, l1_reg, l2_reg):
		'''Weight gradient regularization.
		
		Parameters
		----------
		w_grad : torch.doubleTensor
			The entire weight gradient matrix (amplitude).
		l1_reg : float
			L1 regularization hyperparameter (default = 0.0)
		l2_reg : float
			L2 regularization hyperparameter (default = 0.0)
		'''
		return (w_grad
				+ (l2_reg * self.weights_amp)
				+ (l1_reg * self.weights_amp.sign()))

	def regularize_weight_gradients_phase(self, w_grad, l1_reg, l2_reg):
		'''Weight gradient regularization.
		
		Parameters
		----------
		w_grad : torch.doubleTensor
			The entire weight gradient matrix (phase).
		l1_reg : float
			L1 regularization hyperparameter
		l2_reg : float
			L2 regularization hyperparameter
		'''
		return (w_grad
				+ (l2_reg * self.weights_phase)
				+ (l1_reg * self.weights_phase.sign()))

	def eff_energy_amp(self, v):
		'''The effective energy of the amplitude RBM.

		Parameters
		----------
		v : torch.doubleTensor

		Returns
		----------
		.. math::
			E_{\lambda} = b^{\lambda}v + \sum_{i}\log\sum_{h_{i}^{\lambda}}e^{h_{i}^{\lambda}\left(c_{i}^{\lambda} + W_{i}^{\lambda}v\right)}
		'''
		if len(v.shape) < 2:
			v = v.view(1, -1)

		visible_bias_term = torch.mv(v, self.visible_bias_amp)
		hidden_bias_term = F.softplus(F.linear(v, self.weights_amp, self.hidden_bias_amp)).sum(1)

		return visible_bias_term + hidden_bias_term

	def eff_energy_phase(self, v):
		'''The effective energy of the phase RBM.

		Parameters
		----------
		v : torch.doubleTensor

		Returns
		----------
		.. math::
			E_{\mu} = b^{\mu}v + \sum_{i}\log\sum_{h_{i}^{\mu}}e^{h_{i}^{\mu}\left(c_{i}^{\mu} + W_{i}^{\mu}v\right)}
		'''
		if len(v.shape) < 2:
			v = v.view(1, -1)

		visible_bias_term = torch.mv(v, self.visible_bias_phase)
		hidden_bias_term = F.softplus(F.linear(v, self.weights_phase, self.hidden_bias_phase)).sum(1)

		return visible_bias_term + hidden_bias_term

	def unnormalized_probability_amp(self, v):
		'''The effective energy of the phase RBM.

		Parameters
		----------
		v : torch.doubleTensor
			Visible units.

		Returns
		----------
		.. math::
			p_{\lambda} = e^{E_{\lambda}} 
		''' 
		return self.eff_energy_amp(v).exp()

	def unnormalized_probability_phase(self, v):
		'''The effective energy of the phase RBM.

		Parameters
		----------
		v : torch.doubleTensor
			Visible units.

		Returns
		----------
		.. math::
			p_{\mu} = e^{E_{\mu}}       
		''' 
		return self.eff_energy_phase(v).exp()

	def normalized_wavefunction(self, v):
		'''The RBM wavefunction.

		Parameters
		----------
		v : torch.doubleTensor

		Returns
		----------
		.. math::
			\psi_{\lambda\mu} = \sqrt{\frac{p_{\lambda}}{Z_{\lambda}}}e^{\frac{i\log(p_{\mu})}{2}} 
		''' 
		v_prime   = v.view(-1,self.num_visible)
		temp1     = (self.unnormalized_probability_amp(v_prime)).sqrt()
		temp2     = ((self.unnormalized_probability_phase(v_prime)).log())*0.5

		cos_angle = temp2.cos()
		sin_angle = temp2.sin()
		
		psi       = torch.zeros(2, v_prime.size()[0], dtype = torch.double)
		psi[0]    = temp1*cos_angle
		psi[1]    = temp1*sin_angle

		sqrt_Z    = (self.partition(self.generate_visible_space())).sqrt()

		return psi / sqrt_Z

	def unnormalized_wavefunction(self, v):
		'''The unnormalized RBM wavefunction.

		Parameters
		----------
		v : torch.doubleTensor

		Returns
		----------
		.. math::
			\tilde{\psi}_{\lambda\mu} = \sqrt{p_{\lambda}}e^{\frac{i\log(p_{\mu})}{2}} 
		''' 
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
		'''Picks out the true psi in the correct basis.
		
		Parameters
		----------
		basis : str
			E.g. XZZZX
		'''
		key = ''
		for i in range(len(basis)):
			key += basis[i]
		return self.psi_dictionary[key]

	def overlap(self, visible_space, basis):
		'''Computes the overlap between the RBM and true wavefunctions.
		Parameters
		----------
		visible_space : torch.doubleTensor
			An array of all possible spin configurations.
		basis : str
			E.g. XZZZX

		Returns
		----------
		double
		.. math::
			O = \braket{\psi_{true} | \psi_{\lambda\mu}}
		
		'''
		overlap_ = cplx.inner_prod(self.get_true_psi(basis),
						   self.normalized_wavefunction(visible_space))
		return overlap_

	def fidelity(self, visible_space, basis):
		'''Computed the fidelity of the RBM and true wavefunctions. 
		Parameters
		----------
		visible_space : torch.doubleTensor
			An array of all possible spin configurations.
		basis : str
			E.g. XZZZX

		Returns
		----------
		double 
		.. math::
			F = |O|^2
		
		'''
		return cplx.norm(self.overlap(visible_space, basis))

	def generate_visible_space(self):
		'''Generates all possible spin configurations of "num_visible" spins.
		

		Returns 
		----------
		space : torch.doubleTensor
			An array where each row is a given spin configuration (2^N rows).
		'''    
		space = torch.zeros((2**self.num_visible, self.num_visible),
							device=self.device, dtype=torch.double)
		for i in range(2**self.num_visible):
			d = i
			for j in range(self.num_visible):
				d, r = divmod(d, 2)
				space[i, self.num_visible - j - 1] = int(r)

		return space

	def log_partition(self, visible_space):
		'''Computes the natural log of the partition function.
		
		
		Parameters
		----------
		visible_space : torch.doubleTensor
			An array of all possible spin configurations.


		Returns
		----------
		logZ : double
			The log of the partition function.
		'''

		eff_energies = self.eff_energy_amp(visible_space)
		max_eff_energy = eff_energies.max()

		reduced = eff_energies - max_eff_energy
		logZ = max_eff_energy + reduced.exp().sum().log()

		return logZ

	def partition(self, visible_space):
		'''Computes the partition function.
		
		
		Parameters
		----------
		visible_space : torch.doubleTensor
			An array of all possible spin configurations.


		Returns
		----------
		double
			The partition function.
		'''
	
		return self.log_partition(visible_space).exp()

	def state_generator(self, num_non_trivial_unitaries):
		'''A function that returns all possible configurations of 'num_non_trivial_unitaries' spins. Similar to generate_visible_space.

		Parameters
		----------
		num_non_trivial_unitaries : int
			The number of sites measured in the non-computational basis.

  
		Returns
		----------
		states : torch.doubleTensor
			An array of all possible spin configurations of 'num_non_trivial_unitaries' spins.
		'''
		states = torch.zeros((2**num_non_trivial_unitaries, num_non_trivial_unitaries), device = self.device, dtype=torch.double)
		for i in range(2**num_non_trivial_unitaries):
			temp = i
			for j in range(num_non_trivial_unitaries): 
				temp, remainder = divmod(temp, 2)
				states[i][num_non_trivial_unitaries - j - 1] = remainder
		return states

	def basis_state_generator(self, s):
		'''Only works for binary visible units at the moment. Generates a vector given a spin value (0 or 1).


		Parameters
		----------
		s : double
			A spin's value (either 0 or 1).


		Returns
		----------
		torch.doubleTensor
			If s = 0, this is the (1,0) state in the basis of the measurement. If s = 1, this is the (0,1) state in the basis of the measurement.
		'''
		if s == 0.:
			return torch.tensor([[1., 0.],[0., 0.]], dtype = torch.double)
		if s == 1.:
			return torch.tensor([[0., 1.],[0., 0.]], dtype = torch.double) 
