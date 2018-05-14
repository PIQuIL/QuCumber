import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import make_grid , save_image

def spin_config(number, n_vis): # generates a binary list from a number
	spins = list(map(int, list(format(number, 'b').zfill(n_vis))))
	spins.reverse()
	return spins
	
def spin_list(n_vis): # returns a list of all possible spin configurations for n_vis spins
	spins = [spin_config(number, n_vis) for number in range(2**n_vis)  ]
	spins = Variable(torch.FloatTensor(spins))
	return spins
	
class RBM(nn.Module):
    def __init__(self,
                 n_vis=10,
                 n_hin=50,
                 k=5, sample_length = 1):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hin,n_vis)*1e-2) # randomly initialize weights
        self.v_bias = nn.Parameter(torch.zeros(n_vis))
        self.h_bias = nn.Parameter(torch.zeros(n_hin))
        self.k = k
        self.sample_length = sample_length
        self.n_vis = n_vis

    def sample_from_p(self,p): # Compares p to rand number and returns 0 or 1
        return F.relu(torch.sign(p - Variable(torch.rand(p.size()))))

    def v_to_h(self,v): # sample h, given v
        p_h = F.sigmoid(F.linear(v,self.W,self.h_bias))
        # p (h_j | v ) = sigma(b_j + sum_i v_i w_ij)
        sample_h = p_h.bernoulli()
        return p_h,sample_h

    def h_to_v(self,h): # sample v given h
        p_v = F.sigmoid(F.linear(h,self.W.t(),self.v_bias))
        # p (v_i | h ) = sigma(a_i + sum_j h_j w_ij)
        sample_v = p_v.bernoulli()
        return p_v,sample_v
    
    def forward(self,v):
        pre_h1,h1 = self.v_to_h(v)
        
        h_ = h1
        for _ in range(self.k):
            pre_v_,v_ = self.h_to_v(h_)
            pre_h_,h_ = self.v_to_h(v_)
        
        return v,v_

    def free_energy(self,v): # exp( v_bias^transp*v + sum(log(1+exp(h_bias + W*v))))
        vbias_term = v.mv(self.v_bias) # v_bias^transp*v
        wx_b = F.linear(v,self.W,self.h_bias) # v*W^transp + h_bias
        hidden_term = wx_b.exp().add(1).log().sum(1) # sum indicates over which tensor index we sum
        return (-hidden_term - vbias_term) # returns the free energies of all the input spins in a vector
    
    def F_single_spin(self,v):
        vbias_term = (v * self.v_bias).sum()
        wx_b = F.linear(v,self.W,self.h_bias) # v*W^transp + h_bias
        hidden_term = wx_b.exp().add(1).log().sum() # sum indicates over which tensor index we sum
        return (-hidden_term - vbias_term)
			
    def draw_sample(self):
        v_ = F.relu(torch.sign(Variable(torch.randn(self.n_vis))))
        for _ in range(self.sample_length):
            pre_h_,h_ = self.v_to_h(v_)
            pre_v_,v_ = self.h_to_v(h_)
        return v_

    # -------------------------------------------------------------------------
    # TO DO (for n_hidden > 150 works does not work)
    # Calculate exp( log( p(v))) to avoid exploding exponentials
    # exp ( -epsilon(v) - log(Z) )
    def partition_fct(self, spins):
        return self.free_energy(spins).exp().sum()

    def probability_of_v(self, all_spins, v):
        if len(v.shape)<2:
            epsilon = self.F_single_spin(v).exp()
        else:
                epsilon = self.free_energy(v).exp().sum()
        Z = self.partition_fct(all_spins)
        return epsilon/Z

		
					 
			  
batch_size = 512
filename = 'training_data.txt'
with open(filename, 'r') as fobj:
	all_lines = [[int(num) for num in line.split()] for line in fobj]
data = torch.FloatTensor(all_lines)
filename = 'target_psi.txt'
with open(filename, 'r') as fobj:
	psi = [float(line.split()[0]) for line in fobj]
		

vis = len(data[0]) #input dimension
	
rbm = RBM(n_vis = vis, n_hin = 10, k=1)
train_op = optim.SGD(rbm.parameters(), lr = 0.0001, momentum = 0.9)
# rbm.parameters gives a generator object with the weights and the biases

#Example SGD:
#	 >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
#	 >>> optimizer.zero_grad()
#	 >>> loss_fn(model(input), target).backward()
#	 >>> optimizer.step()

train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
												   shuffle=True)
all_spins = spin_list(10)
for epoch in range(1000):
    loss_ = []
    for _, data in enumerate(train_loader):
        data = Variable(data.view(-1,vis)) # convert tensor to node in computational graph
        # sample_data = data.bernoulli()

        v,v1 = rbm(data) # returns batch before and after Gibbs sampling steps
        loss = (rbm.free_energy(v).mean() - rbm.free_energy(v1).mean()).abs_() # calc difference in free energy before and after Gibbs sampling
        # .mean() for averaging over whole batch
        # KLL =~ F(spins) - log (Z), by looking at the difference between F before and after the iteration, one gets rid of Z
        loss_.append(loss.data[0])
        train_op.zero_grad() # reset gradient to zero (pytorch normally accumulates them to use it e.g. for RNN)
        loss.backward() # calc gradient
        train_op.step() # make one step of SGD

    print( np.mean(loss_))
    if epoch%1 == 0:
        a = 0
        for i in range(len(psi)):
            a += psi[i]*torch.sqrt(rbm.probability_of_v(all_spins, all_spins[i]))
        print('OVERLAPP:', a.data[0])
