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

def overlapp_fct(all_spins, data, psi):
    a = 0
    for i in range(len(data)):
        a += psi[i]*torch.sqrt(rbm.probability_of_v(all_spins, data[i]))
    return a.data[0]
	
class RBM(nn.Module):
    def __init__(self,
                 n_vis=10,
                 n_hin=50,
                 k=5, sample_length = 1, gpu = False):
        super(RBM, self).__init__()
        self.gpu = gpu
        self.W = nn.Parameter(torch.randn(n_hin,n_vis)*1e-2, requires_grad=True) # randomly initialize weights
        self.v_bias = nn.Parameter(torch.zeros(n_vis), requires_grad=True)
        self.h_bias = nn.Parameter(torch.zeros(n_hin), requires_grad=True)
        self.k = k
        self.sample_length = sample_length
        self.n_vis = n_vis
    
    
        if self.gpu:
            self.W = self.W.cuda()
            self.v_bias = self.v_bias.cuda()
            self.h_bias = self.h_bias.cuda()
            
            self.weights_momentum = self.weights_momentum.cuda()
            self.visible_bias_momentum = self.visible_bias_momentum.cuda()
            self.hidden_bias_momentum = self.hidden_bias_momentum.cuda()
    

    def sample_from_p(self,p): # Compares p to rand number and returns 0 or 1
        # can be replaced by p.bernoulli()
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
        if len(v.shape)<2: #if v is just ONE vector
            v = v.view(1, v.shape[0])
        vbias_term = v.mv(self.v_bias) # v_bias^transp*v; should give a scalar for every element of batch
        wx_b = F.linear(v,self.W,self.h_bias) # v*W^transp + h_bias
        # wx_b has dimension batch_size x v_dim
        hidden_term = wx_b.exp().add(1).log().sum(1) # sum indicates over which tensor index we sum
        # hidden_term has dim batch_size
        return (-hidden_term - vbias_term) # returns the free energies of all the input spins in a vector
    
#    def F_single_spin(self,v):
#        vbias_term = (v * self.v_bias).sum()
#        wx_b = F.linear(v,self.W,self.h_bias) # v*W^transp + h_bias
#        # wx_b has dimension v_dim
#        hidden_term = wx_b.exp().add(1).log().sum() # sum indicates over which tensor index we sum
#        return (-hidden_term - vbias_term)

    def draw_sample(self):
        v_ = F.relu(torch.sign(Variable(torch.randn(self.n_vis))))
        for _ in range(self.sample_length):
            pre_h_,h_ = self.v_to_h(v_)
            pre_v_,v_ = self.h_to_v(h_)
        return v_

    # -------------------------------------------------------------------------
    # TO DO (for n_hidden > 150 does not work)
    # Calculate exp( log( p(v))) to avoid exploding exponentials
    # exp ( -epsilon(v) - log(Z) )
    def partition_fct(self, spins):
        return (-self.free_energy(spins)).exp().sum()

    def probability_of_v_TEST(self, all_spins, v):
        epsilon = self.free_energy(v).exp().sum()
        Z = self.partition_fct(all_spins)
        return epsilon/Z, self.free_energy(v), Z

    def probability_of_v(self, all_spins, v):
        epsilon = (-self.free_energy(v)).exp().sum()
        Z = self.partition_fct(all_spins)
        return epsilon/Z

CUDA = torch.cuda.is_available()
CUDA_DEVICE = 0

if CUDA:
    torch.cuda.set_device(CUDA_DEVICE)
					 
			  
batch_size = 500

filename = 'training_data.txt'
with open(filename, 'r') as fobj:
	data = torch.FloatTensor([[int(num) for num in line.split()] for line in fobj])

filename = 'target_psi.txt'
with open(filename, 'r') as fobj:
	psi = torch.FloatTensor([float(line.split()[0]) for line in fobj])

vis = len(data[0]) #input dimension
	
rbm = RBM(n_vis = vis, n_hin = 10, k=10, sample_length = 10)
train_op = optim.SGD(rbm.parameters(), lr = 0.01, momentum = 0.0)
# rbm.parameters gives a generator object with the weights and the biases

#Example SGD:
#	 >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
#	 >>> optimizer.zero_grad()
#	 >>> loss_fn(model(input), target).backward()
#	 >>> optimizer.step()

all_spins = spin_list(10)

if CUDA:
    data = data.cuda()
    psi = psi.cuda()
    rbm = rbm.cuda()
psi = Variable(psi)

# ------------------------------------------------------------------------------
#define a simple training set and check if rbm.draw() returns this after training.
dummy_training = False
if dummy_training:
    data = torch.FloatTensor([[0]*10]*1000) #torch.FloatTensor([[1,0,1,0,1,0,1,0,1,0], [0]*10, [1]*10]*1000)
    test = Variable(torch.FloatTensor([1,1,1,1,1,0,0,0,0,0]))
    psi = Variable(torch.FloatTensor([1/np.sqrt(1)]))
# ------------------------------------------------------------------------------

train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                           shuffle=True)

for epoch in range(1000):
    loss_ = []
    for _, data in enumerate(train_loader):
        data = Variable(data.view(-1,vis)) # convert tensor to node in computational graph
#        sample_data = data.bernoulli()
        v,v1 = rbm(data) # returns batch before and after Gibbs sampling steps
        loss = (rbm.free_energy(v).mean() - rbm.free_energy(v1).mean()) # calc difference in free energy before and after Gibbs sampling
        # .mean() for averaging over whole batch
        # KLL =~ F(spins) - log (Z), by looking at the difference between F before and after the iteration, one gets rid of Z
        loss_.append(loss.data[0])
        train_op.zero_grad() # reset gradient to zero (pytorch normally accumulates them to use it e.g. for RNN)
        loss.backward() # calc gradient
        train_op.step() # make one step of SGD
    print('OVERLAPP:', overlapp_fct(all_spins, all_spins, psi))
    print( np.mean(loss_))

