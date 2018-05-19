import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

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

def outer_product(vecs1, vecs2):
    '''Computes the outer product of batches of vectors
        
        Arguments:
        
        :param vecs1: b 1-D tensors of length m
        :type vecs1: list of torch.Tensor or torch.autograd.Variable
        :param vecs2: b 1-D tensors of length n
        :type vecs2: list of torch.Tensor or torch.autograd.Variable
        :returns: torch.Tensor or torch.autograd.Variable of size (m, n)
        '''
    return torch.bmm(vecs1.unsqueeze(2), vecs2.unsqueeze(1)) #batch-matrix-matrix product
# (b x n x m) @ (b x m x p) = (b x n x p), where b is the batch size, normal matrix multiplication
# unsqueeze(pos) gives a new dimension at position 'pos' with size one.
# x = [1,2,3,4], x.unsqueeze(0) has shape (1,4). x.unsqueeze(1) has shape (4,1)

class RBM(nn.Module):
    def __init__(self,
                 n_vis=10,
                 n_hin=50,
                 k=5, gpu = False, continuous_visible = False, continuous_hidden = False, saved_weights = None):
        super(RBM, self).__init__()
        self.gpu = gpu
        if saved_weights is None:
            self.W = nn.Parameter(torch.randn(n_hin,n_vis)*1e-2, requires_grad = True) # randomly initialize weights
            self.v_bias = nn.Parameter(torch.randn(n_vis)*1e-2, requires_grad=True)
            self.h_bias = nn.Parameter(torch.randn(n_hin)*1e-2, requires_grad=True)
        else:
            self.W = saved_weights[0]
            self.v_bias = saved_weights[1]
            self.h_bias = saved_weights[2]
        self.k = k
        self.n_vis = n_vis
        self.continuous_v = continuous_visible
        self.continuous_h = continuous_hidden
        
        self.W_update = self.W.clone()
        self.h_bias_update = self.h_bias.clone()
        self.v_bias_update = self.v_bias.clone()
        
        if self.gpu:
            self.W_update = self.W_update.cuda()
            self.v_bias_update = self.v_bias_update.cuda()
            self.h_bias_update = self.h_bias_update.cuda()

    def v_to_h(self,v): # sample h, given v
        if (self.gpu and not v.is_cuda):
            v = v.cuda()
        p_h = F.sigmoid(F.linear(v,self.W,self.h_bias))
        # p (h_j | v ) = sigma(b_j + sum_i v_i w_ij)
        sample_h = p_h.bernoulli()
        return p_h if self.continuous_h else sample_h
    
    def h_to_v(self,h): # sample v given h
        if (self.gpu and not h.is_cuda):
            h = h.cuda()
        p_v = F.sigmoid(F.linear(h,self.W.t(),self.v_bias))
        # p (v_i | h ) = sigma(a_i + sum_j h_j w_ij)
        sample_v = p_v.bernoulli()
        return p_v if self.continuous_v else sample_v
    
    def forward(self,v): # forward is pytorch standard fct that defines what happens with input data
        if (self.gpu and not v.is_cuda):
            v = v.cuda()
        h1 = self.v_to_h(v)
        h_ = h1
        for _ in range(self.k):
            v_ = self.h_to_v(h_)
            h_ = self.v_to_h(v_)
        return v,v_
        
    def free_energy(self,v): # exp( v_bias^transp*v + sum(log(1+exp(h_bias + W*v))))
        if (self.gpu and not v.is_cuda):
            v = v.cuda()
        if len(v.shape)<2: #if v is just ONE vector
            v = v.view(1, v.shape[0])
        vbias_term = v.mv(self.v_bias) # v_bias^transp*v; should give a scalar for every element of batch
        wx_b = F.linear(v,self.W,self.h_bias) # v*W^transp + h_bias
        # wx_b has dimension batch_size x v_dim
        hidden_term = wx_b.exp().add(1).log().sum(1) # sum indicates over which tensor index we sum
        # hidden_term has dim batch_size
        return (-hidden_term - vbias_term) # returns the free energies of all the input spins in a vector
    
    def draw_sample(self, sample_length):
        v_ = F.relu(torch.sign(Variable(torch.randn(self.n_vis))))
        for _ in range(sample_length):
            h_ = self.v_to_h(v_)
            v_ = self.h_to_v(h_)
        return v_
    
    # -------------------------------------------------------------------------
    # TO DO (for n_hidden > 150 does not work)
    # Calculate exp( log( p(v))) to avoid exploding exponentials
    # exp ( -epsilon(v) - log(Z) )
    def partition_fct(self, spins):
        return (-self.free_energy(spins)).exp().sum()

    def probability_of_v(self, all_spins, v):
        epsilon = (-self.free_energy(v)).exp().sum()
        Z = self.partition_fct(all_spins)
        return epsilon/Z
    
    def train(self, train_loader, lr= 0.01, weight_decay=0, momentum=0.9, epoch=0):
        loss_ = []
        for _, data in enumerate(train_loader):
            self.data = Variable(data.view(-1,self.n_vis))
            
            if self.gpu:
                self.data = self.data.cuda()
            
            # Get positive phase from the data
            self.vpos = self.data
            self.hpos = self.v_to_h(self.vpos)
            # Get negative phase from the chains
            _, self.vneg = self.forward(self.vpos) # make actual k-step sampling
            self.hneg = self.v_to_h(self.vneg)
            if self.continuous_h == False:
                self.continuous_h = True
                self.hneg_probability = self.v_to_h(self.vneg)
                self.continuous_h = False
            else:
                self.hneg_probability = self.v_to_h(self.vneg)
        
            self.W_update.data      *= momentum
            self.h_bias_update.data *= momentum
            self.v_bias_update.data *= momentum
            
            self.deltaW = (outer_product(self.hpos, self.vpos)- outer_product(self.hneg_probability, self.vneg)).data.mean(0)
            self.deltah = (self.hpos - self.hneg_probability).data.mean(0)
            # change hneg_prob to hneg still works, but more wiggling
            self.deltav = (self.vpos - self.vneg).data.mean(0)
            # mean averages over all batches
            if self.gpu:
                self.W_update.data      += (lr * self.deltaW).cuda()
                self.h_bias_update.data += (lr * self.deltah).cuda()
                self.v_bias_update.data += (lr * self.deltav).cuda()
            else:
                self.W_update.data      += (lr * self.deltaW)
                self.h_bias_update.data += (lr * self.deltah)
                self.v_bias_update.data += (lr * self.deltav)
                # Update rule is W <- W + lr*(h_0 x_0 - h_k x_k)
                # But generally it is defined as W = W - v
                # Therefore v = -lr deltaW --> v in our case is W_update
                # With momentum we get v_t+1 = m*v_t + lr deltaW
                    
                self.W.data      += self.W_update.data
                self.h_bias.data += self.h_bias_update.data
                self.v_bias.data += self.v_bias_update.data
                
                loss_.append(F.mse_loss(self.vneg, self.vpos).data[0])

