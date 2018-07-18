
# coding: utf-8

# In[338]:


import numpy as np
import math
from scipy import optimize


# ## Background
# 
# The values in the short rate lattice are of the form $r_{ks}=a_ke^{b_ks}$
# 
# k is the time (0,1..n)
# s is the state (0,1...k)
# $a_k$ is a measure of aggregate drift, asigned to match the implied term structure
# $b_k$ is the volatility of the logarithim of the short rate from time k-1 to k

# # Short-rate-lattice trees
# 
# -  Maturity T <br>
# -  Time_step 
# <br>
# OBS: T/time_step must be an integer
# 
# The class short_rate_lattice assemble the short-rate lattice in the form of a dictionary whose keys are the periods and values are the discount rates. The lattice can be acessed trough the attribute .sr_lattice. 
# 
# The build method is called at __init__, thus the class it is ready to go. However, the rates and the sigma are randomly generated. We want a procedure in which those are input variables.

# In[205]:


class short_rate_lattice():
    def __init__(self,T,time_step):
        srl.T=T
        self.n_of_periods = T/time_step
        self.time = np.arange(0,T,time_step)
        self.period = np.arange(int(self.n_of_periods))
        self.ir = np.random.rand(int(self.n_of_periods))
        self.sigma = np.random.rand(int(self.n_of_periods))
        self.sigma[0] = 0
        self.build()
    def setup(self,ir,sigma):
        self.ir = ir
        self.sigma = sigma
        self.build()
         
    def build(self):
        self.sr_lattice = {}
        for i in self.period:
            bin_ = np.zeros(i+1)
            x = self.ir[i]
            for j in range(i+1):  
                bin_[j] = x
                x = bin_[j]*np.exp(2*self.sigma[i])
                
            self.sr_lattice[i]=bin_


# In[188]:


srl = short_rate_lattice(T=3,time_step=1)


# In[131]:


ir = [0.0953,0.0923,0.0940]
sigma = [0,0.19,0.172]
srl.setup(ir,sigma)
srl.build_srTree()


# In[117]:


srl.sr_lattice


# # Binomial Trees
# 
# The class tree will assemble a binomial tree with T periods whose face values at maturity are one. The FV can be alterred with the attribute FV. The method build acalculate the tree components and the price. Both can be accessed trough the attributes .tree and .p respectively.
# 
# A tree requires a short-rate-lattice object.

# In[417]:


class Tree():
    def __init__(self,T,srl):
        self.srl = srl
        self.T = T
        self.n_of_periods = srl.n_of_periods
        self.period = np.arange(T)+1
        self.build()
        
    def build(self):
        self.tree = {}
        self.tree[self.T]=np.ones(self.T+1)
        for i in np.flip(self.period,axis=0):
            
            bin_=[]
            for j in np.arange(i):
                
                bin_.append(0.5*(self.tree[i][j]+
                                 self.tree[i][j+1])*
                            np.exp(-self.srl.sr_lattice[i-1][j]))
            
            self.tree[i-1]=bin_
        self.p = self.tree[0][0]
        self.get_vol()
        
    def get_vol(self):
        if self.T !=1:
            rates = np.log(self.tree[1])/(self.T-1)
            self.vol = 0.5*np.log(rates[1]/rates[0])
        else:
            self.vol = 0
        
        pass


# In[238]:


Tr = Tree(2,srl)
print('price: {}'.format(Tr.p))


# In[168]:


Tr.tree


# In[156]:


Tr.vol


# # Building a model

# The BDT class builds a short rate tree and all required tress for the bonds wih different maturities. It is a subclass of the short_rate_lattice and Tree classes. The method loss calculates the loss given a interest rate(ir) and volatility(vol) series.
# 
# Ex. 3 year maturity, with a time step of one year.
# <br> Market_Data = {'Price': [0.9091,0.8116,0.7118],'Vol':[0,.19,.18]}

# In[81]:


T = 3
time_step=1
Market_Data = {'Price': [0.9091,0.8116,0.7118],'Vol':[0,.19,.18]}


# In[488]:


np.random.rand(2)


# In[518]:


class BDT(short_rate_lattice,Tree):
    def __init__(self,T,time_step,mkt_data):
        self.T=T
        self.srl = short_rate_lattice(T,time_step)
        self.build_Trees()
        self.mkt_data = mkt_data
        self.random_seed()
    
    def random_seed(self):
        self.ir = -1*np.divide(np.log(self.mkt_data['Price']),np.arange(len(self.mkt_data['Price']))+1)
        self.vols = np.array(self.mkt_data['Vol'])
        self.vols[0] = 0
        #self.ir = np.array([0.15,0.0923,0.5])
        #self.vols = np.array([0,0.18,0.172])
        self.x0 = np.vstack([self.ir,self.vols])
        
    def build_Trees(self):
        trees_T = np.arange(self.T)+1
        self.trees = {}
        for i in trees_T:
            self.trees[i]= Tree(i,self.srl)
            
    def loss(self,x):
        
        x = np.reshape(np.array(x),[2,len(self.ir)])
        ir = x[0]
        
        vols = np.append([0],x[1][1:])
        self.srl.setup(ir,vols)

        loss = 0
        for j in self.trees.keys():
            self.trees[j].build()
            ps = (-self.trees[j].p + self.mkt_data['Price'][j-1])**2
            loss += 10000*(ps + (-self.trees[j].vol +mkt_data['Vol'][j-1])**2)
        return loss
    def solve(self):
        ir = self.ir #np.array([0.15,0.0923,0.5])
        vols = self.vols #np.array([0,0.18,0.172])
        x0 = np.append(self.ir,self.vols)
        self.opt_results = optimize.minimize(self.loss,x0=x0,method='BFGS')
        print(self.opt_results.x)

        
    


# In[523]:


mkt_data = {'Price': [0.9091,0.8116,0.7118,0.62,0.2],'Vol':[0,.19,.18,.15,0.25]}
bdt = BDT(5,1,mkt_data)
bdt.solve()

