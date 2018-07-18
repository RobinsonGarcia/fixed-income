
from functions2 import *
import numpy as np
import math
from scipy import optimize
import pylab as pl
from IPython import display as dp



class short_rate_lattice():
    def __init__(self,T,time_step):
        self.time_step = time_step
        self.T=T
        self.n_of_periods = T/time_step
        self.time = np.arange(0,T,time_step)
        self.period = np.arange(int(self.n_of_periods))
        self.ir = np.random.rand(int(self.n_of_periods))
        self.sigma = np.random.rand(int(self.n_of_periods))
        self.sigma[0] = 0
        self.build_sr()
    def setup(self,ir,sigma):
        self.ir = ir
        self.sigma = sigma
        self.build_sr()
         
    def build_sr(self):
        self.sr_lattice = {}
        for i in self.period:            
            bin_ = self.ir[i]*np.exp(np.arange(i+1)*2*self.sigma[i])#*np.sqrt(self.time_step))
            self.sr_lattice[i]=bin_

class Tree():
    def __init__(self,T,srl):
        self.srl = srl
        self.T = T
        self.n_of_periods = srl.n_of_periods
        self.period = np.arange(T)+1
        self.tree = {} #new
        self.p = 0 #new
        self.vol = 0 #new
        
        
    def build(self):   
        self.tree = {}
        self.tree[self.T]=np.ones(self.T+1)
        for i in np.flip(self.period,axis=0):                    
            self.tree[i-1] = 0.5*(self.tree[i][:-1] + np.roll(self.tree[i],shift=-1)[:-1])*np.exp(-self.srl.sr_lattice[i-1])
        self.p = self.tree[0][0]
        self.get_vol()
        
        return self.p,self.vol
        
    def get_vol(self):
        
        if self.T !=1:
            rates = np.log(self.tree[1])/(self.T-1)
            self.vol = 0.5*np.log(rates[1]/rates[0])/np.sqrt(self.srl.time_step)
        else:
            self.vol = 0
        
        pass

import multiprocessing as mp

class BDT(Tree):
    def __init__(self,T,time_step,mkt_data,plot=0):
        self.loss_bin = []
        self.T=T
        self.time_step = time_step
        self.n_of_periods = T/time_step
        self.period = np.arange(int(self.n_of_periods))
        self.srl = short_rate_lattice(T,time_step)
        self.build_Trees()
        self.mkt_data = mkt_data
        self.random_seed()
        self.plot = plot
    
    def random_seed(self):
        self.ir = -1*np.log(self.mkt_data['Price'])/(np.arange(self.time_step,self.T+self.time_step,self.time_step))
        self.vols = np.array(self.mkt_data['Vol'])
        self.vols[0] = 0
        self.x0 = np.vstack([self.ir,self.vols])
        
    def build_Trees(self):
        trees_T = np.arange(int(self.n_of_periods))+1
        self.trees = {}
        self.tree_prices = []
        self.tree_vols = []        
        
        for i in trees_T:
            self.trees[i]= Tree(i,self.srl)
            
        pool = mp.Pool(processes=4)
        results = [pool.apply(self.trees[x].build) for x in trees_T]
        self.tree_prices,self.tree_vols = zip(*results)
        pool.close()

            
    def loss(self,x):
        

        ir = x[:len(self.ir)]
        vols = np.append(0,x[len(self.ir):])
        self.srl.setup(ir,vols)
        self.build_Trees()      
        
        a = np.linalg.norm(np.array(self.tree_prices))
        b = np.linalg.norm(self.mkt_data['Price'])
        c = np.linalg.norm(np.array(self.tree_vols[1:]))
        d = np.linalg.norm(self.mkt_data['Vol'][1:])
       
        #ps = (np.array(self.tree_prices)/a - self.mkt_data['Price']/b)
        #vs = (np.array(self.tree_vols[1:])/c - self.mkt_data['Vol'][1:]/d)
        #ps[0] *= np.sqrt(0.1)
    
        ps = (np.array(self.tree_prices) - self.mkt_data['Price'])
        vs = (np.array(self.tree_vols[1:]) - self.mkt_data['Vol'][1:])
        ps[0] *= np.sqrt(0.1)

        loss = np.append(ps,vs)
        
        loss = 10000000*loss.dot(loss)

        if self.plot== 1:
            self.plot_ps(self.tree_prices,self.tree_vols,loss)

        if math.isnan(loss):
            loss = 10**6

        

            
        return loss
    
    def plot_loss(self,loss):
        pl.title('Total loss')
        pl.plot(loss)
        dp.clear_output(wait=True)
        dp.display(pl.gcf())
        
    
    def plot_ps(self,p,v,loss):
        N = np.arange(len(p))
        pindex = np.array(p)/np.array(self.mkt_data['Price'])
        vindex = np.array(v)/self.mkt_data['Vol']
        dp.clear_output(wait=True)
        
        pl.title('Relatives price and vol')
        pl.plot(N,pindex,label='price')
        pl.plot(N[1:],vindex[1:],label='vol')
        plt.legend()
        print(loss)
        dp.display(pl.gcf())        
        
        pl.gcf().clear()
        
    def grad(self,x):
        '''
        self.rate *= 0.95
        if self.rate<1e-6:
            self.rate=1e-6
        print('lr: {}'.format(self.rate)) 
        '''      
        return optimize.approx_fprime(x, self.loss,epsilon=self.rate)       
    
    def solve(self,plot=0,x0=None,method ='BFGS',options={},bounds=[]):
        self.plot=plot
        self.m = np.empty([])
        self.count=0
        self.rate = 1e-3
        self.random_seed()
        
        self.ir =  self.ir 
        self.vols = self.vols 
        
        
        try:
            _ = x0.shape[0]
            self.ir = x0[:len(self.ir)]
            self.vols = np.append(0,x0[len(self.ir):])

        except:
            self.srl.setup(self.ir,self.vols)
            x0 = np.append(self.ir,self.vols[1:])#/np.sqrt(len(self.ir)+len(self.vols))
        
        self.srl.setup(self.ir,self.vols)


        self.opt_results = optimize.minimize(self.loss,x0=x0,#jac=self.grad,
                                            method=method,bounds=bounds,options=options)
        self.ir_op = self.opt_results.x[:len(self.ir)]
        self.vols_op = self.opt_results.x[:len(self.ir)]
        print(self.opt_results)

