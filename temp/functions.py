
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import ipywidgets as widgets
from IPython import display as dp
import pylab as pl
import math



# # RSM2310 - Fixed income securities
# This notebook covers interpolation,bootstrapping,interpolation using Nelson Siegel model, and the implementation of the Vasicek model.

# ## Load data and pre process:
# The csv file data.csv contains the historical rates for a set of trasury bonds.

# In[2]:


def load_data():
    data = pd.read_csv('data.csv')
    data = data.set_index('Time Period')

    NDs=[]
    count =0
    for i in data.index:
        if 'ND'== data.loc[i][1]:
            NDs.append(i)
            count += 1
    print("Excluded {} NDs".format(count))
    print("{}% of the total".format(100*round(count/data.shape[0],2)))
    data_ = data.applymap(f).copy()
    #data_ = data_[data_['RIFLGFCM06_N.B']>0] # keeping only data with at least one strip
    data_ = data_.loc['7/31/2001':]
    return data,data_


# In[3]:


def f(x):
    try:
        return float(x)/100
    except:
        return np.nan


# ## Bootstrapping class:

# In[5]:




def getStrips(y,T):
    rn = {}
    for i in np.arange(len(y)):
        if T[i]<0.5:
            rn[T[i]]=y[i]
    if len(rn.keys())==0:
        print("No strips retrieved!")
    return rn



# In[6]:


class Bootstrapping():
    def __init__(self,F=100,cf=2):
        self.F = F
        self.cf = cf
        
    def getStrips(self,y,T):
        rn = {}
        for i in np.arange(len(y)):
            if T[i]<0.5:
                rn[T[i]]=y[i]
        if len(rn.keys())==0:
            print("No strips retrieved!")
        return rn
   
    def fadjust(self,y,T):
        x = np.isnan(y)
        T_ = []
        y_ = []
        for i in np.arange(len(y)):
            if x[i]==False:
                T_.append(T[i])
                y_.append(y[i])
        return np.array(y_),np.array(T_)
   
    def interpol(self,y,T_):
        self.y=y        
        self.T = np.arange(0.5,30.5,0.5)
        N = self.cf*self.T
        self.N = [int(x) for x in N]
        self.yn = np.interp(self.T,T_,y)
        self.rn = self.getStrips(y,T_) 
    
        
        if len(self.y)>4:      
            self.seed = [self.y[-1],self.y[0]-self.y[-1],self.y[1],self.y[2]]
        else:
            self.seed = [self.y[-1],self.y[0]-self.y[-1],self.y[1],self.y[2]]
    
    def get_price(self,y,T):
        C = y*self.F
        N = T*self.cf
        pv = 0
        for n in np.arange(N):
            pv += C/(1+y)**n
        pv += self.F/(1+y)**N
        return pv
    
    def solve(self,y,T_):      
        self.interpol(y,T_)
        for i in self.N:
            C = self.yn[i-2]*self.F/2
            rate_sum = self.get_price(self.yn[i-2],i/self.cf) 
            for j in np.arange(0.5,i/self.cf,0.5):
                rate_sum -= C*((1+self.rn[j]/self.cf)**(-j))
            self.rn[i/self.cf] = (((self.F+C)/rate_sum)**(1/i)-1)*2 
            
    def plot(self):
        d = pd.DataFrame.from_dict(self.rn,orient='index')
        d.plot(kind='scatter')

        


# ## Nelson Siegel Class
# 
# The Nelon Siegel model follow the equation:
# 
# \begin{equation}
# R(0,t) = \beta_0 + \beta_1 \frac{1-e^{(\frac{-t}{T})}}{t/T} + \beta_2  (\frac{1-e^{(\frac{-t}{T})}}{t/T} -e^{\frac{-t}{T}})
# \end{equation}
# 
# Where:
# 
# -  $R(0,t)$: pure discount rate with maturity t
# -  $\beta_0$: level parameter, the long term rate
# -  $\beta_1$: slope parameter, the spread short/long-term
# -  $\beta_2$: curvature parameter
# -  $T$: scale parameter
# 
# 

# In[7]:


class NelsonSiegel():
    def __init__(self,rn):
        self.rn = rn
        
    def NS(self,w):
        rn = self.rn
        if type(self.rn)==dict:
            t = np.array(list(rn.keys()))
        else:
            t = self.rn

        A = [np.ones(t.shape[0]),(1-np.exp(-t/w[3]))/(t/w[3]),(1-np.exp(-t/w[3]))/(t/w[3])-np.exp(-t/w[3])]

        Ax = w[:3].dot(A)
        return Ax

    def loss(self,w):
        Ax = self.NS(w)
        rn = self.rn
        y = np.array(list(rn.values()))
        error = y - Ax
        Loss = np.sum(error**2)
        return Loss
    
    def get_w(self,x0):
        self.opt_results = optimize.minimize(self.loss,x0=x0,method='BFGS')
        return self.opt_results.x
    
    def plot_termS(self):
        termS = self.NS(self.opt_results.x)
        plt.plot(termS)
    

            


# In[8]:


class TermStructure(NelsonSiegel,Bootstrapping):
    def __init__(self,progress_bar=0):
        self.progress_bar = progress_bar
        self.pbar = widgets.IntProgress(
                value=0,
                min=0,
                max=4000,
                step=1,
                description='Loading:',
                bar_style='', # 'success', 'info', 'warning', 'danger' or ''
                orientation='horizontal'
            )
        pass
    
    def Boot(self,y,T_,F=100,cf=2):
        self.F = F
        self.cf = cf
        self.solve(y,T_)
        self.wmin = self.get_w(self.seed)
    def fit(self,data_,num_of_days=4000):
        
        NScoef = np.empty((0,4))
        self.date = np.empty((0,1))
        count = 0
        
        self.pbar.max = num_of_days
        display(self.pbar)
        for day in data_.index[-num_of_days:]:
            
            
            T_ = [1/12,0.25,0.5,1,2,3,5,7,10,20,30]

            day1 = data_.loc[day]
            y = day1.as_matrix().flatten()
            y,T_ = self.fadjust(y,T_)
            rn = getStrips(y,T_)
            if rn.keys()==0:
                print('no rn?')
                continue
            #TS = TermStructure()
            self.Boot(y,T_)
            NScoef = np.append(NScoef,[self.wmin],axis=0)
            self.date = np.append(self.date,[day])
            count +=1
            self.pbar.value=count
            #if count%100==0:
             #   print("Percent concluded: {}%".format(100*count/num_of_days))
        self.date = pd.to_datetime(self.date)
        self.Results = pd.DataFrame(data=NScoef,index=self.date)
        self.Results.plot()
        self.NScoef = NScoef
    
    def make_ps(self,t,plot=0):
        w_ = self.NScoef
        r = []
        for x in np.arange(len(w_)):
            w = w_[x]

            A = np.array([np.ones(t.shape[0]),(1-np.exp(-t/w[3]))/(t/w[3]),(1-np.exp(-t/w[3]))/(t/w[3])-np.exp(-t/w[3])])

            r.append(w[:3].dot(A))
        self.ps=pd.DataFrame(data=np.vstack(r),columns=t,index=self.date)
        if plot==1:
            self.ps.plot(title='Projected Term Structure',figsize=(10,6));
        pass
         


# In[9]:




# # Vasicek class: solve for model coeficients
# 
# Vasicek's model parameters can be solved by minimzing the loss function:
# 
# \begin{equation}
# loss = \sum{\frac{1}{N}(P-P(0,T))^2} 
# \end{equation}
# 
# The model for $P(0,T)$ is given by the following relations:
# 
# \begin{equation}
# P(0,T) = A(T) exp^{-B(T)r_0}
# \end{equation}
# <br>
# \begin{equation}
# A(T) = exp{{\frac{(B(T)-T)(a^2b-\frac{\sigma^2}{2})}{a^2}}-\frac{\sigma^2B^2(T)}{4a}}
# \end{equation}
# <br>
# \begin{equation}
# B(T) = \frac{1-e^{-aT}}{a}
# \end{equation}
# 
# The class Vasicek has this model implemented. It requires a time series with NelsonSiegel coeficients and the historical spot rates at a given set of maturities (ps). We use the method get_ps() on the previous cell to extract ps using NS.

# In[37]:


class Vasicek():
    def __init__(self,NScoef,ps):
        self.t = np.array([1/12,0.25,0.5,1.0,2.0,3.0,5.0,7.0,10.0,20.0,30.0])
        self.NScoef = NScoef
        self.build_TS(NScoef)
        self.ps=ps
        self.sigma = np.std(ps)
        
    def build_TS(self,NScoef):
        t = self.t
        Nsiegel = NelsonSiegel(t)
        result = []
        for i in range(len(NScoef)):
            result.append(Nsiegel.NS(NScoef[i]))
        self.X = np.vstack(result)
        pass
    
    def Bt(self,t):
        a = self.a
        return 1-np.exp(-a*self.t)/a
    
    def get_AB(self):
        sigma = self.sigma
        x = self.t
        a = self.a
        b = self.b
        B = self.Bt(x)
        A = np.exp(((B-x)*(a**2*b-(sigma**2)/2))/a**2)-(sigma**2*B**2)/(4*a)
        self.B=B
        self.A=A
        pass
    
    def get_P(self,r):
        A = self.A
        B = self.B
        b = np.exp(-np.multiply(B,r))
        return np.multiply(A,b)
    
    def loss(self,x):
        self.a = x[0]
        self.b = x[1]
        self.get_AB()
        sim_p = pd.DataFrame(self.X).apply(self.get_P,axis=1)
        p = 100/(self.ps+1)**self.t
        loss = np.array(p)-np.array(sim_p)
        size = loss.shape
        N = size[0]*size[1]
        loss = np.sum(loss**2)/N
        return loss
    
    def solve(self,x0=[1,1]):
        self.opt_results = optimize.minimize(self.loss,x0=x0,method='BFGS')
        self.a = self.opt_results.x[0]
        self.b = self.opt_results.x[1]
        print(self.opt_results)
    
    def price(self,T,r):
        dic = dict(zip(self.t,self.sigma))
        sigma = dic[T]
        a = self.a
        b = self.b
        B = 1-np.exp(-a*T)/a
        A = np.exp(((B-T)*(a**2*b-(sigma**2)/2))/a**2)-(sigma**2*B**2)/(4*a)
        return A*np.exp(-B*r)

    def rate(self,T,p):
        dic = dict(zip(self.t,self.sigma))
        sigma = dic[T]
        a = self.a
        b = self.b
        B = 1-np.exp(-a*T)/a
        A = np.exp(((B-T)*(a**2*b-(sigma**2)/2))/a**2)-(sigma**2*B**2)/(4*a)
        return -1*np.log(p/A)/B


# In[ ]:


# Pending improvement: std rolling window, or period options. Way to do sensitivity analysis.


# coding: utf-8

# In[338]:




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
            bin_ = self.ir[i]*np.exp(np.arange(i+1)*2*self.sigma[i]*np.sqrt(self.time_step))
            self.sr_lattice[i]=bin_


# In[188]:


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
            self.vol = 0.5*np.log(rates[1]/rates[0])/np.sqrt(self.srl.time_step)
        else:
            self.vol = 0
        
        pass





# # Building a model

# The BDT class builds a short rate tree and all required tress for the bonds wih different maturities. It is a subclass of the short_rate_lattice and Tree classes. The method loss calculates the loss given a interest rate(ir) and volatility(vol) series.
# 
# Ex. 3 year maturity, with a time step of one year.
# <br> Market_Data = {'Price': [0.9091,0.8116,0.7118],'Vol':[0,.19,.18]}
class BDT(short_rate_lattice,Tree):
    def __init__(self,T,time_step,mkt_data):
        self.loss_bin = []
        self.T=T
        self.time_step = time_step
        self.n_of_periods = T/time_step
        self.srl = short_rate_lattice(T,time_step)
        self.build_Trees()
        self.mkt_data = mkt_data
        self.random_seed()
    
    def random_seed(self):
        self.ir = -1*np.divide(np.log(self.mkt_data['Price']),np.arange(self.time_step,self.T+self.time_step,self.time_step))
        self.vols = np.array(self.mkt_data['Vol'])
        self.vols[0] = 0
        self.x0 = np.vstack([self.ir,self.vols])
        
    def build_Trees(self):
        trees_T = np.arange(int(self.n_of_periods))+1
        self.trees = {}
        for i in trees_T:
            self.trees[i]= Tree(i,self.srl)
 
            
    def loss(self,x):
        ir = x[:len(self.ir)]
        vols = np.append(0,x[len(self.ir)+1:])
        self.srl.setup(ir,vols)
        self.build_Trees()

        loss = 0
        count = 0
        prices = []
        volatilities = []
        for j in self.trees.keys():
            
            self.trees[j].build()
            ps = (self.trees[j].p - self.mkt_data['Price'][j-1])**2
            
            loss += 10000*(ps + (self.trees[j].vol - self.mkt_data['Vol'][j-1])**2)
            
            prices.append(self.trees[j].p)
            volatilities.append(self.trees[j].vol) 
            
        
        self.loss_bin.append(loss)
        #self.plot_loss(self.loss_bin[-100:])
        self.plot_ps(prices,volatilities)
        
        if len(self.loss_bin)>500:
            del self.loss_bin[:-110]
            
        return loss
    
    def plot_loss(self,loss):
        pl.title('Total loss')
        pl.plot(loss)
        dp.display(pl.gcf())
        dp.clear_output(wait=True)
    
    def plot_ps(self,p,v):
        pindex = np.array(p)/np.array(self.mkt_data['Price'])
        vindex = np.array(v[1:])/self.mkt_data['Vol'][1:]
        pl.title('Relatives price and vol')
        pl.plot(pindex,label='price')
        pl.plot(vindex,label='vol')

        dp.display(pl.gcf())
        dp.clear_output(wait=True)
        pl.gcf().clear() 
        
    
    def solve(self,method ='BFGS',options={},bounds=[]):
        ir = self.ir 
        vols = self.vols 
        x0 = np.append(self.ir,self.vols)                    
        self.opt_results = optimize.minimize(self.loss,x0=x0,tol=10,
                                             method=method,bounds=bounds,options=options)
        self.ir_op = self.opt_results.x[:len(self.ir)]
        self.vols_op = self.opt_results.x[:len(self.ir)]
        print(self.opt_results)
        