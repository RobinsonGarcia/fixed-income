
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
get_ipython().magic('matplotlib inline')


# # Fixed income securities

# ## Load data and pre process:

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
    data_ = data_[data_['RIFLGFCM06_N.B']>0] # keeping only data with at least one strip
    return data,data_


# In[3]:

def f(x):
    try:
        return np.float(x)/100
    except:
        return np.nan


# In[6]:

def fadjust(y,T):
    x = np.isnan(y)
    T_ = []
    y_ = []
    for i in np.arange(len(y)):
        if x[i]==False:
            T_.append(T[i])
            y_.append(y[i])
    return np.array(y_),np.array(T_)

def getStrips(y,T):
    rn = {}
    for i in np.arange(len(y)):
        if T[i]<1:
            rn[T[i]]=y[i]
    if len(rn.keys())==0:
        print("No strips retrieved!")
    return rn



# In[7]:

class Bootstrapping():
    def __init__(self,F=100,cf=2):
        self.F = F
        self.cf = cf

    def getStrips(self,y,T):
        rn = {}
        for i in np.arange(len(y)):
            if T[i]<1:
                rn[T[i]]=y[i]
        if len(rn.keys())==0:
            print("No strips retrieved!")
        return rn

    def interpol(self,y,T_):
        self.y=y
        self.T = np.arange(1,30.5,0.5)
        N = self.cf*self.T
        self.N = [int(x) for x in N]
        self.yn = np.interp(self.T,T_,y)
        self.rn = self.getStrips(y,T_) #{1/12:y[0],0.25:y[1],0.5:y[2]}

        if len(self.y)>4:
            self.seed = np.random.choice(self.y,4)
        else:
            self.seed = np.random.randn(4)

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

# In[8]:

class NelsonSiegel():
    def __init__(self,rn={}):
        self.rn = rn

    def NS(self,w):
        rn = self.rn
        t = np.array(list(rn.keys()))
        A = [np.ones(t.shape[0]),np.exp(-t/w[3]),(t/w[3]*np.exp(-t/w[3]))]
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


# In[9]:

class TermStructure(NelsonSiegel,Bootstrapping):
    def __init__(self,y,T_,F=100,cf=2):
        self.F = F
        self.cf = cf
        self.solve(y,T_)
        self.wmin = self.get_w(self.seed)


class getTermS():
    def __init__(self,data_,start):
        self.data_ = data_
        self.start = start
    def solve(self):
        data_=self.data_
        NScoef = {}
        count = 0
        num_of_days = data_.shape[0]
        for day in data_.index[data_.index>self.start]:
            T_ = [1/12,0.25,0.5,1,2,3,5,7,10,20,30]

            day1 = data_.loc[day]
            y = day1.as_matrix().flatten()
            y,T_ = fadjust(y,T_)
            rn = getStrips(y,T_)
            if rn.keys()==0:
                continue
            TS = TermStructure(y,T_)
            NScoef[day] = TS.wmin
            count +=1
            if count%100==0:
                print("Percent concluded: {}%".format(100*count/num_of_days))
        self.w = NScoef
