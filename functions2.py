from __future__ import division

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize



def load_data():
    #https://www.federalreserve.gov/datadownload/Choose.aspx?rel=H15
    #https://fred.stlouisfed.org/categories/115
    data = pd.read_csv('data.csv')
    data = data.set_index('Time Period')
    data.index = pd.to_datetime(data.index)


    #data_ = data.dropna()
    data_ = data.applymap(f).copy()
    #data_ = data_[data_['RIFLGFCM06_N.B']>0] # keeping only data with at least one strip
    #data_ = data_.loc['7/31/2001':]
    print('Total of nulls {}'.format(data_.isnull().sum()))
    
    total = data_.shape[0]
    data_ = data_.fillna(0)
    #data_=data_.dropna()
    print('\n Total of nulls {}'.format(data_.isnull().sum().sum()))

    return data,data_

def f(x):
    try:
        return float(x)/100
    except:
        return np.nan


from scipy.interpolate import CubicSpline
class CubicS():
    def __init__(self,x=np.array([1/12,3/13,6/12,1,2,3,5,7,10,20,30]),x_=np.arange(1,30+1/2,1/2)):
        self.x = x
        self.x_ = x_
        
    def spline(self,y):   
        cs = CubicSpline(self.x,y)
        y_ = cs(self.x_)
        return y_

class NelsonSiegel():
    def __init__(self,x=np.array([1/12,3/13,6/12,1,2,3,5,7,10,20,30]),x_=np.arange(1,30+1/2,1/2)):
        self.x = x
        self.x_ = x_
    
    def loss(self,w):
        t = self.x
        A = [np.ones(t.shape[0]),(1-np.exp(-t/w[3]))/(t/w[3]),(1-np.exp(-t/w[3]))/(t/w[3])-np.exp(-t/w[3])]
        Ax = w[:3].dot(A)
        error = self.y-Ax
        return np.sum(error**2)
    
    def get_w(self):
        y = self.y  
        if np.sum(y)==0:
            print('gotcha')
        x0 = [y[-1],np.abs(y[0]-y[1]),y[0],y[-1]]
        self.opt_results = optimize.minimize(self.loss,x0=x0,jac=False,tol=1e-13,method='BFGS')
        self.w = self.opt_results.x
        
    def NS(self,y):
        self.y = y
        self.get_w()
        y = self.y
        w = self.w
        t = self.x_
        A = [np.ones(t.shape[0]),(1-np.exp(-t/w[3]))/(t/w[3]),(1-np.exp(-t/w[3]))/(t/w[3])-np.exp(-t/w[3])]
        return w[:3].dot(A)
    

def apply(f,y):
    results = []
    for i in y.index:
        results.append(f(y.loc[i]))
    return np.vstack(results)

np.seterr(all='ignore') 
# it is expected to ger errors on the log because of
# divisions by zero. Those are removed from the data
# on the last lines
def get_vols(irs,vol_window):
    irs = irs[-vol_window:]
    ratio = irs[1:]/irs.shift(1)[1:]
    ratio = ratio[ratio.notnull()].copy()
    for i in ratio.columns:
        ratio = ratio[ratio[i].notnull()]
        log_ratio = np.log(ratio)
    log_ratio = log_ratio[log_ratio!=np.inf]
    log_ratio = log_ratio[log_ratio!=-np.inf]
    return np.std(log_ratio)

from scipy.interpolate import CubicSpline

class BootStraper():

    def interpol(self,y):
        self.x = np.array([1/12,3/13,6/12,1,2,3,5,7,10,20,30]) #available maturties
        
        try:
            _ = self.x_.shape
        except:
            self.x_ = np.arange(1,30+1/2,1/2) # strips
           
        
        
        
        if self.method == 'NelsonSiegel':
            ns = NelsonSiegel()
            return apply(ns.NS,y)
        
        
        if self.method == 'spline':
            sp = CubicS()
            sp.x_ = self.x_
            return apply(sp.spline,y)     
        
    def BootStrap(self,x_,y,method='spline'):
        '''dataframe info'''
        index = y.index
        columns = [1/12,3/12,6/12]
        
        '''INPUTS'''        
        self.x_ = x_
        self.method = method
        
        '''Interpolation'''
        y_strips = y[y.columns[:3]] #get 1,3 and 6 months strips     
        y = self.interpol(y)

        y = np.hstack([y_strips,y]) #combine strips with interpolated data
        
        '''Vectorized Bootstrapping'''
        n=y.shape[0]
        y = y[-n:].reshape(n,1,y.shape[1])      
        N = y.shape[2]
        n = y.shape[0]
        powers = np.arange(1,N+1)
        ys = np.tile(y,(1,N,1))
        ys = np.swapaxes(ys,2,1)
        f = 100
        F = np.tile(f*np.diag(np.ones(N)),(n,1,1))
        C = ys*f/2
        YS = 1/np.power(ys+1,powers)
        b = np.sum(np.tril((C+F)*YS),axis=2)
        A = np.tril(F + C)
        rates = np.linalg.solve(A,b)
        rates = np.power(1/rates,1/powers) -1
        
        '''return dataframe'''
        return pd.DataFrame(data=rates,index=index,columns=np.append(columns,self.x_))
        