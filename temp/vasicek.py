from functions2 import *
import numpy as np
#from functions import TermStructure,load_data
import numpy as np
import math
from scipy import optimize
import pylab as pl
from IPython import display as dp




class Vasicek():
    def __init__(self,rs,vol):
        self.t = rs.columns
        self.ps= rs[-1:]
        self.sigma = vol        
    
    def get_TheoreticalP(self,x=0):
        sigma = self.sigma
        try:
            _ = x.shape
        except:
            x = self.t
            
        a = self.a
        b = self.b
        B = (1-np.exp(-a*x))/a
        A = np.exp(((B-x)*(a**2*b-(sigma**2)/2))/a**2-(sigma**2*B**2)/(4*a))
        self.B=B
        self.A=A
        self.sim_p = A*np.exp(-B*x)
        self.r = -1*np.log(self.sim_p)/x
        return self.r

    
    def loss(self,x):
        self.a = x[0]
        self.b = x[1]   
        self.sim_rs = apply(self.get_TheoreticalP,self.ps)
        loss = np.array(self.ps.as_matrix())-np.array(self.sim_rs)

        loss = 10000*np.sum(loss**2)
        
        return loss

    
    def solve(self,x0=np.random.rand(2)):
        self.opt_results = optimize.fmin(self.loss,x0=x0)#,tol=1e-10,method='Nelder-Mead',options={'maxiter':1800})
        self.a = self.opt_results[0]
        self.b = self.opt_results[1]
        print(self.opt_results)
    
    def get_price_rate(self,T,r):
        
        sigma = list(self.sigma)[T]
        T = self.t[T]
        a = self.a
        b = self.b
        B = (1-np.exp(-a*T))/a
        A = np.exp(((B-T)*(a**2*b-(sigma**2)/2))/a**2)-(sigma**2*B**2)/(4*a)
        p = A*np.exp(-B*r)
        r = -1*np.log(p)/T
        return p,r


def option_pricing(V,r,t,T,X):
    #print('Expiration: {}'.format(t))
    #print('Maturity: {}'.format(T))
    
    time_dict = dict(zip(V.t,np.arange(len(V.t))))
    
    r = r[-1:][t].item()
    
    P = V.get_price_rate(time_dict[T],r)
    
    p = V.get_price_rate(time_dict[t],r)
    

    
    sigmap = V.sigma[t]*(1/V.a)*(1/np.sqrt(t))*(1-np.exp(-V.a*(T-t)))*np.sqrt((1-np.exp(-2*V.a*t))/(2*V.a))
    
    d = (1/sigmap)*np.log(P[0]/(p[0]*X))+0.5*sigmap
    
    c = P[0]*norm.cdf(d)-X*p[0]*norm.cdf(d-sigmap)
    
    return c