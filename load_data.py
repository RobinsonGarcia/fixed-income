from functions2 import *
import numpy as np
import math
from scipy import optimize
import pylab as pl
from IPython import display as dp

'''
T = 5.5
time_step = 1/12
range_ = np.arange(time_step,T+time_step,time_step)
vol_window = 2*252
term_window= 6000
interpol_method = 'spline' #NelsonSiegel
solver_method = 'BFGS'
'''

def f(x):
    try:
        return float(x)/100
    except:
        return np.nan

#
# it is expected to ger errors on the log because of
# divisions by zero. Those are removed from the data
# on the last lines
def get_vols(irs,vol_window):
    np.seterr(all='ignore') 
    irs = irs[-vol_window:]
    ratio = irs[1:]/irs.shift(1)[1:]
    ratio = ratio[ratio.notnull()].copy()
    for i in ratio.columns:
        ratio = ratio[ratio[i].notnull()]
        log_ratio = np.log(ratio)
    log_ratio = log_ratio[log_ratio!=np.inf]
    log_ratio = log_ratio[log_ratio!=-np.inf]
    return np.std(log_ratio)




class get_data():
    def __init__(self,T,time_step,term_window,interpol_method='spline'):
        range_ = np.arange(time_step,T+time_step,time_step)

        #https://www.federalreserve.gov/datadownload/Choose.aspx?rel=H15
        #https://fred.stlouisfed.org/categories/115
        data = pd.read_csv('data.csv')
        data = data.set_index('Time Period')
        data.index = pd.to_datetime(data.index)

        #data_ = data.dropna()
        data_ = data.applymap(self.f).copy()
        #data_ = data_[data_['RIFLGFCM06_N.B']>0] # keeping only data with at least one strip
        #data_ = data_.loc['7/31/2001':]
        print('Total of nulls {}'.format(data_.isnull().sum()))
        
        total = data_.shape[0]
        data_ = data_.fillna(0)
        #data_=data_.dropna()
        print('\n Total of nulls {}'.format(data_.isnull().sum().sum()))

        TS = BootStraper()
        y = data_[-term_window:]
        y=y[y!=0].dropna() #removing days in which yields where quoted as zero
        x_= None
        r = TS.BootStrap(x_,y,method=interpol_method)
        #r[-100:].plot(legend=False)

        #plt.plot(np.array(r.columns),r[-1:].as_matrix().flatten(),label='Bootstrapped');
        #plt.plot(TS.x,data_[-1:].as_matrix().flatten(),label='yield_curve');
        #plt.legend()

        r=r.dropna()

        interpol = CubicS(x=np.array(r.columns),x_=range_)
        f = interpol.spline
        irs = pd.DataFrame(data=apply(f,r),index=r.index,columns=range_)
        #irs[-2000:].plot(legend=False);
        self.irs = irs
        self.data_=data_
        self.range_ =range_

    def f(self,x):
        try:
            return float(x)/100
        except:
            return np.nan
 
    
    def vols(self,vol_window):
        irs = self.irs
        range_ = self.range_

        vols = get_vols(irs,vol_window)
        plt.plot(vols)
        plt.title('vols, roll back window of {}'.format(vol_window));



        last_term = np.exp(-1*irs[-1:]*range_)
        plt.figure()
        #plt.title('rates')
        #plt.plot(range_,last_term.as_matrix().flatten());


        return {'Price':last_term.as_matrix().flatten(),'Vol':list(vols)}




