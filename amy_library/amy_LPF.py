"""
Low-Pass Filter (LPF) class for filtering high-frequency components from a signal.

Parameters
----------
cutoff_freq : float
    Cutoff frequency of low-pass filter
ts : float
    Sampling time step for the input signal
    
Attributes
----------
ts : float
    Sampling time step
cutoff_freq : float
    Cutoff frequency of low-pass filter
pre_out : float
    Previous output value of the filter (recursive filtering)
tau : float
    Filter coefficient calculated based on the cutoff frequency
"""

import numpy as np

class amy_LPF: #2
    def __init__(self, cutoff_freq, ts): #3
        self.ts = ts #4
        self.cutoff_freq = cutoff_freq #4
        self.pre_out = 0. #5
        self.tau = self.calc_filter_coef() #6 
        
    def calc_filter_coef(self): #7
        w_cut = 2*np.pi*self.cutoff_freq
        return 1/w_cut
        
    def filter(self, data): #8
        out = (self.tau*self.pre_out + self.ts*data) / (self.tau+self.ts) #9
        self.pre_out = out #10
        return out
    
    
    