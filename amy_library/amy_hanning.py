"""
Calculates hanning window

Parameters
----------
N : int
    window length of hanning
    
Returns
-------
- : array
    the value of hanning window
"""

import numpy as np

# Define custom Hanning window function
def amy_hanning(N):
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / (N-1)))

