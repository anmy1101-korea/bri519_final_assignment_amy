"""
Calculates the Fast Fourier Transform (FFT) of a sequence using recursion

Parameters
----------
x : array
    input sequence (time domain signal)
    
Returns
-------
- : array
    FFT of input sequence x (freqeuncy domain signal)
"""

import numpy as np 

def amy_fft(x): #1
	N = len(x) #2
	if N <= 1: #3
		return x
	even = amy_fft(x[0::2]) #4
	odd = amy_fft(x[1::2]) #5
	Wn = [np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)] #6
	return [even[k] + Wn[k] for k in range(N//2)] + [even[k] - Wn[k] for k in range(N // 2)] #7

# def amy_fft(x): #1
# 	N = len(x) #2
# 	if N == 1: #if input length == 1, 
# 		return x 
# 	else :
# 		if N % 2 != 0:
# 			x = np.append(x, 0)
# 			N += 1	
# 		even = amy_fft(x[0::2]) #4
# 		odd = amy_fft(x[1::2]) #5
# 		Wn = [np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)] #6
# 		return [even[k] + Wn[k] for k in range(N//2)] + [even[k] - Wn[k] for k in range(N // 2)] #7

