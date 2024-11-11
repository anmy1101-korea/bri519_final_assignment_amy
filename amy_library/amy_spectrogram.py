"""
Calculates the spectrogram of a signal, showing the frequency components over time
a spectrogram of the short-time FFT using the hanning window

Parameters
----------
x : array
    Input time-domain signal
fs : float
    Sampling frequency of the input signal
windLength : int
    Length of the Hanning window applied to each segment
nonoverlap : int
    Number of samples for non-overlapping portion of each segment
nfft : int
    Number of points in FFT calculation
                        
Returns
-------
frequencies : array
    Array of frequency components up to Nyquist frequency
times : array
    Array of time points for each segment in the spectrogram
spectrogram : 2D array
    Power spectrum of the input signal (dB)
"""

import numpy as np
from amy_library.amy_fft import amy_fft
from amy_library.amy_hanning import amy_hanning

def amy_spectrogram(x,fs,windLength, nonoverlap, nfft) : #1
    wind = amy_hanning(windLength)
    spectrogram = []
    times = []
    overlap = nonoverlap / windLength
    step = int(windLength * (1-overlap))
    
    for start in range(0, len(x) - windLength, step):
        segment = x[start:start + windLength] * wind  

        
        if nfft > windLength :
            segment = np.pad(segment, (0, nfft-windLength), mode='constant')

        fft_result = amy_fft(segment)                
        power_spectrum = np.abs(fft_result[:nfft//2]) ** 2  
        spectrogram.append(power_spectrum)

        times.append(start/fs)
        
    spectrogram = np.array(spectrogram).T  
    spectrogram = 10 * np.log10(spectrogram)
    
    frequencies = np.arange(nfft//2)*fs/nfft
    # times = np.arange(0, len(x)-windLength, step) / fs
    times = np.array(times)

    return frequencies, times, spectrogram