import matplotlib.pyplot as plt 
import numpy as np 
import scipy.signal as sg 
import scipy.io # only use when loading .mat file
# import matplotlib.pylab as plt
import timeit
from amy_library.amy_fft import amy_fft
from amy_library.amy_spectrogram import amy_spectrogram
from amy_library.amy_hanning import amy_hanning
from amy_library.amy_LPF import amy_LPF
from matplotlib.colors import Normalize


#--------------------------------------------------------
# Question 1 --------------------------------------------
#--------------------------------------------------------

# 1 (c)  ------------------------------------------------

fs = 1024; 
time = np.linspace(0,2,fs*2)
y=sg.chirp(time,100,1,150,'quadratic'); 

# (2) FFT

my_out = amy_fft(y) 
nyquist_half = int(len(my_out)/2)
print(nyquist_half)
my_out_nyquist = my_out[0:nyquist_half] 

T = len(time)/fs 
freq = np.arange(len(time)) 
freq = freq / T 
freq_nyquist = freq[0:nyquist_half] 

plt.figure(figsize=(15,9))

plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("FFT of chirp sound")  
plt.grid(True)  
plt.plot(freq_nyquist, my_out_nyquist) 
plt.savefig("1_c.png")

# 1 (d)  ------------------------------------------------

# Comparison 1 --------------------
fig=plt.figure(figsize=(15,5))
ax = plt.subplot(1,2,1)
ax.plot(freq_nyquist, my_out_nyquist) 
ax.set_xlabel("Frequency (Hz)")  
ax.set_ylabel("Magnitude (dB)")  
ax.set_title("my FFT (low level code)")  
ax.grid(True)  

# Code Using fft library for Comprarison
ax = plt.subplot(1,2,2)#5
library_out = np.fft.fft(y) #2
library_out_nyquist = library_out[0:nyquist_half]
ax.plot(freq_nyquist, library_out_nyquist) 
ax.set_xlabel("Frequency (Hz)")  
ax.set_ylabel("Magnitude (dB)")  
ax.set_title("Library FFT code")  
ax.grid(True) 
plt.savefig("1_d_comp1.png")

# Comparison 2 --------------------
difference = np.abs(my_out-library_out)
print("average of difference : {}".format(np.average(difference)))
print("max of difference : {}".format(np.max(difference)))

plt.figure(figsize=(10,5))
plt.xlabel("Frequency (Hz)")  
plt.ylabel("difference of magnitude (dB)")  
plt.title("Difference between my low level fft and library fft")  
plt.grid(True)  
plt.plot(freq, difference) 
plt.savefig("1_d_comp2.png")

# Comparison 3 --------------------
setup_code = """
import matplotlib.pyplot as plt 
import numpy as np 
import scipy.signal as sg 
fs = 1024; #2
time = np.linspace(0,2,fs*2)
y=sg.chirp(time,100,1,200,'quadratic');
def my_fft(x):
	N = len(x)
	if N <= 1:
		return x
	even = my_fft(x[0::2])
	odd = my_fft(x[1::2])
	T = [np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)]
	return [even[k] + T[k] for k in range(N//2)] + [even[k] - T[k] for k in range(N // 2)]
"""

# Measure NumPy FFT runtime
numpy_time = timeit.timeit(stmt='np.fft.fft(y)', setup=setup_code, number=100)

# Measure my low level FFT runtime
my_fft_time = timeit.timeit(stmt='my_fft(y)', setup=setup_code, number=100)

# my_fft_time/numpy_time
runtime_times = my_fft_time/numpy_time

with open("1_d_comp3.txt", "w") as file:
    file.write(f"NumPy Library FFT runtime: {numpy_time:.6f} sec\n")
    file.write(f"My Low level FFT runtime: {my_fft_time:.6f} sec\n")
    file.write(f"My Low level FFT runtime / NumPy Library FFT runtime : {runtime_times:.6f}\n")

# 1 (e)  ------------------------------------------------
windLength = 128; 
overl = windLength-1; 
freqBins = 250;
nfft = 2048 
f, tt, Sxx = amy_spectrogram(y, fs, windLength, overl, nfft=nfft)

# Plot the spectrogram
plt.figure(figsize=(10, 6))
plt.pcolormesh(tt, f, Sxx)
plt.xlabel("Time (sec)")
plt.ylabel("Frequency (Hz)")
plt.title("Spectrogram (amy Hanning Window)")
plt.colorbar(label="Power (dB)")
plt.savefig("1_e.png")

#--------------------------------------------------------
# Question 2 --------------------------------------------
#--------------------------------------------------------

# 2 (c)  ------------------------------------------------
cutoffFrequency = 1e3; # Low pass cutoff frequency 1kHz
binWidth = 5; # 5 Hz bands
maxFreq = 200; # Plot signal power up to 200 Hz
numTrials = 200; # We have data from 200 trials
numSessions = 4; # We have data from this many sessions
fs = 1e4; # Sampling frequency 10kHz
stimOnset = 1000; # Stimulus comes on at 100 ms.
stimOffset = 1500; # Stimulus goes off at 150 ms.
wind = amy_hanning(256); # A 256 element hanning window will do
overl = 255; # Maximal overlap
nfft = np.arange(0, (maxFreq+binWidth), binWidth); # Create the frequency bins 0~maxFreq, binwidth 5

# 2 (d)  ------------------------------------------------
mat_file_name =  "mouseLFP.mat" # file name to call 
mat_data = scipy.io.loadmat(mat_file_name) # call matlab variable file (dictionary)
DATA = mat_data['DATA'] # data parsing from dictionary
dataSamples = len(DATA[0][0][1])

# 2 (e)  ------------------------------------------------
# frequency domain  -------------------------------------

ts = 1/fs 
lpf = amy_LPF(cutoff_freq = 1000, ts = ts) 
time = np.linspace(0, 0.3, dataSamples) 

T = len(time)/fs 
freq = np.arange(len(time)) 
freq = freq / T 
nyquist = int(dataSamples/2) 

filtered_data = np.zeros((numSessions, numTrials, dataSamples)) 

for ii in range(numSessions): 
    for jj in range(numTrials): 
        for kk in range(dataSamples):
            filtered_data[ii, jj, kk] = lpf.filter(DATA[ii][0][jj][kk])  

freq_cut = freq[0:nyquist] 
fig=plt.figure(figsize=(40,5)) 

#-------------------------
# Before LPF
data_fft = np.zeros((numSessions, dataSamples)) 
for ii in range(numSessions): 
    for jj in range(numTrials): 
        data_fft[ii] = data_fft[ii] + np.fft.fft(DATA[ii][0][jj])
    data_fft[ii] = data_fft[ii] / numTrials;
  
data_fft_dB = 10*np.log10((np.abs(data_fft))**2) #5
data_fft_dB_nyquist = np.zeros((numSessions, nyquist)) #6

for i in range(numSessions): 
    ax = plt.subplot(1,4,(i+1))  
    data_fft_dB_nyquist[i] = data_fft_dB[i][0:nyquist] 
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power of Signal (dB)")
    plt.title("Session {}".format(i))
    plt.grid(True)
    plt.plot(freq_cut, data_fft_dB_nyquist[i], label='Before LPF')  
    plt.legend()

#-------------------------
# After LPF
filtered_data_fft = np.zeros((numSessions, dataSamples)) 
for ii in range(numSessions): 
    for jj in range(numTrials): 
        filtered_data_fft[ii] = filtered_data_fft[ii] + np.fft.fft(filtered_data[ii][jj]) 
    filtered_data_fft[ii] = filtered_data_fft[ii] / numTrials;
  
filtered_data_fft_dB = 10*np.log10((np.abs(filtered_data_fft))**2) 
filtered_data_fft_dB_nyquist = np.zeros((numSessions, nyquist)) 

for i in range(numSessions):
    ax = plt.subplot(1,4,(i+1))  
    filtered_data_fft_dB_nyquist[i] = filtered_data_fft_dB[i][0:nyquist]
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power of Signal (dB)")
    plt.title("Session {}".format(i))
    plt.grid(True)
    plt.plot(freq_cut, filtered_data_fft_dB_nyquist[i], label='After LPF')    
    plt.legend()
plt.savefig("2_e_freq.png")

# time domain  ------------------------------------------
fig=plt.figure(figsize=(30,10)) #2
plt.subplots_adjust(hspace=0.5)

# Before LPF
data_avg = np.zeros((numSessions, dataSamples)) #3
for ii in range(numSessions): #4
    for jj in range(numTrials): 
        data_avg[ii] = data_avg[ii] + DATA[ii][0][jj];
    data_avg[ii] = data_avg[ii] / numTrials;

for i in range(numSessions): 
    ax = plt.subplot(2,4,(i+1))  
    plt.xlabel("time (s)")
    plt.ylabel("Membrane Potential (mV)")
    plt.title("Session {}".format(i))
    plt.grid(True)
    plt.plot(time, data_avg[i], label='Before LPF')  
    plt.legend(loc='lower left')    
#-------------------------
# After LPF (my lpf)
filtered_data_avg = np.zeros((numSessions, dataSamples)) #8
for ii in range(numSessions): #9
    for jj in range(numTrials): 
        filtered_data_avg[ii] = filtered_data_avg[ii] + filtered_data[ii][jj] 
    filtered_data_avg[ii] = filtered_data_avg[ii] / numTrials;

for i in range(numSessions):
    ax = plt.subplot(2,4,(i+1))  
    plt.xlabel("time (s)")
    plt.ylabel("Membrane Potential (mV)")
    plt.title("Session {}".format(i))
    plt.grid(True)
    plt.plot(time, filtered_data_avg[i], label='My LPF')    
    plt.legend(loc='lower left')   
plt.savefig("2_e_time.png")

# 2 (f)  ------------------------------------------------
ToneIndices = []
filteredTones = []
for ii in range(numSessions):
    low_tone_indices = np.where(DATA[ii, 4] == np.min(np.unique(DATA[ii, 4])))[0]  
    high_tone_indices = np.where(DATA[ii, 4] == np.max(np.unique(DATA[ii, 4])))[0]  

    ToneIndices.append([low_tone_indices, high_tone_indices]) 
    filteredTones.append([filtered_data[ii, low_tone_indices, :], filtered_data[ii, high_tone_indices, :]]) 


# 2 (g)  ------------------------------------------------
# Calculating means
meanLowTones = np.array([np.mean(filteredTones[ii][0], axis=0) for ii in range(numSessions)]) #1
meanHighTones = np.array([np.mean(filteredTones[ii][1], axis=0) for ii in range(numSessions)]) #2

ylim_min = 1.1*(np.min([meanLowTones, meanHighTones]))
ylim_max = 1.1*(np.max([meanLowTones, meanHighTones]))

for ii in range(numSessions): #4
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    plt.subplots_adjust(hspace=0.5)

    axes[0,0].plot(time, meanLowTones[ii, :], color='k')
    axes[0,0].set_title(f'Mean LFP from Mouse Auditory Cortex (Session {ii}) - Low Tones')
    axes[0,0].set_xlabel('Time (ms)')
    axes[0,0].set_ylabel('Membrane Potential (mV)')
    axes[0,0].set_ylim([ylim_min, ylim_max])
    
    axes[0,1].plot(time, meanHighTones[ii, :], color='k')
    axes[0,1].set_title(f'Mean LFP from Mouse Auditory Cortex (Session {ii}) - High Tones')
    axes[0,1].set_xlabel('Time (ms)')
    axes[0,1].set_ylabel('Membrane Potential (mV)')
    axes[0,1].set_ylim([ylim_min, ylim_max])
    
    norm_lim = Normalize(vmin=-88.8, vmax=6.2)
    over2=127
    
    f, t, Sxx = amy_spectrogram(meanLowTones[ii, :], fs, len(wind), overl, nfft=256)
    axes[1,0].pcolormesh(t, f, 10 * np.log10(Sxx), shading='auto', cmap='jet', norm=norm_lim)
    axes[1,0].set_title('Spectrogram of LFP data - Low Tones')
    axes[1,0].set_ylabel('Frequency (Hz)')
    axes[1,0].set_xlabel('Time (sec)')
    axes[1,0].set_ylim([0, 200])    
    
    f, t, Sxx = amy_spectrogram(meanHighTones[ii, :], fs, len(wind), overl, nfft=256)
    axes[1,1].pcolormesh(t, f, 10 * np.log10(Sxx), shading='auto', cmap='jet', norm=norm_lim)
    axes[1,1].set_title('Spectrogram of LFP data - High Tones')
    axes[1,1].set_ylabel('Frequency (Hz)')
    axes[1,1].set_xlabel('Time (sec)')
    axes[1,1].set_ylim([0, 200])   
    
    plt.savefig(f"2_g_session{ii}.png")

#--------------------------------------------------------

