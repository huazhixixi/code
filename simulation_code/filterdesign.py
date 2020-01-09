from scipy.fft import fftfreq
from scipy.fft import fft,ifft

import numpy as np

def bandpass_filter(signal_samples,fs,fs_high,fs_low):
    signal_samples = np.atleast_2d(signal_samples)
    freq_vector = fftfreq(signal_samples.shape[1],1/fs)
    mask = np.logical_and(freq_vector < fs_high,freq_vector>=fs_low)
    fft_samples = fft(signal_samples,axis=1)
    fft_samples[:,mask] = 0
    return ifft(fft_samples,axis=1)
