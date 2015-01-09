"""
Created on Mon Jan  5 09:39:31 2015

@author: rkp

Contains some basic signal processing functions not found in scipy.
"""

import numpy as np
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt

# TEST SIGNALS
T = np.arange(300)
X = (np.random.uniform(0, 1, 300) > .95).astype(float)
HT = np.arange(-20, 20)
H = np.exp(-HT/7.)
H[H > 1] = 0
Y = np.convolve(X, H, mode='same')

def fftxcorr(x, y, dt=1.):
    """Calculate the cross correlation between two signals using fft.
    
    Returns:
        time vector, cross correlation vector
    """
    
    # get length of signal
    sig_len = len(x)
    
    # build time vector & triangular normalization function
    if sig_len % 2:
        # If odd
        tri_norm_asc = np.arange(np.ceil(sig_len / 2.), sig_len)
        tri_norm_desc = np.arange(sig_len, np.floor(sig_len / 2.), -1)
        t = np.arange(-np.floor(sig_len / 2.), np.ceil(sig_len / 2.))
    else:
        # If even
        tri_norm_asc = np.arange(sig_len / 2., sig_len)
        tri_norm_desc = np.arange(sig_len, sig_len / 2., -1)
        t = np.arange(-np.floor(sig_len / 2.), np.ceil(sig_len / 2.))
    t *= dt
    tri_norm = np.concatenate([tri_norm_asc, tri_norm_desc])
    
    # subtract mean of signals & divide by std
    x_zero_mean = x - x.mean()
    x_clean = x_zero_mean / x_zero_mean.std()
    
    y_zero_mean = y - y.mean()
    y_clean = y_zero_mean / y_zero_mean.std()
    
    # calculate cross correlation
    xy = fftconvolve(x_clean, y_clean[::-1], mode='same')
    
    # normalize signal by triangle function
    xy /= tri_norm
    
    return t, xy
    
def test_fftxcorr():
    fig, axs = plt.subplots(3, 1)
    axs[0].plot(T, X)
    axs[1].plot(T, Y)
    
    TC, XY = fftxcorr(X, Y)
    axs[2].plot(TC, XY)