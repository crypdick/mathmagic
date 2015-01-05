"""
Created on Mon Jan  5 09:39:31 2015

@author: rkp

Contains some basic signal processing functions not found in scipy.
"""

import numpy as np
from scipy.signal import fftconvolve

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