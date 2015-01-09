import pdb
"""
Created on Tue Sep  9 22:21:40 2014

@author: rkp

Code for GLM, ARMA models
"""

import numpy as np
from numpy import concatenate as cc
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
import statsmodels.api as sm

from math_tools.matrix import sliding_window as swm

FILT_COLORS = ['r','g']

class Arma():
    """Purely causal ARMA model
    
    Args:
        sig_filter_lengths: list of signal filter lengths (1 per input channel)
        fdbk_filter_length: feedback filter length
        noise: noise level (used when generating output signals)
    """
    
    def __init__(self, sig_filter_lens=None, fdbk_filter_len=10, noise=.1):
        """Constructor."""
        
        if sig_filter_lens is None:
            sig_filter_lens = [10]
        
        # store filter lengths
        self.sig_filter_lens = sig_filter_lens
        self.fdbk_filter_len = fdbk_filter_len
        
        # store maximum filter length (for zeropadding later)
        max_sig_filter_len = np.max(sig_filter_lens)
        self.max_filter_len = np.max([max_sig_filter_len, fdbk_filter_len])
        
        # store signal dimension
        self.sig_dim = len(sig_filter_lens)
        
        # store noise
        self.noise = noise
        
        # store zero-valued filters
        self.sig_filters = [np.zeros((L,), dtype=float) for L in sig_filter_lens]
        self.fdbk_filter = np.zeros((fdbk_filter_len,), dtype=float)
        
        # store constant
        self.constant = 0.
        
    def gen(self, xs):
        """Generate an output signal from an input signal.
        
        Args:
            xs: list of input signals.
        """
        
        # get signal length
        sig_len = len(xs[0])
        
        # zeropad x's to make convolution easier
        zp = np.zeros((self.max_filter_len,), dtype=float)
        xzps = [None for x in xs]
        for xctr, x in enumerate(xs):
            xzps[xctr] = cc([zp, x])
        
        # generate zeropadded y
        yzp = np.zeros(xzps[0].shape, dtype=float)
        
        # fill in y (t refers to yzp's idx)
        for t in range(self.max_filter_len, self.max_filter_len + sig_len):
            # sequentially add components of y for each signal
            result = 0
            for sctr in range(self.sig_dim):
                # get signal filter, its length, and relevant signal
                sf = self.sig_filters[sctr]
                sf_len = self.sig_filter_lens[sctr]
                sig = xzps[sctr][t-sf_len:t]
                
                # dot filter with signal (remembering to flip filter)
                result += np.dot(sf[::-1], sig)
            
            # add output feedback component
            ## get relevant output
            output = yzp[t-self.fdbk_filter_len:t]
            ## dot filter with output (remembering to flip filter)
            result += np.dot(self.fdbk_filter[::-1], output)
            
            # add constant
            result += self.constant
            
            # add Gaussian noise
            result += np.random.normal(0, self.noise)
            
            # store output in correct position
            yzp[t] = result
        
        # remove initial zeros from yzp and return
        return yzp[self.max_filter_len:]
        
    def fit(self, xs, y):
        """Fit signal & feedback filters from an input & output timeseries.
        
        Args:
            xs: list of input signals.
        """
        
        # convert xs into fittable data array
        ## the format of the data matrix that will be fed into the statsmodels
        ## api is D = [1, x1, x2, x3, ..., y], where rows correspond to 
        ## timepoints, 1 is a column vector of ones, x1 is a sliding window matrix
        ## for the first signal, etc., and y is a toeplitz matrix for the 
        ## output
        
        ## generate sliding window matrices for each x
        xswms = [None] * self.sig_dim
        
        for sctr, x in enumerate(xs):
            sf_len = self.sig_filter_lens[sctr]
            start = self.max_filter_len - sf_len
            xswms[sctr] = swm(x, sf_len, start=start, end=-1)
            
        ## generate sliding window matrix for y
        start = self.max_filter_len - self.fdbk_filter_len
        yswm = swm(y, self.fdbk_filter_len, start=start, end=-1)
        
        ## generate complete input data array
        data_in = cc(xswms + [yswm], axis=1)
        
        # add constant to data
        data_in = sm.add_constant(data_in)
        
        # count rows
        nrows = data_in.shape[0]
        
        # fit data
        model = sm.GLM(y[-nrows:], data_in, family=sm.families.Gaussian())
        
        params = model.fit().params
        
        # store parameters
        self.constant = params[0]
        
        ctr = 1
        for sctr in range(self.sig_dim):
            sf_len = self.sig_filter_lens[sctr]
            self.sig_filters[sctr] = params[ctr:ctr+sf_len][::-1]
            ctr += sf_len
        
        self.fdbk_filter = params[ctr:][::-1]