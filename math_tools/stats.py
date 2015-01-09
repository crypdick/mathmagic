import pdb
"""
Modified Apr 29 2014
Modified Jan 31 2014
Modified Jan 7 2014
Modified Dec 11 2013

Created Nov 29 2013

@author: R. K. Pang

This module provides a set of very useful statistical tools.
"""

# Import numpy
import numpy as np
from scipy import interpolate
import scipy.stats as stats

class dhist():
    """Dynamic histogram.
    
    Example:
    >>> x0 = [1, 7, 6, 8, 9]
    >>> x1 = [2, 3, 1, 7, 9, 8, 7, 7]
    >>> x2 = x0 + x1
    >>> bins = np.array([0,5,10])
    >>> cts, _ = np.histogram(x2, bins)
    >>> cts
    >>> dh = dhist(bins=bins)
    >>> dh.add(x0)
    >>> dh.add(x1)
    >>> dh.cts
    """
    
    def __init__(self, bins=10):
        self.bins = bins
        self.bin_centers = None
        self.count = 0
        self.mean = None
        
    def add(self, data):
        """Add data to histogram and recalculate distribution and mean."""
        
        # If first dataset
        if self.count == 0:
            
            # Compute mean
            self.mean = np.mean(data)
            # Compute counts & bins
            self.cts, self.bins = np.histogram(data, self.bins)
            # Get bin center & width
            self.bin_centers = .5 * (self.bins[:-1] + self.bins[1:])
            self.bin_width = self.bins[1] - self.bins[0]
            
        # If subsequent dataset
        else:
            
            # Update mean
            total_count = float(self.count + len(data))
            old_weight = self.count / total_count
            new_weight = len(data) / total_count
            self.mean = old_weight * self.mean + new_weight * np.mean(data)
            # Add new counts
            self.cts += np.histogram(data, self.bins)[0]
            
        # Update total count
        self.count += len(data)
        
        # Compute normalized counts
        self.normed_cts = self.cts / float(self.count)

def DKL(P,Q,symmetric=True):
    """Compute the Kullback-Liebler divergence between two probability
    distributions."""
    
    dx = 1./P.sum()
    
    # Check to make sure that Q_i = 0 ==> P_i = 0
    if np.any(P[Q==0.]):
        return np.nan
    if symmetric:
        if np.any(Q[P==0.]):
            return np.nan
    
    # Calculate log
    L1 = np.log(P/Q)
    # Set inf's to zero, because they will be zero anyhow
    L1[np.isinf(L1)] = 0.
    DKL1 = (L1*P).sum()
    
    if symmetric:
        L2 = np.log(Q/P)
        L2[np.isinf(L2)] = 0.
        DKL2 = (L2*Q).sum()
        DKL = (DKL1 + DKL2)/2.
    else:
        DKL = DKL1
        
    DKL *= dx
        
    return DKL