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