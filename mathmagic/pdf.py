#!/anaconda/bin/python
import pdb
"""
Modified Jan 31 2013
Modified Jan 7 2013
Modified Dec 11 2013

Created Nov 29 2013

@author: R. K. Pang

This module provides a set of very useful functions, including nonstandard 
random number generators and special plotting functions.
"""

# Most of these functions rely on mathematical tools in numpy
import scipy.io as io
import numpy as np

import mathmagic.fun as mmf

def logmvnpdf(x, mu, K, logdetK=None, opt1='standard'):
    """Calculate the log multivariate normal probability density at x.
    
    logmvnpdf calculates the natural logarithm of the probability density of 
    the samples contained in x.
    
    Args:
        x: Samples to calculate probability for. x can be given as a single 
        vector (1-D array), or as a matrix ((n x d) array). In the latter case
        if mu is a matrix ((n x d) array), and K is a (n x d x d) array, the 
        probability of the i-th row of x will be calculated under the multi-
        variate normal with its mean given by the i-th row of mu, and its 
        covariance given by the i-th plane of K.
        
        mu: Mean of multivariate normal distribution. mu can be given either as
        a single vector (1-D array) or as a matrix ((n x d) array). In the 
        latter case, mu must be the same shape as x.
        
        K: Covariance matrix of multivariate normal distribution. K can be 
        given either as a single matrix ((d x d) array) or as a set of matrices
        ((n x d x d) array). If mu is an (n x d) array, then K must be an
        (n x d x d) array.
        
        logdetK: Natural log of determinant(s) of K. Float (if only one K) or 
        (n x 1) float array (if several K)
        
        opt1: Method of interpreting K. If set to 'standard', K will be 
        interpreted as the standard covariance matrix. If set to 'inverse', K 
        will be interpreted as the inverse covariance matrix.
    
    Returns:
        logprob: the logarithm of the probability density of the samples under
        the given distribution(s). Length (n) float array.
        
    Example call:
        >>> # Calculate probability of one sample under one distribution
        >>> x = np.array([1.,2.])
        >>> mu = np.array([0.,0.])
        >>> K = np.array([[2.,1.],[1.,2.]])
        >>> logmvnpdf(x,mu,K)
        -3.3871832107433999
        
        >>> # Calculate probabiliy of three samples under one distribution
        >>> x = np.array([[1.,2.],[0,0],[-1,-2]])
        >>> mu = np.array([0.,0.])
        >>> K = np.array([[2,1],[1,2]])
        >>> logmvnpdf(x,mu,K)
        array([-3.38718321, -2.38718321, -3.38718321])
        
        >>> # Calculate probability of three samples with three different means
        >>> # and one covariance matrix
        >>> x = np.array([[1.,2.],[0,0],[-1,-2]])
        >>> mu = np.array([[0.,0.],[1,1],[2,2]])
        >>> K = np.array([[2,1],[1,2]])
        >>> logmvnpdf(x,mu,K)
        array([-3.38718321, -2.72051654, -6.72051654])
        
        >>> # Calculate probabiliy of three sample with one mean and three 
        >>> # different covariance matrices
        >>> x = np.array([[1.,2.],[0,0],[-1,-2]])
        >>> mu = np.array([0.,0.])
        >>> K = np.array([[[2,1],[1,2]],[[3,1],[1,3]],[[5,1],[1,5]]])
        >>> logmvnpdf(x,mu,K)
        array([-3.38718321, -2.87759784, -3.86440398])
        
        >>> # Calculate probability of three samples with three mean and three
        >>> # different covariance matrices
        >>> x = np.array([[1.,2.],[0,0],[-1,-2]])
        >>> mu = np.array([[0.,0.],[1,1],[2,2]])
        >>> K = np.array([[[2,1],[1,2]],[[3,1],[1,3]],[[5,1],[1,5]]])
        >>> logmvnpdf(x,mu,K)
        array([-3.38718321, -3.12759784, -5.53107065])
        """
    # Remove extraneous dimension from x and mu
    x = np.squeeze(x)
    mu = np.squeeze(mu)
    # If x has larger dimension than mu, tile mu to match the size of x
    if len(x.shape) > len(mu.shape):
        mu = np.tile(mu,(x.shape[0],1))
    # Create new variable z that is the difference between x and mu
    z = x - mu
    # Make sure there are as many samples as covariance matrices and figure out
    # how many total calculations we'll need to do
    if len(K.shape) == 3: # Multiple K
        if len(z.shape) == 1: # Single z
            z = np.tile(z,(K.shape[0],1))
        num_calcs = K.shape[0]
    else: # Single K
        if len(z.shape) == 2: # Multiple z
            K = np.tile(K,(z.shape[0],1,1))
            num_calcs = K.shape[0]
        else: # Single z
            num_calcs = 1
    # Calculate inverses and log-determinants if necessary
    if not opt1.lower() == 'inverse':
        # Have multiple covariance matrices been supplied?
        if len(K.shape) == 3:
            # Calculate inverses
            Kinv = mmf.mwfun(np.linalg.inv,K)
            # Calculate log determinants
            if logdetK is None:
                logdetK = np.log(mmf.mwfun(np.linalg.det,K))
        else:
            # Calculate inverse
            Kinv = np.linalg.inv(K)
            # Calculate log determinant
            if logdetK is None:
                logdetK = np.log(np.linalg.det(K))
    else:
        Kinv = K
        # Have log-determinants been provided?
        if logdetK is None:
            # Multiple covariance matrices?
            if len(K.shape) == 3:
                K = mwfun(np.linalg.inv,Kinv)
                detK = mmf.mwfun(np.linalg.det,K)
            else:
                K = np.linalg.inv(K)
                detK = np.det(K)
            logdetK = np.log(detK)

    # Calculate matrix product of z*Kinv*z.T for each Kinv and store it in y.
    if num_calcs == 1:
        # Remove extraneous dimension from z
        z = np.squeeze(z)
        temp1 = np.dot(z,Kinv)
        mat_prod = np.dot(temp1,z)
    else:
        temp1 = mmf.mwfun(np.dot,z,Kinv)
        mat_prod = mmf.mwfun(np.dot,temp1,z)

    # Get dimension of system
    if len(z.shape) > 1:
        dim = z.shape[1]
    else:
        dim = z.shape[0]
    
    # Calculate final log probability
    logprob = -.5*(dim*np.log(2*np.pi) + logdetK + mat_prod)
    
    # Remove extraneous dimension
    return np.squeeze(logprob)