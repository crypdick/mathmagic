#!/anaconda/bin/python

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
import pdb

def mwfun(func, *args):
    """Performs a function operation matrix-wise on numpy ndarray objects
    
    mwfun performs the function func on each pair of matrices in the 3D numpy
    arrays provided as arguments. All outputs must have same dimensionality, 
    e.g., they must all be c x d matrices, all be scalars, etc. 
    
    The output vals[ii] is equal to func(args[0][ii],args[1][ii],...). 
    See examples.
    
    Args:
        func: Function object.
        
        *args: Arguments to be passed matrix-wise to func. (n x m x p) numpy
        arrays, where the function operates on matrices args[ii][jj].
        
    Returns:
        vals: output of function. Length (n) 1D array if scalar ouput, (n x c)
        array if vector output. (n x c x d) if matrix output, etc.

    Examples:
        >>> K = np.array([[[3,1],[1,3]],[[5,1],[1,5]]])
        >>> # Calculate determinants
        >>> mwfun(np.linalg.det,K)
        array([8., 24.])
        >>> # Calculate inverses
        >>> mwfun(np.linalg.inv,K)
        array([[[ 0.375     , -0.125     ],
                [-0.125     ,  0.375     ]],

               [[ 0.20833333, -0.04166667],
                [-0.04166667,  0.20833333]]])
        >>> # Calculate matrix products
        >>> x = np.array([[1,1],[2,2]])
        >>> mwfun(np.dot,K,x)
        array([[  4.,   4.],
               [ 12.,  12.]])
    
    """
    # Get number of arguments
    num_args = len(args)
    # Make list of first matrices
    first_mat_list = [args[ii][0] for ii in range(num_args)]
    # Calculate function value of first matrices
    first_val = func(*first_mat_list)
    # Get number of matrices in each np.ndarray argument
    num_mats = args[0].shape[0]    
    
    # Allocate space for all function values
    if not isinstance(first_val,np.ndarray):
        vals = np.empty((num_mats,),dtype=float)
    elif len(first_val.shape) == 1:
        vals = np.empty((num_mats,first_val.shape[0]),dtype=float)
    else:
        vals = np.empty((num_mats,first_val.shape[0],first_val.shape[1]),
                        dtype=float)
    
    # Store first_val in vals
    vals[0] = first_val
    # Run through and calculate function values
    for jj in range(1,num_mats):
        # Get list of ii-th matrices
        jj_mat_list = [args[ii][jj] for ii in range(num_args)]
        # Calculate function value
        vals[jj] = func(*jj_mat_list)
        
    return vals

def statdist(tr_mat):
    """Calculates the stationary distribution of a transition matrix.
    
    statdist returns the stationary distribution of a (right stochastic) 
    transition matrix. This is the left eigenvector of the transition matrix 
    corresponding to the eigenvalue 1
    
    Args:
        tr_mat: transition matrix; right stochastic matrix, i.e., each row 
        should sum to 1; square 2D array
    
    Returns:
        statdistvec: stationary distribution vector; 1D array which sums to 1
    
    Example:
        >>> tr_mat = np.array([[.3, .4, .3],[.2, .7, .1],[.1, .1, .8]])
        >>> statdist(tr_mat)
        array([])
    """
    # Calculate left eigenvectors and eigenvalues of transition matrix
    evs,evecs = np.linalg.eig(tr_mat.transpose())
    # Get eigenvector with eigenvalue 1
    statdistvec = evecs[:,np.argmin(np.abs(evs-1))]
    # Normalize stationary distribution and remove any spurious imaginary part
    statdistvec = np.real(statdistvec/np.sum(statdistvec))
    return statdistvec
    
def logsum(logx):
    """Efficiently calculates the logarithm of a sum from the logarithm of its
    terms.
    
    logsum employes an efficient algorithm for finding the logarithm of a sum
    of numbers when provided with the logarithm of the terms in the sum. This 
    is especially useful when the logarithms are exceptionally large or small
    and would cause numerical errors if exponentiated.
    
    Args:
        logx: Logarithms of terms to sum. n-length float array.
    
    Returns:
        logS: Logarithm of the sum. int.
    
    Example:
        >>> logx = np.array([-1000,-1001,-1002])
        >>> np.log(np.sum(np.exp(logx)))
        -inf
        >>> logsum(logx)
        -999.59239403555557
        >>> logx = np.array([1000,1001,1002])
        >>> np.log(np.sum(np.exp(logx)))
        inf
        >>> logsum(logx)
        1002.4076059644444
    """   
    # Sort elements of logx from highest to lowest
    logx = np.sort(logx)[::-1]
    # Get largest element
    maxlogx = logx[0]
    # Normalize logx by subtracting maxlogx
    logx -= maxlogx
    # Calculate sum of logarithms
    logS = maxlogx + np.log(np.sum(np.exp(logx)))
    return logS
        
def detrend(x, mean_abs_norm=1.):
    """ Subtract mean and normalize absolute value of list of arrays.
    
    Subtract the overall mean (dimension-wise) from a 2-D array or list of 
    2-D arrays. Means are calculated along the first (0) dimension. Normalize
    mean absolute value to mean_abs_val along each dimension.
    
    Args:
        x: 2-D array or list of 2-D arrays.
        
        mean_abs_norm: Positive value to normalize mean absolute value to. 
        Positive float. Can also be list/array of floats if each dimension is
        to have a different normalization constant.
        
    Returns:
        y: Detrended array/list of arrays.
        
    Example:
        >>> x = np.array([[2.,4.],[0.,8.],[4.,0.]])
        >>> detrend(x)
        array([[ 0. ,  0. ],
               [-1.5,  1.5],
               [ 1.5, -1.5]])
        >>> y = np.array([[4.,0.],[0.,8.],[2.,4.]])
        >>> detrend([x,y])
        
    """
    
    # Is x a list?
    if not isinstance(x,list):
        # Is mean_abs_norm a list?
        if isinstance(mean_abs_norm, int) or isinstance(mean_abs_norm, float):
            mean_abs_norm = [mean_abs_norm for ii in range(x.shape[1])]
        # Make sure x is a float
        x = x.astype(float)
        # Calculate mean
        mean_x = np.mean(x,0)
        # Subtract mean from x
        x -= mean_x
        # Calculate mean absolute value
        mean_abs_x = np.mean(np.abs(x),0)
        # Normalize each column of x
        for jj in range(x.shape[1]):
            x[:,jj] /= mean_abs_x[jj]
            x[:,jj] *= mean_abs_norm[jj]
    
    else:
        # Is mean_abs_norm a list?
        if isinstance(mean_abs_norm, int) or isinstance(mean_abs_norm, float):
            mean_abs_norm = [mean_abs_norm for ii in range(x[0].shape[1])]
        # Make sure elements of x are all float
        x = [x[ii].astype(float) for ii in range(len(x))]
        # Calculate mean
        mean_x = np.mean(np.concatenate(x),0)
        # Subtract mean from x
        x = [x[ii] - mean_x for ii in range(len(x))]
        # Calculate mean absolute value
        mean_abs_x = np.mean(np.abs(np.concatenate(x)),0)
        # Normalize each column of x
        for jj in range(x[0].shape[1]):
            for ii in range(len(x)):
                x[ii][:,jj] /= mean_abs_x[jj]
                x[ii][:,jj] *= mean_abs_norm[jj]
                
    return x

def nans(shape):
    """ Create an array of nans.
    
    Useful for allocating space for a matrix.
    
    Args:
        shape: Shape of array to create. Tuple.
        
    Returns:
        x: nan array
        
    Example:
        >>> x = nans((3,3))
        >>> x
        array([[ nan,  nan,  nan],
               [ nan,  nan,  nan],
               [ nan,  nan,  nan]])
    """
    x = np.empty(shape,dtype=float)
    x[:] = np.nan
    return x
    
def symtri(x,center=0,height=1,slope=1):
    """ Symmetric triangle function.
    
    Returns the value of a symmetric trianglel function evaluated at x. This is
    zero if x is less than the intersection of the left arm of the triangle
    with the x-axis or if x is greater than the intersection fo the right arm
    of the triangle with the x-axis. Otherwise it returns the linear function
    of x corresponding to the arm in which x is located.
    
    """
    # Calculate x-intercepts
    x_left = center - height/slope
    x_right = center + height/slope
    
    if x > x_right or x < x_left:
        return 0
    elif x >= x_left and x < center:
        return (x - x_left)*slope
    elif x <= x_right and x >= center:
        return (x_right - x)*slope
        
    