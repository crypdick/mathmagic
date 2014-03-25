"""
Created on Mar 24 2014

@author: rkp

Code for various fancy styles of plots
"""

import matplotlib.pylab as plt
import numpy as np

import mathmagic.rnd as rnd
from mpl_toolkits.mplot3d import Axes3D

def heat_scatter_3D(X,num_pts=500):
    """Create a 3D heat map using sparsely scattered points.
    
    heat_scatter_3D samples an element and its index from X with a probability
    proportional to the value of the element, and then plots that point using
    matplotlib's scatter function, colored according to its value.
    
    Args:
        X: 3D numpy array whose elements represent "temperatures".
    
        num_pts: how many points to plot
        
    Example:
        >>> x,y,z = np.mgrid[-30:30,-30:30,-30:30]
        >>> X = exp(-(x**2-y**2-z**2)/6)
    """
    
    # Create probability distribution by normalizing X
    P = X / np.sum(X)
    
    # Draw num_pts samples from vectorized P
    samples = rnd.catrnd(P.reshape((P.size,)),(num_pts,))
    
    # Make arrays for coordinates of samples and their heat values
    coords = np.empty((num_pts,3),dtype=int)
    heat_vals = np.empty((num_pts,),dtype=float)
    
    # Get shape of X (called xs for convenience)
    xs = X.shape
    
    # Fill in coordinate array
    for sample_idx,sample in enumerate(samples):
        # Get first coordinate
        x_coor = int(np.floor(float(sample)/(xs[0]*xs[1])))
        # Get second coordinate
        sample -= (x_coor*(xs[0]*xs[1]))
        y_coor = int(np.floor(float(sample)/xs[1]))
        # Get third coordinate
        sample -= (y_coor*xs[1])
        z_coor = int(sample)
        
        # Store coordinates and their values
        coords[sample_idx,:] = [x_coor,y_coor,z_coor]
        heat_vals[sample_idx] = X[x_coor,y_coor,z_coor]
        
    # Scatter plot everything
    ax = plt.gcf().add_subplot(111, projection='3d')
    ax.scatter(coords[:,0],coords[:,1],coords[:,2],c=heat_vals)
    
    # Set limits
    ax.set_xlim(0,xs[0])
    ax.set_ylim(0,xs[1])
    ax.set_zlim(0,xs[2])