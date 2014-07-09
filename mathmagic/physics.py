import pdb
"""
Created Jul 07 2013

@author: R. K. Pang

This module provides a set of physics-related functions.
"""

# Import numpy
import numpy as np

def ang_vel(vel,dt=1.):
    """Calculate angular velocity time-series.
    
    Args:
        vel: 3-D velocity time-series. Rows are time points.
    
    Returns:
        Angular velocity time-series.
    """
    dt = float(dt)
    # Calculate normalized velocity vector
    v = vel/((vel**2).sum(1)**.5)[:,None]
    # Calculate angle between each consecutive pair of normalized velocity 
    # vectors
    dtheta = np.arccos((v[:-1,:]*v[1:,:]).sum(1))
    # Calculate magnitude of angular velocity
    a_vel_mag = dtheta/dt
    # Calculate the direction of angular change by computing the cross-
    # product between each consecutive pair of normalized velocity vectors
    cr = np.cross(v[:-1,:],v[1:,:])
    # Normalize the cross product array
    cr /= ((cr**2).sum(1)**.5)[:,None]
    # Create the angular velocity array
    a_vel = cr * a_vel_mag[:,None]
    # Correct the size so that it matches the size of the velocity array
    a_vel_full = np.zeros((a_vel.shape[0]+1,a_vel.shape[1]),dtype=float)
    a_vel_full[:-1] += a_vel
    a_vel_full[1:] += a_vel
    a_vel_full[1:-1] /= 2.
    
    return a_vel_full