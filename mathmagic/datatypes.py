"""
Created on Apr 11 2014

@author: rkp

This encodes several useful mathematical data types.
"""

import numpy as np

class Space():
    """Space data type.
    
    Contains meshgrid-style information about space.
    
    Args:
        edges: Specification of edges in each dimension. Syntax - 
            np.array([xmin,xmax,ymin,ymax,zmin,zmax]).
    """
    
    def __init__(self,edges=np.array([0,512,0,512])):
        """Space constructor."""
        
        # Do something if edges is None
        if edges is None:
            pass
        else:
            # Get dimensionality
            self.dim = len(edges)/2
            # Create meshgrids
            if self.dim == 2:
                self.X, self.Y = np.mgrid[edges[0]:edges[1],edges[2]:edges[3]]
            elif self.dim == 3:
                self.X, self.Y, self.Z = np.mgrid[edges[0]:edges[1],
                    edges[2]:edges[3],edges[4]:edges[5]]
            # Store format as mgrid
            self.format = 'mgrid'
            # Store shape
            self.shape = self.X.shape
            
class ParArray():
    """Parametric array data type.
    
    The parametric array allows one to store data in an array using either the 
    numpy array structure itself or a parametric representation. In the 
    latter case, a function must also be set that indicates how the parameters
    should be used to generate points in the array.
    
    Args:
        Y: Either numpy array or dictionary of parameters.
        
        D: Dimensionality of array.
        
        X: Support (necessary if Y is an array). List of 1-D numpy arrays.
        
        F: Function converting dictionary of parameters to numpy array over
        specified support.
        
    Example:
        >>> Y = {'a':3, 'b':5, 'c':-1}
        >>> def F(Y,x1,x2): Y['a']*x1 + Y['b']*x2 + Y['c']
        >>> q = ParArray(Y=Y,D=1,F=F)
        
    """
    
    def __init__(self,Y,D=2,X=None,F='array'):
        """ParArray constructor."""
        
        self.Y = Y
        self.F = F
        # If Y is not a parametric rep...
        if not isinstance(Y,dict):
            D = len(Y.shape)
            # Has support not been provided?
            if X is None:
                X = [np.arange(N) for N in Y.shape]
            self.X = X
        self.D = D
        
    def arr(self,X=None):
        """Return an array.
        
        Args:
            X: Support of array. List of 1-D numpy arrays (one per dimension).
            
        Returns:
            Array calculated over support, or stored array.
            
        """
        if X is None:
            return self.Y
        else:
            # Create meshgrids
            X_mesh = np.meshgrid(X)
            # Calculate array over support defined by X_mesh
            return self.F(self.Y,*X_mesh)
            