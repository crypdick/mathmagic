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
            self.d = len(edges)/2
            # Create meshgrids
            if self.d == 2:
                self.X, self.Y = np.mgrid[edges[0]:edges[1],edges[2]:edges[3]]
            elif self.d == 3:
                self.X, self.Y, self.Z = np.mgrid[edges[0]:edges[1],
                    edges[2]:edges[3],edges[4]:edges[5]]
            