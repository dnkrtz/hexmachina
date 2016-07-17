"""
A short and sweet matrix library.

"""

import math
import random
import numpy as np
from numpy import norm
from matrix import Matrix

__version__ = "0.1"


def extract_surface(tet_mesh):
    """

    tet_mesh: tetrahedron mesh from tetgen build function
    returns:  list, (n,1)
    """
    surface_faces = []
    for i, tet in enumerate(tet_mesh.elements):
        neighbors = list(tet_mesh.neighbors[i])
        # Try to get the vertex indices of boundary face.
        # @TODO(aidan) This only looks at the first boundary face, consider case where a tet has multiple...
        if -1 in neighbors:
            non_surface_vtx = tet[neighbors.index(-1)]
            tet_cpy = tet.copy()
            tet_cpy.remove(non_surface_vtx)
            surface_faces.append(tet_cpy)
    return surface_faces
