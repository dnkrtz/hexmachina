'''
    File: utils.py
    License: MIT
    Author: Aidan Kurtz
    Created: 09/07/2016
    Python Version: 3.5
    ========================
    This module contains some useful functions and hardcodes.
'''

import numpy as np

# Normalize a vector to unit length.
def normalize(vector):
    return vector / np.linalg.norm(vector)

def tet_centroid(tet_mesh, ti):
    return ( np.array(tet_mesh.points[tet_mesh.elements[ti][0]]) + 
             np.array(tet_mesh.points[tet_mesh.elements[ti][1]]) +
             np.array(tet_mesh.points[tet_mesh.elements[ti][2]]) + 
             np.array(tet_mesh.points[tet_mesh.elements[ti][3]]) ) / 4

# The cubical chiral symmetry group of permutations.
rotational_symmetries = [
    # Identity
    np.matrix([[ 1,  0,  0], [ 0,  1,  0], [ 0,  0,  1]]),
    # 90 degree 4-fold rotations
    np.matrix([[ 0, -1,  0], [ 1,  0,  0], [ 0,  0,  1]]),
    np.matrix([[ 1,  0,  0], [ 0,  0, -1], [ 0,  1,  0]]),
    np.matrix([[ 0,  0,  1], [ 0,  1,  0], [-1,  0,  0]]),
    np.matrix([[ 0,  1,  0], [-1,  0,  0], [ 0,  0,  1]]),
    np.matrix([[ 1,  0,  0], [ 0,  0,  1], [ 0, -1,  0]]),
    np.matrix([[ 0,  0, -1], [ 0,  1,  0], [ 1,  0,  0]]),
    # 180 degree 4-fold rotations
    np.matrix([[-1,  0,  0], [ 0, -1,  0], [ 0,  0,  1]]),
    np.matrix([[ 1,  0,  0], [ 0, -1,  0], [ 0,  0, -1]]),
    np.matrix([[-1,  0,  0], [ 0,  1,  0], [ 0,  0, -1]]),
    # 120 degree 3-fold rotations
    np.matrix([[ 0,  1,  0], [ 0,  0, -1], [-1,  0,  0]]),
    np.matrix([[ 0,  0, -1], [ 1,  0,  0], [ 0, -1,  0]]),
    np.matrix([[ 0,  0,  1], [-1,  0,  0], [ 0, -1,  0]]),
    np.matrix([[ 0, -1,  0], [ 0,  0, -1], [ 1,  0,  0]]),
    np.matrix([[ 0,  0,  1], [ 1,  0,  0], [ 0,  1,  0]]),
    np.matrix([[ 0,  1,  0], [ 0,  0,  1], [ 1,  0,  0]]),
    np.matrix([[ 0, -1,  0], [ 0,  0,  1], [-1,  0,  0]]),
    np.matrix([[ 0,  0, -1], [-1,  0,  0], [ 0,  1,  0]]),
    # 180 degree 2-fold rotations
    np.matrix([[ 0,  1,  0], [ 1,  0,  0], [ 0,  0, -1]]),
    np.matrix([[ 0, -1,  0], [-1,  0,  0], [ 0,  0, -1]]),
    np.matrix([[-1,  0,  0], [ 0,  0, -1], [ 0, -1,  0]]),
    np.matrix([[-1,  0,  0], [ 0,  0,  1], [ 0,  1,  0]]),
    np.matrix([[ 0,  0,  1], [ 0, -1,  0], [ 1,  0,  0]]),
    np.matrix([[ 0,  0, -1], [ 0, -1,  0], [-1,  0,  0]]), #double check this
]