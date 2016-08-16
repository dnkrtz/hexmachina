'''
    File: utils.py
    License: MIT
    Author: Aidan Kurtz
    Created: 09/07/2016
    Python Version: 3.5
    ========================
    This module contains some small functions.
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