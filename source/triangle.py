"""
A short and sweet matrix library.

"""

import math
import random
import numpy as np
from numpy import norm
from matrix import Matrix
from numpy import vstack

__version__ = "0.1"


def is_facing(triangles, normals, reference):
    '''
    When deciding if a polygon is facing the camera, you need
    only calculate the dot product of the normal vector of
    that polygon, with a vector from the reference point to one
    of the polygon's vertices.

    triangles: vertices of triangles, (n,3,3)
    returns:   boolean, (n,1)
    '''
    return [not (np.dot(normals[i], reference - t[0])) < 0.0 for i, t in enumerate(triangles)]


def cnormal(face):
    '''
    # Compute normal

    face:
    return:
    '''
    return np.cross(np.array(tet_mesh.points[face[0]]) - np.array(tet_mesh.points[face[1]]),
                    np.array(tet_mesh.points[face[2]]) - np.array(tet_mesh.points[face[1]]))

def

def compute_avg(face):
    return (np.array(tet_mesh.points[face[0]])
          + np.array(tet_mesh.points[face[1]])
          + np.array(tet_mesh.points[face[2]])) / 3

face_normals = [compute_normal(face) for face in surface_faces]
face_center = [compute_avg(face) for face in surface_faces]

# Correct normal
center = vstack(tet_mesh.points).mean()
triangles = [np.array([tet_mesh.points[f[0]], tet_mesh.points[f[1]], tet_mesh.points[f[2]]]) for f in surface_faces]

facing = is_facing(triangles, face_normals, center)
for i, verdict in enumerate(facing):
    if verdict:
        face_normals[i] = -face_normals[i]
