'''
    File: framefield.py
    License: MIT
    Author: Aidan Kurtz
    Created: 06/08/2016
    Python Version: 3.5
    ========================
    This module contains all things 3D frame field.
'''

from visual import plot_framefield
from utils import tet_centroid

import math
import numpy as np
from scipy import spatial

class Frame(object):
    def __init__(self, u, v, w, location):
        self.u, self.v, self.w = u, v, w
        self.location = location

def init_framefield(tet_mesh, surf_mesh):
    boundary_frames = []
    # The frame field is initialized at the boundary,
    # based on the curvature cross-field and normals.
    for fi, surf_face in enumerate(surf_mesh.faces):
        # Retrieve the tet this face belongs to.
        ti = tet_mesh.adjacent_elements[surf_mesh.face_map.inv[fi]][0]
        tet = tet_mesh.elements[ti]
        # Ignore faces which have 0 curvature.
        if math.isclose(surf_mesh.k1[surf_face[0]], 0) and math.isclose(surf_mesh.k2[surf_face[0]], 0):
            continue
        # @TODO(aidan) Find actual face values, not vertex.
        boundary_frames.append(Frame( surf_mesh.pdir1[surf_face[0]],
                                      surf_mesh.pdir2[surf_face[0]],
                                      surf_mesh.vertex_normals[surf_face[0]],
                                      tet_centroid(tet_mesh, ti) ))

    # Prepare a KDTree of boundary frame coords for quick spatial queries.
    tree = spatial.KDTree(np.vstack([frame.location for frame in boundary_frames]))

    frames = []
    # Now propagate the boundary frames throughout the tet mesh.
    for ti, tet in enumerate(tet_mesh.elements):
        location = tet_centroid(tet_mesh, ti)
        nearest_ti = tree.query(location)[1] # Find closest boundary frame
        # Create new frame
        frames.append(Frame( boundary_frames[nearest_ti].u,
                             boundary_frames[nearest_ti].v,
                             boundary_frames[nearest_ti].w,
                             location ))
        
    plot_framefield(frames)

def singular_graph(tet_mesh, frames):
    pass

def optimize_framefield(tet_mesh, frames):
    pass