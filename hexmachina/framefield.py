'''
    File: framefield.py
    License: MIT
    Author: Aidan Kurtz
    Created: 06/08/2016
    Python Version: 3.5
    ========================
    This module contains all things 3D frame field.
'''

import math
import numpy as np
from scipy import spatial
import itertools

from visual import *
from utils import *
from transforms import *
from tetmesh import *

def init_framefield(tet_mesh, surf_mesh):
    boundary_frames = []
    boundary_ids = {}
    # The frame field is initialized at the boundary,
    # based on the curvature cross-field and normals.
    for fi, face in enumerate(surf_mesh.faces):
        # Retrieve the tet this face belongs to.
        ti = tet_mesh.mesh.adjacent_elements[surf_mesh.face_map.inv[fi]][0]
        tet = tet_mesh.mesh.elements[ti]
        # Ignore faces which have 0 curvature.
        if math.isclose(surf_mesh.k1[face[0]], 0) and math.isclose(surf_mesh.k2[face[0]], 0):
            continue
        # @TODO(aidan) Find actual face values, not vertex values.
        uvw = np.hstack((np.vstack(surf_mesh.pdir1[face[0]]),
                         np.vstack(surf_mesh.pdir2[face[0]]),
                         np.vstack(surf_mesh.vertex_normals[face[0]])))
        boundary_frames.append(Frame(uvw, tet_centroid(tet_mesh.mesh, ti)))
        boundary_ids[ti] = len(boundary_frames) - 1

    # Prepare a KDTree of boundary frame coords for quick spatial queries.
    tree = spatial.KDTree(np.vstack([frame.location for frame in boundary_frames]))

    # Now propagate the boundary frames throughout the tet mesh.
    # Each tet frame takes the value of its nearest boundary tet.
    for ti, tet in enumerate(tet_mesh.mesh.elements):
        location = tet_centroid(tet_mesh.mesh, ti)
        if ti in boundary_ids:
            tet_mesh.frames.append(Frame(boundary_frames[boundary_ids[ti]].uvw, location, True))
        else:
            nearest_ti = tree.query(location)[1] # Find closest boundary frame
            tet_mesh.frames.append(Frame(boundary_frames[nearest_ti].uvw, location, False))

def compute_onerings(tet_mesh, surf_mesh):
    # Compute the one ring of tets surrounding each internal edge.
    for ei, edge in enumerate(tet_mesh.mesh.edges):
        # Make sure this is an internal edge, skip if it isn't.
        if (edge[0] in surf_mesh.vertex_map and edge[1] in surf_mesh.vertex_map):
            continue
        # If it is, construct its one ring.
        one_ring = []
        finished = False
        one_ring.append(tet_mesh.mesh.edge_adjacent_elements[ei])
        # Walk around the edge until we've closed the one ring.
        while not finished:
            finished = True
            for neigh_ti in tet_mesh.mesh.neighbors[one_ring[-1]]:
                neighbor = tet_mesh.mesh.elements[neigh_ti]
                # Make sure this neighbor is a viable pick.
                if (neigh_ti == -1 or neigh_ti in one_ring):
                    continue
                # Make sure this neighbor shares the edge.
                if (edge[0] in neighbor and edge[1] in neighbor ):
                    # Add it to the ring.
                    one_ring.append(neigh_ti)
                    finished = False
                    break
        # Store it in our ring dictionary (don't tell golem).
        tet_mesh.one_rings[ei] = one_ring

def singular_graph(tet_mesh):
    # Compute the matchings for all pairs of face-adjacent tets.
    matchings = {}
    for pair in tet_mesh.mesh.adjacent_elements:
        args = []
        # If boundary face, skip.
        if -1 in pair:
            continue
        # Find the best permutation to characterize closeness.
        for permutation in chiral_symmetries:
            arg = tet_mesh.frames[pair[0]].uvw - np.dot(tet_mesh.frames[pair[1]].uvw, permutation.T)
            args.append(np.linalg.norm(arg))
        # Store the matching
        matchings[tuple(pair)] = chiral_symmetries[np.argmin(args)]

    # Classify the internal edges by type, and find the singular graph.
    # The edge type is determined via concatenation of the matchings around the edge's one-ring.
    lines = []
    for ei, edge in enumerate(tet_mesh.mesh.edges):
        try:
            one_ring = tet_mesh.one_rings[ei]
        except KeyError:
            continue
        # Concatenate the matchings around the edge to find its type.
        type = np.identity(3)
        for i in range(len(one_ring)):
            matching = []
            pair = (one_ring[i], one_ring[(i + 1) % len(one_ring)])
            # If pair order is reversed, invert permutation matrix
            if pair not in matchings:
                pair = pair[::-1] # reverse
                matching = np.linalg.inv(matchings[pair])
            else:
                matching = matchings[pair]
            # Concatenate transforms     
            type = np.dot(type, matchings[pair])

        # Singular edge.
        if not np.array_equal(type, np.identity(3)):
            lines.append(edge)

    # Plot singular edges.    
    plot_lines(lines, tet_mesh.mesh.points)


def compute_closeness(frames, s, t):
    pass

# Quantify closeness of the matching to the chiral symmetry group.
def rotation_energy(tet_mesh, s, t):
    # Approximate permutation for the matching.
    P = tet_mesh.frames[t].uvw.T * tet_mesh.frames[s].uvw
    # Since our initialized framefield is orthogonal, we can easily quantify
    # closeness of the permutation to the chiral symmetry group G. The cost
    # function should drive each row/column to have a single non-zero value.
    E = 0
    for i in range(3):
        E += P[i,0]**2 * P[i,1]**2 + P[i,1]**2 * P[i,2]**2 + P[i,2]**2 * P[i,0]**2
        E += P[0,i]**2 * P[1,i]**2 + P[1,i]**2 * P[2,i]**2 + P[2,i]**2 * P[0,i]**2
    return E

# Function E to minimize via L-BFGS.
def global_energy(tet_mesh, euler_angles):
    E = 0

    for frame in tet_mesh.frames:
        pass

    # All internal edges.
    for ei, edge in enumerate(tet_mesh.mesh.edges):
        if ei not in tet_mesh.one_rings:
            continue
        # All combinations of s, t around the edges' one ring.
        for combo in itertools.combinations(tet_mesh.one_rings[ei], 2):
            pass

# Optimize the framefield.
def optimize_framefield(tet_mesh):

    # Define all frames in terms of euler angles.
    euler_angles = [ np.zeros(3) for _ in range(len(tet_mesh.mesh.elements)) ]
    
    for ti, tet in enumerate(tet_mesh.mesh.elements):
        # If boundary tet
        if -1 in tet:
            pass
        else:
            R = tet_mesh.frames[ti].uvw
            tet_mesh.frames[ti].euler = convert_to_euler(R)



            


    