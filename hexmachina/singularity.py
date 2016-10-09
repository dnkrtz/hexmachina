'''
    File: singularity.py
    License: MIT
    Author: Aidan Kurtz
    Created: 09/09/2016
    Python Version: 3.5
    ========================
    This module contains functions related to the volumetric
    singularities found in the tetrahedral mesh frame field.
'''

import numpy as np

from transforms import *
from utils import *


def compute_matchings(machina):
    """Compute the matchings for all pairs of face-adjacent tets.
    We use matchings to characterize closeness between two frames s and t.
    This matching is the chiral permutation most closely defines the rotation."""
    # Loop through all pairs of face-adjacent tets.
    for fi, pair in enumerate(machina.tet_mesh.adjacent_elements):
        args = []
        if -1 in pair:
            continue # Boundary face
        # Find the best permutation to characterize closeness.
        for permutation in chiral_symmetries:
            arg = machina.frames[pair[0]].uvw - machina.frames[pair[1]].uvw * permutation.T
            args.append(np.linalg.norm(arg))
        # Store the matching, as an index into chiral symmetry group.
        machina.matchings[fi] = np.argmin(args)

def compute_edge_types(machina, edge_index):
    """Classify the internal edges by type, and find the singular graph.
    The edge type is determined by concatenating the matchings around the edges one-ring."""
    # For each internal edge of the tetrahedral mesh.
    for ei in edge_index:
        try:
            one_ring = machina.one_rings[ei]
        except KeyError:
            continue # Not an internal edge.
        
        # Concatenate the matchings around the edge to find its type.
        edge_type = np.identity(3)
        for fi in one_ring['faces']:
            matching = []
            # Recall that in the one-ring, if 'fi' is negative, it is
            # a 't-s' pair, as opposed to a 's-t' pair.
            # If pair order is reversed, invert/transpose rotation matrix.
            # Use copysign to distinguish +0 from -0.
            if np.copysign(1, fi) > 0:
                matching = chiral_symmetries[machina.matchings[fi]]
            else:
                matching = chiral_symmetries[machina.matchings[-fi]].T
            # Concatenate transforms
            edge_type = np.dot(edge_type, matching)
        
        # Locate singular (not identity) and improper (not restricted) edges.
        is_singular, is_improper = True, True
        for si, restricted_type in enumerate(chiral_symmetries[0:9]):
            if np.allclose(edge_type, restricted_type):
                if si == 0 : is_singular = False
                is_improper = False
                break

        # Classify as proper(0), singular(1), improper (2)
        if is_singular: machina.edge_types[ei] = 1
        if is_improper: machina.edge_types[ei] = 2
        
def singular_graph(machina):
    """Computes singular graph, that is, all non-identity (singular) edges.
    Returns the singular edges, improper edges and singular vertices."""
    # Compute matchings.
    compute_matchings(machina)
    compute_edge_types(machina, range(len(machina.tet_mesh.edges)))
   
    # Store them in lists for output.
    singular_edges = []
    improper_edges = []
    singular_vertices = dict()
    # Store edges and vertices
    for ei, edge in enumerate(machina.tet_mesh.edges):     
        if machina.edge_types[ei] > 0:
            singular_edges.append(ei)
            # Store the vertices with associated singularity type.
            singular_vertices[edge[0]] = machina.edge_types[ei]
            singular_vertices[edge[1]] = machina.edge_types[ei]
        if machina.edge_types[ei] == 2:
            improper_edges.append(ei)
    # Output each graph as a .vtk.
    vtk_lines(machina.tet_mesh.points,
             [machina.tet_mesh.edges[ei] for ei in singular_edges],
             'singular')
    vtk_lines(machina.tet_mesh.points,
             [machina.tet_mesh.edges[ei] for ei in improper_edges],
             'improper')

    return singular_edges, improper_edges, singular_vertices