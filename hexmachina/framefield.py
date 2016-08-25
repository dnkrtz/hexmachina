'''
    File: framefield.py
    License: MIT
    Author: Aidan Kurtz
    Created: 06/08/2016
    Python Version: 3.5
    ========================
    Might be moving this somewhere else soon...
'''

import bisect
import math
import numpy as np

from visual import *
from tetmesh import *
from utils import *

# Compute the matchings for all pairs of face-adjacent tets.
# We use matchings to characterize closeness between two frames s and t.
# It is essentially the chiral permutation that most closely defines
# the rotation between them.
def compute_matchings(tet_mesh):
    # Loop through all pairs of face-adjacent tets.
    for pair in tet_mesh.mesh.adjacent_elements:
        args = []
        if -1 in pair:
            continue # Boundary face
        # Find the best permutation to characterize closeness.
        for permutation in chiral_symmetries:
            arg = tet_mesh.frames[pair[0]].uvw - tet_mesh.frames[pair[1]].uvw * permutation.T
            args.append(np.linalg.norm(arg))
        # Store the matching
        tet_mesh.matchings[tuple(pair)] = np.argmin(args)

# Greedy matching adjustment (section 3.3.1)
def matching_adjustment(tet_mesh):
    # Recompute the matchings.
    tet_mesh.compute_matchings()

    # Create edge dictionary.
    edges = {}
    for ei, edge in enumerate(tet_mesh.mesh.edges):
        edges[(sorted(edge))] = ei

    # Loop over all edges.
    for ei in improper_edges:
        edge = tet_mesh.mesh.edges[ei]
        ti = tet_mesh.mesh.edge_adjacent_elements[ei]
        for neigh_ti in tet_mesh.mesh.neighbors[ti]:
            # Make sure this neighbor shares the edge.
            if is_on_edge(tet_mesh.mesh.elements[ti], tet_mesh.mesh.edges[ei]):
                break
        
        # Face to match.
        tet_pair = (ti, neigh_ti)
        if tet_pair not in tet_mesh.matchings:
            tet_pair = tet_pair[1::] # reverse

        tet = tet_mesh.mesh.elements[tet_pair[0]]
        neigh = tet_mesh.mesh.neighbors[tet_pair[1]]

        # Define fi and other edges.
        fi = 0
        face = tet_mesh.mesh.faces[fi]
        other_vi = face.remove(edge[0]).remove(edge[1])
        edge1 = sorted([other_vi, edge[0]])
        edge2 = sorted([other_vi, edge[1]])
        ei_1 = edges[edge1]
        ei_2 = edges[edge2]
        
        # Rank the 24 possible chiral symmetries.
        sorted_permutations = []
        for permutation in chiral_symmetries:
            arg = tet_mesh.frames[pair[0]].uvw - tet_mesh.frames[pair[1]].uvw * permutation.T
            bisect.insort_left(sorted_permutations, np.linalg.norm(arg))
            
        # Test each permutation, in order.
        for arg in sorted_permutations:
            tet_mesh.matchings[pair] = chiral_symmetries[arg]
            types = [ compute_edge_type(tet_mesh, ei),
                      compute_edge_type(tet_mesh, ei_1),
                      compute_edge_type(tet_mesh, ei_2) ]
            

        

def compute_edge_type(tet_mesh, ei):
    # Classify the internal edges by type, and find the singular graph.
    # The edge type is determined via concatenation of the matchings around 
    # the edge's tetrahedral one-ring.
    try:
        one_ring = tet_mesh.one_rings[ei]
    except KeyError:
        return # Not an internal edge.
    
    # Concatenate the matchings around the edge to find its type.
    edge_type = np.identity(3)
    for i in range(len(one_ring)):
        matching = []
        pair = (one_ring[i], one_ring[(i + 1) % len(one_ring)])
        # If pair order is reversed, invert/transpose rotation matrix.
        if pair not in tet_mesh.matchings:
            pair = pair[::-1] # reverse
            matching = chiral_symmetries[tet_mesh.matchings[pair]].T
        else:
            matching = chiral_symmetries[tet_mesh.matchings[pair]]
        # Concatenate transforms
        edge_type = np.dot(edge_type, matching)
    
    # Locate singular (not identity) and improper (not restricted) edges.
    is_singular, is_improper = True, True
    for si, restricted_type in enumerate(chiral_symmetries[0:9]):
        if np.allclose(edge_type, restricted_type):
            if si == 0 : is_singular = False
            is_improper = False
            break

    # Classify as proper(0), improper(1) or singular(2)
    if is_improper: tet_mesh.edge_types[ei] = 1
    if is_singular: tet_mesh.edge_types[ei] = 2

    return tet_mesh.edge_types[ei]

def singular_graph(tet_mesh):
    # Compute matchings if it hasn't been done yet.
    compute_matchings(tet_mesh)
    for i in range(len(tet_mesh.mesh.edges)):
        compute_edge_type(tet_mesh, i)
   
    # Store them in lists for output.
    singular_edges = []
    improper_edges = []
    for ei in range(len(tet_mesh.mesh.edges)):
        if tet_mesh.edge_types[ei] > 0:
            improper_edges.append(ei)           
        if tet_mesh.edge_types[ei] == 2:
            singular_edges.append(ei)

    # Output each graph as a .vtk.
    vtk_lines(tet_mesh.mesh.points,
             [tet_mesh.mesh.edges[ei] for ei in singular_edges],
             'singular')
    vtk_lines(tet_mesh.mesh.points,
             [tet_mesh.mesh.edges[ei] for ei in improper_edges],
             'improper')

    return singular_edges, improper_edges






            


    