'''
    File: parametrization.py
    License: MIT
    Author: Aidan Kurtz
    Created: 25/08/2016
    Python Version: 3.5
    ========================
    Might be moving this somewhere else soon...
'''

import bisect
import math
import numpy as np

from visual import *
from machina import *
from utils import *

# Compute the matchings for all pairs of face-adjacent tets.
# We use matchings to characterize closeness between two frames s and t.
# It is essentially the chiral permutation that most closely defines
# the rotation between them.
def compute_matchings(machina):
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

# Greedy matching adjustment (section 3.3.1)
def matching_adjustment(machina):
    # Recompute the matchings.
    machina.compute_matchings(machina)

    # Create edge dictionary.
    edges = {}
    for ei, edge in enumerate(machina.tet_mesh.edges):
        edges[(sorted(edge))] = ei

    # Loop over all edges.
    for ei in improper_edges:
        edge = machina.tet_mesh.edges[ei]
        ti = machina.tet_mesh.edge_adjacent_elements[ei]
        tet_pairs = []
        for neigh_ti in machina.tet_mesh.neighbors[ti]:
            # Make sure this neighbor shares the edge.
            if is_on_edge(machina.tet_mesh.elements[ti], machina.tet_mesh.edges[ei]):
                tet_pair.append(ti, neigh_ti)
        
        # Face to match.
        tet_pair = (ti, neigh_ti)
        if tet_pair not in machina.matchings:
            tet_pair = tet_pair[1::] # reverse

        tet = machina.tet_mesh.elements[tet_pair[0]]
        neigh = machina.tet_mesh.neighbors[tet_pair[1]]

        # Define fi and other edges.
        fi = 0
        face = machina.tet_mesh.faces[fi]
        other_vi = face.remove(edge[0]).remove(edge[1])
        edge1 = sorted([other_vi, edge[0]])
        edge2 = sorted([other_vi, edge[1]])
        ei_1 = edges[edge1]
        ei_2 = edges[edge2]
        
        # Rank the 24 possible chiral symmetries.
        sorted_permutations = []
        for permutation in chiral_symmetries:
            arg = machina.frames[pair[0]].uvw - machina.frames[pair[1]].uvw * permutation.T
            bisect.insort_left(sorted_permutations, np.linalg.norm(arg))
            
        # Test each permutation, in order.
        for arg in sorted_permutations:
            machina.matchings[pair] = chiral_symmetries[arg]
            types = [ compute_edge_type(machina, ei),
                      compute_edge_type(machina, ei_1),
                      compute_edge_type(machina, ei_2) ]
            if np.all(types) and 2 in types:
                continue 
        

def compute_edge_types(machina, edge_index):
    # Classify the internal edges by type, and find the singular graph.
    # The edge type is determined via concatenation of the matchings around 
    # the edge's tetrahedral one-ring.
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
    # Compute matchings.
    compute_matchings(machina)
    compute_edge_types(machina, range(len(machina.tet_mesh.edges)))
   
    # Store them in lists for output.
    singular_edges = []
    improper_edges = []
    for ei in range(len(machina.tet_mesh.edges)):     
        if machina.edge_types[ei] > 0:
            singular_edges.append(ei)     
        if machina.edge_types[ei] == 2:
            improper_edges.append(ei)

    # Output each graph as a .vtk.
    vtk_lines(machina.tet_mesh.points,
             [machina.tet_mesh.edges[ei] for ei in singular_edges],
             'singular')
    vtk_lines(machina.tet_mesh.points,
             [machina.tet_mesh.edges[ei] for ei in improper_edges],
             'improper')

    return singular_edges, improper_edges

def parametrize_volume(machina, h):

    # Translational gaps (parallel to tet_mesh.faces)
    gaps = []
    # Minimum spanning tree of tetrahedral mesh (subset of dual edges).
    mst_edges = []
    visited_tets = set()
    # Span until all tets have been visited.
    ti = 0
    while len(visited_tets) != len(machina.tet_mesh.elements):
        for neigh_ti in machina.tet_mesh.neighbors[ti]:
            if neigh_ti in visited_tets:
                continue
            mst_edges.add(machina.dual_edges[frozenset([ti, neigh_ti])])
            visited_tets.add(ti)

    cons = ({'type': 'eq', 'fun': lambda x:  x[0] - 2 * x[1] + 2})

    
    
    







    # Each vertex may have multiple initial map values, depending
    # on the number of tets it's a part of. We narrow down later.
    f_map = [ [] for _ in range(len(machina.points)) ]
    f_map_grad = [ [] for _ in range(len(machina.points)) ]

    # Compute the framefield gradient.
    for tet_pair, matching in machina.matchings.items():
        for vi in tet_pair[0]:
            f_map[vi].append(machina.frames[tet_pair[0]])
            f_map_grad[vi].append(np.diag(machina.frames[tet_pair[0]]))

        for vi in tet_pair[1]:
            f_map[vi].append(machina.frames[tet_pair[1]])
    
    for vi, vertex_map in enumarate(f_map):
        ti = 0 # what?
        score = tet_volume(machina, ti)
        for f in vertex_map:
            vol = tet_volume(machina, ti)
            D = np.linalg.norm(h * frames_grad[:,0] - machina.frames[ti].uvw[:,0])**2 + \
                np.linalg.norm(h * frames_grad[:,1] - machina.frames[ti].uvw[:,1])**2 + \
                np.linalg.norm(h * frames_grad[:,2] - machina.frames[ti].uvw[:,2])**2
            





    
