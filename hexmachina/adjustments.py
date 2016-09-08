'''
    File: adjustments.py
    License: MIT
    Author: Aidan Kurtz
    Created: 08/09/2016
    Python Version: 3.5
    ========================
    Various greedy adjustments used to fix artifacts.
'''

import numpy as np

# CURRENTLY WIP, UNUSED
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
        fi = 0 # TODO@ wut?
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