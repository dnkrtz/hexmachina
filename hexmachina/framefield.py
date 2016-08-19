'''
    File: framefield.py
    License: MIT
    Author: Aidan Kurtz
    Created: 06/08/2016
    Python Version: 3.5
    ========================
    Might be moving this somewhere else soon...
'''

import math
import numpy as np

from visual import *
from tetmesh import *

def singular_graph(tet_mesh):
    # We use matchings to characterize closeness between two frames s and t.
    # It is essentially the chiral permutation that best most closesly defines
    # the rotation between them.
    matchings = {}
    # Compute the matchings for all pairs of face-adjacent tets.
    for pair in tet_mesh.mesh.adjacent_elements:
        args = []
        if -1 in pair:
            continue # Boundary face
        # Find the best permutation to characterize closeness.
        for permutation in chiral_symmetries:
            arg = tet_mesh.frames[pair[0]].uvw - tet_mesh.frames[pair[1]].uvw * permutation.T
            args.append(np.linalg.norm(arg))
        # Store the matching
        matchings[tuple(pair)] = chiral_symmetries[np.argmin(args)]

    # Classify the internal edges by type, and find the singular graph.
    # The edge type is determined via concatenation of the matchings around 
    # the edge's tetrahedral one-ring.
    singular_edges = []
    for ei, edge in enumerate(tet_mesh.mesh.edges):
        try:
            one_ring = tet_mesh.one_rings[ei]
        except KeyError:
            continue # Not an internal edge.
        # Concatenate the matchings around the edge to find its type.
        edge_type = np.identity(3)
        for i in range(len(one_ring)):
            matching = []
            pair = (one_ring[i], one_ring[(i + 1) % len(one_ring)])
            # If pair order is reversed, invert/transpose rotation matrix.
            if pair not in matchings:
                pair = pair[::-1] # reverse
                matching = matchings[pair].T
            else:
                matching = matchings[pair]
            # Concatenate transforms
            edge_type = np.dot(edge_type, matching)

        # Singular edge.
        if not np.allclose(edge_type, np.identity(3)):
            singular_edges.append(edge)

    return singular_edges






            


    