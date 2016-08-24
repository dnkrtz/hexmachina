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
            if pair not in tet_mesh.matchings:
                pair = pair[::-1] # reverse
                matching = tet_mesh.matchings[pair].T
            else:
                matching = tet_mesh.matchings[pair]
            # Concatenate transforms
            edge_type = np.dot(edge_type, matching)

        # Singular edge.
        if not np.allclose(edge_type, np.identity(3)):
            singular_edges.append(edge)

    return singular_edges






            


    