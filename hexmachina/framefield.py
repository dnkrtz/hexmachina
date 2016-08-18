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
        etype = np.identity(3)
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
            etype = np.dot(etype, matchings[pair])

        # Singular edge.
        if not np.array_equal(etype, np.identity(3)):
            lines.append(edge)

    # Plot singular edges.    
    plot_lines(lines, tet_mesh.mesh.points)






            


    