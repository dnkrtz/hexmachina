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
from scipy import sparse
import timeit

from visual import *
from machina import *
from utils import *
from transforms import *

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

def var_index(ti, vi, ci):
    return ( 4 * ti + 3 * vi + ci )

def constraint_matrix(machina, mst_edges):
    
    ne = len(machina.tet_mesh.elements)
    L = sparse.lil_matrix( (4 * ne, 4 * ne) )
    C = sparse.lil_matrix( (3 * 12 * ne, 12*ne) )
    ccount = 0

    for fi, adj_ti in enumerate(machina.tet_mesh.adjacent_elements):
        s, t = adj_ti[0], adj_ti[1]
        # Boundary face.
        if -1 in [s, t]:
            t = s if s != -1 else t # tet index
            vi_t = [] # local tet vertex indices of face.
            for vi in machina.tet_mesh.faces[fi]:
                vi_t.append(machina.tet_mesh.elements[t].index(vi))
            # Constrain surface normal.
            for i in [1, 2]: # qr
                C[ccount, var_index(t, vi_t[0], 2)] = 1
                C[ccount, var_index(t, vi_t[i], 2)] = -1
                ccount += 1
            
        # Internal face with two tets in common.
        else:
            match = chiral_symmetries[machina.matchings[fi]]
            # Get local tet vertex indices of shared face vertices.
            vi_s, vi_t = [], []
            for vi in machina.tet_mesh.faces[fi]:
                vi_s.append(machina.tet_mesh.elements[s].index(vi))
                vi_t.append(machina.tet_mesh.elements[t].index(vi))

            # Store laplacian.
            # for p in [0,1,2]: # points pqr
            #     i = 4 * t + vi_t[p]
            #     j = 4 * s + vi_s[p]
            #     L[i,i] -= 1
            #     L[j,j] -= 1
            #     L[i,j] += 1
            #     L[j,i] += 1

            # Next, apply constraints.
            # If gap is 0 (minimum spanning tree).
            if fi in mst_edges:
                for i in [0,1,2]: # points pqr
                    for j in [0,1,2]: # coords uvw
                        C[ccount, var_index(t, vi_t[i], j)] = - 1
                        for k in [0,1,2]:
                            C[ccount, var_index(s, vi_s[i], k)] = match[j, k]
                        ccount += 1
            # If gap isn't 0, enforce that it be constant.
            else:
                for i in [1,2]: # points qr
                    for j in [0,1,2]: # coords uvw
                        C[ccount, var_index(t, vi_t[0], j)] = 1
                        C[ccount, var_index(t, vi_t[i], j)] = - 1
                        for k in [0,1,2]: # permutation
                            C[ccount, var_index(s, vi_s[0], k)] = - match[j, k]
                            C[ccount, var_index(s, vi_s[i], k)] = match[j, k]
                        ccount += 1
    
    C = C.tocsr()
    num_nonzeros = np.diff(C.indptr)
    C = C[num_nonzeros != 0] # remove zero-rows

    # Expand the laplacian for uvw format.
    # L = sparse.kron(L, sparse.eye(3))

    L = sparse.diags([1,1,1,-3,1,1,1],[-9,-6,-3,0,3,6,9],(12*ne,12*ne))

    return L, C

def icall(xk):
    print(xk)

def parametrize_volume(machina):

    # Each vertex has multiple values, depending
    # on the number of tets it's a part of.
    ne = len(machina.tet_mesh.elements)
    f_map = np.zeros(12 * ne)

    # Minimum spanning tree of dual mesh as list of face indices.
    # Span until all tets have been visited.
    ti = 0
    mst_edges = set()
    visited_tets = set()
    while ti < len(machina.tet_mesh.elements):
        for neigh_ti in machina.tet_mesh.neighbors[ti]:
            if neigh_ti in visited_tets or neigh_ti == -1:
                continue
            # Get face index from s-t tet pair.
            fi = machina.dual_edges[frozenset([ti, neigh_ti])]
            mst_edges.add(fi)
            visited_tets.add(ti)
        ti += 1

    # Remove translational freedom with constraints.
    laplacian, cons = constraint_matrix(machina, mst_edges)
    n_cons = cons.get_shape()[0]

    A = sparse.bmat(([[laplacian, cons.transpose()],[cons, None]]), dtype=np.int8)

    # Discrete frame divergence.
    b = np.zeros(12*ne + n_cons)
    for ti in range(ne):
        frame = machina.frames[ti]
        div = [ np.sum(frame.uvw[0,0]),
                np.sum(frame.uvw[1,1]),
                np.sum(frame.uvw[2,2]) ]
        b[12*ti:12*(ti+1)] = np.hstack([div for _ in range(4)])

    print("Beginning Conjugate Gradient...")

    x = sparse.linalg.cg(A, b, tol = 1e-2)
    print(x)

    # scipy.optimize.minimize(
    #     fun = parametrization_energy,
    #     x0 = f_map,
    #     args = machina,
    #     method = 'CG',
    #     options = {
    #         'disp': True,
    #         ''
    #     }
    # )
    

def parametrization_energy(f_map, machina):

    result = 0

    # Build the summation.
    for ti, tet in enumerate(machina.tet_mesh.elements):
        volume = tet_volume(machina, ti)
        # On each tetrahedron, we have 4 values of (u,v,w)
        pts = [ machina.tet_mesh.points[tet[i]] for i in range(4) ]
        u = [ f_map[ti, i, 0] for i in range(4) ]
        v = [ f_map[ti, i, 1] for i in range(4) ]
        w = [ f_map[ti, i, 2] for i in range(4) ]
        # Interpolating the finite difference gives us the gradient
        # with respect to euclidean coords (x,y,z).
        du = (1/3)*((u[0] - u[1]) / (pts[0] - pts[1]) + \
                    (u[0] - u[2]) / (pts[0] - pts[2]) + \
                    (u[0] - u[3]) / (pts[0] - pts[3]))
        dv = (1/3)*((v[0] - v[1]) / (pts[0] - pts[1]) + \
                    (v[0] - v[2]) / (pts[0] - pts[2]) + \
                    (v[0] - v[3]) / (pts[0] - pts[3]))
        dw = (1/3)*((w[0] - w[1]) / (pts[0] - pts[1]) + \
                    (w[0] - w[2]) / (pts[0] - pts[2]) + \
                    (w[0] - w[3]) / (pts[0] - pts[3]))
        # Compute D for the tetrahedron.
        D = np.linalg.norm(h * du - machina.frames[ti].uvw[:,0])**2 + \
            np.linalg.norm(h * dv - machina.frames[ti].uvw[:,1])**2 + \
            np.linalg.norm(h * dw - machina.frames[ti].uvw[:,2])**2

        result += volume * D

    return result
        




    
