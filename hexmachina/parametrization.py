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
import sys
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
    singular_vertices = set()
    # Store edges and vertices
    for ei, edge in enumerate(machina.tet_mesh.edges):     
        if machina.edge_types[ei] > 0:
            singular_edges.append(ei)
            singular_vertices.update([edge[0],edge[1]])
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

def var_index(ti, vi, ci):
    return ( 4 * ti + 3 * vi + ci )

def linear_system(machina, mst_edges, singular_vertices):
    
    ne = len(machina.tet_mesh.elements)
    L = sparse.lil_matrix( (4 * ne, 4 * ne) )
    C = sparse.lil_matrix( (3 * 12 * ne, 12*ne) )
    integer_vars = set()
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
            w_pqr = [ var_index(t, vi_t[i], 2) for i in range(3) ]
            for i in range(2):
                C[ccount, w_pqr[0]] = 1
                C[ccount, w_pqr[i]] = -1
                ccount += 1
                integer_vars.update(w_pqr)
            
        # Internal face with two tets in common.
        else:
            match = chiral_symmetries[machina.matchings[fi]]
            # Get local tet vertex indices of shared face vertices.
            vi_s, vi_t = [], []
            for vi in machina.tet_mesh.faces[fi]:
                vi_s.append(machina.tet_mesh.elements[s].index(vi))
                vi_t.append(machina.tet_mesh.elements[t].index(vi))

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

    # Check singular edges (must be integer).
    for ti, tet in enumerate(machina.tet_mesh.elements):
        for local_vi, vi in enumerate(tet):
            if vi in singular_vertices:
                integer_vars.update(range(12*ti + 3*local_vi, 12*ti + 3*local_vi + 2))

    C = C.tocsr()
    num_nonzeros = np.diff(C.indptr)
    C = C[num_nonzeros != 0] # remove zero-rows

    # Create laplacian of tetrahedrons.
    L = sparse.diags([1,1,1,-3,1,1,1],[-9,-6,-3,0,3,6,9],(12*ne,12*ne))

    return L, C, integer_vars

# def drop_rows(M, idx_to_drop, drop_cols=False):
#     C = M.tocoo()
#     keep = ~np.in1d(C.row, idx_to_drop)
#     C.data, C.row, C.col = C.data[keep], C.row[keep], C.col[keep]
#     C.row -= idx_to_drop.searchsorted(C.row)
#     C._shape = (C._shape[0] - len(idx_to_drop), C._shape[1])
#     if drop_cols:
#         keep = ~np.in1d(C.col, idx_to_drop)
#         C.data, C.row, C.col = C.data[keep], C.row[keep], C.col[keep]
#         C.col -= idx_to_drop.searchsorted(C.col)
#         C._shape = (C._shape[0], C._shape[1] - len(idx_to_drop))
    
#     return C.tocsr()

def drop_rows(M, var_i):
    M = M.tolil()
    M.rows = np.delete(M.rows, var_i)
    M.data = np.delete(M.data, var_i)
    M._shape = (M._shape[0] - len(var_i), M._shape[1])
    return M

# Remove variables from system.
# The 'b' matrix should have its i value(s) set before calling.
def reduce_system(A, x, b, var_i):

    # Convert all the lil format.
    A = sparse.lil_matrix(A)
    x = sparse.lil_matrix(x.reshape((len(x),1)))
    b = sparse.lil_matrix(b.reshape((len(b),1)))

    # Update rhs b (absorbs vars).
    for i in var_i:
        b = b - x[i,0] * A.getcol(i)

    # Drop rows form b vector.
    b = drop_rows(b, var_i) 
    # Drop rows from the x vector.
    x = drop_rows(x, var_i)
    # Drop rows from the A matrix.
    A = drop_rows(x, var_i)
    # Drop cols from the A matrix.    
    A.transpose()
    A = drop_rows(x, var_i)    
    A.transpose()

    return A, x, b

def adaptive_rounding(machina):
    pass

def something(machina, f_map):
    fun = a * f_map[0] + b * f_map[0]
    pass

def parametrize_volume(machina, singular_vertices):

    # Each vertex has multiple values, depending
    # on the number of tets it's a part of.
    ne = len(machina.tet_mesh.elements)
    f_atlas = np.zeros(12 * ne)

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

    print('Computing laplacian and constraints...')

    # Create linear system based on laplacian and constraints.
    laplacian, cons, int_vars = linear_system(machina, mst_edges, singular_vertices)
    n_cons = cons.get_shape()[0]

    # tet_weights = [ tet_volume(machina.tet_mesh, ti) for ti in range(ne) ]
    # for ti in range(ne):
    #     laplacian[:,12*ti:12*(ti+1)] = tet_weights[ti] * laplacian[:,12*ti:12*(ti+1)]

    A = sparse.bmat(([[laplacian, cons.transpose()],[cons, None]]), dtype=np.int8)

    # Discrete frame divergence.
    b = np.zeros((12*ne + n_cons))
    for ti in range(ne):
        frame = machina.frames[ti]
        div = [ np.sum(frame.uvw[0,0]),
                np.sum(frame.uvw[1,1]),
                np.sum(frame.uvw[2,2]) ]
        b[12*ti:12*(ti+1)] = np.hstack([div for _ in range(4)])

    print("(Conjugate Gradient)...", end=" ")
    sys.stdout.flush()

    x, info = sparse.linalg.cg(A, b, tol = 1e-2)

    say_ok()

    print(len(x))

    # Enforce integer variables
    vars_to_remove = []
    for vi in sorted(int_vars,reverse=True):
        value = x[vi]
        rounded = int(round(value))
        if np.abs(value - rounded) > 1e-4:
            continue
        # Otherwise, delta is small enough to round.
        x[vi] = rounded
        vars_to_remove.append(vi)
    vars_to_remove = np.array(vars_to_remove)

    # Update linear system.
    A, x, b = reduce_system(A, x, b, vars_to_remove)

    print(x.shape[0])

    print(x[:12*ne,0])


    # # Recompute gaps
    # gaps = np.zeros( (len(machina.tet_mesh.faces),3) )
    # for fi, adj_ti in enumerate(machina.tet_mesh.adjacent_elements):
    #     if -1 in adj_ti:
    #         continue
    #     # Get local tet vertex indices of shared face vertices.
    #     s, t = adj_ti[0], adj_ti[1]
    #     vi = machina.tet_mesh.faces[fi][0]
    #     vi_s = machina.tet_mesh.elements[s].index(vi)
    #     vi_t = machina.tet_mesh.elements[t].index(vi)
    #     f_s = x[var_index(s, vi_s, 0):var_index(s, vi_s, 3)]
    #     f_t = x[var_index(t, vi_t, 0):var_index(t, vi_t, 3)]
    #     gaps[fi,:] = f_t - np.dot(chiral_symmetries[machina.matchings[fi]], f_s)

    # print(gaps)


    



