'''
    File: parametrization.py
    License: MIT
    Author: Aidan Kurtz
    Created: 25/08/2016
    Python Version: 3.5
    ========================
    Hexahedral parametrization based on the discrete 3D frame field.
    (This module is currently broken)
'''

import bisect
import numpy as np
from scipy import sparse
import sys

from machina import *
from transforms import *
from utils import *
from visual import *

# 
def var_index(ti, vi, ci):
    """The flattened index corresponding to the tet index ti,
    local vertex index vi, and coordinate index ci.
    """
    if not isinstance(ci, range):
        ci = [ ci ]
    return [ ( 12 * ti + 3 * vi + i ) for i in ci ]

def drop_rows(M, i):
    """Remove row(s) i from matrix"""
    M = M.tolil()
    M.rows = np.delete(M.rows, i)
    M.data = np.delete(M.data, i)
    M._shape = (M._shape[0] - len(i), M._shape[1])
    return M

# Remove variables from system.
# The 'b' matrix should have its i value(s) set before calling.
def reduce_system(A, x, b, i):
    """Remove variable(s) i from system.
    Row(s) i of matrix 'b' must be set before this gets called."""

    # Convert all the lil format.
    A = sparse.lil_matrix(A)
    x = sparse.lil_matrix(x.reshape((len(x),1)))
    b = sparse.lil_matrix(b.reshape((len(b),1)))

    # Update rhs b (absorbs vars).
    for i in var_i:
        b = b - x[i,0] * A.getcol(i)

    # Drop rows form b vector.
    b = drop_rows(b, i) 
    # Drop rows from the x vector.
    x = drop_rows(x, i)
    # Drop rows from the A matrix.
    A = drop_rows(A, i)
    # Drop cols from the A matrix.
    A = drop_rows(A.transpose(), i)

    return A, x, b

def linear_system(machina, mst_edges, singular_vertices):
    """Define linear system that represents the parametrization.
    This involes an atlas of maps defining a uvw iso-value at each vertex.
    A single vertex can have multiple uvw values."""
    
    ne = len(machina.tet_mesh.elements)
    C = sparse.lil_matrix( (9 * 12 * ne, 12*ne) )
    ccount = 0 # constraint counter

    for fi, adj_ti in enumerate(machina.tet_mesh.adjacent_elements):
        s, t = adj_ti[0], adj_ti[1]
        # Boundary face.
        if -1 in [s, t]:
            t = s if s != -1 else t # tet index
            vi_t = [] # local tet vertex indices of face.
            for vi in machina.tet_mesh.faces[fi]:
                vi_t.append(machina.tet_mesh.elements[t].index(vi))
            # Constrain surface normal.
            pqr_w = [ var_index(t, vi_t[i], 2) for i in range(3) ]
            for i in [1,2]: # points qr
                C[ccount, pqr_w[0]] = 1
                C[ccount, pqr_w[i]] = -1
                ccount += 1
            
        # Internal face with two tets in common.
        else:
            match = chiral_symmetries[machina.matchings[fi]]
            # Get local tet vertex indices of shared face vertices.
            vi_s, vi_t = [], []
            for vi in machina.tet_mesh.faces[fi]:
                # Store the ordered indices of each vertex on the face.
                # In other words, vi_s[0] and vi_t[0] are the same vertex.
                vi_s.append(machina.tet_mesh.elements[s].index(vi))
                vi_t.append(machina.tet_mesh.elements[t].index(vi))
            # The point variable index range for the uvw values of each point.
            pqr_t = [ var_index(t, vi_t[i], range(3)) for i in range(3) ]
            pqr_s = [ var_index(s, vi_s[i], range(3)) for i in range(3) ]
            # Next, apply constraints.
            # If gap is 0 (minimum spanning tree).
            if fi in mst_edges:
                for i in [0,1,2]: # points pqr
                    # Enforce 0 translation, but possible chiral rotation.
                    C[ccount:ccount+3, pqr_t[i]] = - sparse.eye(3)
                    C[ccount:ccount+3, pqr_s[i]] = match
                    ccount += 3
            else:
                # If gap isn't 0, enforce that it be constant.
                # In other words, constrain edges.
                for i in [1,2]: # points qr
                    # Constraint.
                    C[ccount:ccount+3, pqr_t[0]] = sparse.eye(3)
                    C[ccount:ccount+3, pqr_t[i]] = - sparse.eye(3)
                    C[ccount:ccount+3, pqr_s[0]] = - match
                    C[ccount:ccount+3, pqr_s[i]] = match
                    ccount += 3

    # Remove zero-rows from constraint matrix.
    C = C.tocsr()
    num_nonzeros = np.diff(C.indptr)
    C = C[num_nonzeros != 0]

    # Create laplacian of local tetrahedron connectivity.
    L = sparse.diags([1,1,1,-3,1,1,1],[-9,-6,-3,0,3,6,9],(12*ne,12*ne))

    return L, C

def flag_integer_vars(machina, singular_vertices):
    """Compute which variables are integer-constrained.
    Return the indices of these variables."""

    int_vars = set()

    # Iterate through all variables
    for ti, tet in enumerate(machina.tet_mesh.elements):
        for local_vi, vi in enumerate(tet):
            # Singular vertices constrain two of their variables.
            # If singularity type is Jw, then u, v must be integers.
            if vi in singular_vertices:
                if singular_vertices[vi] < 4:
                    int_vars.add(var_index(ti, local_vi, 0))
                    int_vars.add(var_index(ti, local_vi, 1))
                elif singular_vertices[vi] < 7:
                    int_vars.add(var_index(ti, local_vi, 1))
                    int_vars.add(var_index(ti, local_vi, 2))
                else: # Doesn't check for improper.
                    int_vars.add(var_index(ti, local_vi, 2))
                    int_vars.add(var_index(ti, local_vi, 0))
            # Surface vertices must be integer in w.
            if vi in machina.surf_mesh.vertex_map:
                int_vars.add(var_index(ti, local_vi, 2))
    
    return int_vars

def adaptive_rounding(machina, A, x, b, singular_vertices):
    """Adaptively round the solution vector in a greedy manner."""

    int_vars = flag_integer_vars(machina, singular_vertices)
    # Enforce integer variables
    vars_fixed = dict()
    ne = len(machina.tet_mesh.elements)
    # The reduction array is used to keep track of global vs. reduced indices
    # as we progressively round the variables.
    # row index: reduced index, col 0: global index, col 1: is_int boolean.
    reduction_arr = np.zeros((12*ne, 2))
    reduction_arr[:,0] = np.arange(12*ne)
    for vi in int_vars:
        reduction_arr[vi,1] = 1

    # Loop until all integer variables are fixed.
    while (len(vars_fixed) < len(int_vars)):
        # Identify integer variables not yet fixed.
        vars_left = dict()
        for ri in range(reduction_arr.shape[0]):
            if reduction_arr[ri,1]:
                vars_left[ri] = reduction_arr[ri,0]

        print('Conjugate gradient... (%i integers left)' % len(vars_left))

        # Identify fixeable variables
        # First, variables with a small deviation should
        # be rounded to their nearest integer.
        vars_to_fix = []
        # gvi: global variable index, rvi: reduced variable index.
        for rvi, gvi in vars_left.items():
            value = x[rvi]
            rounded = int(round(value))
            if np.abs(value - rounded) > 1e-4:
                continue
            # Otherwise, delta is small enough to round.
            x[rvi] = rounded
            vars_fixed[gvi] = rounded
            vars_to_fix.append(rvi)

        # If no variable is fixed, fix the one with the smallest
        # deviation from its rounded integer.
        if len(vars_to_fix) == 0:
            key_list = list(vars_left.keys())
            rvi = np.argmin([ np.abs(x[rvi] - round(x[rvi])) for rvi in key_list ])
            rvi = key_list[rvi]
            x[rvi] = round(x[rvi])
            vars_fixed[vars_left[rvi]] = x[rvi]
            vars_to_fix.append(rvi)

        # Update linear system.
        A, x, b = reduce_system(A, x, b, vars_to_fix)
        b = b.toarray()
        x = x.toarray()

        # Run conjugate gradient on reduced system.
        x, info = sparse.linalg.cg(A, b, x0=x, tol = 1e-2)

        # Update the reduction array.
        reduction_arr = np.delete(reduction_arr, vars_to_fix, axis=0)

    # Final map.
    uvw_map = np.zeros(12*ne)
    count = 0
    for i in range(12*ne):
        if i in vars_fixed:
            uvw_map[i] = vars_fixed[i]
            count += 1
        else:
            uvw_map[i] = x[i - count]

    return uvw_map

def parametrize_volume(machina, singular_vertices, h):
    """Parametrize the volume as an atlas of maps based on the 3d frame field.
    Returns the discretized uvw map atlas (vertex-based)."""

    # Each vertex has multiple values, depending
    # on the number of tets it's a part of.
    ne = len(machina.tet_mesh.elements)

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
    laplacian, cons = linear_system(machina, mst_edges, singular_vertices)
    n_cons = cons.get_shape()[0]

    A = sparse.bmat(([[laplacian, cons.transpose()],[cons, None]]), dtype=np.int32)

    # Discrete frame divergence.
    b = np.zeros((12*ne + n_cons))
    for ti in range(ne):
        tet_vol = tet_volume(machina.tet_mesh, ti)
        frame = machina.frames[ti]
        div = [ np.sum(frame.uvw[:,0]),
                np.sum(frame.uvw[:,1]),
                np.sum(frame.uvw[:,2]) ]
        b[12*ti : 12*(ti+1)] = np.hstack([ div for _ in range(4)])

    b = np.divide(b, h)

    print("Conjugate Gradient... (Round 1)", end=" ")
    sys.stdout.flush()

    x, info = sparse.linalg.cg(A, b, tol = 1e-4)

    say_ok()

    print('Adaptive rounding...')

    # uvw_map = adaptive_rounding(machina, A, x, b, singular_vertices)

    uvw_map = x

    return uvw_map
