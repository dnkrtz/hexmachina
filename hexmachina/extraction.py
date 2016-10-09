'''
    File: extraction.py
    License: MIT
    Author: Aidan Kurtz
    Created: 08/09/2016
    Python Version: 3.5
    ========================
    This module contains function that extract the hexahedral grid
    from the parametrized uvw maps.
'''

import numpy as np

from utils import vtk_points

# Computes the euclidean coordinates of the point with the desired
# uvw-map value. Value is interpolated over the linear tetrahedral.
def barycentric_interp(values, coords, desired_val):
    # Determine barycentric coords for desired value.
    Q = np.zeros((4,4))
    Q[0:3,:] = values.T
    Q[3,:] = [1, 1, 1, 1]
    v = np.zeros((4,))
    v[0:3] = desired_val
    v[3] = 1
    # Try to solve system, if singular, return None
    try:
        bary = np.linalg.solve(Q, v)
    except np.linalg.linalg.LinAlgError:
        return None
    # If any barycentric coordinate is negative, we are outside
    # the tetrahedral, return None.
    if (np.any([ e < 0 for e in bary])):
        return None
    # Convert barycentric coords to euclidean coords.
    R = np.zeros((4,4))
    R[0:3,:] = coords.T
    R[3,:] = [1, 1, 1, 1]

    pt = (R @ bary)[0:3]

    return pt

def extract_isolines(machina, f_map):

    iso_pts = []

    for ti, tet in enumerate(machina.tet_mesh.elements):

        p = np.zeros((4,3)) # xyz location of each point
        f = np.zeros((4,3)) # uvw values at each point
        for vi in [0,1,2,3]: # tet points pqrs
            p[vi,:] = machina.tet_mesh.points[tet[vi]]
            f[vi,:] = f_map[12*ti+3*vi : 12*ti+3*vi+3]
        
        # Integer iso-values in this tet.
        u_int, v_int, w_int = [], [], []
        u_vtx = f[:,0]
        v_vtx = f[:,1]
        w_vtx = f[:,2]

        # Find which integer points are in this tet.
        u_min, u_max = np.amin(u_vtx), np.amax(u_vtx)
        v_min, v_max = np.amin(v_vtx), np.amax(v_vtx)
        w_min, w_max = np.amin(w_vtx), np.amax(w_vtx)

        # print("Min: %f" %v_min)
        # print("Max: %f" %v_max)

        # Increment on integers in the range.
        u_cur = np.ceil(u_min)
        while (u_cur < u_max):
            u_int.append(u_cur)
            u_cur += 1
        v_cur = np.ceil(v_min)
        while (v_cur < v_max):
            v_int.append(v_cur)
            v_cur += 1
        w_cur = np.ceil(w_min)
        while (w_cur < w_max):
            w_int.append(w_cur)
            w_cur += 1

        # print(w_int)

        # Make sure there is a uvw-integer intersection in this tet.
        if len(u_int) > 0 and len(v_int) > 0 and len(w_int) > 0:
            
            uvw = [[u,v,w] for u in u_int for v in v_int for w in w_int]

            # Interpolate using linear barycentric interpolation.
            for val in uvw:
                iso_pt = barycentric_interp(f, p, val)
                # Iso-points that are none 
                if iso_pt != None:
                    # uvw_lookup[val] = len(iso_pts)
                    iso_pts.append(iso_pt)        

    vtk_points(iso_pts, 'isopoints')
    