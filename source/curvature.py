'''
    File: curvature.py
    License: MIT
    Author: Aidan Kurtz
    Created: 09/07/2016
    Python Version: 3.5
    ========================
    This module estimates principal curvatures over a triangle mesh.

    It is a Python implementation of Szymon Rusinkiewicz' paper
    "Estimating Curvatures and Their Derivatives on Triangle Meshes".
    http://gfx.cs.princeton.edu/pubs/_2004_ECA/curvpaper.pdf 
    It is heavily based on his C++ code in trimesh2.
'''

import numpy as np
from utils import normalize

# Rotate a coordinate system to be perpendicular to the given normal.
def rotate_coord_sys(old_u, old_v, new_norm):
    new_u = old_u
    new_v = old_v
    old_norm = np.cross(old_u, old_v)
    # Project old normal onto new normal
    ndot = np.dot(old_norm, new_norm)
    # If projection is leq to -1, simply reverse
    if ndot <= -1:
        new_u = -new_u
        new_v = -new_v
        return new_u, new_v
    # Otherwise, compute new normal
    perp_old = new_norm - ndot * old_norm
    dperp = (old_norm + new_norm) / (1 + ndot)
    new_u -= dperp * np.dot(new_u, perp_old)
    new_v -= dperp * np.dot(new_v, perp_old)
    return new_u, new_v

# Reproject curvature tensor from the basis spanned by old uv to the new uv basis.
def project_curvature(old_u, old_v, old_ku, old_kuv, old_kv, new_u, new_v):
    old_normal = np.cross(old_u, old_v)
    # Rotate new coord system to be normal to old, for reprojection
    r_new_u, r_new_v = rotate_coord_sys(new_u, new_v, old_normal)
    u1 = np.dot(r_new_u, old_u)
    v1 = np.dot(r_new_u, old_v)
    u2 = np.dot(r_new_v, old_u)
    v2 = np.dot(r_new_v, old_v)
    new_ku  = old_ku * u1*u1 + old_kuv * (2 * u1*v1) + old_kv * v1*v1
    new_kuv = old_ku * u1*u2 + old_kuv * (u1*v2 + u2*v1) + old_kv * v1*v2
    new_kv  = old_ku * u2*u2 + old_kuv * (2 * u2*v2) + old_kv * v2*v2

    return new_ku, new_kuv, new_kv

# Given a curvature tensor, diagonalize to find principal directions and curvatures.
def diagonalize_curvature(old_u, old_v, ku, kuv, kv, new_norm):
    # Rotate old coord system to be normal to new.
    r_old_u, r_old_v = rotate_coord_sys(old_u, old_v, new_norm)
    c = 1
    s = 0
    tt = 0
    if kuv != 0:
        # Jacobi rotation to diagonalize.
        h = 0.5 * (kv - ku) / kuv
        if h < 0:
            tt = 1 / (h - np.sqrt(1 + h*h))
        else:
            tt = 1 / (h + np.sqrt(1 + h*h))
        c = 1 / np.sqrt(1 + tt*tt)
        s = tt * c
    # Compute principal curvatures.
    k1 = ku - tt * kuv
    k2 = kv + tt * kuv
    # Compute principal directions.
    if abs(k1) >= abs(k2):
        pdir1 = c * r_old_u - s * r_old_v
    else:
        k1, k2 = k2, k1 # swap
        pdir1 = s * r_old_u + c * r_old_v
    pdir2 = np.cross(new_norm, pdir1)
    # Return all the things.
    return pdir1, pdir2, k1, k2

# Compute the area "belonging" to each vertex or each corner
# of a triangle (defined as Voronoi area restricted to the 1-ring of
# a vertex, or to the triangle).
def compute_pointareas(vertices, faces):

    cornerareas = np.zeros( (len(faces), 3) )
    pointareas = np.zeros(len(vertices),)

    for i, face in enumerate(faces):
        # Face edges
        e = np.array([ vertices[face[2]] - vertices[face[1]],
                       vertices[face[0]] - vertices[face[2]],
                       vertices[face[1]] - vertices[face[0]] ])
        # Compute corner weights
        area = 0.5 * np.linalg.norm(np.cross(e[0], e[1]))
        l2 = [ np.linalg.norm(e[0]) ** 2, 
               np.linalg.norm(e[1]) ** 2,
               np.linalg.norm(e[2]) ** 2 ]
        ew = [ l2[0] * (l2[1] + l2[2] - l2[0]),
               l2[1] * (l2[2] + l2[0] - l2[1]),
               l2[2] * (l2[0] + l2[1] - l2[2]) ]
        # Case by case based on edge weight
        if ew[0] <= 0:
            cornerareas[i,1] = -0.25 * l2[2] * area / np.dot(e[0], e[2])
            cornerareas[i,2] = -0.25 * l2[1] * area / np.dot(e[0], e[1])
            cornerareas[i,0] = area - cornerareas[i,1] - cornerareas[i,2]
        elif ew[1] <= 0:
            cornerareas[i,2] = -0.25 * l2[0] * area / np.dot(e[1], e[0])
            cornerareas[i,0] = -0.25 * l2[2] * area / np.dot(e[1], e[2])
            cornerareas[i,1] = area - cornerareas[i,2] - cornerareas[i,0]
        elif ew[2] <= 0:
            cornerareas[i,0] = -0.25 * l2[1] * area / np.dot(e[2], e[1])
            cornerareas[i,1] = -0.25 * l2[0] * area / np.dot(e[2], e[0])
            cornerareas[i,2] = area - cornerareas[i,0] - cornerareas[i,1]
        else:
            ewscale = 0.5 * area / (ew[0] + ew[1] + ew[2])
            for j in range(3):
                cornerareas[i,j] = ewscale * (ew[(j+1)%3] + ew[(j+2)%3])
    
        pointareas[face[0]] += cornerareas[i,0]
        pointareas[face[1]] += cornerareas[i,1]
        pointareas[face[2]] += cornerareas[i,2]

    return pointareas, cornerareas

# Given the faces, vertices and vertex normals.
# Compute principal curvatures and directions.
def compute_curvatures(vertices, faces, normals):
    
    # Initialize lists
    # @TODO(aidan) Make these objects variables
    curv1 = np.zeros(len(vertices),)
    curv2 = np.zeros(len(vertices),)
    curv12 = np.zeros(len(vertices),)
    pdir1 = [ [] for _ in range(len(vertices)) ]
    pdir2 = [ [] for _ in range(len(vertices)) ]

    # Compute pointareas
    pointareas, cornerareas = compute_pointareas(vertices, faces)
    
    # Set up an initial coordinate system per-vertex
    for i, face in enumerate(faces):
        pdir1[face[0]] = vertices[face[1]] - vertices[face[0]]
        pdir1[face[1]] = vertices[face[2]] - vertices[face[1]]
        pdir1[face[2]] = vertices[face[0]] - vertices[face[2]]

    for i, vertex in enumerate(vertices):
        pdir1[i] = normalize(np.cross(pdir1[i], normals[i]))
        pdir2[i] = np.cross(normals[i], pdir1[i]) 
        
    # Compute curvature per-face
    for i, face in enumerate(faces):
        # Face edges
        e = np.array([ vertices[face[2]] - vertices[face[1]],
                       vertices[face[0]] - vertices[face[2]],
                       vertices[face[1]] - vertices[face[0]] ])
        # N-T-B coordinate system per-face
        t = normalize(e[0])
        n = np.cross(e[0], e[1])
        b = normalize(np.cross(n, t))

        # Estimate curvature based on variation of
        # normals along edges.
        m = np.zeros(3,)
        w = np.zeros( (3,3) )
        for j in range(3):
            u =  np.dot(e[j], t)
            v =  np.dot(e[j], b)
            w[0,0] += u*u
            w[0,1] += u*v
            w[2,2] += v*v
            dn = normals[face[(j+2)%3]] - normals[face[(j+1)%3]]
            dnu = np.dot(dn, t)
            dnv = np.dot(dn, b)
            m[0] += dnu*u
            m[1] += dnu*v + dnv*u
            m[2] += dnv*v
        w[1,1] = w[0,0] + w[2,2]
        w[1,2] = w[0,1]
        
        # Least squares solution.
        x, residuals, rank, s = np.linalg.lstsq(w,m)

        # Push it back out to the vertices.
        for j in range(3):
            vj = face[j]
            c1, c12, c2 = project_curvature(t, b, x[0], x[1], x[2],
                                            pdir1[vj], pdir2[vj])
            weight = cornerareas[i,j] / pointareas[vj]
            curv1[vj] += weight * c1
            curv12[vj] += weight * c12
            curv2[vj] += weight * c2
        
    # Compute principal directions and curvatures at each vertex.
    for i, vertex in enumerate(vertices):
        pdir1[i], pdir2[i], curv1[i], curv2[i] = \
            diagonalize_curvature(pdir1[i], pdir2[i], curv1[i], 
                                  curv12[i], curv2[i], normals[i])
    
    # Sliced bread.
    return curv1, curv2, pdir1, pdir2