'''
    File: normals.py
    License: MIT
    Author: Aidan Kurtz
    Created: 09/07/2016
    Python Version: 3.5
    ========================
    This module computes face and vertex normals.
'''

import numpy as np
from utils import normalize

def compute_normals(faces, vertices):

    f_norms = [ np.zeros(3,) for _ in range(len(faces)) ]
    v_norms = [ np.zeros(3,) for _ in range(len(vertices)) ]

    # Compute face normals, easy as cake.
    for fi, face in enumerate(faces):
        f_norms[fi] = np.cross(vertices[face[1]] - vertices[face[0]],
                               vertices[face[2]] - vertices[face[0]])
    
    # Next, compute the vertex normals.
    for fi, face in enumerate(faces):
        v_norms[face[0]] += f_norms[fi]
        v_norms[face[1]] += f_norms[fi]
        v_norms[face[2]] += f_norms[fi]

    # Normalize all vectors
    for i, f_norm in enumerate(f_norms):
        f_norms[i] = normalize(f_norm)
    for i, v_norm in enumerate(v_norms):
        v_norms[i] = normalize(v_norm)

    return f_norms, v_norms