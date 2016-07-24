#!/usr/bin/env python3

'''
    File: visual.py
    License: MIT
    Author: Aidan Kurtz
    Created: 09/07/2016
    Python Version: 3.5
    ========================
    For now, this is the main script.
'''

from normals import compute_normals
from curvature import compute_curvatures
from visual import plot_vectors

import meshpy.tet as TetGen
import numpy as np
import trimesh


tri_mesh = trimesh.load_mesh('../tests/data/icosphere.stl')

# Define MeshPy options
opt = TetGen.Options(switches='pq', edgesout=True, facesout=True, neighout=True)
# Generate mesh
mesh_info = TetGen.MeshInfo()
mesh_info.set_points(tri_mesh.vertices)
faces = [list(map(lambda x: int(x), i)) for i in tri_mesh.faces]
mesh_info.set_facets(faces)
tet_mesh = TetGen.build(mesh_info, opt, max_volume=10)
# Output tetrahedral mesh
tet_mesh.write_vtk("../tests/data/test.vtk")


# Extract surface triangle mesh from volumetric tetrahedral mesh.
surf_faces = []
surf_vertices = []
global2surf = dict()

for ti, tet in enumerate(tet_mesh.elements):
    # Within the neighbors list of each tet, position 'i' contains the index of
    # the face adjacent to the tet at face opposing vertex 'i'. A value of -1 
    # indicates that the face has no neighbor (i.e. it's a boundary face).
    # So, let's find all such occurences in the current tet.
    outliers = [i for i, x in enumerate(tet_mesh.neighbors[ti]) if x == -1]
    for bound_id in outliers:
        v_indices = list(range(4))
        v_indices.remove(bound_id)
        # If vertex 1 or vertex 3 are not part of the face, the order of the 
        # vertices must be reversed to obtain an outward facing triangle. Refer
        # to TetGen documentation to see why that is.
        if 1 not in v_indices or 3 not in v_indices:
            v_indices.reverse()
        # Get the global vertex indices for the face
        face = [tet[i] for i in v_indices]
        
        # For each vertex on the surface
        for vi in face:
            # If currently not mapped
            if vi not in global2surf:
                # Keep track of global to surface
                global2surf[vi] = len(surf_vertices)
                # Append to the surface vertex list
                surf_vertices.append(np.array(tet_mesh.points[vi]))
        
        # Translate using the global 2 surface vertex indices map
        face = list(map(lambda f: global2surf[f], face))
        
        surf_faces.append(face)

# Compute surface and vertex normals.
f_norms, v_norms = compute_normals(surf_faces, surf_vertices)

plot_vectors(v_norms, surf_vertices)

# Compute principal curvatures.
# k1, k2, dir1, dir2 = compute_curvatures(np.array(surf_vertices), surf_faces, v_norms)