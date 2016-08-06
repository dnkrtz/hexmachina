#!/usr/bin/env python3
'''
    File: core.py
    License: MIT
    Author: Aidan Kurtz, Sebastien Dery
    Created: 09/07/2016
    Python Version: 3.5
    ========================
    For now, this is the main script.
'''

from normals import compute_normals
from curvature import compute_curvatures
from visual import plot_vectors, plot_mesh

from bidict import bidict
import meshpy.tet as TetGen
import numpy as np
import trimesh

tri_mesh = trimesh.load_mesh('../io/cylinder.stl')

# Define MeshPy options
opt = TetGen.Options(switches='pq', edgesout=True, facesout=True, neighout=True)
# Generate tetrahedral mesh
mesh_info = TetGen.MeshInfo()
mesh_info.set_points(tri_mesh.vertices)
faces = [list(map(lambda x: int(x), i)) for i in tri_mesh.faces]
mesh_info.set_facets(faces)
tet_mesh = TetGen.build(mesh_info, opt, max_volume=10)
# Output tetrahedral mesh
tet_mesh.write_vtk("../io/test.vtk")

# Extract surface triangle mesh from volumetric tetrahedral mesh.
surf_faces = []
surf_vertices = []
# Volume-to-surface index maps (bi-directional).
vertex_map, face_map = bidict(), bidict()
# Loop through all faces.
for fi, face in enumerate(tet_mesh.faces):
    # If face marker is 0, face is internal.
    if (tet_mesh.face_markers[fi] == 0):
        continue
    # Otherwise, face is at boundary.
    for vi in face:
        # If vertex is currently not mapped
        if vi not in vertex_map:
            # Keep track of volume-to-surface index
            vertex_map[vi] = len(surf_vertices)
            # Append to the surface vertex list
            surf_vertices.append(np.array(tet_mesh.points[vi]))
    
    # Store surface vertex indices.
    face = list(map(lambda f: vertex_map[f], face))
    face_map[fi] = len(surf_faces)
    surf_faces.append(face)

# Compute face and vertex normals.
f_norms, v_norms = compute_normals(surf_faces, surf_vertices)

# Compute principal curvatures and directions.
k1, k2, dir1, dir2 = compute_curvatures(surf_vertices, surf_faces, v_norms)

# Plot principal curvature.
plot_vectors(dir1, surf_vertices)

# Initialize 3D frame field as an array of (U, V, W) frames.
# This field is parallel to the tet list (i.e. each tet has a frame).
frames = np.zeros( (3, len(tet_mesh.elements)) )

