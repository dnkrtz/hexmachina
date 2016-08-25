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

import math
import numpy as np
import trimesh

from surfacemesh import SurfaceMesh
from tetmesh import TetrahedralMesh
from framefield import *
from utils import *
from optimization import *

print('Reading triangle mesh...')
tri_mesh = trimesh.load_mesh('../io/cylinder.stl')

# Instantiate tetrahedral mesh
print('Generating tetrahedral mesh...', end=" ")
tet_mesh = TetrahedralMesh(tri_mesh)
# Output tetrahedral mesh
vtk_tetmesh(tet_mesh.mesh, 'tet_mesh')
print_ok()

# Construct boundary surface of tetrahedral mesh.
print('Extracting surface mesh and curvatures/normals...', end=" ")
surf_mesh = SurfaceMesh(tet_mesh.mesh)
# Compute face and vertex normals.
surf_mesh.compute_normals()
# Compute principal curvatures and directions.
surf_mesh.compute_curvatures()
# Output curvature crossfield to .vtk file.
vtk_curvature(surf_mesh, 'curvature')
print_ok()

# Construct 3D frame field as an array of (U, V, W) frames.
# This field is parallel to the tet list (i.e. each tet has a frame).
print('Initializing framefield...', end=" ")
# Compute the tetrahedral one-rings of the mesh.
tet_mesh.compute_onerings(surf_mesh)
# Construct frame field.
tet_mesh.init_framefield(surf_mesh)
print_ok()

print(tet_mesh.mesh.voro_edge)

# # Optimize 3D frame field by L-BFGS minimization.
# print('Optimizing framefield...')
# tet_mesh.optimize_framefield()

# Output frame field to .vtk file.
vtk_framefield(tet_mesh.frames, 'field')

# Compute the pair matchings.
print("Computing all pair matchings...", end=" ")
compute_matchings(tet_mesh)
print_ok()

# Determine the singular edges of the framefield.      
print("Computing singular graph...", end=" ")
singular_graph(tet_mesh)
print_ok()





