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
print('Generating tetrahedral mesh...')
tet_mesh = TetrahedralMesh(tri_mesh)
# Output tetrahedral mesh

vtk_tetmesh(tet_mesh.mesh)

print('\033[92m Tetrahedral mesh generation succesful. \033[0m')

# Construct boundary surface of tetrahedral mesh.
print('Extracting surface mesh...')
surf_mesh = SurfaceMesh(tet_mesh.mesh)

print('Computing normals and curvatures...')
# Compute face and vertex normals.
surf_mesh.compute_normals()
# Compute principal curvatures and directions.
surf_mesh.compute_curvatures()
# Output curvature crossfield to .vtk file.
vtk_curvature(surf_mesh)

print('\033[92m Boundary computations succesful. \033[0m')

# Construct 3D frame field as an array of (U, V, W) frames.
# This field is parallel to the tet list (i.e. each tet has a frame).
print('Initializing framefield...')
# Compute the tetrahedral one-rings of the mesh.
tet_mesh.compute_onerings(surf_mesh)
# Construct frame field.
tet_mesh.init_framefield(surf_mesh)

print('\033[92m Frame initialization succesful. \033[0m')

# Optimize 3D frame field by L-BFGS minimization.
print('Optimizing framefield...')
tet_mesh.optimize_framefield()

print('\033[92m Framefield optimization succesful. \033[0m')

# Output frame field to .vtk file.
vtk_framefield(tet_mesh.frames)

# Compute the pair matchings.
print("Computing all pair matchings...")
tet_mesh.compute_matchings()

# Determine the singular edges of the framefield.      
print("Computing singular graph...")
# Find the singular graph.
singular_edges = singular_graph(tet_mesh)
# Output singular graph to .vtk file.
vtk_lines(tet_mesh.mesh.points, singular_edges)



