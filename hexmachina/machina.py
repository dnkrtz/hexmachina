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
from visual import *
from utils import *

print('Reading triangle mesh...')
tri_mesh = trimesh.load_mesh('../io/cylinder.stl')

# Instantiate tetrahedral mesh
print('Generating tetrahedral mesh...')
tet_mesh = TetrahedralMesh(tri_mesh)
# Output tetrahedral mesh

vtk_tetmesh(tet_mesh.mesh)

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

# Compute the tetrahedral one-rings of the mesh.
print('Computing tetrahedral one-rings...')
tet_mesh.compute_onerings(surf_mesh)

# Construct 3D frame field as an array of (U, V, W) frames.
# This field is parallel to the tet list (i.e. each tet has a frame).
print('Initializing framefield...')
tet_mesh.init_framefield(surf_mesh)

print(tet_mesh.frames[0].uvw)

# Optimize 3D frame field by L-BFGS minimization.
print('Optimizing framefield...')
tet_mesh.optimize_framefield()

print(tet_mesh.frames[0].uvw)

# Output frame field to .vtk file.
vtk_framefield(tet_mesh.frames)

# Determine the singular edges of the framefield.      
print("Computing singular graph...")
singular_edges = singular_graph(tet_mesh)
# Output singular graph to .vtk file.
vtk_lines(tet_mesh.mesh.points, singular_edges)



