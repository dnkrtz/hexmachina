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

from smesh import SurfaceMesh
from framefield import init_framefield, singular_graph
from visual import *

import math
import meshpy.tet as TetGen
import numpy as np
import trimesh

tri_mesh = trimesh.load_mesh('../io/cylinder.stl')

# Define MeshPy options
opt = TetGen.Options(switches='pqnn', facesout=True, edgesout=True)
# Generate tetrahedral mesh
mesh_info = TetGen.MeshInfo()
mesh_info.set_points(tri_mesh.vertices)
faces = [list(map(lambda x: int(x), i)) for i in tri_mesh.faces]
mesh_info.set_facets(faces)
tet_mesh = TetGen.build(mesh_info, opt, max_volume=10)
# Output tetrahedral mesh
tet_mesh.write_vtk("../io/test.vtk")

# Construct boundary surface of tetrahedral mesh.
surf_mesh = SurfaceMesh(tet_mesh)

# Compute face and vertex normals.
surf_mesh.compute_normals()

# Compute principal curvatures and directions.
surf_mesh.compute_curvatures()

# Construct 3D frame field as an array of (U, V, W) frames.
# This field is parallel to the tet list (i.e. each tet has a frame).
frames = init_framefield(tet_mesh, surf_mesh)

# Determine the singular edges of the framefield.      
singular_graph(tet_mesh, surf_mesh, frames)

