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

import sys
import math
import numpy as np
import trimesh

from extraction import *
from machina import HexMachina
from optimization import *
from parametrization import *
from utils import *

print("""\
_______________________________________________________
                                   _     _             
  /\  /\_____  __ /\/\   __ _  ___| |__ (_)_ __   __ _ 
 / /_/ / _ \ \/ //    \ / _` |/ __| '_ \| | '_ \ / _` |
/ __  /  __/>  </ /\/\ \ (_| | (__| | | | | | | | (_| |
\/ /_/ \___/_/\_\/    \/\__,_|\___|_| |_|_|_| |_|\__,_|                                       
_______________________________________________________
""")

print('Reading triangle mesh...', end=" ")
sys.stdout.flush()
tri_mesh = trimesh.load_mesh('../io/cylinder.stl')
say_ok()

# Instantiate tetrahedral mesh
print('Generating tetrahedral mesh...', end=" ")
sys.stdout.flush()
machina = HexMachina(tri_mesh, max_vol = 5)
# Output tetrahedral mesh
vtk_tetmesh(machina.tet_mesh, 'tet_mesh')
say_ok()

# Construct boundary surface of tetrahedral mesh.
print('Computing surface curvatures/normals...', end=" ")
sys.stdout.flush()
# Compute face and vertex normals.
machina.surf_mesh.compute_normals()
# Compute principal curvatures and directions.
machina.surf_mesh.compute_curvatures()
# Output curvature crossfield to .vtk file.
vtk_curvature(machina.surf_mesh, 'curvature')
say_ok()

# Compute the tetrahedral one-rings of the mesh.
print('Extracting dual voronoi...', end=" ")
sys.stdout.flush()
machina.compute_dual()
say_ok()

# Construct 3D frame field as an array of (U, V, W) frames.
# This field is parallel to the tet list (i.e. each tet has a frame).
print('Initializing framefield...', end=" ")
sys.stdout.flush()
# Construct frame field.
machina.init_framefield()
say_ok()

# # Optimize 3D frame field by L-BFGS minimization.
# print('Optimizing framefield...')
# machina.optimize_framefield()

# Output frame field to .vtk file.
vtk_framefield(machina.frames, 'field')

# # Compute the pair matchings.
# print("Computing all pair matchings...", end=" ")
# sys.stdout.flush()
# compute_matchings(machina)
# say_ok()

# Determine the singular edges of the framefield.      
print("Computing singular graph...", end=" ")
sys.stdout.flush()
singular_vertices = singular_graph(machina)[2]
say_ok()

print("Parametrizing volume...")
uvw_map = parametrize_volume(machina, singular_vertices, 2.0)
say_ok()

print("Extracting hexahedrons...", end=" ")
iso_pts = extract_isolines(machina, uvw_map)
say_ok()





