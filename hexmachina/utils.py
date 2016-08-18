'''
    File: utils.py
    License: MIT
    Author: Aidan Kurtz
    Created: 09/07/2016
    Python Version: 3.5
    ========================
    This module contains some small functions.
'''

import numpy as np
from pyvtk import *

# Normalize a vector to unit length.
def normalize(vector):
    return vector / np.linalg.norm(vector)

def tet_centroid(tet_mesh, ti):
    return ( np.array(tet_mesh.points[tet_mesh.elements[ti][0]]) + 
             np.array(tet_mesh.points[tet_mesh.elements[ti][1]]) +
             np.array(tet_mesh.points[tet_mesh.elements[ti][2]]) + 
             np.array(tet_mesh.points[tet_mesh.elements[ti][3]]) ) / 4

# Write tetrahedral mesh as .vtk file for paraview.
def vtk_tetmesh(mesh):
    mesh.write_vtk("../io/vtk/tetmesh.vtk")

# Write 3D frame field as .vtk file for paraview.
def vtk_framefield(frames):
    points = []
    for frame in frames:
        points.append(frame.location + frame.uvw[:,0] / 2)
        points.append(frame.location - frame.uvw[:,0] / 2)
        points.append(frame.location + frame.uvw[:,1] / 2)
        points.append(frame.location - frame.uvw[:,1] / 2)
        points.append(frame.location + frame.uvw[:,2] / 2)
        points.append(frame.location - frame.uvw[:,2] / 2)
    lines = []
    line_colors = []
    for i in range(len(frames)):
        lines.append([6*i, 6*i + 1])
        line_colors.append(0)
        lines.append([6*i + 2, 6*i + 3])
        line_colors.append(1)
        lines.append([6*i + 4, 6*i + 5])
        line_colors.append(2)

    structure = UnstructuredGrid(points, line=lines)
    line_data = CellData(Scalars(line_colors, name='line_colors'))

    vtk = VtkData(structure, line_data, 'Volumetric frame-field')

    vtk.tofile('../io/vtk/field')

# Outputs the curvature cross-field as a .vtk file for paraview.
def vtk_curvature(surf_mesh):
    points = []
    for vi, vertex in enumerate(surf_mesh.vertices):
        points.append(vertex + surf_mesh.pdir1[vi] / 2)
        points.append(vertex - surf_mesh.pdir1[vi] / 2)
        points.append(vertex + surf_mesh.pdir2[vi] / 2)
        points.append(vertex - surf_mesh.pdir2[vi] / 2)
    lines = []
    for i in range(len(surf_mesh.vertices)):
        lines.append([4*i, 4*i + 1])
        lines.append([4*i + 2, 4*i + 3])

    vtk = VtkData(\
        UnstructuredGrid(points, line=lines),
        'Curvature cross-field')

    vtk.tofile('../io/vtk/curvature')