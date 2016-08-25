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

def print_ok():
    print('\033[92m OK. \033[0m')

# Normalize a vector to unit length.
def normalize(vector):
    return vector / np.linalg.norm(vector)

def tet_centroid(tet_mesh, ti):
    return ( np.array(tet_mesh.points[tet_mesh.elements[ti][0]]) + 
             np.array(tet_mesh.points[tet_mesh.elements[ti][1]]) +
             np.array(tet_mesh.points[tet_mesh.elements[ti][2]]) + 
             np.array(tet_mesh.points[tet_mesh.elements[ti][3]]) ) / 4

# Shape and edge are indices into mesh.points.
def is_on_edge(shape, edge):
    if ei[0] in shape and ei[1] in shape:
        return True
    return False

# Write tetrahedral mesh as .vtk file for paraview.
def vtk_tetmesh(mesh, filename):
    mesh.write_vtk("../io/vtk/%s.vtk" % filename)

# Write 3D frame field as .vtk file for paraview.
def vtk_framefield(frames, filename):
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
        lines.append([6*i + 2, 6*i + 3])
        lines.append([6*i + 4, 6*i + 5])

    structure = PolyData(points=points, lines=lines)
    vtk = VtkData(structure, 'Volumetric frame-field')

    vtk.tofile('../io/vtk/%s' % filename)

# Outputs the curvature cross-field as a .vtk file for paraview.
def vtk_curvature(surf_mesh, filename):
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


    structure = PolyData(points=points, lines=lines)

    vtk = VtkData(structure, 'Curvature cross-field')

    vtk.tofile('../io/vtk/%s' % filename)

# Outputs a set of lines as a .vtk file for paraview.
def vtk_lines(points, lines, filename):
    vtk = VtkData(\
          UnstructuredGrid(points, line=lines),
          'Singular graph')

    vtk.tofile('../io/vtk/%s' % filename)
