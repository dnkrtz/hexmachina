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


def say_ok():
    print('\033[92m OK. \033[0m')


def normalize(vector):
    """Normalize a vector to unit length."""
    return vector / np.linalg.norm(vector)


def tet_centroid(tet_mesh, ti):
    """Compute the centroid of tetrahedron ti."""
    return ( np.array(tet_mesh.points[tet_mesh.elements[ti][0]]) + 
             np.array(tet_mesh.points[tet_mesh.elements[ti][1]]) +
             np.array(tet_mesh.points[tet_mesh.elements[ti][2]]) + 
             np.array(tet_mesh.points[tet_mesh.elements[ti][3]]) ) / 4


def tet_volume(tet_mesh, ti):
    """Compute the volume of tetrahedron ti."""
    a = np.array(tet_mesh.points[tet_mesh.elements[ti][0]])
    b = np.array(tet_mesh.points[tet_mesh.elements[ti][1]])
    c = np.array(tet_mesh.points[tet_mesh.elements[ti][2]])
    d = np.array(tet_mesh.points[tet_mesh.elements[ti][3]])
    vol = np.matrix([ [a[0], b[0], c[0], d[0]],
                      [a[1], b[1], c[1], d[1]],
                      [a[2], b[2], c[2], d[2]],
                      [1.0, 1.0, 1.0, 1.0] ])
                    
    return (np.linalg.det(vol) / 6)


def is_on_edge(shape, edge):
    """Shape and edge are lists of indices into mesh.points.
    Returns true if the edge is on the shape (triangle)."""
    if ei[0] in shape and ei[1] in shape:
        return True
    return False


def vtk_tetmesh(mesh, filename):
    """Save tetrahedral mesh as a .vtk file for paraview."""
    mesh.write_vtk("../data/vtk/%s.vtk" % filename)


def vtk_framefield(frames, filename):
    """Save 3D frame field as a .vtk file for paraview."""
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
    vtk.tofile('../data/vtk/%s' % filename)


def vtk_curvature(surf_mesh, filename):
    """Save curvature cross-field as a .vtk file for paraview."""
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
    vtk.tofile('../data/vtk/%s' % filename)


def vtk_lines(points, lines, filename):
    """Save set of lines as a .vtk file for paraview."""
    vtk = VtkData(\
          UnstructuredGrid(points, line=lines),
          'Singular graph')
    vtk.tofile('../data/vtk/%s' % filename)


def vtk_points(points, filename):
    """Save set of points as a .vtk file for paraview."""
    vtk = VtkData(\
          PolyData(points, vertices = [ range(len(points)) ]),
          'Points')
    vtk.tofile('../data/vtk/%s' % filename)