"""
A short and sweet matrix library.

"""

import math
import random
import numpy as np
from numpy import norm
from matrix import Matrix

__version__ = "0.1"

def get_vertices_from_faces(v):
    ring = list(set(np.concatenate([vert for ind, vert in surrounding_faces[v]])))
    ring.remove(v)
    return ring

def tensor_curvature(vertices, normals):
    max_curv = np.zeros((3, len(vertices)))
    min_curv = np.zeros((3, len(vertices)))

    for vi, v in enumerate(vertices):
        position = np.array(tet_mesh.points[v])
        # Vertex normal as a matrix
        n0 = np.array(normals[v]) / norm(normals[v])
        Nvi = Matrix([[n0[0]], [n0[1]], [n0[2]]])

        # Unsorted ring
        ring = get_vertices_from_faces(v)

        # Calculate face weightings, wij
        n = len(ring)
        wij = np.zeros((n,))
        for j in range(n):
            vec0 = tet_mesh.points[int(ring[(j+(n-1))%n])] - position
            vec1 = tet_mesh.points[int(ring[j])] - position
            vec2 = tet_mesh.points[int(ring[(j+1)%n])] - position
            # Assumes closed manifold
            # TODO: handle boundaries
            wij[j] = 0.5 * (length(cross(vec0, vec1)) + length(cross(vec1, vec2)))

        # Sum
        wijSum = wij.sum()

        """
        Princeton curvature

        # Calculate Mvi
        I = Matrix.identity(3)
        Mvi = Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

        curvature = np.zeros((len(ring),))
        Mvi = np.zeros((3, 3))

        for i, j in enumerate(ring):
            n1 = np.array(v_norms[j])
            vec = np.array(position) - tet_mesh.points[int(j)]
            # edgeAsMatrix = Matrix([vec[0], vec[1], vec[2]])
            curvature[i] = dot((n0 - n1), vec) / (length(vec) ** 2)
            Mvi += curvature[i] * transpose(np.matrix(vec)) * vec
        """

        # Calculate Mvi
        I = Matrix.identity(3)
        Mvi = Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

        for i, j in enumerate(ring):
            vec = tet_mesh.points[int(j)] - position
            edgeAsMatrix = Matrix([[vec[0]], [vec[1]], [vec[2]]])
            Tij = (I - (Nvi * Nvi.transpose())) * edgeAsMatrix
            Tij *= (1 / Tij.getFrobeniusNorm())
            kij = (Nvi.transpose() * 2 * edgeAsMatrix).getEntry(0, 0) / (length(vec) ** 2)
            Mvi += Tij.multiply(Tij.transpose()).scalarMultiply((wij[i] / wijSum) * kij)

        # Get eigenvalues and eigenvectors for Mvi
        # maxc, minc = givens_rotation(Mvi._array)
        evals, evecs = eig(Mvi._array)

        # If zero-matrix
        if (np.array(evecs) == np.zeros((3, 3))).all():
            max_curv.append(np.zeros((3,)))
            min_curv.append(np.zeros((3,)))
            continue

        evecs = [np.array((evecs[0,y], evecs[1,y], evecs[2,y])) for y in range(3)]
        for e in evecs:
            e /= norm(e)
        evecs = [evecs[y] * evals[y] for y in range(3)]
        sortv = list(zip([abs(i) for i in evals], evals, evecs))
        sortv.sort(key=lambda x: [x[0], x[1]])

        maxc = sortv[1][-1] / norm(sortv[1][-1])
        minc = sortv[2][-1] / norm(sortv[2][-1])

        max_curv.append(maxc)
        min_curv.append(minc)
