"""

"""

import numpy as np
from triangle import is_facing
from matrix import Matrix
from frame import Frame
from numpy import linalg, transpose, multiply
from numpy import vstack, dot, cross
from numpy.linalg import eig, norm

__version__ = "0.1"


def length(v):
    return norm(v)


class Tetmesh(object):
    """A simple tetrahedral mesh class."""

    def __init__(self, mesh):
        """Constructor (defaults to empty 2D array)."""
        self._mesh = mesh
        self._surface_mesh = None
        self._frame_field = None

    def __str__(self):
        """Return 2D array as string."""
        return ""

    def extract_surface_mesh(self):
        '''Compute and return the surface mesh.

        tet_mesh: tetrahedron mesh from tetgen build function
        returns:  list, (n,1)
        '''
        surface_faces = []
        for i, tet in enumerate(self._mesh.elements):
            neighbors = list(self._mesh.neighbors[i])
            if -1 in neighbors:
                non_surface_vtx = tet[neighbors.index(-1)]
                tet_cpy = tet.copy()
                tet_cpy.remove(non_surface_vtx)
                surface_faces.append(tet_cpy)

        # Get vertex indices
        dedup_vertices = set(list(np.concatenate(surface_faces)))

        # Compute neighbouring faces (ring)
        surrounding_faces = {}
        for i, v in enumerate(dedup_vertices):
            surrounding_faces[v] = list(filter(lambda f: f[1][0] == v or f[1][1] == v or f[1][2] == v, zip(range(0, len(surface_faces)), surface_faces)))

        self._surface_mesh = {
            "vertices": dedup_vertices,
            "neighbours": surrounding_faces,
            "faces": surface_faces,
            "normals": [self.face_normal(face) for face in surface_faces],
            "center": [self.face_center(face) for face in surface_faces]
        }
        return self._surface_mesh

    def surface_vertex_ring(self, v):
        ring = list(set(np.concatenate([vert for ind, vert in self._surface_mesh["neighbours"][v]])))
        ring.remove(v)
        return ring

    def face_normal(self, face):
        '''
        '''
        return np.cross(np.array(self._mesh.points[face[0]]) - np.array(self._mesh.points[face[1]]),
                        np.array(self._mesh.points[face[2]]) - np.array(self._mesh.points[face[1]]))

    def face_center(self, face):
        '''
        '''
        return (np.array(self._mesh.points[face[0]]) + np.array(self._mesh.points[face[1]]) + np.array(self._mesh.points[face[2]])) / 3

    def correct_normal_by_cullfacing(self):
        '''
        '''
        center = vstack(self._mesh.points).mean()
        triangles = [np.array([self._mesh.points[f[0]], self._mesh.points[f[1]], self._mesh.points[f[2]]]) for f in surface_faces]
        facing = is_facing(triangles, self._surface_mesh["normals"], center)
        for i, verdict in enumerate(facing):
            if verdict:
                self._surface_mesh["normals"][i] = -self._surface_mesh["normals"][i]

    def compute_surface_tensor_curvature(self):
        points = self._mesh.points
        vertices = self._surface_mesh["vertices"]
        normals = self._surface_mesh["normals"]

        max_curv = np.zeros((3, len(vertices)))
        min_curv = np.zeros((3, len(vertices)))

        for vi, v in enumerate(vertices):
            position = np.array(points[v])
            # Vertex normal as a matrix
            n0 = np.array(normals[v]) / norm(normals[v])
            Nvi = Matrix([[n0[0]], [n0[1]], [n0[2]]])

            # Unsorted ring
            ring = self.surface_vertex_ring(v)

            # Calculate face weightings, wij
            n = len(ring)
            wij = np.zeros((n,))
            for j in range(n):
                vec0 = points[int(ring[(j + (n - 1)) % n])] - position
                vec1 = points[int(ring[j])] - position
                vec2 = points[int(ring[(j + 1) % n])] - position
                # Assumes closed manifold
                # TODO: handle boundaries
                wij[j] = 0.5 * (length(cross(vec0, vec1)) + length(cross(vec1, vec2)))

            # Sum
            wijSum = wij.sum()

            # Calculate Mvi
            I = Matrix.identity(3)
            Mvi = Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

            for i, j in enumerate(ring):
                vec = points[int(j)] - position
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
                max_curv[vi] = np.zeros((3,))
                min_curv[vi] = np.zeros((3,))
                continue

            evecs = [np.array((evecs[0, y], evecs[1, y], evecs[2, y])) for y in range(3)]
            for e in evecs:
                e /= norm(e)
            evecs = [evecs[y] * evals[y] for y in range(3)]
            sortv = list(zip([abs(i) for i in evals], evals, evecs))
            sortv.sort(key=lambda x: [x[0], x[1]])

            max_curv[vi] = sortv[1][-1] / norm(sortv[1][-1])
            min_curv[vi] = sortv[2][-1] / norm(sortv[2][-1])
