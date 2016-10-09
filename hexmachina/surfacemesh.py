'''
    File: surfacemesh.py
    License: MIT
    Author: Aidan Kurtz
    Created: 09/07/2016
    Python Version: 3.5
    ========================
    This module involves computations related to the boundary surface of the
    tetrahedral mesh. The boundary surface is extracted as a triangle mesh, and
    normals/curvatures are computed.
'''

from utils import normalize
from bidict import bidict
import math
import meshpy.tet
import numpy as np

class SurfaceMesh(object):

    def __init__(self, tet_mesh):
        """Extract surface triangle mesh from volumetric tetrahedral mesh."""
        # Vertices as array of coordinates.
        # Faces as triplets of vertex indices.
        self.vertices = []
        self.faces = []
        # Bidirectional volume-to-surface index maps.
        self.vertex_map = bidict()
        self.face_map = bidict()

        # Loop through all faces.
        for fi, face in enumerate(tet_mesh.faces):
            # If face marker is 0, face is internal.
            if (tet_mesh.face_markers[fi] == 0):
                continue
            # Otherwise, face is at boundary.
            for vi in face:
                # If vertex is currently not mapped
                if vi not in self.vertex_map:
                    # Keep track of volume-to-surface index
                    self.vertex_map[vi] = len(self.vertices)
                    # Append to the surface vertex list
                    self.vertices.append(np.array(tet_mesh.points[vi]))
            
            # Store surface vertex indices.
            face = [ self.vertex_map[vi] for vi in face ]
            self.face_map[fi] = len(self.faces)
            self.faces.append(face)

        # Normal vectors.
        self.vertex_normals = [ np.zeros(3) for _ in range(len(self.vertices)) ]
        self.face_normals = [ np.zeros(3) for _ in range(len(self.faces)) ]
        # Principal curvatures and directions
        self.k1 = np.zeros( (len(self.vertices),1) )
        self.k2 = np.zeros( (len(self.vertices),1) )
        self.pdir1 = np.zeros( (len(self.vertices),3) )
        self.pdir2 = np.zeros( (len(self.vertices),3) )

    def compute_normals(self):
        """Compute vertex and face normals of the triangular mesh."""

        # Compute face normals, easy as cake.
        for fi, face in enumerate(self.faces):
            self.face_normals[fi] = np.cross(self.vertices[face[2]] - self.vertices[face[0]],
                                             self.vertices[face[1]] - self.vertices[face[0]])
        
        # Next, compute the vertex normals.
        for fi, face in enumerate(self.faces):
            self.vertex_normals[face[0]] += self.face_normals[fi]
            self.vertex_normals[face[1]] += self.face_normals[fi]
            self.vertex_normals[face[2]] += self.face_normals[fi]

        # Normalize all vectors
        for i, f_norm in enumerate(self.face_normals):
            self.face_normals[i] = normalize(f_norm)
        for i, v_norm in enumerate(self.vertex_normals):
            self.vertex_normals[i] = normalize(v_norm)

    """"
    Below is a Python implementation of Szymon Rusinkiewicz' paper
    "Estimating Curvatures and Their Derivatives on Triangle Meshes".
    http://gfx.cs.princeton.edu/pubs/_2004_ECA/curvpaper.pdf 
    """

    @staticmethod
    def rotate_coord_sys(old_u, old_v, new_norm):
        """Rotate a coordinate system to be perpendicular to the given normal."""
        new_u = old_u
        new_v = old_v
        old_norm = np.cross(old_u, old_v)
        # Project old normal onto new normal
        ndot = np.dot(old_norm, new_norm)
        # If projection is leq to -1, simply reverse
        if ndot <= -1:
            new_u = -new_u
            new_v = -new_v
            return new_u, new_v
        # Otherwise, compute new normal
        perp_old = new_norm - ndot * old_norm
        dperp = (old_norm + new_norm) / (1 + ndot)
        new_u -= dperp * np.dot(new_u, perp_old)
        new_v -= dperp * np.dot(new_v, perp_old)
        return new_u, new_v

    @classmethod
    def project_curvature(cls, old_u, old_v, old_ku, old_kuv, old_kv, new_u, new_v):
        """Reproject curvature tensor from the basis spanned by old uv to the new uv basis."""
        old_normal = np.cross(old_u, old_v)
        # Rotate new coord system to be normal to old, for reprojection
        r_new_u, r_new_v = cls.rotate_coord_sys(new_u, new_v, old_normal)
        u1 = np.dot(r_new_u, old_u)
        v1 = np.dot(r_new_u, old_v)
        u2 = np.dot(r_new_v, old_u)
        v2 = np.dot(r_new_v, old_v)
        new_ku  = old_ku * u1*u1 + old_kuv * (2 * u1*v1) + old_kv * v1*v1
        new_kuv = old_ku * u1*u2 + old_kuv * (u1*v2 + u2*v1) + old_kv * v1*v2
        new_kv  = old_ku * u2*u2 + old_kuv * (2 * u2*v2) + old_kv * v2*v2

        return new_ku, new_kuv, new_kv

    @classmethod
    def diagonalize_curvature(cls, old_u, old_v, ku, kuv, kv, new_norm):
        """Given a curvature tensor, diagonalize to find principal directions and curvatures."""
        # Rotate old coord system to be normal to new.
        r_old_u, r_old_v = cls.rotate_coord_sys(old_u, old_v, new_norm)
        c = 1
        s = 0
        tt = 0
        if not math.isclose(kuv, 0):
            # Jacobi rotation to diagonalize.
            h = 0.5 * (kv - ku) / kuv
            if h < 0:
                tt = 1 / (h - math.sqrt(1 + h*h))
            else:
                tt = 1 / (h + math.sqrt(1 + h*h))
            c = 1 / math.sqrt(1 + tt*tt)
            s = tt * c
        # Compute principal curvatures.
        k1 = ku - tt * kuv
        k2 = kv + tt * kuv

        # Compute principal directions.
        if abs(k1) >= abs(k2):
            pdir1 = c * r_old_u - s * r_old_v
        else:
            k1, k2 = k2, k1 # swap
            pdir1 = s * r_old_u + c * r_old_v
        pdir2 = np.cross(new_norm, pdir1)
        # Return all the things.
        return pdir1, pdir2, k1, k2

    def compute_pointareas(self):
        """Compute the area "belonging" to each vertex or each corner
        of a triangle (defined as Voronoi area restricted to the 1-ring of
        a vertex, or to the triangle)."""

        cornerareas = np.zeros( (len(self.faces), 3) )
        pointareas = np.zeros( (len(self.vertices), 1) )

        for i, face in enumerate(self.faces):
            # Face edges
            e = np.array([ self.vertices[face[2]] - self.vertices[face[1]],
                           self.vertices[face[0]] - self.vertices[face[2]],
                           self.vertices[face[1]] - self.vertices[face[0]] ])
            # Compute edge and corner weights
            area = 0.5 * np.linalg.norm(np.cross(e[0], e[1]))
            l2 = [ np.linalg.norm(e[0]) ** 2, 
                np.linalg.norm(e[1]) ** 2,
                np.linalg.norm(e[2]) ** 2 ]
            ew = [ l2[0] * (l2[1] + l2[2] - l2[0]),
                l2[1] * (l2[2] + l2[0] - l2[1]),
                l2[2] * (l2[0] + l2[1] - l2[2]) ]
            # Case by case based on edge weight
            if ew[0] <= 0:
                cornerareas[i,1] = -0.25 * l2[2] * area / np.dot(e[0], e[2])
                cornerareas[i,2] = -0.25 * l2[1] * area / np.dot(e[0], e[1])
                cornerareas[i,0] = area - cornerareas[i,1] - cornerareas[i,2]
            elif ew[1] <= 0:
                cornerareas[i,2] = -0.25 * l2[0] * area / np.dot(e[1], e[0])
                cornerareas[i,0] = -0.25 * l2[2] * area / np.dot(e[1], e[2])
                cornerareas[i,1] = area - cornerareas[i,2] - cornerareas[i,0]
            elif ew[2] <= 0:
                cornerareas[i,0] = -0.25 * l2[1] * area / np.dot(e[2], e[1])
                cornerareas[i,1] = -0.25 * l2[0] * area / np.dot(e[2], e[0])
                cornerareas[i,2] = area - cornerareas[i,0] - cornerareas[i,1]
            else:
                ewscale = 0.5 * area / (ew[0] + ew[1] + ew[2])
                for j in range(3):
                    cornerareas[i,j] = ewscale * (ew[(j+1)%3] + ew[(j+2)%3])
        
            pointareas[face[0]] += cornerareas[i,0]
            pointareas[face[1]] += cornerareas[i,1]
            pointareas[face[2]] += cornerareas[i,2]

        return pointareas, cornerareas

    def compute_curvatures(self):
        """Given the faces, vertices and vertex normals.
        Compute principal curvatures and directions."""
        
        # Since we diagonalize the matrix later, this isn't an object variable.
        curv12 = np.zeros( (len(self.vertices),1) )

        # Compute pointareas
        pointareas, cornerareas = self.compute_pointareas()
        
        # Set up an initial coordinate system per-vertex
        for i, face in enumerate(self.faces):
            self.pdir1[face[0],:] = self.vertices[face[1]] - self.vertices[face[0]]
            self.pdir1[face[1],:] = self.vertices[face[2]] - self.vertices[face[1]]
            self.pdir1[face[2],:] = self.vertices[face[0]] - self.vertices[face[2]]

        for i, vertex in enumerate(self.vertices):
            self.pdir1[i,:] = normalize(np.cross(self.pdir1[i,:], self.vertex_normals[i]))
            self.pdir2[i,:] = np.cross(self.vertex_normals[i], self.pdir1[i,:])
            
        # Compute curvature per-face
        for i, face in enumerate(self.faces):
            # Face edges
            e = np.array([ self.vertices[face[2]] - self.vertices[face[1]],
                           self.vertices[face[0]] - self.vertices[face[2]],
                           self.vertices[face[1]] - self.vertices[face[0]] ])
            # N-T-B coordinate system per-face
            t = normalize(e[0])
            n = np.cross(e[0], e[1])
            b = normalize(np.cross(n, t))

            # Estimate curvature based on variation of
            # normals along edges.
            m = np.zeros( (3,1) )
            w = np.zeros( (3,3) )
            for j in range(3):
                u =  np.dot(e[j], t)
                v =  np.dot(e[j], b)
                w[0,0] += u*u
                w[0,1] += u*v
                w[2,2] += v*v
                dn = self.vertex_normals[face[(j+2)%3]] - \
                     self.vertex_normals[face[(j+1)%3]]
                dnu = np.dot(dn, t)
                dnv = np.dot(dn, b)
                m[0] += dnu*u
                m[1] += dnu*v + dnv*u
                m[2] += dnv*v
            w[1,1] = w[0,0] + w[2,2]
            w[1,2] = w[0,1]
            # @TODO(aidan) Do we need this? I think so.
            w[2,1] = w[1,2]
            w[1,0] = w[0,1]

            # Least squares solution.
            x = np.linalg.lstsq(w,m)[0]

            # Push it back out to the vertices.
            for j in range(3):
                vj = face[j]
                c1, c12, c2 = self.project_curvature(t, b, x[0], x[1], x[2],
                                                     self.pdir1[vj], self.pdir2[vj])
                weight = cornerareas[i,j] / pointareas[vj]
                self.k1[vj] += weight * c1
                curv12[vj] += weight * c12
                self.k2[vj] += weight * c2
            
        # Compute principal directions and curvatures at each vertex.
        for i, vertex in enumerate(self.vertices):
            self.pdir1[i,:], self.pdir2[i,:], self.k1[i], self.k2[i] = \
                self.diagonalize_curvature(self.pdir1[i,:], self.pdir2[i,:], self.k1[i], 
                                           curv12[i], self.k2[i], self.vertex_normals[i])