'''
    File: tetmesh.py
    License: MIT
    Author: Aidan Kurtz
    Created: 09/07/2016
    Python Version: 3.5
    ========================
    This module includes both the tetrahedral mesh and
    3D frame field initialization/optimization.
'''


import itertools
import meshpy.tet
import numpy as np
from scipy import spatial, optimize

from transforms import *
from utils import *

class Frame(object):
    def __init__(self, uvw, location, is_boundary = False):
        self.uvw = uvw
        self.location = location
        self.is_boundary = is_boundary
        self.euler = np.zeros(3)

class TetrahedralMesh(object):
    def __init__(self, tri_mesh):
        mesh_info = meshpy.tet.MeshInfo()
        # Define MeshPy options.
        # @TODO Investigate why the -Y switch messes up surface extraction (facets?)
        opt = meshpy.tet.Options(switches='pqnn', facesout=True, edgesout=True)
        # Generate tetrahedral mesh.
        mesh_info.set_points(tri_mesh.vertices)
        faces = [list(map(lambda x: int(x), i)) for i in tri_mesh.faces]
        mesh_info.set_facets(faces)
        mesh_info = meshpy.tet.build(mesh_info, opt, max_volume=15)
        # Set instance variables.
        self.mesh = mesh_info
        self.one_rings = {}
        self.frames = []

    def compute_onerings(self, surf_mesh):
        # Compute the one ring of tets surrounding each internal edge.
        one_rings = {}
        for ei, edge in enumerate(self.mesh.edges):
            # Make sure this is an internal edge, skip if it isn't.
            if (edge[0] in surf_mesh.vertex_map and edge[1] in surf_mesh.vertex_map):
                continue
            # If it is, construct its one ring.
            one_ring = []
            finished = False
            one_ring.append(self.mesh.edge_adjacent_elements[ei])
            # Walk around the edge until we've closed the one ring.
            while not finished:
                finished = True
                for neigh_ti in self.mesh.neighbors[one_ring[-1]]:
                    # Make sure this neighbor is a viable pick.
                    if (neigh_ti == -1 or neigh_ti in one_ring):
                        continue
                    neighbor = self.mesh.elements[neigh_ti]
                    # Make sure this neighbor shares the edge.
                    if (edge[0] in neighbor and edge[1] in neighbor):
                        # Add it to the ring.
                        one_ring.append(neigh_ti)
                        finished = False
                        break
            # Store it in our ring dictionary (don't tell golem).
            self.one_rings[ei] = one_ring

    def init_framefield(self, surf_mesh):
        boundary_frames = []
        boundary_ids = {}
        # The frame field is initialized at the boundary,
        # based on the curvature cross-field and normals.
        for fi, face in enumerate(surf_mesh.faces):
            # Retrieve the tet this face belongs to.
            ti = self.mesh.adjacent_elements[surf_mesh.face_map.inv[fi]][0]
            tet = self.mesh.elements[ti]
            # Ignore faces which have 0 curvature.
            if np.isclose(surf_mesh.k1[face[0]], 0) and np.isclose(surf_mesh.k2[face[0]], 0):
                continue
            # @TODO(aidan) Find actual face values, not vertex values.
            uvw = np.hstack((np.vstack(surf_mesh.pdir1[face[0]]),
                            np.vstack(surf_mesh.pdir2[face[0]]),
                            np.vstack(surf_mesh.vertex_normals[face[0]])))
            boundary_frames.append(Frame(uvw, tet_centroid(self.mesh, ti)))
            boundary_ids[ti] = len(boundary_frames) - 1

        # Prepare a KDTree of boundary frame coords for quick spatial queries.
        tree = spatial.KDTree(np.vstack([frame.location for frame in boundary_frames]))

        # Now propagate the boundary frames throughout the tet mesh.
        # Each tet frame takes the value of its nearest boundary tet.
        for ti, tet in enumerate(self.mesh.elements):
            location = tet_centroid(self.mesh, ti)
            if ti in boundary_ids:
                self.frames.append(Frame(boundary_frames[boundary_ids[ti]].uvw, location, True))
            else:
                nearest_ti = tree.query(location)[1] # Find closest boundary frame
                self.frames.append(Frame(boundary_frames[nearest_ti].uvw, location, False))

    # Quantify adjacent frame smoothness.
    @staticmethod
    def pair_energy(F_s, F_t):
        # Approximate permutation for the transformation from s-t.
        P = F_t.T * F_s
        # Since our initialized framefield is orthogonal, we can easily quantify
        # closeness of the permutation to the chiral symmetry group G. The cost
        # function should drive each row/column to have a single non-zero value.
        E_st = 0
        for i in range(3):
            E_st += P[i,0]**2 * P[i,1]**2 + P[i,1]**2 * P[i,2]**2 + P[i,2]**2 * P[i,0]**2
            E_st += P[0,i]**2 * P[1,i]**2 + P[1,i]**2 * P[2,i]**2 + P[2,i]**2 * P[0,i]**2
        return E_st

    # Quantify adjacent frame smoothness derivative (for energy gradient).
    @staticmethod
    def pair_energy_deriv(F_s, F_t, dF_s, dF_t):
        # Approximate permutation and its derivative (chain rule).
        P = F_t.T * F_s
        dP = dF_t.T * F_s + F_t.T * dF_s
        # More chain rule in the energy function H(n).
        dE_st = 0
        for i in range(3):
            dE_st += (2 * dP[i,0] * P[i,0] * P[i,1]**2) + (2 * P[i,0]**2 * dP[i,1] * P[i,1]) + \
                     (2 * dP[i,1] * P[i,1] * P[i,2]**2) + (2 * P[i,1]**2 * dP[i,2] * P[i,2]) + \
                     (2 * dP[i,2] * P[i,2] * P[i,0]**2) + (2 * P[i,2]**2 * dP[i,0] * P[i,0])
            dE_st += (2 * dP[0,i] * P[0,i] * P[1,i]**2) + (2 * P[0,i]**2 * dP[1,i] * P[1,i]) + \
                     (2 * dP[1,i] * P[1,i] * P[2,i]**2) + (2 * P[1,i]**2 * dP[2,i] * P[2,i]) + \
                     (2 * dP[2,i] * P[2,i] * P[0,i]**2) + (2 * P[2,i]**2 * dP[0,i] * P[0,i])
        return dE_st


    # Function E to minimize via L-BFGS.
    def global_energy(self, euler_angles):

        E = 0
        # All internal edges.
        for ei, edge in enumerate(self.mesh.edges):
            if ei not in self.one_rings:
                continue
            # All combinations of s, t around the edges' one ring.
            for combo in itertools.combinations(self.one_rings[ei], 2):
                UVW = []
                for i in [combo[0], combo[1]]:
                    frame = self.frames[i]
                    R = convert_to_R(frame, euler_angles[3*i], euler_angles[3*i + 1], euler_angles[3*i + 2])
                    UVW.append(R)
                    
                E += self.pair_energy(UVW[0], UVW[1])
        
        return E

    def global_gradient(self, euler_angles):
        
        # Partial derivative wrt each euler angle
        Eg = np.zeros( (len(self.mesh.elements), 3) )

        # All internal edges.
        for ei, edge in enumerate(self.mesh.edges):
            if ei not in self.one_rings:
                continue
            # All combinations of s, t around the edges' one ring.
            for combo in itertools.combinations(self.one_rings[ei], 2):
                UVW = []
                dUVW = []
                for i in [combo[0], combo[1]]:
                    frame = self.frames[i]
                    R = convert_to_R(frame, euler_angles[3*i], euler_angles[3*i + 1], euler_angles[3*i + 2])
                    UVW.append(R)
                    # Partial derivative wrt each angle.
                    dR = convert_to_dR(frame, euler_angles[3*i], euler_angles[3*i + 1], euler_angles[3*i + 2])
                    dUVW.append(dR)

                for i in range(3):
                    Eg[combo[0],i] += self.pair_energy_deriv(UVW[0], UVW[1], dUVW[0][i], np.zeros((3,3)) )
                    Eg[combo[1],i] += self.pair_energy_deriv(UVW[0], UVW[1], np.zeros((3,3)), dUVW[1][i])

        return Eg.flatten()

    # Optimize the framefield.
    def optimize_framefield(self):

        # Define all frames in terms of euler angles.
        euler_angles = np.zeros( (len(self.mesh.elements), 3) )
        
        for ti, tet in enumerate(self.mesh.elements):
            if self.frames[ti].is_boundary:
                continue
            else:
                R = self.frames[ti].uvw
                euler_angles[ti,:] = convert_to_euler(R)

        euler_angles = euler_angles.flatten()

        # @TODO Compute gradient and pass it here, numerical approximation is super expensive...
        opti = optimize.minimize(self.global_energy, euler_angles, method='L-BFGS-B', jac = self.global_gradient,
                                 options={'ftol': 1e-4, 'maxiter': 100, 'disp': True}).x

        # Once optimization is complete, save results
        for fi, frame in enumerate(self.frames):
            # Make sure to store as ndarray, not matrix. Else pyvtk goes bonkers.
            frame.uvw = convert_to_R(frame, opti[3 * fi], opti[3 * fi + 1], opti[3 * fi + 2]).getA()