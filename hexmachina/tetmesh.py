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
                    neighbor = self.mesh.elements[neigh_ti]
                    # Make sure this neighbor is a viable pick.
                    if (neigh_ti == -1 or neigh_ti in one_ring):
                        continue
                    # Make sure this neighbor shares the edge.
                    if (edge[0] in neighbor and edge[1] in neighbor ):
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

    # Quantify closeness of the matching to the chiral symmetry group.
    @staticmethod
    def pair_energy(UVW_s, UVW_t):
        # Approximate permutation for the transformation from s-t.
        P = UVW_t.T * UVW_s
        # Since our initialized framefield is orthogonal, we can easily quantify
        # closeness of the permutation to the chiral symmetry group G. The cost
        # function should drive each row/column to have a single non-zero value.
        E_st = 0
        for i in range(3):
            E_st += P[i,0]**2 * P[i,1]**2 + P[i,1]**2 * P[i,2]**2 + P[i,2]**2 * P[i,0]**2
            E_st += P[0,i]**2 * P[1,i]**2 + P[1,i]**2 * P[2,i]**2 + P[2,i]**2 * P[0,i]**2
        return E_st

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
                for i in range(2):
                    frame = self.frames[combo[i]]
                    if frame.is_boundary:
                        theta = euler_angles[3 * combo[i] + 1]
                        UVW.append(np.array([ np.cos(theta) * frame.uvw[:,0] + \
                                              np.sin(theta) * frame.uvw[:,1],
                                              - np.sin(theta) * frame.uvw[:,0] + \
                                              np.cos(theta) * frame.uvw[:,1],
                                              frame.uvw[:,2] ]))
                    else:
                        R = convert_to_R(euler_angles[3 * combo[i]], euler_angles[3 * combo[i] + 1], euler_angles[3 * combo[i] + 2])
                        UVW.append(R)
                    
                E += self.pair_energy(UVW[0], UVW[1])
        
        return E

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

        # @TODO Compute gradient and pass it here, numerical approximation is super expensive...
        opti = optimize.minimize(self.global_energy, euler_angles, method='L-BFGS-B', jac = False,
                                 options={'ftol': 1e-2, 'maxiter': 2, 'disp': True}).x

        # Once optimization is complete, save results
        for fi, frame in enumerate(self.frames):
            if frame.is_boundary:
                theta = opti[3 * fi + 1]
                frame.uvw = np.hstack( (np.vstack(np.cos(theta) * frame.uvw[:,0] + np.sin(theta) * frame.uvw[:,1]),
                                        np.vstack(- np.sin(theta) * frame.uvw[:,0] + np.cos(theta) * frame.uvw[:,1]),
                                        np.vstack(frame.uvw[:,2])) )
            else:
                frame.uvw = convert_to_R(opti[3 * fi], opti[3 * fi + 1], opti[3 * fi + 2])