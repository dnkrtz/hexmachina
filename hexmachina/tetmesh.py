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

from optimization import *
from transforms import *
from utils import *

class Frame(object):
    def __init__(self, uvw, location, is_boundary = False):
        self.uvw = uvw
        self.location = location
        self.is_boundary = is_boundary

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
        self.matchings = {}

    # Compute the one ring of tets surrounding each internal edge.
    def compute_onerings(self, surf_mesh):
        # Loop over all edges.
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

    # Compute the matchings for all pairs of face-adjacent tets.
    # We use matchings to characterize closeness between two frames s and t.
    # It is essentially the chiral permutation that most closely defines
    # the rotation between them.
    def compute_matchings(self):
        # Loop through all pairs of face-adjacent tets.
        for pair in self.mesh.adjacent_elements:
            args = []
            if -1 in pair:
                continue # Boundary face
            # Find the best permutation to characterize closeness.
            for permutation in chiral_symmetries:
                arg = self.frames[pair[0]].uvw - self.frames[pair[1]].uvw * permutation.T
                args.append(np.linalg.norm(arg))
            # Store the matching
            self.matchings[tuple(pair)] = chiral_symmetries[np.argmin(args)]

    # Initialize the frame field based on surface curvature and normals.
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

    # Optimize the framefield.
    def optimize_framefield(self):
        # Define all frames in terms of euler angles.
        euler_angles = np.zeros( (len(self.mesh.elements), 3) )
        # Loop over the tets.
        for ti, tet in enumerate(self.mesh.elements):
            # Boundary tets have only one degree of freedom in the
            # optimization, we inialize this angle to zero.
            if self.frames[ti].is_boundary:
                continue
            # Internal tets have three degrees of freedom. Their
            # current frame must be converted to an euler angle
            # representation as the initial data.
            else:
                R = self.frames[ti].uvw
                euler_angles[ti,:] = convert_to_euler(R)

        # This gets flattened by scipy.optimize, so do it preemptively.
        euler_angles = euler_angles.flatten()

        # Use scipy's L-BFGS to minimize the energy function.
        opti = optimize.minimize(global_energy, euler_angles, args=(self,),
                                 method='L-BFGS-B', jac = True,
                                 options={'ftol': 1e-4, 'maxiter': 12, 'disp': True}).x

        # Once optimization is complete, save results
        for fi, frame in enumerate(self.frames):
            # Make sure to store as ndarray, not matrix. Otherwise pyvtk goes bonkers.
            frame.uvw = convert_to_R(frame, opti[3 * fi], opti[3 * fi + 1], opti[3 * fi + 2]).getA()