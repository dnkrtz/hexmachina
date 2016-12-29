'''
    File: machina.py
    License: MIT
    Author: Aidan Kurtz
    Created: 09/07/2016
    Python Version: 3.5
    ========================
    This module includes both the tetrahedral mesh and
    3D frame field initialization/optimization.
'''

from collections import deque
import itertools
import meshpy.tet
import numpy as np
from scipy import spatial, optimize
import time

from optimization import *
from transforms import *
from surfacemesh import *
from utils import *

class Frame(object):
    """
    Frame object defining a uvw tensor, a location and 
    whether or not it lies at the boudary of the field.
    """
    def __init__(self, uvw, location, is_boundary = False):
        self.uvw = uvw
        self.location = location
        self.is_boundary = is_boundary

class HexMachina(object):
    def __init__(self, tri_mesh, max_vol):
        """Generate tetrahedral mesh and extract surface boundary."""
        mesh_info = meshpy.tet.MeshInfo()
        # Define MeshPy options.
        # @TODO Investigate why the -Y switch messes up surface extraction (facets?)
        opt = meshpy.tet.Options(switches='pqnn', facesout=True, edgesout=True)
        # Generate tetrahedral mesh.
        mesh_info.set_points(tri_mesh.vertices)
        faces = [list(map(lambda x: int(x), i)) for i in tri_mesh.faces]
        mesh_info.set_facets(faces)
        mesh_info = meshpy.tet.build(mesh_info, opt, max_volume = max_vol)
        # Save to self.
        self.tet_mesh = mesh_info
        self.surf_mesh = SurfaceMesh(self.tet_mesh)
        self.dual_edges = [] # aka face adjacency
        self.one_rings = {} # aka edge onerings.
        self.frames = []
        self.matchings = {} # parallel to tet_mesh.faces
        self.edge_types = np.zeros(len(mesh_info.edges))

    def compute_dual(self):
        """Compute dual graph topology information."""
        # Dual edges as a dictionary where they key is a set of tets,
        # and the value is the index of the face they share.
        self.dual_edges = {}
        for fi, face in enumerate(self.tet_mesh.faces):
            edge_key = frozenset(self.tet_mesh.adjacent_elements[fi])
            self.dual_edges[edge_key] = fi
        
        # The dual faces represent edges of the primal. The edges of
        # this face are called the one-ring of tets around the edge.
        for ei, edge in enumerate(self.tet_mesh.edges):
            # Make sure this is an internal edge, skip if it isn't.
            if all(vi in self.surf_mesh.vertex_map for vi in edge):
                continue
            # If it is, construct its one ring.
            one_ring = [ self.tet_mesh.edge_adjacent_elements[ei] ]
            finished = False
            # Walk around the edge until we've closed the one ring.
            while not finished:
                finished = True
                for neigh_ti in self.tet_mesh.neighbors[one_ring[-1]]:
                    # Make sure this neighbor is a viable pick.
                    if (neigh_ti == -1 or neigh_ti in one_ring):
                        continue
                    neighbor = self.tet_mesh.elements[neigh_ti]
                    # Make sure this neighbor shares the edge.
                    if all(vi in neighbor for vi in edge):
                        # Add it to the ring.
                        one_ring.append(neigh_ti)
                        finished = False
                        break
            
            # The one-ring can also be an ordered list of face indices
            # between each pair of adjacent tets 's' and 't'.
            # If the pair is in a different order than the adjacency info,
            # store fi as a negative value.
            face_sequence = []
            for i in range(len(one_ring)):
                st_key = frozenset([one_ring[i], one_ring[(i + 1) % len(one_ring)]])
                fi = self.dual_edges[st_key]
                
                if self.tet_mesh.adjacent_elements[fi][0] == one_ring[i]:
                    face_sequence.append(fi)
                else:
                    face_sequence.append(-fi)

            # Store it in our ring dictionary (don't tell golem).
            self.one_rings[ei] =  { 'tets' : one_ring, 'faces' : face_sequence }

    def init_framefield(self):
        """Initialize the frame field based on surface curvature and normals."""
        boundary_frames = []
        boundary_ids = {}
        # The frame field is initialized at the boundary,
        # based on the curvature cross-field and normals.
        for fi, face in enumerate(self.surf_mesh.faces):
            # Retrieve the tet this face belongs to.
            ti = self.tet_mesh.adjacent_elements[self.surf_mesh.face_map.inv[fi]][0]
            tet = self.tet_mesh.elements[ti]
            # Ignore faces which have 0 curvature.
            if np.isclose(self.surf_mesh.k1[face[0]], 0) and np.isclose(self.surf_mesh.k2[face[0]], 0):
                continue
            # @TODO(aidan) Find actual face values, not vertex values.
            uvw = np.hstack((np.vstack(self.surf_mesh.pdir1[face[0]]),
                             np.vstack(self.surf_mesh.pdir2[face[0]]),
                             np.vstack(self.surf_mesh.vertex_normals[face[0]])))
            boundary_frames.append(Frame(uvw, tet_centroid(self.tet_mesh, ti)))
            boundary_ids[ti] = len(boundary_frames) - 1

        # Prepare a KDTree of boundary frame coords for quick spatial queries.
        tree = spatial.KDTree(np.vstack([frame.location for frame in boundary_frames]))

        # Now propagate the boundary frames throughout the tet mesh.
        # Each tet frame takes the value of its nearest boundary tet.
        for ti, tet in enumerate(self.tet_mesh.elements):
            location = tet_centroid(self.tet_mesh, ti)
            if ti in boundary_ids:
                self.frames.append(Frame(boundary_frames[boundary_ids[ti]].uvw, location, True))
            else:
                nearest_ti = tree.query(location)[1] # Find closest boundary frame
                self.frames.append(Frame(boundary_frames[nearest_ti].uvw, location, False))

    def optimize_framefield(self):
        """Optimize frame field smoothness using L-BFGS"""
        # Define all frames in terms of euler angles.
        euler_angles = np.zeros( (len(self.tet_mesh.elements), 3) )
        # Loop over the tets.
        for ti, tet in enumerate(self.tet_mesh.elements):
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
                                 options={'ftol': 1e-4, 'maxiter': 80, 'disp': True}).x

        # Once optimization is complete, save results
        for fi, frame in enumerate(self.frames):
            # Make sure to store as ndarray, not matrix. Otherwise pyvtk goes bonkers.
            frame.uvw = convert_to_R(frame, opti[3*fi : 3*fi+3]).getA()