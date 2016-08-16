'''
    File: tetmesh.py
    License: MIT
    Author: Aidan Kurtz
    Created: 09/07/2016
    Python Version: 3.5
    ========================
    This module wraps MeshPy's TetGen binding to include other
    parameters relevant to our problem.
'''

import numpy as np

import meshpy.tet

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

        