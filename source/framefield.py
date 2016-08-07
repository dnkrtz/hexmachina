'''
    File: framefield.py
    License: MIT
    Author: Aidan Kurtz
    Created: 06/08/2016
    Python Version: 3.5
    ========================
    This module contains all things 3D frame field.
'''

class Frame(object):
    def __init__(self, u, v, w, loc):
        self.u = u
        self.v = v
        self.w = w
        self.location = loc


def initialize_framefield(tet_mesh):
    frames = np.array(len(tet_mesh.elements))
    frame_loc = [ np.zeros(3,) for _ in range(len(tet_mesh.elements))]
    # The frame field is initialized at the boundary, based on the curvature
    # cross-field and normals.
    for fi, surf_face in enumerate(surf_faces):
        # Retrieve the tet this face belongs to.
        ti = tet_mesh.adjacent_elements[face_map.inv[fi]][0]
        tet = tet_mesh.elements[ti]
        # Ignore faces which have 0 curvature (they will take the value of their nearest boundary neighbor).
        if math.isclose(k1[surf_face[0]], 0) and math.isclose(k2[surf_face[0]], 0):
            continue
        # @TODO Find actual face values, not vertex.
        frames[ti] = Frame(dir1[surf_face[0]], dir2[surf_face[0]], v_norms[surf_face[0]], surf_vertices
        
        frame_loc[ti] = surf_vertices[surf_face[0]]