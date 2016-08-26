'''
    File: parametrization.py
    License: MIT
    Author: Aidan Kurtz
    Created: 25/08/2016
    Python Version: 3.5
    ========================
    Volume parametrization module.
'''

from utils import *

class IsoMap(object):
    # Piece-wise linear volume parametrization.
    def __init__(self):
        # Atlas of maps f at each vertex.
        raw_f = []
        f = []
        pass

def parametrize_volume(tet_mesh, h):

    # Each vertex may have multiple initial map values, depending
    # on the number of tets it's a part of. We narrow down later.
    f_map = [ [] for _ in range(len(tet_mesh.points)) ]

    for tet_pair, matching in tet_mesh.matchings.items():
        for vi in tet_pair[0]:
            f_map[vi].append(tet_mesh.frames[tet_pair[0]])
        for vi in tet_pair[1]:
            f_map[vi].append(tet_mesh.frames[tet_pair[1]])


    for vi, vertex_map in enumarate(f_map):
        ti = 0 # what?
        score = tet_volume(tet_mesh, ti)
        for f in vertex_map:
            vol = tet_volume(tet_mesh, ti)
            D = np.linalg.norm(h * frames_grad[:,0] - tet_mesh.frames[ti].uvw[:,0])**2 +
                np.linalg.norm(h * frames_grad[:,1] - tet_mesh.frames[ti].uvw[:,1])**2 +
                np.linalg.norm(h * frames_grad[:,2] - tet_mesh.frames[ti].uvw[:,2])**2
    





    
