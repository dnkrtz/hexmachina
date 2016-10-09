'''
    File: transforms.py
    License: MIT
    Author: Aidan Kurtz
    Created: 14/08/2016
    Python Version: 3.5
    ========================
    This module contains rotation things.
'''

import numpy as np

def convert_to_euler(R):
    """Compute the euler angles of this rotation.
    Refer to [http://www.staff.city.ac.uk/~sbbh653/publications/euler.pdf]"""
    alpha, beta, gamma = 0, 0, 0
    if not np.isclose(np.abs(R[2,0]), 1):
        beta = - np.arcsin(R[2,0])
        alpha = np.arctan2(R[2,1] / np.cos(beta), R[2,2] / np.cos(beta))
        gamma = np.arctan2(R[1,0] / np.cos(beta), R[0,0] / np.cos(beta))
    else:
        gamma = 0
        if np.isclose(R[2,0], -1):
            beta = np.pi / 2
            alpha = gamma + np.arctan2(R[0,1], R[0,2])
        else:
            beta = - np.pi / 2
            alpha = - gamma + np.arctan2(-R[0,1], -R[0,2])
    return np.array([alpha, beta, gamma])

def convert_to_R(frame, euler):
    """Computes the rotation matrix for a given set of euler angles."""
    alpha, beta, gamma = euler[0], euler[1], euler[2]
    # Full rotation.
    c = [ np.cos(alpha), np.cos(beta), np.cos(gamma) ]
    s = [ np.sin(alpha), np.sin(beta), np.sin(gamma) ]
    R = np.identity(3)
    if frame.is_boundary:
        R = np.matrix(np.hstack((
            c[1] * frame.uvw[:,0:1] + s[1] * frame.uvw[:,1:2],
            - s[1] * frame.uvw[:,0:1] + c[1] * frame.uvw[:,1:2],
            frame.uvw[:,2:3]
        )))
    else:
        R = np.matrix([
            [c[1]*c[0], s[2]*s[1]*c[0] - c[2]*s[0], c[2]*s[1]*c[0] + s[2]*s[0]],
            [c[1]*s[0], s[2]*s[1]*s[0] + c[2]*c[0], c[2]*s[1]*s[0] - s[2]*c[0]],
            [-s[1], s[2]*c[1], c[2]*c[1]]
        ])
    return R

def convert_to_dR(frame, euler):
    """Compute the partial derivatives of R wrt to euler angles."""
    alpha, beta, gamma = euler[0], euler[1], euler[2]
    c = [ np.cos(alpha), np.cos(beta), np.cos(gamma) ]
    s = [ np.sin(alpha), np.sin(beta), np.sin(gamma) ]
    # We will have one partial derivative per euler angle.
    dR = [ np.zeros( (3,3) ) for _ in range(3) ]
    if frame.is_boundary:
        dR[1] = np.matrix(np.hstack((
                -s[1] * frame.uvw[:,0:1] + c[1] * frame.uvw[:,1:2],
                -c[1] * frame.uvw[:,0:1] - s[1] * frame.uvw[:,1:2],
                [[0],[0],[0]] )))
    else:
        dR[0] = np.matrix([
                [-c[1]*s[0], - s[2]*s[1]*s[0] - c[2]*c[0], - c[2]*s[1]*s[0] + s[2]*c[0]],
                [c[1]*c[0], s[2]*s[1]*c[0] - c[2]*s[0], c[2]*s[1]*c[0] + s[2]*s[0]],
                [0, 0, 0] ])
        dR[1] = np.matrix([
                [- s[1]*c[0], s[2]*c[1]*c[0], c[2]*c[1]*c[0]],
                [- s[1]*s[0], s[2]*c[1]*s[0], c[2]*s[1]*c[0]],
                [- c[1], - s[2]*s[1], - c[2]*s[1]] ])
        dR[2] = np.matrix([
                [0, c[2]*s[1]*c[0] + s[2]*s[0], - s[2]*s[1]*c[0] + c[2]*s[0]],
                [0, c[2]*s[1]*s[0] - s[2]*c[0], - s[2]*s[1]*s[0] - c[2]*c[0]],
                [0, c[2]*c[1], - s[2]*c[1]] ])

    return dR   


# The cubical chiral symmetry group G.
chiral_symmetries = [
    # Identity
    np.matrix([[ 1,  0,  0], [ 0,  1,  0], [ 0,  0,  1]]),
    # 90 and 180 degree 4-fold rotations
    np.matrix([[ 0, -1,  0], [ 1,  0,  0], [ 0,  0,  1]]), # Jw
    np.matrix([[-1,  0,  0], [ 0, -1,  0], [ 0,  0,  1]]), # Jw
    np.matrix([[ 0,  1,  0], [-1,  0,  0], [ 0,  0,  1]]), # Jw
    np.matrix([[ 1,  0,  0], [ 0,  0, -1], [ 0,  1,  0]]), # Ju
    np.matrix([[ 1,  0,  0], [ 0,  0,  1], [ 0, -1,  0]]), # Ju
    np.matrix([[ 1,  0,  0], [ 0, -1,  0], [ 0,  0, -1]]), # Ju
    np.matrix([[ 0,  0,  1], [ 0,  1,  0], [-1,  0,  0]]), # Jv
    np.matrix([[ 0,  0, -1], [ 0,  1,  0], [ 1,  0,  0]]), # Jv
    np.matrix([[-1,  0,  0], [ 0,  1,  0], [ 0,  0, -1]]), # Jv
    # 120 degree 3-fold rotations
    np.matrix([[ 0,  1,  0], [ 0,  0, -1], [-1,  0,  0]]),
    np.matrix([[ 0,  0, -1], [ 1,  0,  0], [ 0, -1,  0]]),
    np.matrix([[ 0,  0,  1], [-1,  0,  0], [ 0, -1,  0]]),
    np.matrix([[ 0, -1,  0], [ 0,  0, -1], [ 1,  0,  0]]),
    np.matrix([[ 0,  0,  1], [ 1,  0,  0], [ 0,  1,  0]]),
    np.matrix([[ 0,  1,  0], [ 0,  0,  1], [ 1,  0,  0]]),
    np.matrix([[ 0, -1,  0], [ 0,  0,  1], [-1,  0,  0]]),
    np.matrix([[ 0,  0, -1], [-1,  0,  0], [ 0,  1,  0]]),
    # 180 degree 2-fold rotations
    np.matrix([[ 0,  1,  0], [ 1,  0,  0], [ 0,  0, -1]]),
    np.matrix([[ 0, -1,  0], [-1,  0,  0], [ 0,  0, -1]]),
    np.matrix([[-1,  0,  0], [ 0,  0, -1], [ 0, -1,  0]]),
    np.matrix([[-1,  0,  0], [ 0,  0,  1], [ 0,  1,  0]]),
    np.matrix([[ 0,  0,  1], [ 0, -1,  0], [ 1,  0,  0]]),
    np.matrix([[ 0,  0, -1], [ 0, -1,  0], [-1,  0,  0]]), # TODO(aidan) double check this
]