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

# Compute the euler angles of this rotation. Implementation of:
# http://www.staff.city.ac.uk/~sbbh653/publications/euler.pdf
def convert_to_euler(R):
    alpha, beta, gamma = 0, 0, 0
    if np.abs(R[2,0]) != 1:
        beta = - np.arcsin(R[2,0])
        alpha = np.arctan2(R[2,1] / np.cos(beta), R[2,2] / np.cos(beta))
        gamma = np.arctan2(R[1,0] / np.cos(beta), R[0,0] / np.cos(beta))
    else:
        gamma = 0
        if R[2,0] == -1:
            beta = np.pi / 2
            alpha = gamma + np.arctan2(R[0,1], R[0,2])
        else:
            beta = - np.pi / 2
            alpha = - gamma + np.arctan2(-R[0,1], -R[0,2])
    return np.array([alpha, beta, gamma])

# Computes the rotation matrix for a given set of euler angles.
def convert_to_R(alpha, beta, gamma):
    # Alpha rotation (Rx)
    c, s = np.cos(alpha), np.sin(alpha)
    Rx = np.matrix([[1, 0, 0], [0, c, -s], [0, s, c]])
    # Beta rotation (Ry)
    c, s = np.cos(beta), np.sin(beta)
    Ry = np.matrix([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    # Gamma rotation (Rz)
    c, s = np.cos(gamma), np.sin(gamma)
    Rz = np.matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    # Concatenate
    return Rx * Ry * Rz

# The cubical chiral symmetry group G.
chiral_symmetries = [
    # Identity
    np.matrix([[ 1,  0,  0], [ 0,  1,  0], [ 0,  0,  1]]),
    # 90 degree 4-fold rotations
    np.matrix([[ 0, -1,  0], [ 1,  0,  0], [ 0,  0,  1]]),
    np.matrix([[ 1,  0,  0], [ 0,  0, -1], [ 0,  1,  0]]),
    np.matrix([[ 0,  0,  1], [ 0,  1,  0], [-1,  0,  0]]),
    np.matrix([[ 0,  1,  0], [-1,  0,  0], [ 0,  0,  1]]),
    np.matrix([[ 1,  0,  0], [ 0,  0,  1], [ 0, -1,  0]]),
    np.matrix([[ 0,  0, -1], [ 0,  1,  0], [ 1,  0,  0]]),
    # 180 degree 4-fold rotations
    np.matrix([[-1,  0,  0], [ 0, -1,  0], [ 0,  0,  1]]),
    np.matrix([[ 1,  0,  0], [ 0, -1,  0], [ 0,  0, -1]]),
    np.matrix([[-1,  0,  0], [ 0,  1,  0], [ 0,  0, -1]]),
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
    np.matrix([[ 0,  0, -1], [ 0, -1,  0], [-1,  0,  0]]), #double check this
]