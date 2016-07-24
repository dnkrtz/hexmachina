'''
    File: utils.py
    License: MIT
    Author: Aidan Kurtz
    Created: 09/07/2016
    Python Version: 3.5
    ========================
    This code contains some useful functions and hardcodes.
'''

import numpy as np

# Normalize a vector to unit length.
def normalize(vector):
    return vector /= np.linalg.norm(vector)

# The cubical chiral symmetry group of permutations.
rotational_symmetries = [
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
    np.matrix([[ 0,  0, -1], [ 0, -1,  0], [=1,  0,  0]]),
]