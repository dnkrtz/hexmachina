'''
    File: optimization.py
    License: MIT
    Author: Aidan Kurtz
    Created: 20/08/2016
    Python Version: 3.5
    ========================
    This module involves the 3D framefield optimization based
    on an energy function and its gradient. The efficient
    L-BFGS optimization method is used, with multiprocessing.
    Very slow nonetheless.
'''

import itertools
from scipy import optimize
import multiprocessing as mp
import random

from transforms import *

# Quantify adjacent frame smoothness.
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
def pair_energy_diff(F_s, F_t, dF_s, dF_t):
    # Approximate permutation and its derivative (chain rule).
    P = F_t.T * F_s
    dP = dF_t.T * F_s + F_t.T * dF_s
    # More chain rule in the permutation energy function H(n).
    dE_st = 0
    for i in range(3):
        dE_st += (2 * dP[i,0] * P[i,0] * P[i,1]**2) + (2 * P[i,0]**2 * dP[i,1] * P[i,1]) + \
                    (2 * dP[i,1] * P[i,1] * P[i,2]**2) + (2 * P[i,1]**2 * dP[i,2] * P[i,2]) + \
                    (2 * dP[i,2] * P[i,2] * P[i,0]**2) + (2 * P[i,2]**2 * dP[i,0] * P[i,0])
        dE_st += (2 * dP[0,i] * P[0,i] * P[1,i]**2) + (2 * P[0,i]**2 * dP[1,i] * P[1,i]) + \
                    (2 * dP[1,i] * P[1,i] * P[2,i]**2) + (2 * P[1,i]**2 * dP[2,i] * P[2,i]) + \
                    (2 * dP[2,i] * P[2,i] * P[0,i]**2) + (2 * P[2,i]**2 * dP[0,i] * P[0,i])
    return dE_st

# Quantify smoothness around an internal tetrahedral edge.
# Returns the result and its sparse gradient.
def edge_energy(args):
    ei, one_rings, frames, euler_angles = args

    E = 0
    dE = np.zeros( 3 * len(frames) )

    if ei not in one_rings:
        return E, dE # Not internal.
    
    # All combinations of s, t around the edges' one ring.
    for combo in itertools.combinations(one_rings[ei]['tets'], 2):
        F = []
        dF = []
        # Loop frame index (fi) combos.
        for fi in [combo[0], combo[1]]:
            frame = frames[fi]
            # The frame is represented by its euler angles.
            R = convert_to_R(frame, euler_angles[3*fi], 
                euler_angles[3*fi + 1], euler_angles[3*fi + 2])
            F.append(R)
            # The partial derivative wrt each angle.
            dR = convert_to_dR(frame, euler_angles[3*fi], 
                    euler_angles[3*fi + 1], euler_angles[3*fi + 2])
            dF.append(dR)
        
        # Add pair energy to the one-ring energy.
        E += pair_energy(F[0], F[1])
        # Add pair energy gradients.
        for i in range(3):
            dE[3 * combo[0] + i] += pair_energy_diff(F[0], F[1], dF[0][i], np.zeros((3,3)))
            dE[3 * combo[1] + i] += pair_energy_diff(F[0], F[1], np.zeros((3,3)), dF[1][i])

    return E, dE

# Returns the energy function and its gradient.
# Minimizing via L-BFGS smoothens the framefield.
def global_energy(euler_angles, machina):

    # Edge energies and their gradient.
    E = 0
    dE = [ [] for _ in range(len(machina.tet_mesh.edges)) ]
    
    # Relevant data
    one_rings = machina.one_rings
    frames = machina.frames

    # Multiprocessing setup and execution.
    def parameters():
        for ei in range(len(machina.tet_mesh.edges)):
            yield (ei, one_rings, frames, euler_angles)
    pool = mp.Pool()
    results = pool.map(edge_energy, parameters())
    E = np.sum([ res[0] for res in results ])
    dE = np.sum([ res[1] for res in results ], axis=0)
    pool.close()
    pool.join()
    
    return E, dE