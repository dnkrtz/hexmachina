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
from scipy import sparse
import multiprocessing as mp
import random

from transforms import *

def pair_energy(F_s, F_t):
    """Quantify adjacent frame smoothness"""
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

def pair_energy_diff(F_s, F_t, dF_s, dF_t):
    """Quantify adjacent frame smoothness derivative (for energy gradient)."""
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


def edge_energy(args):
    """Quantify smoothness around an internal tetrahedral edge.
    Returns the result and its sparse gradient."""
    # Parse input args
    ei, one_rings, R, dR = args

    E = 0
    dE = sparse.lil_matrix( (1, 3*len(R)) )

    if ei not in one_rings:
        return E, dE # Not internal.
    
    # All combinations of s, t around the edges' one ring.
    for combo in itertools.combinations(one_rings[ei]['tets'], 2):
        # The frame matrices (euler XYZ)
        Fs, Ft = R[combo[0]], R[combo[1]]
        # The partial derivatives wrt each euler angle.
        dFs, dFt = dR[combo[0]], dR[combo[1]]
        
        # Add pair energy to the one-ring energy.
        E += pair_energy(Fs, Ft)
        # Add pair energy gradients.
        for i in range(3):
            dE[0, 3 * combo[0] + i] += pair_energy_diff(Fs, Ft, dFs[i], np.zeros((3,3)))
            dE[0, 3 * combo[1] + i] += pair_energy_diff(Fs, Ft, np.zeros((3,3)), dFt[i])

    return E, dE.tocsr()

def global_energy(euler_angles, machina):
    """Global smoothness energy function being minimized.
    Returns the energy function and its gradient.
    """
    # Relevant data
    one_rings = machina.one_rings
    frames = machina.frames

    # Precompute R and dR for each frame.
    R = [ convert_to_R(frames[ti], euler_angles[3*ti:3*ti+3]) for ti in range(len(frames)) ]
    dR = [ convert_to_dR(frames[ti], euler_angles[3*ti:3*ti+3]) for ti in range(len(frames)) ]

    # Multiprocessing setup and execution.
    def parameters():
        for ei in range(len(machina.tet_mesh.edges)):
            yield (ei, one_rings, R, dR)
    pool = mp.Pool()
    results = pool.map(edge_energy, parameters())
    # Edge energies and their gradient.
    E = np.sum([ res[0] for res in results ])
    dE = np.sum([ res[1] for res in results ]).toarray()[0,:]
    pool.close()
    pool.join()
    
    return E, dE