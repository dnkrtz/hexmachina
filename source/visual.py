'''
    File: visual.py
    License: MIT
    Author: Aidan Kurtz
    Created: 09/07/2016
    Python Version: 3.5
    ========================
    This module contains some visualization tools.
'''

import numpy as np
from mpl_toolkits.mplot3d import proj3d
import mpl_toolkits.mplot3d as a3
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def plot_vectors(vectors, points):
    fig = plt.figure(num=None, figsize=(12, 10), dpi=80)
    ax = fig.add_subplot(111, projection='3d')

    for i, point in enumerate(points):
        ax.quiver(point[0], point[1], point[2], vectors[i][0], vectors[i][1], vectors[i][2])

    plt.show()

def plot_framefield(frames):
    fig = plt.figure(num=None, figsize=(12, 10), dpi=80)
    ax = fig.add_subplot(111, projection='3d')

    for frame in frames:
        ax.quiver(frame.location[0], frame.location[1], frame.location[2], frame.u[0], frame.u[1], frame.u[2]) # U
        ax.quiver(frame.location[0], frame.location[1], frame.location[2], frame.v[0], frame.v[1], frame.v[2]) # V
        ax.quiver(frame.location[0], frame.location[1], frame.location[2], frame.w[0], frame.w[1], frame.w[2]) # W
    
    plt.show()

def plot_mesh(vertices, faces):
    fig = plt.figure(num=None, figsize=(12, 10), dpi=80)
    ax = fig.add_subplot(111, projection='3d')

    verts = np.array(vertices)

    ax.plot_trisurf(verts[:,0], verts[:,1], verts[:,2], triangles=faces)

    plt.show()