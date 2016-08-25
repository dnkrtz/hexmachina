'''
    File: visual.py
    License: MIT
    Author: Aidan Kurtz
    Created: 09/07/2016
    Python Version: 3.5
    ========================
    This module contains some matplotlib plotting functions.
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

def plot_lines(lines, points):
    fig = plt.figure(num=None, figsize=(12, 10), dpi=80)
    ax = fig.add_subplot(111, projection='3d')

    for line in lines:
        x = [ points[line[0]][0], points[line[1]][0] ]
        y = [ points[line[0]][1], points[line[1]][1] ]
        z = [ points[line[0]][2], points[line[1]][2] ]
        ax.plot(x, y, z)
    
    plt.show()

def plot_mesh(vertices, faces):
    fig = plt.figure(num=None, figsize=(12, 10), dpi=80)
    ax = fig.add_subplot(111, projection='3d')

    verts = np.array(vertices)

    ax.plot_trisurf(verts[:,0], verts[:,1], verts[:,2], triangles=faces)

    plt.show()