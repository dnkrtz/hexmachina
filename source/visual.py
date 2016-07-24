'''
    File: visual.py
    License: MIT
    Author: Aidan Kurtz
    Created: 09/07/2016
    Python Version: 3.5
    ========================
    This code contains some visualization tools.
'''

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