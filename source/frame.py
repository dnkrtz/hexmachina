"""
A simple wrapper for u, v, w, coordinate map

"""

import numpy as np

__version__ = "0.1"


class Frame(object):
    """A simple UVW class stored individually for readability."""

    def __init__(self, u, v, w):
        """Constructor."""
        self.u = u
        self.v = v
        self.w = w

    def __str__(self):
        """Return coordinate as string."""
        return "({}, {}, {})".format(self.u, self.v, self.w)

    @classmethod
    def init_to_zeros(self):
        return self(np.zeros((3,)), np.zeros((3,)), np.zeros((3,)))
