"""
A short and sweet matrix library.

"""

import math
import random

__version__ = "0.1"


class Matrix(object):
    """A simple matrix class. Stored in a 2D array."""
    
    def __init__(self, array=[[]]):
        """Constructor (defaults to empty 2D array)."""
        self._array = array
    
    def __str__(self):
        """Return 2D array as string."""
        return self._array.__str__()
        
    @staticmethod
    def identity(n):
        return Matrix([[ 1 if j == i else 0 for j in range(n)] for i in range(n)])
        
    @staticmethod
    def random(n):
        return Matrix([[ random.uniform(0, 10) for j in range(n)] for i in range(n)])
        
    def subtract(self, B):
        """Return matrix B subtracted from this matrix."""
        if self.shape == B.shape:
            _A = self._array
            _B = B.asList()
            n = self.shape[0]
            m = self.shape[1]
            return Matrix([[ _A[i][j]-_B[i][j] for j in range(m)] for i in range(n)])
        else:
            raise Exception("""Matrix dimensions do not match!
                            Attempted: (%ix%i) - (%ix%i)""" % (n, m, B.shape[0], B.shape[1]))
            
    def add(self, B):
        """Return matrix B added to this matrix."""
        if self.shape == B.shape:
            _A = self._array
            _B = B.asList()
            n = self.shape[0]
            m = self.shape[1]
            return Matrix([[ _A[i][j]+_B[i][j] for j in range(m)] for i in range(n)])
        else:
            raise Exception("""Matrix dimensions do not match!
                            Attempted: (%ix%i) + (%ix%i)""" % (n, m, B.shape[0], B.shape[1]))
            
    def scalarMultiply(self, k):
        """Return the scalar product with k"""
        n = self.shape[0]
        m = self.shape[1]
        return Matrix([[ self._array[i][j]*k for j in range(m)] for i in range(n)])
        
    def transpose(self):
        """Returns the transpose."""
        n = self.shape[0]
        m = self.shape[1]
        return Matrix([[ self._array[i][j] for i in range(n)] for j in range(m)])
        
    def multiply(self, B):
        """Returns the matrix product with matrix B."""
        m = self.shape[1]
        if m == B.shape[0]: # columns A == rows B
            _A = self._array
            _B = B.asList()
            n = self.shape[0]
            p = B.shape[1]
            return Matrix([[sum(_A[i][k]*_B[k][j] for k in range(m)) for j in range(p)] for i in range(n)])
        else:
            raise Exception("""Matrix dimensions do not match!
                            Attempted: (%ix%i)(%ix%i)""" % (n, m, m, p))
    
    def scalarSubtract(self, k):
        """Return scalar k subtracted from this matrix."""
        n = self.shape[0]
        m = self.shape[1]
        return Matrix([[ self._array[i][j]-k for j in range(m)] for i in range(n)])
        
    def scalarAdd(self, k):
        """Return scalar k added to this matrix."""
        n = self.shape[0]
        m = self.shape[1]
        return Matrix([[ self._array[i][j]+k for j in range(m)] for i in range(n)])
    
    def getFrobeniusNorm(self):
        """Returns the Frobenius norm."""
        n = self.shape[0]
        m = self.shape[1]
        return math.sqrt(sum(math.pow(abs(self._array[i][j]), 2) for j in range(m) for i in range(n)))
        
    def normalize(self):
        return self.scalarMultiply(1.0/self.getFrobeniusNorm())
        
    def getEntry(self, u, v):
        """Returns a specific entry from the matrix."""
        return self._array[u][v]
    
    def asList(self):
        """Returns a copy of the 2D array."""
        return [self._array[i][:] for i in range(self.shape[0])]
        
        
    def getColumn(self, n):
        """Returns a matrix containing the specified column."""
        return Matrix([[self._array[i][n]] for i in range(self.shape[0])])
        
    @property
    def shape(self):
        """Returns the matrix dimensions as a tuple, in the form (n, m)."""
        return (len(self._array), len(self._array[0]))
        
    def trace(self):
        n = self.shape[0]
        m = self.shape[1]
        return sum(self._array[i][j] for j in range(m) for i in range(n) if i == j)
        
    def det(self):
        if self.shape[0] != self.shape[1]:
            raise Exception("Matrix is not square!")
        A = self._array
        if self.shape == (2, 2):
            return A[0][0]*A[1][1] - A[0][1]*A[1][0]
        elif self.shape == (3, 3):
            return A[0][0]*A[1][1]*A[2][2] + A[0][1]*A[1][2]*A[2][1] + A[0][2]*A[1][1]*A[2][0] \
                   - A[0][2]*A[1][1]*A[2][0] - A[0][1]*A[1][0]*A[2][2] - A[0][0]*A[1][2]*A[2][1]
        else:
            raise Exception("""'det' method only works for 2x2 and 3x3 matrices.
                            This one is %ix%i""" % (self.shape[0], self.shape[1]))

    # Operators...
    
    def __eq__(self, mat):
        return (mat.asList() == self._array)
        
    def __mul__(self, b):
        if isinstance(b, (int, float, complex)):
            return self.scalarMultiply(b)
        else:
            return self.multiply(b)
            
    def __rmul__(self, a):
        return self.scalarMultiply(a)
        
    def __add__(self, b):
        if isinstance(b, (int, float, complex)):
            return self.scalarAdd(b)
        else:
            return self.add(b)
            
    def __radd__(self, a):
        return self.scalarAdd(a)
        
    def __sub__(self, b):
        if isinstance(b, (int, float, complex)):
            return self.scalarSubtract(b)
        else:
            return self.subtract(b)
            
    def __rsub__(self, a):
        return self.scalarSubtract(a)
        
    def __imul__(self, b):
        temp = self.scalarMultiply(b)
        self._array = temp.asList()
        return self
        
    def __iadd__(self, B):
        temp = self.add(B)
        self._array = temp.asList()
        return self
        
    def __isub__(self, B):
        temp = self.subtract(B)
        self._array = temp.asList()
        return self