
import numpy as np
from numpy.linalg import inv


class ArdDct1D:

    c_matrix = []
    x_size = 0

    def __init__(self, size):
        self.x_size = size
        n = np.arange(0, self.x_size, 1)

        c_temp = []
        for k in range(self.x_size):
            c_temp.append(((np.pi * k) / (self.x_size-1)) * n)

        self.c_matrix = np.cos(np.array(c_temp))

    def dct(self, x_n):
        x_k = x_n.dot(self.c_matrix)
        return x_k

    def idct(self, x_k):
        x_n = self.c_matrix.dot(x_k)
        return x_n*(2/self.x_size)

