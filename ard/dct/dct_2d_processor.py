
import numpy as np


class ArdDct2D:
    c_matrix_x = []
    c_matrix_y = []
    x_size = 0
    y_size = 0

    def __init__(self, x_size, y_size):
        self.x_size = x_size
        self.y_size = y_size

        n_x = np.arange(0, self.x_size, 1)
        n_y = np.arange(0, self.y_size, 1)

        c_temp_x = []
        for k in range(self.x_size):
            c_temp_x.append(((np.pi * k) / self.x_size) * n_x)

        c_temp_y = []
        for k in range(self.y_size):
            c_temp_y.append(((np.pi * k) / self.y_size) * n_y)

        self.c_matrix_x = np.cos(np.array(c_temp_x))
        self.c_matrix_y = np.cos(np.array(c_temp_y))

    def dct(self, x_n):
        x_k = self.c_matrix_y.dot(x_n.dot(self.c_matrix_x))
        # return x_k / (self.x_size*self.y_size)
        return x_k

    def idct(self, x_k):
        # x_n = self.c_matrix_y_inv.dot(x_k.dot(self.c_matrix_x_inv))
        x_n = (self.c_matrix_x.dot((self.c_matrix_y.dot(x_k)).T)).T
        return x_n*(4/(self.x_size*self.y_size))
