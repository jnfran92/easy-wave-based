
import numpy as np
from numpy.linalg import inv


class ArdDct3D:

    c_matrix_x = []
    c_matrix_y = []
    c_matrix_z = []
    x_size = 0
    y_size = 0
    z_size = 0

    def __init__(self, x_size, y_size, z_size):
        self.x_size = x_size
        self.y_size = y_size
        self.z_size = z_size

        n_x = np.arange(0, self.x_size, 1)
        n_y = np.arange(0, self.y_size, 1)
        n_z = np.arange(0, self.z_size, 1)

        c_temp_x = []
        for k in range(self.x_size):
            c_temp_x.append(((np.pi * k) / self.x_size) * n_x)

        c_temp_y = []
        for k in range(self.y_size):
            c_temp_y.append(((np.pi * k) / self.y_size) * n_y)

        c_temp_z = []
        for k in range(self.z_size):
            c_temp_z.append(((np.pi * k) / self.z_size) * n_z)

        self.c_matrix_x = np.cos(np.array(c_temp_x))
        self.c_matrix_y = np.cos(np.array(c_temp_y))
        self.c_matrix_z = np.cos(np.array(c_temp_z))

    def dct(self, x_n):
        x_k = ((((x_n.dot(self.c_matrix_x).transpose(0, 2, 1)).dot(self.c_matrix_y)).transpose(2, 1, 0)).dot(self.c_matrix_z)).transpose(2, 0, 1)
        return x_k

    def idct(self, x_k):
        x_n = (self.c_matrix_x.dot(self.c_matrix_z.dot((self.c_matrix_y.dot(x_k))).transpose([0, 2, 1]))).transpose([1, 2, 0])
        return x_n*(8/(self.x_size * self.y_size * self.z_size))

