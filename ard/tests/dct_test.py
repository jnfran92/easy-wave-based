
from ard.dct.dct_2d_processor import ArdDct2D
import numpy as np
import matplotlib.pyplot as plt

m_size = 128
n_size = 128
# x = np.ones(shape=[m_size, n_size])

k = 4
N = n_size
n = np.arange(0, n_size, 1)
n_line = (np.cos((2*np.pi*k*n) / (N-1))).reshape([N, 1])

k = 5
N = m_size
n = np.arange(0, n_size, 1)
m_line = (np.cos((2*np.pi*k*n) / (N-1))).reshape([1, N])

# Space domain
x_n = n_line.dot(m_line)
plt.figure()
plt.imshow(x_n, cmap='hot', interpolation='nearest')
plt.show()

# Time domain
proc = ArdDct2D(m_size, n_size)
x_k = proc.dct(x_n)

plt.figure()
plt.imshow(x_k, cmap='hot', interpolation='nearest')
plt.show()

