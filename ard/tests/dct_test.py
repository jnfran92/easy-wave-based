
from ard.dct.dct_2d_processor import ArdDct2D
import numpy as np
import matplotlib.pyplot as plt

from scipy.fftpack import fft

m_size = 16
n_size = 16
# x = np.ones(shape=[m_size, n_size])

k = 2
N = m_size
n = np.arange(0, n_size, 1)
m_line = (np.cos((2*np.pi*k*n) / (N-1))).reshape([N, 1])

k = 2
N = n_size
n = np.arange(0, n_size, 1)
n_line = (np.cos((2*np.pi*k*n) / (N-1))).reshape([1, N])


# 2D Space Domain Matrix
x_n = m_line.dot(n_line)


# 1D FFT
fft_out = fft(m_line.T)
print(np.abs(fft_out))


plt.plot(m_line)
plt.plot(np.abs(fft_out))






plt.figure()
plt.imshow(x_n, cmap='hot', interpolation='nearest')
plt.show()



# Time domain
proc = ArdDct2D(m_size, n_size)
x_k = proc.dct(x_n)

plt.figure()
plt.imshow(x_k, cmap='hot', interpolation='nearest')
plt.show()

