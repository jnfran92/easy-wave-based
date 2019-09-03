
#   Sound Propagation in 2 dimensions using ARD
#   Wave Equation Solver
#   Wave Equation: \frac{\partial^2p(\textbf{x},t)}{\partial t^2}  - c^2 \nabla^{2}p(\textbf{x},t) = f(\textbf{x},t)
#   Author: Juan Chango

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from scipy.fftpack import idct, fft, ifft, fft2, ifft2
from ard.dct.dct_2d_processor import ArdDct2D
import matplotlib.cm as cm


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


c = 342  # speed of sound
lx = 342 / 2  # length in meters
ly = 342 / 2
t = 2  # time in seconds

# TIME
Fs_t = 1000  # samples/second time is dependent of space

# SPACE
Fs_xy = 2  # samples/meter
num_div_x = int(lx * Fs_xy)  # divisions of all the space
num_div_y = int(ly * Fs_xy)  # divisions of all the space

# Simulation steps in Time
num_div_t = int(Fs_t * t)
delta_t = t / num_div_t

t_axis = np.arange(0, t, delta_t)

# number of divisions in x axis
delta_x = lx / num_div_x
delta_y = ly / num_div_y

x_axis = np.arange(0, lx, delta_x)
y_axis = np.arange(0, ly, delta_y)

# force signal

t_values = np.arange(0, num_div_t, 1)

x_values = np.arange(0, num_div_x, 1)
y_values = np.arange(0, num_div_y, 1)

x_n = np.zeros([num_div_t, num_div_y, num_div_x])

k_x = 1

k_t = 1 / ((2 * lx) / (k_x * c))

A = 100
pos_x = int(num_div_x/2)
pos_y = int(num_div_y/2)

# pos_x = 0
# pos_y = 0

# Sinusoidal input
x_n[:, pos_y, pos_x] = A * np.sin((2 * np.pi * k_t / Fs_t) * t_values)

# Gaussian input
x_n[:, pos_y, pos_x] = A * gaussian(t_values, 5, 1) - A*gaussian(t_values, 10, 1)


plt.figure()
plt.plot(x_n[:, pos_y, pos_x])

print("num_div_t %i " % num_div_t)
print("num_div_x %i " % num_div_x)
print("num_div_y %i " % num_div_y)

# x_n = np.cos(np.pi * k * x_values / num_div_x)

# plt.figure()
# plt.plot(x_n)


# Init
proc = ArdDct2D(num_div_x, num_div_y)

# Apply 2D DCT
x_k_list = []
for i in range(num_div_t):
    x_k_list.append(proc.dct(x_n[i]))

x_k = np.array(x_k_list)

# Plot DCT
plt.figure()
plt.imshow(x_k[:, 0, :], cmap='hot')


# Init Simulation ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
# Create m^{n}_{i} matrix
m = np.zeros([num_div_y, num_div_x, num_div_t, num_div_t])
f = np.zeros([num_div_y, num_div_x, num_div_t, 1])

for j in range(0, int(num_div_y)):
    for i in range(0, int(num_div_x)):
        if i == 0 & j == 0:
            continue

        m_temp = np.zeros([num_div_t, num_div_t + 2])
        w_ij = c * ((np.pi ** 2) * (((i ** 2) / (lx ** 2)) + ((j ** 2) / (ly ** 2)))) ** 0.5

        for n in range(num_div_t):
            m_temp[n][0 + n] = 1
            m_temp[n][1 + n] = -2 * np.cos(w_ij * delta_t)
            m_temp[n][2 + n] = 1

        f[j][i] = (2 * x_k[:, j, i] * (1 / w_ij ** 2) * (1 - np.cos(w_ij * delta_t))).reshape([num_div_t, 1])

        #   m[0] and m[1] = 0
        m[j][i] = m_temp[0:num_div_t, 2:num_div_t + 2]

# Solution
s = np.zeros([num_div_t, num_div_y, num_div_x])
for j in range(0, int(num_div_y)):
    for i in range(0, int(num_div_x)):
        # print('iteration %d' % i)
        if i == 0 & j == 0:
            continue
        z = inv(m[j][i]).dot(f[j][i])
        s[2:num_div_t, j, i] = z[2:num_div_t, :].reshape(num_div_t - 2)

# iDCT 2D
s_n_list = []
for i in range(num_div_t):
    s_n_list.append(idct(idct(s[i].T).T))

s_n = np.array(s_n_list)


# Animation
plt.figure()
for i in range(num_div_t):
    # print("time step: %d" % i)
    plt.clf()
    plt.imshow(s_n[i], cmap='gray', vmin=s_n.min()/4, vmax=s_n.max()/4)
    plt.pause(0.0001)


