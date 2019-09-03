#  Sound Propagation in 1 dimension using ARD
#  Wave Equation: \frac{\partial^2p(\textbf{x},t)}{\partial t^2}  - c^2 \nabla^{2}p(\textbf{x},t) = f(\textbf{x},t)
#  Author: Juan Chango

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from scipy.fftpack import idct

#   Parameters
from ard.dct.dct_1d_processor import ArdDct1D


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


c = 342  # speed of sound
lx = 342/2  # length in meters
t = 2   # time in seconds

# TIME
Fs_t = 2000    # samples/second time is dependent of space

# SPACE
Fs_x = 2      # samples/meter
num_div_x = int(lx*Fs_x)        # divisions of all the space


# Simulation steps in Time
num_div_t = int(Fs_t*t)
delta_t = t / num_div_t

t_axis = np.arange(0, t, delta_t)


# number of divisions in x axis
delta_x = lx / num_div_x

x_axis = np.arange(0, lx, delta_x)


# force signal

t_values = np.arange(0, num_div_t, 1)
x_values = np.arange(0, num_div_x, 1)

x_n = np.zeros([num_div_t, num_div_x])

k_x = 40
# x_n[:, 0] = np.cos((np.pi * k_x / num_div_x) * x_values)

k_t = 1 / ((2 * lx) / (k_x*c))
A = 100
pos_x = 0
# x_n[:, pos_x] = A * np.sin((2*np.pi * k_t / Fs_t) * t_values)
# x_n[:, pos_x] = gaussian(t_values, 5, 1) - gaussian(t_values, 10, 1)
# x_n[:, pos_x + 100] = gaussian(t_values, 5, 1) - gaussian(t_values, 10, 1)
# x_n[:, pos_x] = A*gaussian(t_values, 38, 9) - A*gaussian(t_values, 74, 9)

offset = 30
x_n[:, pos_x] = A*gaussian(t_values, 38 + offset, 9) - A*gaussian(t_values, 74 + offset, 9)


# plt.figure()
# plt.imshow(x_n, cmap='hot')

plt.figure()
plt.plot(x_n[:, pos_x])


print("num_div_t %i " % num_div_t)
print("num_div_x %i " % num_div_x)




# x_n = np.cos(np.pi * k * x_values / num_div_x)

# plt.figure()
# plt.plot(x_n)

# plt.plot(idct(x_k*2.5)[:,pos_x])


# Init
proc = ArdDct1D(num_div_x)

# Apply DCT
x_k = proc.dct(x_n)
# x_k = dct(x_n)
# x_k = np.real(fft(x_n))



# Plot DCT
plt.figure()
plt.imshow(x_k, cmap='hot')


# Create m^{n}_{i} matrix
m = np.zeros([num_div_x, num_div_t, num_div_t])
f = np.zeros([num_div_x, num_div_t, 1])


for i in range(1, int(num_div_x)):
    m_temp = np.zeros([num_div_t, num_div_t + 2])
    w_i = c * np.pi * (i / lx)

    for n in range(num_div_t):
        m_temp[n][0 + n] = 1
        m_temp[n][1 + n] = -2 * np.cos(w_i * delta_t)
        m_temp[n][2 + n] = 1

    f[i] = (2 * x_k[:, i] * (1/w_i**2) * (1 - np.cos(w_i*delta_t))).reshape([num_div_t, 1])

    #   m[0] and m[1] = 0
    m[i] = m_temp[0:num_div_t, 2:num_div_t + 2]


# Solution
s = np.zeros([num_div_t, num_div_x])
for i in range(1, int(num_div_x)):
    # print('iteration %d' % i)
    z = inv(m[i]).dot(f[i])
    s[2:num_div_t, i] = z[2:num_div_t, :].reshape(num_div_t-2)


# iDCT
# s_n = proc.idct(s)
s_n = idct(s)
# s_n = np.real(ifft(s))
# Plot Sound Propagation

plt.figure()
for i in range(280):
    # print("time step: %d" % i)
    plt.clf()

    plt.subplot(3, 1, 1)
    plt.plot(s_n[i])
    plt.axis([0, num_div_x, -np.max(s_n), np.max(s_n)])

    plt.subplot(3, 1, 2)
    plt.plot(s[i])
    plt.xscale('log')
    plt.axis([0, num_div_x, -np.max(s), np.max(s)])

    plt.subplot(3, 1, 3)
    plt.plot(x_values, x_n[i], 'o', color='r')
    plt.axis([0, num_div_x, -np.max(x_n), np.max(x_n)])

    plt.pause(0.0001)


# Spring movement
plt.figure()
print("num div x %d" % num_div_x)
for i in range(10):
    print("n mode: %d of %d" % (i, num_div_x))
    plt.clf()
    plt.plot(s[:, i])
    plt.pause(1.0)


# Spectrum
plt.figure()
for i in range(100):
    # print("n mode: %d of %d" % (i, num_div_t))
    plt.clf()
    plt.plot(s[i])
    plt.xscale('log')
    plt.pause(0.1)

