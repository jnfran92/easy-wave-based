
#  Sound Propagation in 1 dimension using ARD
#  Wave Equation: \frac{\partial^2p(\textbf{x},t)}{\partial t^2}  - c^2 \nabla^{2}p(\textbf{x},t) = f(\textbf{x},t)
#  Author: Juan Chango

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from scipy.fftpack import idct, dct

#   Parameters
from ard.dct.dct_1d_processor import ArdDct1D


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


c = 342  # speed of sound
lx = 342/4  # length in meters
t = 0.25   # time in seconds

# TIME
Fs_t = 8000    # samples/second time is dependent of space

# SPACE
Fs_x = 4     # samples/meter
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

k_x = 10

k_t = 1 / ((2 * lx) / (k_x*c))
A = 100
pos_x = 0

x_n[:, pos_x] = A*gaussian(t_values, 80*4, 80) - A*gaussian(t_values, 80*4*2, 80)
# x_n[:, 0] = np.cos((np.pi * k_x / num_div_x) * x_values)

plt.figure()
plt.plot(x_n[:, pos_x])

print("num_div_t %i " % num_div_t)
print("num_div_x %i " % num_div_x)


# Init Simulation ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# Init
proc = ArdDct1D(num_div_x)
# Create m^{n}_{i} matrix, source in first partition, All partitions are the same
n_partitions = 1    # Dont change, it is a constant

# Time stepping calculation
m_minus_1 = np.zeros(shape=[n_partitions, num_div_x, 1])
m_actual = np.zeros(shape=[n_partitions, num_div_x, 1])
# m_plus_1 = np.zeros(shape=[n_partitions, num_div_x, 1])
f_field = np.zeros(shape=[n_partitions, num_div_x, 1])

# Pressure Field
p_field = np.zeros(shape=[n_partitions, num_div_x])

kernel = np.zeros(shape=[3, num_div_x, 1])
kernel[0] = np.ones([1, num_div_x, 1])
kernel[2] = np.ones([1, num_div_x, 1])

w_i = c * np.pi * (x_values / lx)
w_i = w_i.reshape([num_div_x, 1])
kernel[1] = 2 * np.cos(w_i * delta_t)

# Init force
force_k = proc.dct(x_n[1])

# Rec p_fields
p_field_list = []
for time_step in range(2, int(num_div_t)):

    #   Partition #1 -----
    n_p = 0

    # Force to Freq
    f_field[n_p] = (2 * force_k.reshape([num_div_x, 1]) * (1 / w_i ** 2) * (1 - np.cos(w_i * delta_t)))
    f_field[n_p, 0, 0] = 0

    m_plus_1 = (kernel[1]*m_actual[n_p]) - (kernel[0] * m_minus_1[n_p]) + f_field[n_p]
    # m_plus_1[int(num_div_x / 4):num_div_x, :] = 0

    #   iDCT
    m_k = m_plus_1.reshape(num_div_x)
    p_field[n_p] = proc.idct(m_k)

    p_field_list.append(p_field[n_p].copy())

    #   Update m's
    m_minus_1[n_p] = m_actual[n_p].copy()
    m_actual[n_p] = m_plus_1.copy()

    # Update Force
    force_k = proc.dct(x_n[time_step])


# Plot Simulation Results ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
plot_step = 100

plt.figure()
for i in range(0, len(p_field_list), plot_step):
    plt.clf()
    plt.plot(p_field_list[i])
    plt.axis([0, num_div_x, p_field_list[i].min(), p_field_list[i].max()])
    plt.pause(0.00001)
