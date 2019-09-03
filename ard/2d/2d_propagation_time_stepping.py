
#   Sound Propagation in 2 dimensions using ARD
#   Wave Equation Solver
#   Wave Equation: \frac{\partial^2p(\textbf{x},t)}{\partial t^2}  - c^2 \nabla^{2}p(\textbf{x},t) = f(\textbf{x},t)
#   Author: Juan Chango

import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import idct, fft, ifft, fft2, ifft2
from ard.dct.dct_2d_processor import ArdDct2D


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


c = 342  # speed of sound
lx = 342 / 1  # length in meters
ly = 342 / 1
t = 2  # time in seconds

# TIME
Fs_t = 200  # samples/second time is dependent of space

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
pos_y = 0

# pos_x = 0
# pos_y = 0

# Sinusoidal input
x_n[:, pos_y, pos_x] = A * np.sin((2 * np.pi * k_t / Fs_t) * t_values)

# Gaussian input
x_n[:, pos_y, pos_x] = A * gaussian(t_values, 5, 1)


plt.figure()
plt.plot(x_n[:, pos_y, pos_x])

print("num_div_t %i " % num_div_t)
print("num_div_x %i " % num_div_x)
print("num_div_y %i " % num_div_y)


# Init
proc = ArdDct2D(num_div_x, num_div_y)

# Init Simulation ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
n_partitions = 1
# Time step Calculation :create m^{n}_{i} matrix
m_minus_1 = np.zeros(shape=[n_partitions, num_div_y, num_div_x, 1])
m_plus_1 = np.zeros(shape=[n_partitions, num_div_y, num_div_x, 1])
m_actual = np.zeros(shape=[n_partitions, num_div_y, num_div_x, 1])

f_field = np.zeros(shape=[n_partitions, num_div_y, num_div_x, 1])

# Pressure Field
p_field = np.zeros(shape=[n_partitions, num_div_y, num_div_x])


w_i_matrix = np.zeros(shape=[num_div_y, num_div_x, 1])
for y in range(num_div_y):
    for x in range(num_div_x):
        w_i_matrix[y, x, 0] = c * ((np.pi ** 2) * (((x ** 2) / (lx ** 2)) + ((y ** 2) / (ly ** 2)))) ** 0.5


beta = 2 * np.cos(w_i_matrix * delta_t)

# Init force
force_k = proc.dct(x_n[1])

# Rec p_fields
p_field_list = []

for time_step in range(2, int(num_div_t)):
    #   Partition #1 -----
    n_p = 0

    # Force to Freq
    f_field[n_p] = (2 * force_k.reshape([num_div_y, num_div_x, 1]) * (1 / w_i_matrix ** 2)
                    * (1 - np.cos(w_i_matrix * delta_t)))
    f_field[n_p, 0, 0, 0] = 0

    m_plus_1[n_p] = (beta*m_actual[n_p]) - (m_minus_1[n_p]) + f_field[n_p]

    m_k = m_plus_1[n_p].reshape(num_div_y, num_div_x)
    p_field[n_p] = proc.idct(m_k)

    p_field_list.append(p_field[n_p].copy())

    #   Update m's
    m_minus_1[n_p] = m_actual[n_p].copy()
    m_actual[n_p] = m_plus_1[n_p].copy()

    # Update Force
    force_k = proc.dct(x_n[time_step])


# Plot Simulation Results
max_lim = p_field_list[int(num_div_t/2)].max()
min_lim = p_field_list[int(num_div_t/2)].min()

plot_step = 10
plt.figure()
for i in range(0, len(p_field_list), plot_step):
    plt.clf()
    plt.imshow(p_field_list[i], cmap='jet', vmin=min_lim, vmax=max_lim)
    plt.pause(0.01)
