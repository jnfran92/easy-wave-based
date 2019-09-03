
#   Sound Propagation in 2 dimensions using ARD - Domain Decomposition Method 3 Partitions (x and y axis)
#   Wave Equation Solver
#   Wave Equation: \frac{\partial^2p(\textbf{x},t)}{\partial t^2}  - c^2 \nabla^{2}p(\textbf{x},t) = f(\textbf{x},t)
#   Author: Juan Chango

import matplotlib.pyplot as plt
import numpy as np
from ard.dct.dct_2d_processor import ArdDct2D


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


c = 342  # speed of sound
lx = 342 / 2  # length in meters
ly = 342 / 2
t = 3  # time in seconds

n_partitions = 3    # For this script this is a constant value, dont change

# TIME
Fs_t = 1200  # samples/second time is dependent of space

# SPACE
Fs_xy = 1  # samples/meter
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

t_values = np.arange(0, num_div_t, 1)

x_values = np.arange(0, num_div_x, 1)
y_values = np.arange(0, num_div_y, 1)

# Boundary Condition info --- --- --- --- --- --- --- --- --- --- --- --- --- ---
courant_number = ((delta_t*c)/delta_x) + ((delta_t*c)/delta_y)

print("n_partitions %i " % n_partitions)
print("num_div_t %i " % num_div_t)
print("num_div_x %i " % num_div_x)
print("num_div_y %i " % num_div_y)

print("delta t: %f" % delta_t)
print("delta x: %f" % delta_x)
print("delta y: %f" % delta_y)

print("CFL Condition in x 1D %f" % (delta_x/((3**0.5)*c)))

print("Courant number in 2D %f" % courant_number)

print("Cmax %f" % (1/(3**0.5)))


# force signal --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
force_n = np.zeros(shape=[n_partitions, num_div_t, num_div_y, num_div_x])
k_x = 8    # Space domain discrete freq
k_t = 1 / ((2 * lx) / (k_x * c))    # Time domain discrete freq

A = 100     # Amplitude signal

# Position of source
pos_x = int(num_div_x/2)
pos_y = int(num_div_y/2)

# Gaussian and Sin inputs - First partition
# force_n[0, :, pos_y, pos_x] = A * gaussian(t_values, 68, 10)
force_n[0, :, pos_y, pos_x] = A * np.sin((2 * np.pi * k_t / Fs_t) * t_values)
# force_n[0, :, pos_y + 30, pos_x + 30] = A * np.sin((2 * np.pi * k_t / Fs_t) * t_values)

plt.figure()
plt.plot(force_n[0, :, pos_y, pos_x])

# Init fdtd interface handling coefficients ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
fdtd_kernel_6 = np.array([2, -27, 270, -490, 270, -27, 2])*(1/180)

fdtd_k_size = int((fdtd_kernel_6.shape[0] - 1) / 2)     # Interface kernel size

fdtd_kernel_matrix_temp = np.zeros(shape=[fdtd_k_size, fdtd_k_size])
fdtd_kernel_matrix = np.zeros(shape=[fdtd_k_size * 2, fdtd_k_size * 2])

for i in range(fdtd_k_size):
    fdtd_kernel_matrix_temp[i, 2 - i:fdtd_k_size] = fdtd_kernel_6[0:i + 1]

for i in range(2):
    fdtd_kernel_matrix[0 + i * fdtd_k_size:fdtd_k_size + i * fdtd_k_size, 0:fdtd_k_size] = -1 * fdtd_kernel_matrix_temp
    fdtd_kernel_matrix[0 + i * fdtd_k_size:fdtd_k_size + i * fdtd_k_size, fdtd_k_size:2 * fdtd_k_size] = np.fliplr(fdtd_kernel_matrix_temp)

    fdtd_kernel_matrix_temp = -1 * np.flipud(fdtd_kernel_matrix_temp)

lambda_x_axis = (c / delta_x) ** 2
lambda_y_axis = (c / delta_y) ** 2

fdtd_kernel_matrix_x_axis = fdtd_kernel_matrix.copy()*lambda_x_axis
fdtd_kernel_matrix_y_axis = fdtd_kernel_matrix.copy()*lambda_y_axis


# Init Simulation ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

# Init DCT
proc = ArdDct2D(num_div_x, num_div_y)

# Time step Calculation :create m^{n}_{i} matrix
m_minus_1 = np.zeros(shape=[n_partitions, num_div_y, num_div_x, 1])
m_plus_1 = np.zeros(shape=[n_partitions, num_div_y, num_div_x, 1])
m_actual = np.zeros(shape=[n_partitions, num_div_y, num_div_x, 1])

f_n_field = np.zeros(shape=[n_partitions, num_div_y, num_div_x, 1])
f_k_field = np.zeros(shape=[n_partitions, num_div_y, num_div_x, 1])

# Pressure Field
p_field = np.zeros(shape=[n_partitions, num_div_y, num_div_x])

# Constant Matrix
w_i_matrix = np.zeros(shape=[num_div_y, num_div_x, 1])
for y in range(num_div_y):
    for x in range(num_div_x):
        w_i_matrix[y, x, 0] = c * ((np.pi ** 2) * (((x ** 2) / (lx ** 2)) + ((y ** 2) / (ly ** 2)))) ** 0.5

#   beta*M^{n} constant term
beta = 2 * np.cos(w_i_matrix * delta_t)

# Init force
force_k = np.zeros(shape=[n_partitions, num_div_y, num_div_x])

# In All Partitions
for n_p in range(n_partitions):
    # force_k[n_p] = proc.dct(force_n[n_p, 1])
    f_n_field[n_p] = force_n[n_p, 1].copy().reshape(num_div_y, num_div_x, 1)

# Rec p_fields
p_field_list = []

for time_step in range(2, int(num_div_t)):

    # Air partitions calculation
    for n_p in range(n_partitions):
        #   Partition #1 -----
        # n_p = 0
        # Force to Freq
        force_k[n_p] = proc.dct(f_n_field[n_p].reshape([num_div_y, num_div_x]))
        f_k_field[n_p] = (2 * force_k[n_p].reshape([num_div_y, num_div_x, 1]) * (1 / w_i_matrix ** 2)
                          * (1 - np.cos(w_i_matrix * delta_t)))
        f_k_field[n_p, 0, 0, 0] = 0

        m_plus_1[n_p] = (beta * m_actual[n_p]) - (m_minus_1[n_p]) + f_k_field[n_p]

        m_k = m_plus_1[n_p].reshape(num_div_y, num_div_x)
        p_field[n_p] = proc.idct(m_k)

        #   Update m's
        m_minus_1[n_p] = m_actual[n_p].copy()
        m_actual[n_p] = m_plus_1[n_p].copy()

        # Update Force
        f_n_field[n_p] = force_n[n_p, time_step].copy().reshape([num_div_y, num_div_x, 1])

    # Interface Handling along y axis, x axis pressure field is used to compute forces
    for y_ih in range(num_div_y):
        p_field_mini = np.zeros(shape=[2 * fdtd_k_size, 1])

        p_field_mini[0:fdtd_k_size, :] = p_field[0, y_ih, -fdtd_k_size:num_div_x].copy().reshape([fdtd_k_size, 1])

        p_field_mini[fdtd_k_size:2 * fdtd_k_size, :] = p_field[1, y_ih, 0:fdtd_k_size].copy().reshape(fdtd_k_size, 1)

        f_mini_update = fdtd_kernel_matrix_x_axis.dot(p_field_mini)*0.25

        f_n_field[0, y_ih, -fdtd_k_size:num_div_x, :] += f_mini_update[0:fdtd_k_size, :]
        f_n_field[1, y_ih, 0:fdtd_k_size, :] += f_mini_update[fdtd_k_size:2 * fdtd_k_size, :]

    # Interface Handling along x axis, y axis pressure field is used to compute forces
    for x_ih in range(num_div_x):
        p_field_mini = np.zeros(shape=[2 * fdtd_k_size, 1])

        p_field_mini[0:fdtd_k_size, :] = p_field[0, -fdtd_k_size:num_div_y, x_ih].copy().reshape([fdtd_k_size, 1])

        p_field_mini[fdtd_k_size:2 * fdtd_k_size, :] = p_field[2, 0:fdtd_k_size, x_ih].copy().reshape(fdtd_k_size, 1)

        f_mini_update = fdtd_kernel_matrix_y_axis.dot(p_field_mini)*0.25

        f_n_field[0, -fdtd_k_size:num_div_y, x_ih, :] += f_mini_update[0:fdtd_k_size, :]
        f_n_field[2, 0:fdtd_k_size, x_ih, :] += f_mini_update[fdtd_k_size:2 * fdtd_k_size, :]

    p_field_list.append(p_field.copy())


# Plot ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
global_grid = np.zeros(shape=[2*num_div_y, 2*num_div_x]) - 1

plot_step = 10
plt.figure()
for i in range(0, len(p_field_list), plot_step):
    plt.clf()

    global_grid[0:num_div_y, 0:num_div_x] = p_field_list[i][0]
    global_grid[0:num_div_y, 0+num_div_x:2*num_div_x] = p_field_list[i][1]
    global_grid[0+num_div_y:2*num_div_y, 0:num_div_x] = p_field_list[i][2]

    plt.imshow(global_grid, cmap='jet', vmin=-0.00005, vmax=0.00005)

    plt.xticks([])
    plt.yticks([])

    plt.pause(0.001)

