#  Sound Propagation in 1 dimension using ARD DDM Domain Decomposition Method
#  Wave Equation: \frac{\partial^2p(\textbf{x},t)}{\partial t^2}  - c^2 \nabla^{2}p(\textbf{x},t) = f(\textbf{x},t)
#  Author: Juan Chango

import matplotlib.pyplot as plt
import numpy as np

#   Parameters
from ard.dct.dct_1d_processor import ArdDct1D


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


c = 342  # speed of sound
lx = 342/2  # length in meters
t = 2   # time in seconds

# TIME
Fs_t = 8000    # samples/second time is dependent of space

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

k_x = 1
# x_n[:, 0] = np.cos((np.pi * k_x / num_div_x) * x_values)

k_t = 1 / ((2 * lx) / (k_x*c))
A = 100
# pos_x = int(num_div_x/2)
# pos_x = int(8*num_div_x/10)
pos_x = 0

# x_n[:, pos_x] = A * np.sin((2*np.pi * k_t / Fs_t) * t_values)
# x_n[:, pos_x] = A*gaussian(t_values, 38, 9) - A*gaussian(t_values, 74, 9)
# x_n[:, pos_x + 100] = gaussian(t_values, 5, 1) - gaussian(t_values, 10, 1)

sigma = 180
x_n[:, pos_x] = A*gaussian(t_values, sigma*4, sigma) - A*gaussian(t_values, sigma*4*2, sigma)

plt.figure()
plt.plot(x_n[:, pos_x])


print("num_div_t %i " % num_div_t)
print("num_div_x %i " % num_div_x)

print("delta t: %f" % delta_t)
print("delta x: %f" % delta_x)
print("CFL Condition(delta_t must be less than this value:) %f" % (delta_x/((3**0.5)*c)))


# Init DCT
proc = ArdDct1D(num_div_x)
n_partitions = 2

# Compute Laplace residual operator
k_matrix_global = np.zeros(shape=[n_partitions*num_div_x, n_partitions*num_div_x])

fdtd_kernel_6 = np.array([2, -27, 270, -490, 270, -27, 2])*(1/180)
# fdtd_kernel_6 = np.array([2, -27, 270, -490, 270, -27, 2])

# Creating K Laplace operator matrix
k_matrix_temp = np.zeros(shape=[n_partitions*num_div_x, n_partitions*num_div_x + 6])
for i in range(n_partitions*num_div_x):
    k_matrix_temp[i, i:i+7] = fdtd_kernel_6.copy()


k_matrix_global = k_matrix_temp[:, 3:-3].copy()

# Rigid walls Nuemann boundary condition (partial_p/partial_x)=0 when x=0 and x=l_x
k_matrix_global[:, 0:3] = k_matrix_global[:, 0:3] + np.fliplr(k_matrix_temp[:, 0:3])
k_matrix_global[:, -3:n_partitions*num_div_x] = k_matrix_global[:, -3:n_partitions*num_div_x] + np.fliplr(k_matrix_temp[:, -3:n_partitions*num_div_x + 6])

# Two Partitions of equal size
k_matrix_local = k_matrix_global.copy()
p_1_k_matrix = k_matrix_local[0:int(n_partitions*num_div_x/2), :]
p_2_k_matrix = k_matrix_local[int(n_partitions*num_div_x/2):n_partitions*num_div_x, :]

# Both with boundaries conditions (partial_p/partial_x)=0 when x=0 and x=l_x
p_1_k_matrix[:, int(n_partitions*num_div_x/2)-3:int(n_partitions*num_div_x/2)] = p_1_k_matrix[:, int(n_partitions*num_div_x/2)-3:int(n_partitions*num_div_x/2)] \
                                                      + np.fliplr(p_1_k_matrix[:, int(n_partitions*num_div_x/2):int(n_partitions*num_div_x/2)+3])

p_1_k_matrix[:, int(n_partitions*num_div_x/2):int(n_partitions*num_div_x/2)+3] = p_1_k_matrix[:, int(n_partitions*num_div_x/2):int(n_partitions*num_div_x/2)+3] \
                                                       - p_1_k_matrix[:, int(n_partitions*num_div_x/2):int(n_partitions*num_div_x/2)+3]


p_2_k_matrix[:, int(n_partitions*num_div_x/2):int(n_partitions*num_div_x/2)+3] = p_2_k_matrix[:, int(n_partitions*num_div_x/2):int(n_partitions*num_div_x/2)+3] \
                                                      + np.fliplr(p_2_k_matrix[:, int(n_partitions*num_div_x/2)-3:int(n_partitions*num_div_x/2)])

p_2_k_matrix[:, int(n_partitions*num_div_x/2)-3:int(n_partitions*num_div_x/2)] = p_2_k_matrix[:, int(n_partitions*num_div_x/2)-3:int(n_partitions*num_div_x/2)] \
                                                       - p_2_k_matrix[:, int(n_partitions*num_div_x/2)-3:int(n_partitions*num_div_x/2)]

#  Laplace operator Residual = global - local
k_matrix_res = k_matrix_global - k_matrix_local
k_mini_matrix_res = k_matrix_res[int(n_partitions*num_div_x/2)-3:int(n_partitions*num_div_x/2)+3, int(n_partitions*num_div_x/2)-3:int(n_partitions*num_div_x/2)+3]

# Terms
lambda_2 = (c / delta_x) ** 2

k_matrix_local = k_matrix_local*lambda_2
k_matrix_res = k_matrix_res*lambda_2
k_mini_matrix_res = k_mini_matrix_res*lambda_2


#   Init Simulation ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

# Create m^{n}_{i} matrix, source in first partition, All partitions are the same size
# n_partitions = 2

m = np.zeros([n_partitions, num_div_x, num_div_t, num_div_t])
f = np.zeros([n_partitions, num_div_x, num_div_t, 1])

# Time stepping calculation
m_n_minus1 = np.zeros(shape=[n_partitions, num_div_x, 1])
m_n = np.zeros(shape=[n_partitions, num_div_x, 1])
m_n_plus1 = np.zeros(shape=[n_partitions, num_div_x, 1])

# Force field
f_n_field = np.zeros(shape=[n_partitions, num_div_x, 1])
f_k_field = np.zeros(shape=[n_partitions, num_div_x, 1])

# Force shared for update global pressures
f_n_shared = np.zeros(shape=[n_partitions*num_div_x, 1])

# Pressure Field
p_field_n_plus1 = np.zeros(shape=[n_partitions, num_div_x])

# Init constants (kernel)
kernel = np.zeros(shape=[3, num_div_x, 1])
kernel[0] = np.ones([1, num_div_x, 1])
kernel[2] = np.ones([1, num_div_x, 1])

w_i = c * np.pi * (x_values / lx)
w_i = w_i.reshape([num_div_x, 1])
kernel[1] = 2 * np.cos(w_i * delta_t)

# Update Forces Initial Partitions
f_n_field[0] = x_n[1].copy().reshape([num_div_x, 1])   # partition 1
# f_n_field[1] = x_n[1].copy().reshape([num_div_x, 1])   # partition 2 no forced

p_n_mini = np.zeros(shape=[6, 1])
p_n_mini[0:3, :] = p_field_n_plus1[0, -3:num_div_x].reshape([3, 1])
p_n_mini[3:6, :] = p_field_n_plus1[1, 0:3].reshape(3, 1)

f_mini_update = k_mini_matrix_res.dot(p_n_mini)

f_n_field[0, -3:num_div_x, :] = f_n_field[0, -3:num_div_x, :] + f_mini_update[0:3, :]
f_n_field[1, 0:3, :] = f_n_field[1, 0:3, :] + f_mini_update[3:6, :]

# recording fields
p_field_list = []


for i in range(2, int(num_div_t)):

    #   Partition #1 ----- ----- ----- ----- ----- ----- ----- -----
    n_p = 0

    # Force to Freq
    force_k = proc.dct(f_n_field[n_p].reshape(num_div_x))

    f_k_field[n_p] = (2 * force_k.reshape([num_div_x, 1]) * (1 / w_i ** 2) * (1 - np.cos(w_i * delta_t)))
    f_k_field[n_p, 0, 0] = 0

    m_n_plus1[n_p] = (kernel[1] * m_n[n_p]) - (kernel[0] * m_n_minus1[n_p]) + f_k_field[n_p]

    #   iDCT
    m_k = m_n_plus1[n_p].reshape(num_div_x)
    p_field_n_plus1[n_p] = proc.idct(m_k)

    # Update local force
    f_n_field[n_p] = x_n[i].copy().reshape([num_div_x, 1])

    #   Partition #2 ----- ----- ----- ----- ----- ----- ----- -----
    n_p = 1

    # Force to Freq
    force_k = proc.dct(f_n_field[n_p].reshape(num_div_x))

    f_k_field[n_p] = (2 * force_k.reshape([num_div_x, 1]) * (1 / w_i ** 2) * (1 - np.cos(w_i * delta_t)))
    f_k_field[n_p, 0, 0] = 0

    m_n_plus1[n_p] = (kernel[1] * m_n[n_p]) - (kernel[0] * m_n_minus1[n_p]) + f_k_field[n_p]

    #   iDCT
    m_k = m_n_plus1[n_p].reshape(num_div_x)
    p_field_n_plus1[n_p] = proc.idct(m_k)

    # Update local force No Force in Second Partition
    f_n_field[n_p] = np.zeros(shape=[num_div_x, 1])

    # global force update ----- ----- ----- ----- ----- ----- -----
    p_n_mini = np.zeros(shape=[6, 1])
    p_n_mini[0:3, :] = p_field_n_plus1[0, -3:num_div_x].reshape([3, 1])
    p_n_mini[3:6, :] = p_field_n_plus1[1, 0:3].reshape(3, 1)

    f_mini_update = k_mini_matrix_res.dot(p_n_mini)

    f_n_field[0, -3:num_div_x, :] += f_mini_update[0:3, :]
    f_n_field[1, 0:3, :] += f_mini_update[3:6, :]

    #   Update m's ----- ----- ----- ----- ----- ----- -----
    n_p = 0
    m_n_minus1[n_p] = m_n[n_p].copy()
    m_n[n_p] = m_n_plus1[n_p].copy()

    #   Update m's
    n_p = 1
    m_n_minus1[n_p] = m_n[n_p].copy()
    m_n[n_p] = m_n_plus1[n_p].copy()

    # REC
    p_field_list.append(p_field_n_plus1.copy())


# Plots ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

min_lim = p_field_list[int(num_div_t/100)][0].min()
max_lim = p_field_list[int(num_div_t/100)][0].max()

plot_step = 100
plt.figure()
for i in range(0, len(p_field_list), plot_step):
    # print(i)
    plt.clf()

    n_p = 0
    plt.subplot(1, 2, 1)
    plt.plot(p_field_list[i][n_p])
    plt.axis([0, num_div_x, p_field_list[i][n_p].min(), p_field_list[i][n_p].max()])

    n_p = 1
    plt.subplot(1, 2, 2)
    plt.plot(p_field_list[i][n_p])
    plt.axis([0, num_div_x, p_field_list[i][n_p].min(), p_field_list[i][n_p].max()])

    plt.pause(0.0001)


