
import numpy as np
import matplotlib.pyplot as plt


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
# pos_x = int(num_div_x/2)
pos_x = 0

# x_n[:, pos_x] = A * np.sin((2*np.pi * k_t / Fs_t) * t_values)
offset = 30
x_n[:, pos_x] = A*gaussian(t_values, 38 + offset, 9) - A*gaussian(t_values, 74 + offset, 9)
# x_n[:, pos_x + 100] = gaussian(t_values, 5, 1) - gaussian(t_values, 10, 1)

# plt.figure()
# plt.imshow(x_n, cmap='hot')

plt.figure()
plt.plot(x_n[:, pos_x])


print("num_div_t %i " % num_div_t)
print("num_div_x %i " % num_div_x)

print("delta t: %f" % delta_t)
print("CFL Condition %f" % (delta_x/((3**0.5)*c)))



#   Init Simulation time-stepping scheme----

p_n_minus1 = np.zeros(shape=[num_div_x, 1])
p_n = np.zeros(shape=[num_div_x, 1])
p_n_plus1 = np.zeros(shape=[num_div_x, 1])

k_matrix = np.zeros(shape=[num_div_x, num_div_x])

fdtd_kernel_2 = np.array([1, -2, 1])

# Creating K Laplace operator matrix
k_matrix_temp = np.zeros(shape=[num_div_x, num_div_x + 2])
for i in range(num_div_x):
    k_matrix_temp[i, i:i+3] = fdtd_kernel_2.copy()


k_matrix = k_matrix_temp[:, 1:-1].copy()

# Rigid walls Nuemann boundary condition (partial_p/partial_x)=0 when x=0 and x=l_x
k_matrix[:, 0] = k_matrix[:, 0] + k_matrix_temp[:, 0]
k_matrix[:, -1] = k_matrix[:, -1] + k_matrix_temp[:, -1]


# Terms
lambda_2 = (c * delta_t / delta_x) ** 2

k_matrix = k_matrix*lambda_2

plt.figure()
for time_step in range(num_div_t):
    f = np.zeros(shape=[num_div_x, 1])
    f = (x_n[time_step, :].reshape([num_div_x, 1])).copy()
    p_n_plus1 = 2 * p_n - p_n_minus1 + (k_matrix.dot(p_n)) + (delta_t ** 2) * f
    #   Update last temporal terms
    p_n_minus1 = p_n.copy()
    p_n = p_n_plus1.copy()

    # Plot
    #   Plot
    plt.clf()

    plt.subplot(2, 1, 1)
    plt.plot(p_n_plus1)
    plt.axis([0, num_div_x, np.min(p_n_plus1), np.max(p_n_plus1)])

    plt.subplot(2, 1, 2)
    plt.plot(x_values, f.reshape([num_div_x]), 'o', color='r')
    plt.axis([0, num_div_x, -np.max(x_n), np.max(x_n)])

    plt.pause(0.001)



