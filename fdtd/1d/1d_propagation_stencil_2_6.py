
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

k_x = 10
# x_n[:, 0] = np.cos((np.pi * k_x / num_div_x) * x_values)

k_t = 1 / ((2 * lx) / (k_x*c))
A = 100
# pos_x = int(num_div_x/2)
pos_x = 0

x_n[:, pos_x] = A * np.sin((2*np.pi * k_t / Fs_t) * t_values)
# offset = 30
# x_n[:, pos_x] = A*gaussian(t_values, 38 + offset, 9) - A*gaussian(t_values, 74 + offset, 9)
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

fdtd_kernel_6 = np.array([2, -27, 270, -490, 270, -27, 2])*(1/180)

# Creating K Laplace operator matrix
k_matrix_temp = np.zeros(shape=[num_div_x, num_div_x + 6])
for i in range(num_div_x):
    k_matrix_temp[i, i:i+7] = fdtd_kernel_6.copy()


k_matrix = k_matrix_temp[:, 3:-3].copy()

# Rigid walls Nuemann boundary condition (partial_p/partial_x)=0 when x=0 and x=l_x
k_matrix[:, 0:3] = k_matrix[:, 0:3] + np.fliplr(k_matrix_temp[:, 0:3])
k_matrix[:, -3:num_div_x] = k_matrix[:, -3:num_div_x] + np.fliplr(k_matrix_temp[:, -3:num_div_x + 6])


# Terms
lambda_2 = (c * delta_t / delta_x) ** 2

k_matrix = k_matrix*lambda_2

# Rec p_fields
p_field_list = []

# Init forces
f = np.zeros(shape=[num_div_x, 1])
f = (x_n[1, :].reshape([num_div_x, 1])).copy()

# plt.figure()
for time_step in range(2, num_div_t):

    p_n_plus1 = 2 * p_n - p_n_minus1 + (k_matrix.dot(p_n)) + (delta_t ** 2) * f
    #   Update last temporal terms
    p_n_minus1 = p_n.copy()
    p_n = p_n_plus1.copy()

    # Update Forces
    f = (x_n[time_step, :].reshape([num_div_x, 1])).copy()

    p_field_list.append(p_n_plus1)
    # # Plot
    # #   Plot
    # plt.clf()
    #
    # plt.subplot(2, 1, 1)
    # plt.plot(p_n_plus1)
    # plt.axis([0, num_div_x, np.min(p_n_plus1), np.max(p_n_plus1)])
    #
    # plt.subplot(2, 1, 2)
    # plt.plot(x_values, f.reshape([num_div_x]), 'o', color='r')
    # plt.axis([0, num_div_x, -np.max(x_n), np.max(x_n)])
    #
    # plt.pause(0.001)


# Plot Simulation Results
data = np.array(p_field_list)
max_lim = data.max()
min_lim = data.min()

plot_step = 10

plt.figure()
for i in range(0, data.shape[0], plot_step):
    # print(i)
    plt.clf()
    plt.plot(data[i])
    plt.axis([0, num_div_x, min_lim, max_lim])
    plt.pause(0.0001)




