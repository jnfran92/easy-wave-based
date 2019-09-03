#
# import numpy as np
# import matplotlib.pyplot as plt
# from ard_lib.two_dim.ard_2d_group import ProcessingGroup, ArdForce
# from ard_lib.two_dim.ard_2d_interface import InterfaceUnit, InterfaceHandler
#
#
# def gaussian(x, mu, sig):
#     return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
#
#
# c = 342  # speed of sound
# lx = 30  # length in meters
# ly = 30
# t = 3  # time in seconds
#
# # TIME
# Fs_t = 1300  # samples/second time is dependent of space
#
# # SPACE
# Fs_xy = 1  # samples/meter
# num_div_x = int(lx * Fs_xy)  # divisions of all the space
# num_div_y = int(ly * Fs_xy)  # divisions of all the space
#
# # Simulation steps in Time
# num_div_t = int(Fs_t * t)
# delta_t = t / num_div_t
#
# t_axis = np.arange(0, t, delta_t)
#
# # number of divisions in x axis
# delta_x = lx / num_div_x
# delta_y = ly / num_div_y
#
# x_axis = np.arange(0, lx, delta_x)
# y_axis = np.arange(0, ly, delta_y)
#
# t_values = np.arange(0, num_div_t, 1)
#
# x_values = np.arange(0, num_div_x, 1)
# y_values = np.arange(0, num_div_y, 1)
#
# # Boundary Condition info --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# courant_number = ((delta_t * c) / delta_x) + ((delta_t * c) / delta_y)
#
# print("num_div_t %i " % num_div_t)
# print("num_div_x %i " % num_div_x)
# print("num_div_y %i " % num_div_y)
#
# print("delta t: %f" % delta_t)
# print("delta x: %f" % delta_x)
# print("delta y: %f" % delta_y)
#
# print("CFL Condition in x 1D %f" % (delta_x / ((3 ** 0.5) * c)))
#
# print("Courant number in 2D %f" % courant_number)
#
# print("Cmax %f" % (1 / (3 ** 0.5)))
#
# # point force signal --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# # force_n = np.zeros(shape=[num_div_t])
# k_x = 15  # Space domain discrete freq
# k_t = 1 / ((2 * lx) / (k_x * c))  # Time domain discrete freq
#
# A = 100  # Amplitude signal
#
# # Gaussian and Sin inputs - First partition
# force_n = A * gaussian(t_values, 68, 10)
# # force_n = A * np.sin((2 * np.pi * k_t / Fs_t) * t_values)
# # force_n[0, :, pos_y + 30, pos_x + 30] = A * np.sin((2 * np.pi * k_t / Fs_t) * t_values)
#
# plt.figure()
# plt.plot(force_n)
#
# # Groups --------
# # processing groups
# processing_groups = []
# n_partitions = 2
# pg1 = ProcessingGroup(delta_y, delta_x, num_div_y, num_div_x, delta_t, n_partitions)
# processing_groups.append(pg1)
#
# n_partitions = 2
# pg2 = ProcessingGroup(delta_y, delta_x, int(num_div_y / 2), int(num_div_x * 2), delta_t, n_partitions)
# processing_groups.append(pg2)
#
# # Interface
# interface_handler = InterfaceHandler(delta_y, delta_x)
#
# for i in range(int(num_div_y / 2)):
#     interface_handler.x_units.append(InterfaceUnit(((0, 0), (1, 0)), (i, i)))
#
# for i in range(int(num_div_y / 2)):
#     interface_handler.x_units.append(InterfaceUnit(((1, 0), (0, 1)), (i, i)))
#
# for i in range(num_div_x):
#     interface_handler.y_units.append(InterfaceUnit(((0, 0), (1, 1)), (i, i)))
#
# ard_forces = []
# p_field_list = []
# p_field_list2 = []
# p_field_list3 = []
# p_field_list4 = []
#
# for i in range(2, 3200):
#     for pg in processing_groups:
#         ard_forces.clear()
#         # ard_forces.append(ArdForce(int(num_div_y/2), int(num_div_x/2), 0, force_n[i]))
#         if i < num_div_t:
#             if pg.num_div_y == num_div_y:
#                 ard_forces.append(ArdForce(int(num_div_y / 2), int(num_div_x / 2), 0, force_n[i]))
#         # ard_forces.append(ArdForce(0, 0, 1, force_n[i]))
#         # ard_forces.append(ArdForce(60, 60, 2, force_n[i]))
#
#         pg.solve_local_all(ard_forces)
#
#     interface_handler.solve_interfaces(processing_groups)
#
#     p_field_list.append(processing_groups[0].r_partitions[0].p_field.copy())
#     p_field_list2.append(processing_groups[0].r_partitions[1].p_field.copy())
#     p_field_list3.append(processing_groups[1].r_partitions[0].p_field.copy())
#     p_field_list4.append(processing_groups[1].r_partitions[1].p_field.copy())
#
# # Plot Global ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------
#
# global_pressure = np.zeros(shape=[num_div_y*2, num_div_x*4]) - 1
#
# # global_pressure_1 = global_pressure[0:num_div_y, 0:num_div_x]
# # global_pressure_2 = global_pressure[0:int(num_div_y/2), num_div_x:3*num_div_x]
# # global_pressure_3 = global_pressure[0:num_div_y, 3*num_div_x:4*num_div_x]
# # global_pressure_4 = global_pressure[num_div_y:num_div_y + int(num_div_y/2), 0:2*num_div_x]
#
# plot_step = 1
# plt.figure()
#
# for i in range(0, len(p_field_list), plot_step):
#     plt.clf()
#
#     global_pressure[0:num_div_y, 0:num_div_x] = p_field_list[i]
#     global_pressure[0:int(num_div_y / 2), num_div_x:3 * num_div_x] = p_field_list3[i]
#     global_pressure[0:num_div_y, 3 * num_div_x:4 * num_div_x] = p_field_list2[i]
#     global_pressure[num_div_y:num_div_y + int(num_div_y / 2), 0:2 * num_div_x] = p_field_list4[i]
#
#     plt.imshow(global_pressure, cmap='jet', vmin=-0.00005, vmax=0.00005)
#     plt.pause(0.001)
#
#
# # Plot By Parts ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------
#
# plot_step = 20
# plt.figure()
# for i in range(0, len(p_field_list), plot_step):
#     plt.clf()
#
#     plt.subplot(2, 3, 1)
#     plt.imshow(p_field_list[i], cmap='jet', vmin=-0.00005, vmax=0.00005)
#
#     plt.subplot(2, 3, 2)
#     plt.imshow(p_field_list3[i], cmap='jet', vmin=-0.00005, vmax=0.00005)
#
#     plt.subplot(2, 3, 3)
#     plt.imshow(p_field_list2[i], cmap='jet', vmin=-0.00005, vmax=0.00005)
#
#     plt.subplot(2, 3, 4)
#     plt.imshow(p_field_list4[i], cmap='jet', vmin=-0.00005, vmax=0.00005)
#
#     plt.pause(0.001)
