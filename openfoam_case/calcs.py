from cmath import pi
import matplotlib.pyplot as plt
import numpy as np

delta = 1
Re_tau = 180
nu = 4.7679e-05
Ub = 0.1335

Lx = 3.14
nx = 38
Lz = 1.57
nz = 28

u_tau = Re_tau*nu/delta
print('u_tau:' + str(u_tau))

Re_b = Ub*2*delta/nu
print('Bulk Reybolds number:' + str(Re_b))

delta_x_plus = Lx/nx*u_tau/nu
delta_z_plus = Lz/nz*u_tau/nu
print('delta_x_plus:' + str(delta_x_plus) + ', delta_z_plus:'+ str(delta_z_plus))

t_star = nu/u_tau**2
print('t_star:' + str(t_star))
delta_t_ctrl_plus = 0.6
delta_t_ctrl = delta_t_ctrl_plus*t_star
print('delta_t_ctrl_plus:'+ str(delta_t_ctrl_plus) + ', delta_t_ctrl:' + str(delta_t_ctrl))

delta_t_CFD = 0.02
delta_t_CFD_plus = delta_t_CFD/t_star
print('delta_t_CFD_plus:'+ str(delta_t_CFD_plus) + ', delta_t_CFD_plus:' + str(delta_t_CFD_plus))

y_plus_sample = 15
y_sample = y_plus_sample*nu/u_tau
print('y_sample:' + str(y_sample))
