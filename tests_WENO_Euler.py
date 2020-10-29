import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from define_WENO_Network import WENONetwork
from define_Euler_system import Euler_system

train_model = WENONetwork()
torch.set_default_dtype(torch.float64)
params=None
problem = Euler_system
problem_main = problem(space_steps=1000, time_steps=None, params = params)
params = problem_main.get_params()

q, rho, u, p = train_model.run_weno_Euler(problem_main, vectorized=True, trainable = False, just_one_time_step = False)
_,x,t = problem_main.transformation(u)
# plt.figure(1)
# plt.plot(x,r)
# plt.figure(2)
# plt.plot(x,u)
# plt.figure(3)
# plt.plot(x,p)

x_ex, p_ex, rho_ex, u_ex, c_ex, mach_ex = problem_main.exact()

plt.figure(1)
plt.plot(x,rho,x_ex,rho_ex)
plt.figure(2)
plt.plot(x,p,x_ex,p_ex)
plt.figure(3)
plt.plot(x,u,x_ex,u_ex)
