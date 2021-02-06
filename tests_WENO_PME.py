import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from define_problem_Digital import Digital_option
from define_WENO_Network_2 import WENONetwork_2
from scipy.stats import norm
from define_problem_heat_eq import heat_equation
from define_problem_PME import PME

torch.set_default_dtype(torch.float64)

train_model = WENONetwork_2()
train_model = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Digital_Option_Test/Models/Model_17/7999.pt")

params=None

problem= PME

problem_main = problem(space_steps=80, time_steps=None, params = params)
params = problem_main.get_params()

u_init, nn = train_model.init_run_weno(problem_main, vectorized=True, just_one_time_step=False)
# parameters needed for the computation of exact solution
params_main = problem_main.params
T = params_main['T']
u_ex = problem_main.exact(T)

# u_t = u_init
# with torch.no_grad():
#     for k in range(nn):
#         u_t = train_model.run_weno(problem_main, u_t, mweno=True, mapped=False, vectorized=True, trainable=True, k=k)
#     V_t, _, _ = problem_main.transformation(u_t)

u_nt = u_init
for k in range(nn):
    u_nt = train_model.run_weno(problem_main, u_nt, mweno=True, mapped=False, vectorized=True, trainable=False, k=k)
V_nt, S, _ = problem_main.transformation(u_nt)

# error_t_mean = np.sqrt(12 / 80) * (np.sqrt(np.sum((u_t.detach().numpy() - u_ex) ** 2)))
# error_nt_mean = np.sqrt(12 / 80) * (np.sqrt(np.sum((u_nt.detach().numpy() - u_ex) ** 2)))
# error_nt_max = np.max(np.absolute(u_ex - u_nt.detach().numpy()))
# error_t_max = np.max(np.absolute(u_ex - u_t.detach().numpy()))
#
# err_mat = np.zeros((4,1))
# err_mat[0,:] = np.array(error_nt_max)
# err_mat[1,:] = np.array(error_t_max)
# err_mat[2,:] = np.array(error_nt_mean)
# err_mat[3,:] = np.array(error_t_mean)

# plt.figure(2)
plt.plot(S,V_nt, S, u_ex)
# plt.plot(S,V_nt, S, V_t, S, u_ex)

# VV = V.detach().numpy()
# X, Y = np.meshgrid(x, t, indexing="ij")
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(X, Y, VV, cmap=cm.viridis)


