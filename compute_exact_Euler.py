import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from define_WENO_Network import WENONetwork
from define_WENO_Euler import WENONetwork_Euler
from define_Euler_system import Euler_system

train_model = WENONetwork_Euler()
torch.set_default_dtype(torch.float64)
params=None
problem = Euler_system
sp_st = 2048
init_cond = "Sod"
problem_main = problem(space_steps=sp_st, init_cond = init_cond, time_steps=None, params = params, time_disc=None, init_mid=False)
params = problem_main.get_params()
gamma = params['gamma']
tt = problem_main.t

#R_left, R_right = train_model.comp_eigenvectors_matrix(problem_main, problem_main.initial_condition)
q_0, q_1, q_2, lamb, nn, h = train_model.init_Euler(problem_main, vectorized = True, just_one_time_step=False)
q_0_nt, q_1_nt, q_2_nt, lamb_nt = q_0, q_1, q_2, lamb

_,x,t = problem_main.transformation(q_0)

q_0_nt = q_0_nt.detach().numpy()
q_1_nt = q_1_nt.detach().numpy()
q_2_nt = q_2_nt.detach().numpy()

x_ex = np.linspace(0, 1, 2048+1)
p_ex = torch.zeros((x_ex.shape[0],t.shape[0]))
rho_ex = torch.zeros((x_ex.shape[0],t.shape[0]))
u_ex = torch.zeros((x_ex.shape[0],t.shape[0]))

for k in range(0,t.shape[0]):
    p_ex[:,k], rho_ex[:,k], u_ex[:,k], _,_ = problem_main.exact(x_ex, t[k])
#
# X, Y = np.meshgrid(x_ex[0:2048 + 1:8], t, indexing="ij")
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(X, Y, rho_ex.detach().numpy()[0:2048 + 1:8], cmap=cm.viridis)

rho_ex_s = rho_ex[0:2048 + 1:8]
u_ex_s = u_ex[0:2048 + 1:8]
p_ex_s = p_ex[0:2048 + 1:8]
# q0_ex = torch.Tensor(q0_ex)
# q1_ex = torch.Tensor(q1_ex)
# q2_ex = torch.Tensor(q2_ex)
torch.save(rho_ex, "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Euler_System_Test/rho_ex")
torch.save(u_ex, "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Euler_System_Test/u_ex")
torch.save(p_ex, "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Euler_System_Test/p_ex")