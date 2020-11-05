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
sp_st = 128
init_cond = "Sod"
problem_main = problem(space_steps=sp_st, init_cond = init_cond, time_steps=None, params = params)
params = problem_main.get_params()
gamma = params['gamma']

q_0, q_1, q_2, lamb, nn = train_model.init_Euler(problem_main, vectorized = True, just_one_time_step=False)
#q_0_t, q_1_t, q_2_t, lamb_t = q_0, q_1, q_2, lamb
q_0_nt, q_1_nt, q_2_nt, lamb_nt = q_0, q_1, q_2, lamb

# for k in range(nn):
#     q_0_t, q_1_t, q_2_t, lamb_t = train_model.run_weno(problem_main, mweno = True, mapped = False, q_0=q_0_t, q_1=q_1_t, q_2=q_2_t, lamb=lamb_t, vectorized=True, trainable=True, k=k)
for k in range(nn):
    q_0_nt, q_1_nt, q_2_nt, lamb_nt = train_model.run_weno(problem_main, mweno = True, mapped = False, q_0=q_0_nt, q_1=q_1_nt, q_2=q_2_nt, lamb=lamb_nt, vectorized=True, trainable=False, k=k)
_,x,t = problem_main.transformation(q_0)

# q_0_t = q_0_t.detach().numpy()
# q_1_t = q_1_t.detach().numpy()
# q_2_t = q_2_t.detach().numpy()

q_0_nt = q_0_nt.detach().numpy()
q_1_nt = q_1_nt.detach().numpy()
q_2_nt = q_2_nt.detach().numpy()

# X, Y = np.meshgrid(x, t, indexing="ij")
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(X, Y, q_1_np/q_0_np, cmap=cm.viridis)

x_ex = np.linspace(0, 1, 128+1)
p_ex = torch.zeros((x_ex.shape[0],t.shape[0]))
rho_ex = torch.zeros((x_ex.shape[0],t.shape[0]))
u_ex = torch.zeros((x_ex.shape[0],t.shape[0]))
c_ex = torch.zeros((x_ex.shape[0],t.shape[0]))
mach_ex = torch.zeros((x_ex.shape[0],t.shape[0]))

for k in range(0,t.shape[0]):
    p_ex[:,k], rho_ex[:,k], u_ex[:,k], c_ex[:,k], mach_ex[:,k] = problem_main.exact(x_ex, t[k])

# X, Y = np.meshgrid(x_ex, t, indexing="ij")
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(X, Y, rho_ex.detach().numpy(), cmap=cm.viridis)

# rho_t = q_0_t
# u_t = q_1_t/rho_t
# E_t = q_2_t
# p_t = (gamma - 1)*(E_t-0.5*rho_t*u_t**2)

rho_nt = q_0_nt
u_nt = q_1_nt/rho_nt
E_nt = q_2_nt
p_nt = (gamma - 1)*(E_nt-0.5*rho_nt*u_nt**2)

# error_rho_nt_max = np.max(np.abs(rho_nt - rho_ex.detach().numpy()[:,-1]))
# error_rho_t_max = np.max(np.abs(rho_t - rho_ex.detach().numpy()[:,-1]))
# error_rho_nt_mean = np.mean((rho_nt - rho_ex.detach().numpy()[:,-1]) ** 2)
# error_rho_t_mean = np.mean((rho_t - rho_ex.detach().numpy()[:,-1]) ** 2)
# error_u_nt_max = np.max(np.abs(u_nt - u_ex.detach().numpy()[:,-1]))
# error_u_t_max = np.max(np.abs(u_t - u_ex.detach().numpy()[:,-1]))
# error_u_nt_mean = np.mean((u_nt - u_ex.detach().numpy()[:,-1]) ** 2)
# error_u_t_mean = np.mean((u_t - u_ex.detach().numpy()[:,-1]) ** 2)
# error_p_nt_max = np.max(np.abs(p_nt - p_ex.detach().numpy()[:,-1]))
# error_p_t_max = np.max(np.abs(p_t - p_ex.detach().numpy()[:,-1]))
# error_p_nt_mean = np.mean((p_nt - p_ex.detach().numpy()[:,-1]) ** 2)
# error_p_t_mean = np.mean((p_t - p_ex.detach().numpy()[:,-1]) ** 2)
#
# err_mat = np.zeros((4,3))
# err_mat[0,:] = np.array([error_rho_nt_max, error_p_nt_max, error_u_nt_max])
# err_mat[1,:] = np.array([error_rho_t_max, error_p_t_max, error_u_t_max])
# err_mat[2,:] = np.array([error_rho_nt_mean, error_p_nt_mean, error_u_nt_mean])
# err_mat[3,:] = np.array([error_rho_t_mean, error_p_t_mean, error_u_t_mean])
#
# plt.figure(1)
# plt.plot(x,rho_t,x_ex,rho_ex[:,-1].detach().numpy())
# plt.figure(2)
# plt.plot(x,p_t,x_ex,p_ex[:,-1].detach().numpy())
# plt.figure(3)
# plt.plot(x,u_t,x_ex,u_ex[:,-1].detach().numpy())

plt.figure(4)
plt.plot(x,rho_nt,x_ex,rho_ex[:,-1].detach().numpy())
plt.figure(5)
plt.plot(x,p_nt,x_ex,p_ex[:,-1].detach().numpy())
plt.figure(6)
plt.plot(x,u_nt,x_ex,u_ex[:,-1].detach().numpy())

# plt.figure(1)
# plt.plot(x,rho_nt)
# plt.figure(2)
# plt.plot(x,p_nt)
# plt.figure(3)
# plt.plot(x,u_nt)