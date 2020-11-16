import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from define_WENO_Network import WENONetwork
from define_WENO_Euler import WENONetwork_Euler
from define_Euler_system import Euler_system

train_model = WENONetwork_Euler()
#train_model = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Euler_System_Test/Models/Model_08/4")
torch.set_default_dtype(torch.float64)
params=None
problem = Euler_system
sp_st = 64
init_cond = "Sod"
problem_main = problem(space_steps=sp_st, init_cond = init_cond, time_steps=None, params = params, time_disc="adaptive", init_mid=False)
params = problem_main.get_params()
gamma = params['gamma']
tt = problem_main.t

#R_left, R_right = train_model.comp_eigenvectors_matrix(problem_main, problem_main.initial_condition)
q_0, q_1, q_2, lamb, nn, h = train_model.init_Euler(problem_main, vectorized = True, just_one_time_step=False)
q_0_nt, q_1_nt, q_2_nt, lamb_nt = q_0, q_1, q_2, lamb
t_update = 0
t = 0.9*h/lamb_nt
T = params['T']
while t_update < T:
    if (t_update + t) > T:
        t=T-t_update
    t_update = t_update + t
    q_0_nt, q_1_nt, q_2_nt, lamb_nt = train_model.run_weno(problem_main, mweno=False, mapped=False, method="char",q_0=q_0_nt, q_1=q_1_nt, q_2=q_2_nt, lamb=lamb_nt, vectorized=True, trainable=False, k=0, dt=t)
    t = 0.9*h/lamb_nt

# matr_0 = np.zeros((2049,40))
# matr_1 = np.zeros((2049,40))
# matr_2 = np.zeros((2049,40))
# for k in range(nn):
#     q_0_nt, q_1_nt, q_2_nt, lamb_nt = train_model.run_weno(problem_main, mweno = True, mapped = False, method="char", q_0=q_0_nt, q_1=q_1_nt, q_2=q_2_nt, lamb=lamb_nt, vectorized=True, trainable=False, k=k, dt=None)
#     # matr_0[:, k] = q_0_nt.detach().numpy()
#     # matr_1[:, k] = q_1_nt.detach().numpy()
#     # matr_2[:, k] = q_2_nt.detach().numpy()
#     print(k)
_,x,t = problem_main.transformation(q_0)

q_0_nt = q_0_nt.detach().numpy()
q_1_nt = q_1_nt.detach().numpy()
q_2_nt = q_2_nt.detach().numpy()

x_ex = np.linspace(0, 1, 64+1)
# p_ex = torch.zeros((x_ex.shape[0],t.shape[0]))
# rho_ex = torch.zeros((x_ex.shape[0],t.shape[0]))
# u_ex = torch.zeros((x_ex.shape[0],t.shape[0]))
# c_ex = torch.zeros((x_ex.shape[0],t.shape[0]))
# mach_ex = torch.zeros((x_ex.shape[0],t.shape[0]))
#
# for k in range(0,t.shape[0]):
#     p_ex[:,k], rho_ex[:,k], u_ex[:,k], c_ex[:,k], mach_ex[:,k] = problem_main.exact(x_ex, t[k])

# p_ex, rho_ex, u_ex, _,_ = problem_main.exact(x_ex, T/2+tt)
p_ex, rho_ex, u_ex, _,_ = problem_main.exact(x_ex, T)

# plt.figure(1)
# plt.plot(x_ex,rho_ex.detach().numpy())
# plt.figure(2)
# plt.plot(x_ex,p_ex.detach().numpy())
# plt.figure(3)
# plt.plot(x_ex,u_ex.detach().numpy())


rho_nt = q_0_nt
u_nt = q_1_nt/rho_nt
E_nt = q_2_nt
p_nt = (gamma - 1)*(E_nt-0.5*rho_nt*u_nt**2)

# plt.figure(1)
# plt.plot(x,rho_nt)
# plt.figure(2)
# plt.plot(x,p_nt)
# plt.figure(3)
# plt.plot(x,u_nt)

plt.figure(1)
plt.plot(x,rho_nt,x_ex,rho_ex.detach().numpy())
plt.figure(2)
plt.plot(x,p_nt,x_ex,p_ex.detach().numpy())
plt.figure(3)
plt.plot(x,u_nt,x_ex,u_ex.detach().numpy())

# q0_ex = q_0_nt[0:2048 + 1:8]
# q1_ex = q_1_nt[0:2048 + 1:8]
# q2_ex = q_2_nt[0:2048 + 1:8]
# rho_nt = q0_ex
# u_nt = q1_ex/rho_nt
# E_nt = q2_ex
# p_nt = (gamma - 1)*(E_nt-0.5*rho_nt*u_nt**2)
# # q0_ex = torch.Tensor(q0_ex)
# # q1_ex = torch.Tensor(q1_ex)
# # q2_ex = torch.Tensor(q2_ex)
# torch.save(rho_nt, "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Euler_System_Test/rho_ex")
# torch.save(u_nt, "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Euler_System_Test/u_ex")
# torch.save(p_nt, "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Euler_System_Test/p_ex")


