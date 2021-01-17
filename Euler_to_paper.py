import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from define_WENO_Network import WENONetwork
from define_WENO_Euler import WENONetwork_Euler
from define_Euler_system import Euler_system

def monotonicity_loss(u):
    monotonicity = np.sum(np.abs(np.minimum(u[:-1]-u[1:], 0)))
    loss = monotonicity
    return loss

def monotonicity_loss_mid(u, x):
    monotonicity = np.zeros(x.shape[0])
    for k in range(x.shape[0]-1):
        if x[k] <= 0.5:
            monotonicity[k] = (np.abs(np.maximum((u[:-1]-u[1:])[k], 0)))
        elif x[k] > 0.5:
            monotonicity[k] = (np.abs(np.minimum((u[:-1]-u[1:])[k], 0)))
    loss = np.sum(monotonicity)
    return loss

train_model = WENONetwork_Euler()
train_model = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Euler_System_Test/Models/Model_87/483.pt")  # 72 tento, 142, 168, 179, 101 good?, 483
torch.set_default_dtype(torch.float64)
params=None
problem = Euler_system
sp_st = 64 #*2*2*2 #*2*2*2
init_cond = "Sod"
time_disc = None
problem_main = problem(space_steps=sp_st, init_cond = init_cond, time_steps=None, params = params, time_disc=time_disc, init_mid=False, init_general=False)
params = problem_main.get_params()
gamma = params['gamma']
method = "char"
T = params['T']

q_0, q_1, q_2, lamb, nn, h = train_model.init_Euler(problem_main, vectorized = True, just_one_time_step=False)
q_0_t_input, q_1_t_input, q_2_t_input, lamb_t = q_0, q_1, q_2, lamb
q_0_nt, q_1_nt, q_2_nt, lamb_nt = q_0, q_1, q_2, lamb
q_0_nt_JS, q_1_nt_JS, q_2_nt_JS, lamb_nt_JS = q_0, q_1, q_2, lamb

time_numb=0
t_update = 0
t = 0.9*h/lamb_nt
while t_update < T:
    if (t_update + t) > T:
        t=T-t_update
    t_update = t_update + t
    q_0_nt, q_1_nt, q_2_nt, lamb_nt = train_model.run_weno(problem_main, mweno=True, mapped=False, method="char",q_0=q_0_nt, q_1=q_1_nt, q_2=q_2_nt, lamb=lamb_nt, vectorized=True, trainable=False, k=0, dt=t)
    t = 0.9*h/lamb_nt
    time_numb = time_numb+1

t_update = 0
t = 0.9*h/lamb_nt_JS
while t_update < T:
    if (t_update + t) > T:
        t=T-t_update
    t_update = t_update + t
    q_0_nt_JS, q_1_nt_JS, q_2_nt_JS, lamb_nt_JS = train_model.run_weno(problem_main, mweno=False, mapped=False, method="char",q_0=q_0_nt_JS, q_1=q_1_nt_JS, q_2=q_2_nt_JS, lamb=lamb_nt_JS, vectorized=True, trainable=False, k=0, dt=t)
    t = 0.9*h/lamb_nt_JS

with torch.no_grad():
    t_update = 0
    t = 0.9*h/lamb_t
    T = params['T']
    while t_update < T:
        if (t_update + t) > T:
            t=T-t_update
        t_update = t_update + t
        q_0_t, q_1_t, q_2_t, lamb_t = train_model.run_weno(problem_main, mweno=True, mapped=False, method="char",q_0=q_0_t_input, q_1=q_1_t_input, q_2=q_2_t_input, lamb=lamb_t, vectorized=True, trainable=True, k=0, dt=t)
        t = 0.9*h/lamb_t
        q_0_t_input = q_0_t.detach().numpy()
        q_1_t_input = q_1_t.detach().numpy()
        q_2_t_input = q_2_t.detach().numpy()
        # lamb_t_input = lamb_t.detach().numpy()
        q_0_t_input = torch.Tensor(q_0_t_input)
        q_1_t_input = torch.Tensor(q_1_t_input)
        q_2_t_input = torch.Tensor(q_2_t_input)
        # lamb_t_input = torch.Tensor(lamb_t_input)

_,x,t = problem_main.transformation(q_0)


q_0_t = q_0_t.detach().numpy()
q_1_t = q_1_t.detach().numpy()
q_2_t = q_2_t.detach().numpy()

q_0_nt = q_0_nt.detach().numpy()
q_1_nt = q_1_nt.detach().numpy()
q_2_nt = q_2_nt.detach().numpy()

q_0_nt_JS = q_0_nt_JS.detach().numpy()
q_1_nt_JS = q_1_nt_JS.detach().numpy()
q_2_nt_JS = q_2_nt_JS.detach().numpy()

# X, Y = np.meshgrid(x, t, indexing="ij")
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(X, Y, q_1_np/q_0_np, cmap=cm.viridis)

x_ex = np.linspace(0, 1, sp_st+1)

# for k in range(0,t.shape[0]):
#     p_ex[:,k], rho_ex[:,k], u_ex[:,k], c_ex[:,k], mach_ex[:,k] = problem_main.exact(x_ex, t[k])

# X, Y = np.meshgrid(x_ex, t, indexing="ij")
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(X, Y, rho_ex.detach().numpy(), cmap=cm.viridis)

rho_t = q_0_t
u_t = q_1_t/rho_t
E_t = q_2_t
p_t = (gamma - 1)*(E_t-0.5*rho_t*u_t**2)

rho_nt = q_0_nt
u_nt = q_1_nt/rho_nt
E_nt = q_2_nt
p_nt = (gamma - 1)*(E_nt-0.5*rho_nt*u_nt**2)

rho_nt_JS = q_0_nt_JS
u_nt_JS = q_1_nt_JS/rho_nt_JS
E_nt_JS = q_2_nt_JS
p_nt_JS = (gamma - 1)*(E_nt_JS-0.5*rho_nt_JS*u_nt_JS**2)

p_ex, rho_ex, u_ex, _,_ = problem_main.exact(x_ex, T)
E_ex = p_ex/(gamma-1)+0.5*rho_ex*u_ex**2

error_rho_nt_max = np.max(np.abs(rho_nt - rho_ex.detach().numpy()))
error_rho_nt_JS_max = np.max(np.abs(rho_nt_JS - rho_ex.detach().numpy()))
error_rho_t_max = np.max(np.abs(rho_t - rho_ex.detach().numpy()))
error_rho_nt_mean = np.sqrt(1 / 128) * (np.sqrt(np.sum((rho_nt - rho_ex.detach().numpy()) ** 2)))
error_rho_nt_JS_mean = np.sqrt(1 / 128) * (np.sqrt(np.sum((rho_nt_JS - rho_ex.detach().numpy()) ** 2)))
error_rho_t_mean = np.sqrt(1 / 128) * (np.sqrt(np.sum((rho_t - rho_ex.detach().numpy()) ** 2)))
error_rho_nt_l1 = (1 / 128) * (np.sum(np.abs(rho_nt - rho_ex.detach().numpy())))
error_rho_t_l1 = (1 / 128) * (np.sum(np.abs(rho_t - rho_ex.detach().numpy())))
error_u_nt_max = np.max(np.abs(u_nt - u_ex.detach().numpy()))
error_u_nt_JS_max = np.max(np.abs(u_nt_JS - u_ex.detach().numpy()))
error_u_t_max = np.max(np.abs(u_t - u_ex.detach().numpy()))
error_u_nt_mean = np.sqrt(1 / 128) * (np.sqrt(np.sum((u_nt - u_ex.detach().numpy()) ** 2)))
error_u_nt_JS_mean = np.sqrt(1 / 128) * (np.sqrt(np.sum((u_nt_JS - u_ex.detach().numpy()) ** 2)))
error_u_t_mean = np.sqrt(1 / 128) * (np.sqrt(np.sum((u_t - u_ex.detach().numpy()) ** 2)))
error_u_nt_l1 = (1 / 128) * (np.sum(np.abs(u_nt - u_ex.detach().numpy())))
error_u_t_l1 = (1 / 128) * (np.sum(np.abs(u_t - u_ex.detach().numpy())))
error_p_nt_max = np.max(np.abs(p_nt - p_ex.detach().numpy()))
error_p_nt_JS_max = np.max(np.abs(p_nt_JS - p_ex.detach().numpy()))
error_p_t_max = np.max(np.abs(p_t - p_ex.detach().numpy()))
error_p_nt_mean = np.sqrt(1 / 128) * (np.sqrt(np.sum((p_nt - p_ex.detach().numpy()) ** 2)))
error_p_nt_JS_mean = np.sqrt(1 / 128) * (np.sqrt(np.sum((p_nt_JS - p_ex.detach().numpy()) ** 2)))
error_p_t_mean = np.sqrt(1 / 128) * (np.sqrt(np.sum((p_t - p_ex.detach().numpy()) ** 2)))
error_p_nt_l1 = (1 / 128) * (np.sum(np.abs(p_nt - p_ex.detach().numpy())))
error_p_t_l1 = (1 / 128) * (np.sum(np.abs(p_t - p_ex.detach().numpy())))

err_mat = np.zeros((6,3))
err_mat[0,:] = np.array([error_rho_nt_JS_max, error_p_nt_JS_max, error_u_nt_JS_max])
err_mat[1,:] = np.array([error_rho_nt_max, error_p_nt_max, error_u_nt_max])
err_mat[2,:] = np.array([error_rho_t_max, error_p_t_max, error_u_t_max])
err_mat[3,:] = np.array([error_rho_nt_JS_mean, error_p_nt_JS_mean, error_u_nt_JS_mean])
err_mat[4,:] = np.array([error_rho_nt_mean, error_p_nt_mean, error_u_nt_mean])
err_mat[5,:] = np.array([error_rho_t_mean, error_p_t_mean, error_u_t_mean])
err_mat = err_mat.T

rng = 3
ratio_inf = np.zeros((rng))
for i in range(rng):
    ratio_inf[i] = min(err_mat[i,0],err_mat[i,1])/err_mat[i,2]
ratio_l2 = np.zeros((rng))
for i in range(rng):
    ratio_l2[i] = min(err_mat[i,3],err_mat[i,4])/err_mat[i,5]

err_mat_ratios = np.zeros((8,3))
err_mat_ratios[0,:] = np.array([error_rho_nt_JS_max, error_p_nt_JS_max, error_u_nt_JS_max])
err_mat_ratios[1,:] = np.array([error_rho_nt_max, error_p_nt_max, error_u_nt_max])
err_mat_ratios[2,:] = np.array([error_rho_t_max, error_p_t_max, error_u_t_max])
err_mat_ratios[3,:] = ratio_inf
err_mat_ratios[4,:] = np.array([error_rho_nt_JS_mean, error_p_nt_JS_mean, error_u_nt_JS_mean])
err_mat_ratios[5,:] = np.array([error_rho_nt_mean, error_p_nt_mean, error_u_nt_mean])
err_mat_ratios[6,:] = np.array([error_rho_t_mean, error_p_t_mean, error_u_t_mean])
err_mat_ratios[7,:] = ratio_l2
err_mat_ratios = err_mat_ratios.T

import pandas as pd
# pd.DataFrame(err_mat).to_csv("err_mat.csv")
pd.DataFrame(err_mat_ratios).to_latex()

# plt.figure(1)
# plt.plot(x,rho_nt_JS,x,rho_nt,x,rho_t,x,rho_ex.detach().numpy())
# plt.figure(2)
# plt.plot(x,p_nt_JS, x,p_nt, x,p_t,x,p_ex.detach().numpy())
# plt.figure(3)
# plt.plot(x,u_nt_JS, x,u_nt, x,u_t,x,u_ex.detach().numpy())

from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# # SOD u
# fig, ax = plt.subplots()
# ax.plot(x, u_nt_JS, color='blue') #, marker='o')
# ax.plot(x, u_nt, color='green') #, marker='o')
# ax.plot(x, u_t, color='red')
# ax.plot(x_ex,u_ex, color='black')
# ax.legend(('WENO-JS', 'WENO-Z', 'WENO-DS', 'ref. sol.'), loc=3)
# ax.set_xlabel('x')
# ax.set_ylabel('u')
# #axins = zoomed_inset_axes(ax, 1.5, loc=1)  # zoom = 6
# axins = inset_axes(ax, width=1, height=2, loc=1)
# axins.plot(x, u_nt_JS, color='blue')
# axins.plot(x, u_nt, color='green')
# axins.plot(x, u_t, color='red')
# axins.plot(x_ex, u_ex, color='black')
# axins.set_xlim(0.7, 0.77)  # Limit the region for zoom
# axins.set_ylim(0.00, 0.5)
# plt.xticks(visible=False)  # Not present ticks
# plt.yticks(visible=False)
# axins2 = inset_axes(ax, width=2, height=0.5, loc=2)
# axins2.plot(x, u_nt_JS, color='blue')
# axins2.plot(x, u_nt, color='green')
# axins2.plot(x, u_t, color='red')
# axins2.plot(x_ex, u_ex, color='black')
# axins2.set_xlim(0.5, 0.725)  # Limit the region for zoom
# axins2.set_ylim(1.355, 1.37)
# plt.xticks(visible=False)  # Not present ticks
# plt.yticks(visible=False)
# mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
# mark_inset(ax, axins2, loc1=1, loc2=4, fc="none", ec="0.5")
# plt.draw()
# plt.show()
# plt.savefig("Sod_u.pdf", bbox_inches='tight')
#
# # SOD_rho
# fig, ax = plt.subplots()
# ax.plot(x, rho_nt_JS, color='blue') #, marker='o')
# ax.plot(x, rho_nt, color='green') #, marker='o')
# ax.plot(x, rho_t, color='red')
# ax.plot(x,rho_ex, color='black')
# ax.legend(('WENO-JS', 'WENO-Z', 'WENO-DS', 'ref. sol.'), loc=3)
# ax.set_xlabel('x')
# ax.set_ylabel(r'$\rho$')
# #axins = zoomed_inset_axes(ax, 1.5, loc=1)  # zoom = 6
# axins = inset_axes(ax, width=1, height=2, loc=1)
# axins.plot(x, rho_nt_JS, color='blue')
# axins.plot(x, rho_nt, color='green')
# axins.plot(x, rho_t, color='red')
# axins.plot(x, rho_ex, color='black')
# axins.set_xlim(0.68, 0.76)  # Limit the region for zoom
# axins.set_ylim(0.124, 0.35)
# plt.xticks(visible=False)  # Not present ticks
# plt.yticks(visible=False)
# plt.xticks(visible=False)  # Not present ticks
# plt.yticks(visible=False)
# mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
# plt.draw()
# plt.show()
# plt.savefig("Sod_rho.pdf", bbox_inches='tight')
#
# # SOD_p
# fig, ax = plt.subplots()
# ax.plot(x, p_nt_JS, color='blue') #, marker='o')
# ax.plot(x, p_nt, color='green') #, marker='o')
# ax.plot(x, p_t, color='red')
# ax.plot(x, p_ex, color='black')
# ax.legend(('WENO-JS', 'WENO-Z', 'WENO-DS', 'ref. sol.'), loc=1)
# ax.set_xlabel('x')
# ax.set_ylabel('p')
# #axins = zoomed_inset_axes(ax, 1.5, loc=1)  # zoom = 6
# axins = inset_axes(ax, width=1, height=3, loc=3)
# axins.plot(x, p_nt_JS, color='blue')
# axins.plot(x, p_nt, color='green')
# axins.plot(x, p_t, color='red')
# axins.plot(x, p_ex, color='black')
# axins.set_xlim(0.68, 0.74)  # Limit the region for zoom
# axins.set_ylim(0.0998, 0.475)
# plt.xticks(visible=False)  # Not present ticks
# plt.yticks(visible=False)
# plt.xticks(visible=False)  # Not present ticks
# plt.yticks(visible=False)
# mark_inset(ax, axins, loc1=1, loc2=4, fc="none", ec="0.5")
# plt.draw()
# plt.show()
# plt.savefig("Sod_p.pdf", bbox_inches='tight')

## LAX rho, p, u
# plt.figure(1)
# plt.plot(x, rho_nt_JS, color='blue') #, marker='o')
# plt.plot(x, rho_nt, color='green') #, marker='o')
# plt.plot(x, rho_t, color='red')
# plt.plot(x_ex, rho_ex, color='black')
# plt.legend(('WENO-JS', 'WENO-Z', 'WENO-DS', 'ref. sol.'), loc=0)
# plt.xlabel('x')
# plt.ylabel(r'$\rho$')
# plt.savefig("rho.pdf", bbox_inches='tight')
# plt.figure(2)
# plt.plot(x, p_nt_JS, color='blue') #, marker='o')
# plt.plot(x, p_nt, color='green') #, marker='o')
# plt.plot(x, p_t, color='red')
# plt.plot(x_ex, p_ex, color='black')
# plt.legend(('WENO-JS', 'WENO-Z', 'WENO-DS', 'ref. sol.'), loc=0)
# plt.xlabel('x')
# plt.ylabel('p')
# plt.savefig("p.pdf", bbox_inches='tight')
# plt.figure(3)
# plt.plot(x, u_nt_JS, color='blue') #, marker='o')
# plt.plot(x, u_nt, color='green') #, marker='o')
# plt.plot(x, u_t, color='red')
# plt.plot(x_ex, u_ex, color='black')
# plt.legend(('WENO-JS', 'WENO-Z', 'WENO-DS', 'ref. sol.'), loc=0)
# plt.xlabel('x')
# plt.ylabel('u')
# plt.savefig("u.pdf", bbox_inches='tight')