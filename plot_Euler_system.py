import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from define_WENO_Network import WENONetwork
from define_WENO_Euler import WENONetwork_Euler
from define_Euler_system import Euler_system
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
train_model = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Euler_System_Test/Models/Model_60/69.pt")  # 39/62 works good obtained with old IC, 44/46
torch.set_default_dtype(torch.float64)
params=None
problem = Euler_system
sp_st = 64 #*2*2*2 #*2*2*2
init_cond = "Lax"
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

time_numb=0
t_update = 0
t = 0.9*h/lamb_nt_JS
while t_update < T:
    if (t_update + t) > T:
        t=T-t_update
    t_update = t_update + t
    q_0_nt_JS, q_1_nt_JS, q_2_nt_JS, lamb_nt_JS = train_model.run_weno(problem_main, mweno=False, mapped=False, method="char",q_0=q_0_nt_JS, q_1=q_1_nt_JS, q_2=q_2_nt_JS, lamb=lamb_nt_JS, vectorized=True, trainable=False, k=0, dt=t)
    t = 0.9*h/lamb_nt_JS
    time_numb = time_numb+1

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

x_ex = np.linspace(0, 1, sp_st+1)

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
error_rho_nt_mean = np.mean((rho_nt - rho_ex.detach().numpy()) ** 2)
error_rho_nt_JS_mean = np.mean((rho_nt_JS - rho_ex.detach().numpy()) ** 2)
error_rho_t_mean = np.mean((rho_t - rho_ex.detach().numpy()) ** 2)
error_u_nt_max = np.max(np.abs(u_nt - u_ex.detach().numpy()))
error_u_nt_JS_max = np.max(np.abs(u_nt_JS - u_ex.detach().numpy()))
error_u_t_max = np.max(np.abs(u_t - u_ex.detach().numpy()))
error_u_nt_mean = np.mean((u_nt - u_ex.detach().numpy()) ** 2)
error_u_nt_JS_mean = np.mean((u_nt_JS - u_ex.detach().numpy()) ** 2)
error_u_t_mean = np.mean((u_t - u_ex.detach().numpy()) ** 2)
error_p_nt_max = np.max(np.abs(p_nt - p_ex.detach().numpy()))
error_p_nt_JS_max = np.max(np.abs(p_nt_JS - p_ex.detach().numpy()))
error_p_t_max = np.max(np.abs(p_t - p_ex.detach().numpy()))
error_p_nt_mean = np.mean((p_nt - p_ex.detach().numpy()) ** 2)
error_p_nt_JS_mean = np.mean((p_nt_JS - p_ex.detach().numpy()) ** 2)
error_p_t_mean = np.mean((p_t - p_ex.detach().numpy()) ** 2)

err_mat = np.zeros((3,6))
err_mat[:,0] = np.array([error_rho_nt_JS_max, error_p_nt_JS_max, error_u_nt_JS_max])
err_mat[:,1] = np.array([error_rho_nt_JS_mean, error_p_nt_JS_mean, error_u_nt_JS_mean])
err_mat[:,2] = np.array([error_rho_nt_max, error_p_nt_max, error_u_nt_max])
err_mat[:,3] = np.array([error_rho_nt_mean, error_p_nt_mean, error_u_nt_mean])
err_mat[:,4] = np.array([error_rho_t_max, error_p_t_max, error_u_t_max])
err_mat[:,5] = np.array([error_rho_t_mean, error_p_t_mean, error_u_t_mean])

import pandas as pd
#pd.DataFrame(err_mat).to_csv("err_mat.csv")
pd.DataFrame(err_mat).to_latex()

# fig, ax = plt.subplots()
# ax.plot(x, u_nt_JS, color='blue') #, marker='o')
# ax.plot(x, u_nt, color='green') #, marker='o')
# ax.plot(x, u_t, color='red')
# ax.plot(x_ex,u_ex, color='black')
# ax.legend(('WENO-JS', 'WENO-Z', 'WENO-ML', 'ref. sol.'), loc=(0.1,0.2))
# ax.set_xlabel('x')
# ax.set_ylabel('rho')
# #axins = zoomed_inset_axes(ax, 1.5, loc=1)  # zoom = 6
# axins = inset_axes(ax, width=1, height=2, loc=1)
# axins.plot(x, u_nt_JS, color='blue')
# axins.plot(x, u_nt, color='green')
# axins.plot(x, u_t, color='red')
# axins.plot(x_ex, u_ex, color='black')
# axins.set_xlim(0.66, 0.73)  # Limit the region for zoom
# axins.set_ylim(0.00, 0.5)
# plt.xticks(visible=False)  # Not present ticks
# plt.yticks(visible=False)
# axins2 = inset_axes(ax, width=2, height=0.5, loc=2)
# axins2.plot(x, u_nt_JS, color='blue')
# axins2.plot(x, u_nt, color='green')
# axins2.plot(x, u_t, color='red')
# axins2.plot(x_ex, u_ex, color='black')
# axins2.set_xlim(0.47, 0.69)  # Limit the region for zoom
# axins2.set_ylim(0.925, 0.95)
# plt.xticks(visible=False)  # Not present ticks
# plt.yticks(visible=False)
# mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
# mark_inset(ax, axins2, loc1=1, loc2=3, fc="none", ec="0.5")
# plt.draw()
# plt.show()
# plt.savefig("foo.pdf", bbox_inches='tight')

# plt.figure(1)
# plt.plot(x, rho_nt_JS, color='blue') #, marker='o')
# plt.plot(x, rho_nt, color='green') #, marker='o')
# plt.plot(x, rho_t, color='red')
# plt.plot(x_ex, rho_ex, color='black')
# plt.legend(('WENO-JS', 'WENO-Z', 'WENO-ML', 'ref. sol.'), loc=0)
# plt.xlabel('x')
# plt.ylabel('rho')
# plt.savefig("foo.pdf", bbox_inches='tight')
# plt.figure(2)
# plt.plot(x,p_nt, x,p_t,x,p_ex.detach().numpy())
# plt.figure(3)
# plt.plot(x,u_nt, x,u_t,x,u_ex.detach().numpy())