import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from validation_problems import validation_problems
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from define_problem_Digital import Digital_option
from define_WENO_Network_2 import WENONetwork_2
from scipy.stats import norm
from define_problem_heat_eq import heat_equation
from define_problem_PME import PME
from initial_condition_generator import init_PME
import random
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

torch.set_default_dtype(torch.float64)

train_model = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/Models/Model_58/195.pt") #45/500 #46/650 # 47/999
# problem= PME
# example = "Barenblatt"
# valid_problems = validation_problems.validation_problems_barenblatt_default_3

train_model = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/Models_2d/Model_11/30.pt") #45/500 #46/650 # 47/999
problem= PME
example = "Barenblatt_2d"
valid_problems = validation_problems.validation_problems_barenblatt_2d_default

_, rng = valid_problems(0)
err_nt_max_vec = np.zeros(rng)
err_nt_mean_vec = np.zeros(rng)
err_t_max_vec = np.zeros(rng)
err_t_mean_vec = np.zeros(rng)

for j in range(rng):
    print(j)
    params, rng = valid_problems(j)
    problem_main = problem(sample_id = None, example=example, space_steps=64, time_steps=None, params=params)
    params = problem_main.get_params()
    print(params)
    if example == "Barenblatt":
        u_init, nn = train_model.init_run_weno(problem_main, vectorized=True, just_one_time_step=False)
        u_t = u_init
        with torch.no_grad():
            for k in range(nn):
                u_t = train_model.run_weno(problem_main, u_t, mweno=True, mapped=False, vectorized=True, trainable=True, k=k)
                V_t, _, _ = problem_main.transformation(u_t)
        u_nt = u_init
        for k in range(nn):
            u_nt = train_model.run_weno(problem_main, u_nt, mweno=True, mapped=False, vectorized=True, trainable=False, k=k)
            V_nt, S, _ = problem_main.transformation(u_nt)
    elif example == "Barenblatt_2d":
        u_init, nn = train_model.init_run_weno(problem_main, vectorized=True, just_one_time_step=False, dim=2)
        u_nt = u_init
        for k in range(nn):
            u_nt = train_model.run_weno_2d(problem_main, u_nt, mweno=True, mapped=False, vectorized=True, trainable=False, k=k)
            V_nt, S, _ = problem_main.transformation(u_nt)
        with torch.no_grad():
            u_t = u_init
            for k in range(nn):
                u_t = train_model.run_weno_2d(problem_main, u_t, mweno=True, mapped=False, vectorized=True, trainable=True, k=k)
                V_t, _, _ = problem_main.transformation(u_t)
    params_main = problem_main.params
    T = params_main['T']
    L = params_main['L']
    sp_st = problem_main.space_steps
    u_ex = problem_main.exact(T)
    if example == "Barenblatt":
        error_t_mean = np.sqrt(1 / sp_st) * (np.sqrt(np.sum((V_t.detach().numpy() - u_ex) ** 2)))
        error_nt_mean = np.sqrt(1 / sp_st) * (np.sqrt(np.sum((V_nt.detach().numpy() - u_ex) ** 2)))
        error_nt_max = np.max(np.absolute(u_ex - V_nt.detach().numpy()))
        error_t_max = np.max(np.absolute(u_ex - V_t.detach().numpy()))
    elif example == "Barenblatt_2d":
        error_nt_max = np.max(np.max(np.abs(u_ex - V_nt.detach().numpy())))
        error_t_max = np.max(np.max(np.abs(u_ex - V_t.detach().numpy())))
        error_nt_mean = (1 / sp_st) * (np.sqrt(np.sum((u_ex - V_nt.detach().numpy()) ** 2)))
        error_t_mean = (1 / sp_st) * (np.sqrt(np.sum((u_ex - V_t.detach().numpy()) ** 2)))
    err_nt_max_vec[j] = error_nt_max
    err_t_max_vec[j] = error_t_max
    err_nt_mean_vec[j] = error_nt_mean
    err_t_mean_vec[j] = error_t_mean
    # plt.figure(j + 1)
    # plt.plot(S, V_nt, S, V_t, S, u_ex)

err_mat = np.zeros((rng,4))
err_mat[:,0] = err_nt_max_vec
err_mat[:,2] = err_nt_mean_vec
err_mat[:,1] = err_t_max_vec
err_mat[:,3] = err_t_mean_vec

ratio_inf = np.zeros((rng))
for i in range(rng):
    ratio_inf[i] = err_mat[i,0]/err_mat[i,1]
ratio_l2 = np.zeros((rng))
for i in range(rng):
    ratio_l2[i] = err_mat[i,2]/err_mat[i,3]

err_mat_ratios = np.zeros((rng,6))
err_mat_ratios[:,0] = err_nt_max_vec
err_mat_ratios[:,3] = err_nt_mean_vec
err_mat_ratios[:,1] = err_t_max_vec
err_mat_ratios[:,4] = err_t_mean_vec
err_mat_ratios[:,2] = ratio_inf
err_mat_ratios[:,5] = ratio_l2

import pandas as pd
# pd.DataFrame(err_mat).to_csv("err_mat.csv")
pd.DataFrame(err_mat_ratios).to_latex()


V_nt = V_nt.detach().numpy()
V_t = V_t.detach().numpy()

# X, Y = np.meshgrid(S, S, indexing="ij")
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('u')
# ax.plot_surface(X, Y, V_t, cmap=cm.viridis)
# plt.savefig("PME_2d_2.pdf", bbox_inches='tight')

# plt.figure(2)
# plt.contour(X,Y,V_t, 20)

# X, Y = np.meshgrid(S, S, indexing="ij")
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(X, Y, V_nt, cmap=cm.viridis)
# plt.figure(4)
# plt.contour(X,Y,V_nt, 20)


# fig, ax = plt.subplots(figsize=(5.0, 5.0))
# ax.plot(S, V_nt, color='blue') #, marker='o')
# ax.plot(S, V_t, color='red', marker='x')
# ax.plot(S, u_ex, color='black')
# ax.legend(('WENO-Z', 'WENO-DS', 'ref. sol.'), loc=1)
# ax.set_xlabel('x')
# ax.set_ylabel('u')
# # axins = inset_axes(ax, width=1.5, height=1.5, loc=6)
# # axins.plot(S, V_nt, color='blue')
# # axins.plot(S, V_t, color='red', marker='x')
# # axins.plot(S, u_ex, color='black')
# # axins.set_xlim(-4.55, -4.25)  # Limit the region for zoom
# # axins.set_ylim(-0.01, 0.03)
# # plt.xticks(visible=False)  # Not present ticks
# # plt.yticks(visible=False)
# # axins2 = inset_axes(ax, width=1.5, height=1.5, loc=7)
# # axins2.plot(S, V_nt, color='blue') #, marker='o')
# # axins2.plot(S, V_t, color='red', marker='x')
# # axins2.plot(S, u_ex, color='black')
# # axins2.set_xlim(4.25, 4.55)  # Limit the region for zoom
# # axins2.set_ylim(-0.01, 0.03)
# # plt.xticks(visible=False)  # Not present ticks
# # plt.yticks(visible=False)
# # mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.5")
# # mark_inset(ax, axins2, loc1=3, loc2=4, fc="none", ec="0.5")
# # plt.draw()
# # plt.show()
# plt.savefig("PME_2_main.pdf", bbox_inches='tight')
#
# fig, (ax1, ax2) = plt.subplots(2,1,figsize=(2, 5.0))
# # fig.suptitle('Vertically stacked subplots')
# ax1.plot(S, V_nt, color='blue')
# ax1.plot(S, V_t, color='red', marker='x')
# ax1.plot(S, u_ex, color='black')
# ax1.set_xlim(-4.55, -4.25)  # Limit the region for zoom
# ax1.set_ylim(-0.001, 0.022)
# ax1.set_yticks([])
# ax2.plot(S, V_nt, color='blue')
# ax2.plot(S, V_t, color='red', marker='x')
# ax2.plot(S, u_ex, color='black')
# ax2.set_xlim(4.25, 4.55)  # Limit the region for zoom
# ax2.set_ylim(-0.001, 0.022)
# ax2.set_yticks([])
# plt.savefig("PME_2_cut.pdf", bbox_inches='tight')
#
#
# fig3 = plt.figure(constrained_layout=True)
# gs = fig3.add_gridspec(3,3)
# f3_ax1 = fig3.add_subplot(gs[0, :])
# f3_ax2 = fig3.add_subplot(gs[1, :-1])
# f3_ax3 = fig3.add_subplot(gs[1:, -1])
# f3_ax4 = fig3.add_subplot(gs[-1, 0])
# f3_ax5 = fig3.add_subplot(gs[-1, -2])

# fig3 = plt.figure(constrained_layout=True, figsize=(7.0, 5.0))
# gs = fig3.add_gridspec(2,2, width_ratios=[3,2], height_ratios=[1,1])
# f3_ax1 = fig3.add_subplot(gs[:, 0])
# f3_ax2 = fig3.add_subplot(gs[0, 1])
# f3_ax3 = fig3.add_subplot(gs[1, 1])
# f3_ax1.plot(S, V_nt, color='blue')
# f3_ax1.plot(S, V_t, color='red', marker='x')
# f3_ax1.plot(S, u_ex, color='black')
# f3_ax1.legend(('MWENO', 'WENO-DS', 'ref. sol.'), loc=1)
# f3_ax1.set_xlabel('x')
# f3_ax1.set_ylabel('u')
# f3_ax2.plot(S, V_nt, color='blue')
# f3_ax2.plot(S, V_t, color='red', marker='x')
# f3_ax2.plot(S, u_ex, color='black')
# f3_ax2.set_xlim(-4.55, -4.25)  # Limit the region for zoom
# f3_ax2.set_ylim(-0.001, 0.022)
# # ax1.set_yticks([])
# f3_ax3.plot(S, V_nt, color='blue')
# f3_ax3.plot(S, V_t, color='red', marker='x')
# f3_ax3.plot(S, u_ex, color='black')
# f3_ax3.set_xlim(4.25, 4.55)  # Limit the region for zoom
# f3_ax3.set_ylim(-0.001, 0.022)
# # ax2.set_yticks([])
# plt.savefig("PME_2_cut.pdf", bbox_inches='tight')

# # m = 2
# fig3 = plt.figure(constrained_layout=True, figsize=(7.0, 5.0))
# gs = fig3.add_gridspec(2,2, width_ratios=[2,2], height_ratios=[1,1])
# f3_ax1 = fig3.add_subplot(gs[0, :])
# f3_ax2 = fig3.add_subplot(gs[1, 0])
# f3_ax3 = fig3.add_subplot(gs[1, 1])
# f3_ax1.plot(S, V_nt, color='blue')
# f3_ax1.plot(S, V_t, color='red', marker='x')
# f3_ax1.plot(S, u_ex, color='black')
# f3_ax1.legend(('MWENO', 'WENO-DS', 'ref. sol.'), loc=1)
# f3_ax1.set_xlabel('x')
# f3_ax1.set_ylabel('u')
# f3_ax2.plot(S, V_nt, color='blue')
# f3_ax2.plot(S, V_t, color='red', marker='x')
# f3_ax2.plot(S, u_ex, color='black')
# f3_ax2.set_xlim(-4.55, -4.25)  # Limit the region for zoom
# f3_ax2.set_ylim(-0.001, 0.022)
# f3_ax2.set_xlabel('x')
# f3_ax2.set_ylabel('u')
# f3_ax3.plot(S, V_nt, color='blue')
# f3_ax3.plot(S, V_t, color='red', marker='x')
# f3_ax3.plot(S, u_ex, color='black')
# f3_ax3.set_xlim(4.25, 4.55)  # Limit the region for zoom
# f3_ax3.set_ylim(-0.001, 0.022)
# f3_ax3.set_xlabel('x')
# f3_ax3.set_ylabel('u')
# plt.savefig("PME_2.pdf", bbox_inches='tight')

# # m = 3
# fig3 = plt.figure(constrained_layout=True, figsize=(7.0, 5.0))
# gs = fig3.add_gridspec(2,2, width_ratios=[2,2], height_ratios=[1,1])
# f3_ax1 = fig3.add_subplot(gs[0, :])
# f3_ax2 = fig3.add_subplot(gs[1, 0])
# f3_ax3 = fig3.add_subplot(gs[1, 1])
# f3_ax1.plot(S, V_nt, color='blue')
# f3_ax1.plot(S, V_t, color='red', marker='x')
# f3_ax1.plot(S, u_ex, color='black')
# f3_ax1.legend(('MWENO', 'WENO-DS', 'ref. sol.'), loc=1)
# f3_ax1.set_xlabel('x')
# f3_ax1.set_ylabel('u')
# f3_ax2.plot(S, V_nt, color='blue')
# f3_ax2.plot(S, V_t, color='red', marker='x')
# f3_ax2.plot(S, u_ex, color='black')
# f3_ax2.set_xlim(-4.35, -4.05)  # Limit the region for zoom
# f3_ax2.set_ylim(-0.001, 0.075)
# f3_ax2.set_xlabel('x')
# f3_ax2.set_ylabel('u')
# f3_ax3.plot(S, V_nt, color='blue')
# f3_ax3.plot(S, V_t, color='red', marker='x')
# f3_ax3.plot(S, u_ex, color='black')
# f3_ax3.set_xlim(4.05, 4.35)  # Limit the region for zoom
# f3_ax3.set_ylim(-0.001, 0.075)
# f3_ax3.set_xlabel('x')
# f3_ax3.set_ylabel('u')
# plt.savefig("PME_3.pdf", bbox_inches='tight')
#
# # m = 4
# fig3 = plt.figure(constrained_layout=True, figsize=(7.0, 5.0))
# gs = fig3.add_gridspec(2,2, width_ratios=[2,2], height_ratios=[1,1])
# f3_ax1 = fig3.add_subplot(gs[0, :])
# f3_ax2 = fig3.add_subplot(gs[1, 0])
# f3_ax3 = fig3.add_subplot(gs[1, 1])
# f3_ax1.plot(S, V_nt, color='blue')
# f3_ax1.plot(S, V_t, color='red', marker='x')
# f3_ax1.plot(S, u_ex, color='black')
# f3_ax1.legend(('MWENO', 'WENO-DS', 'ref. sol.'), loc=1)
# f3_ax1.set_xlabel('x')
# f3_ax1.set_ylabel('u')
# f3_ax2.plot(S, V_nt, color='blue')
# f3_ax2.plot(S, V_t, color='red', marker='x')
# f3_ax2.plot(S, u_ex, color='black')
# f3_ax2.set_xlim(-4.5, -4.275)  # Limit the region for zoom
# f3_ax2.set_ylim(-0.001, 0.02)
# f3_ax2.set_xlabel('x')
# f3_ax2.set_ylabel('u')
# f3_ax3.plot(S, V_nt, color='blue')
# f3_ax3.plot(S, V_t, color='red', marker='x')
# f3_ax3.plot(S, u_ex, color='black')
# f3_ax3.set_xlim(4.275, 4.5)  # Limit the region for zoom
# f3_ax3.set_ylim(-0.001, 0.02)
# f3_ax3.set_xlabel('x')
# f3_ax3.set_ylabel('u')
# plt.savefig("PME_4.pdf", bbox_inches='tight')

