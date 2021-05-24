import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from define_WENO_Network_2 import WENONetwork_2
from scipy.stats import norm
from define_problem_Buckley_Leverett import Buckley_Leverett
from validation_problems import validation_problems
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

torch.set_default_dtype(torch.float64)

train_model = WENONetwork_2()
train_model = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Models/Model_42/4800.pt")

problem= Buckley_Leverett

example = "gravity"
# example = "degenerate"

valid_problems = validation_problems.validation_problems_BL
_, rng, folder = valid_problems(0)
u_exs = validation_problems.exacts_test_BL(folder)

err_nt_max_vec = np.zeros(rng)
err_nt_mean_vec = np.zeros(rng)
err_t_max_vec = np.zeros(rng)
err_t_mean_vec = np.zeros(rng)

for j in range(1,2):
    params, _, _ = valid_problems(j)
    problem_main = problem(sample_id=None, example = example, space_steps=128, time_steps=None, params=params)
    print(problem_main.params)
    u_init, nn = train_model.init_run_weno(problem_main, vectorized=True, just_one_time_step=False)
    u_nt = u_init
    for k in range(nn):
        u_nt = train_model.run_weno(problem_main, u_nt, mweno=True, mapped=False, vectorized=True, trainable=False, k=k)
    V_nt, S, _ = problem_main.transformation(u_nt)
    with torch.no_grad():
        u_t = u_init
        for k in range(nn):
            u_t = train_model.run_weno(problem_main, u_t, mweno=True, mapped=False, vectorized=True, trainable=True, k=k)
        V_t, S, _ = problem_main.transformation(u_t)
    u_ex = u_exs[j][:, -1]
    R = problem_main.params['R']
    sp_st = problem_main.space_steps
    error_t_mean = np.sqrt(R / sp_st) * (np.sqrt(np.sum((u_t.detach().numpy() - u_ex.detach().numpy()) ** 2)))
    error_nt_mean = np.sqrt(R / sp_st) * (np.sqrt(np.sum((u_nt.detach().numpy() - u_ex.detach().numpy()) ** 2)))
    error_nt_max = np.max(np.absolute(u_ex.detach().numpy() - u_nt.detach().numpy()))
    error_t_max = np.max(np.absolute(u_ex.detach().numpy() - u_t.detach().numpy()))
    err_nt_max_vec[j] = error_nt_max
    err_t_max_vec[j] = error_t_max
    err_nt_mean_vec[j] = error_nt_mean
    err_t_mean_vec[j] = error_t_mean
    plt.figure(j + 1)
    plt.plot(S, V_nt, S, V_t, S, u_ex)

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

# # BL C=1, G=5
# fig, ax = plt.subplots(figsize=(5.0, 5.0))
# ax.plot(S, u_nt, color='blue')
# ax.plot(S, u_t, color='red', marker='x')
# ax.plot(S, u_ex, color='black')
# ax.legend(('WENO', 'WENO-DS', 'ref. sol.'), loc=2)
# ax.set_xlabel('x')
# ax.set_ylabel('u')
# axins = inset_axes(ax, width=1, height=2, loc=7)
# axins.plot(S, u_nt, color='blue')
# axins.plot(S, u_t, color='red', marker='x')
# axins.plot(S, u_ex, color='black')
# axins.set_xlim(0.532, 0.56)  # Limit the region for zoom
# axins.set_ylim(0.8, 1.01)
# plt.xticks(visible=False)  # Not present ticks
# plt.yticks(visible=False)
# # axins2 = inset_axes(ax, width=2, height=0.5, loc=2)
# # axins2.plot(x, u_nt_JS, color='blue')
# # axins2.plot(x, u_nt, color='green')
# # axins2.plot(x, u_t, color='red')
# # axins2.plot(x_ex, u_ex, color='black')
# # axins2.set_xlim(0.5, 0.725)  # Limit the region for zoom
# # axins2.set_ylim(1.355, 1.37)
# # plt.xticks(visible=False)  # Not present ticks
# # plt.yticks(visible=False)
# mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")
# # mark_inset(ax, axins2, loc1=1, loc2=4, fc="none", ec="0.5")
# # plt.draw()
# # plt.show()
# plt.savefig("BL_01.pdf", bbox_inches='tight')

# # BL C=1, G=0
# fig, ax = plt.subplots(figsize=(5.0, 5.0))
# ax.plot(S, u_nt, color='blue')
# ax.plot(S, u_t, color='red', marker='x')
# ax.plot(S, u_ex, color='black')
# ax.legend(('WENO', 'WENO-DS', 'ref. sol.'), loc=2)
# ax.set_xlabel('x')
# ax.set_ylabel('u')
# axins = inset_axes(ax, width=1, height=2, loc=7)
# axins.plot(S, u_nt, color='blue')
# axins.plot(S, u_t, color='red', marker='x')
# axins.plot(S, u_ex, color='black')
# axins.set_xlim(0.43, 0.47)  # Limit the region for zoom
# axins.set_ylim(0.8, 1.01)
# plt.xticks(visible=False)  # Not present ticks
# plt.yticks(visible=False)
# # axins2 = inset_axes(ax, width=2, height=0.5, loc=2)
# # axins2.plot(x, u_nt_JS, color='blue')
# # axins2.plot(x, u_nt, color='green')
# # axins2.plot(x, u_t, color='red')
# # axins2.plot(x_ex, u_ex, color='black')
# # axins2.set_xlim(0.5, 0.725)  # Limit the region for zoom
# # axins2.set_ylim(1.355, 1.37)
# # plt.xticks(visible=False)  # Not present ticks
# # plt.yticks(visible=False)
# mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")
# # mark_inset(ax, axins2, loc1=1, loc2=4, fc="none", ec="0.5")
# # plt.draw()
# # plt.show()
# plt.savefig("BL_00.pdf", bbox_inches='tight')