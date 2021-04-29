import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

#train_model = WENONetwork_2()
# train_model = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/Models/Model_41/690.pt") #41/690 for boxes
# train_model = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/Models_boxes/Model_5/999.pt") #5/999 for boxes
train_model = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/Models/Model_13/40.pt") #45/500 #46/650 # 47/999

def validation_problems(j):
    params_vld = []
    params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 2, 'd': 1})
    params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 3, 'd': 1})
    params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 4, 'd': 1})
    params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 5, 'd': 1})
    params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 6, 'd': 1})
    params_vld.append({'T': 1.2, 'e': 1e-13, 'L': 6, 'power': 7, 'd': 1})
    params_vld.append({'T': 1.2, 'e': 1e-13, 'L': 6, 'power': 8, 'd': 1})
    return params_vld[j]

# a = random.uniform(2, 8)
# b = random.uniform(2, 8)
# c = random.uniform(2, 8)
# d = random.uniform(2, 8)
# e = random.uniform(2, 8)
# print(a,b,c,d,e)
#
a = 2.157
aa = 3.012
b = 3.697
bb = 3.987
c = 4.158
cc = 4.572
d = 4.723
dd = 5.041
e = 5.568
ee = 6.087
f = 6.284
ff = 7.124
g = 7.958
print(a,b,c,d,e,f,g)

# def validation_problems(j):
#     params_vld = []
#     params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': a, 'd': 1})
#     params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': aa, 'd': 1})
#     params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': b, 'd': 1})
#     params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': bb, 'd': 1})
#     params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': c, 'd': 1})
#     params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': cc, 'd': 1})
#     params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': d, 'd': 1})
#     params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': dd, 'd': 1})
#     params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': e, 'd': 1})
#     params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': ee, 'd': 1})
#     params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': f, 'd': 1})
#     params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': ff, 'd': 1})
#     params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': g, 'd': 1})
#     return params_vld[j]

# def validation_problems(j):
#     params_vld = []
#     params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': a, 'd': 1})
#     params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': b, 'd': 1})
#     params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': c, 'd': 1})
#     params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': d, 'd': 1})
#     params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': e, 'd': 1})
#     return params_vld[j]

# def validation_problems(j):
#     params_vld = []
#     params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 2.4, 'd': 1})
#     params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 3.8, 'd': 1})
#     params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 4.2, 'd': 1})
#     params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 5.7, 'd': 1})
#     return params_vld[j]

def validation_problems_boxes(j):
    params_vld = []
    params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': 2, 'd': 1})
    params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': 3, 'd': 1})
    params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': 4, 'd': 1})
    params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': 5, 'd': 1})
    params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': 6, 'd': 1})
    return params_vld[j]
u_ex_0 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Basic_test_set/u_ex_0")
u_ex_1 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Basic_test_set/u_ex_1")
u_ex_2 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Basic_test_set/u_ex_2")
u_ex_3 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Basic_test_set/u_ex_3")
u_ex_4 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Basic_test_set/u_ex_4")
u_exs = [u_ex_0[0:1024 + 1:16, :], u_ex_1[0:1024 + 1:16, :], u_ex_2[0:1024 + 1:16, :], u_ex_3[0:1024 + 1:16, :], u_ex_4[0:1024 + 1:16, :]]

# df=pd.read_csv("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Test_set/parameters.txt")
# def validation_problems_boxes(j):
#     power = float(df[df.sample_id==j]["power"])
#     params_vld = []
#     params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': power, 'd': 1})
#     params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': power, 'd': 1})
#     params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': power, 'd': 1})
#     params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': power, 'd': 1})
#     params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': power, 'd': 1})
#     return params_vld[j]
# u_exs = []
# for j in range(5):
#     u_exs.append(torch.Tensor(np.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Test_set/u_exact64_{}.npy".format(j))))

problem= PME
# example = "boxes"
# rng = 1
example = "Barenblatt"
rng = 5
err_nt_max_vec = np.zeros(rng)
err_nt_mean_vec = np.zeros(rng)
err_t_max_vec = np.zeros(rng)
err_t_mean_vec = np.zeros(rng)

for j in range(rng):
    print(j)
    if example == "Barenblatt":
        params = validation_problems(j)
        problem_main = problem(sample_id = None, example=example, space_steps=64, time_steps=None, params=params)
    else:
        params = validation_problems_boxes(j)
        problem_main = problem(sample_id = None, example=example, space_steps=64, time_steps=None, params=params)
    # params = {'T': 2, 'e': 1e-13, 'L': 6, 'power': 8, 'd': 1}
    # problem_main = problem(example=example, space_steps=64, time_steps=None, params = params)
    params = problem_main.get_params()
    print(params)
    # problem_main.initial_condition, _ = init_PME(problem_main.x, height=1)
    # problem_main.initial_condition = torch.Tensor(problem_main.initial_condition)
    u_init, nn = train_model.init_run_weno(problem_main, vectorized=True, just_one_time_step=False)
    u_t = u_init
    # plt.figure(1)
    with torch.no_grad():
        for k in range(nn):
            u_t = train_model.run_weno(problem_main, u_t, mweno=True, mapped=False, vectorized=True, trainable=True, k=k)
            V_t, _, _ = problem_main.transformation(u_t)
            # plt.plot(u_t)
    u_nt = u_init
    #plt.figure(2)
    for k in range(nn):
        u_nt = train_model.run_weno(problem_main, u_nt, mweno=True, mapped=False, vectorized=True, trainable=False, k=k)
        V_nt, S, _ = problem_main.transformation(u_nt)
        #plt.plot(V_nt)
    if example == "Barenblatt":
        params_main = problem_main.params
        T = params_main['T']
        L = params_main['L']
        sp_st = problem_main.space_steps
        u_ex = problem_main.exact(T)
        error_t_mean = np.sqrt(2*L / sp_st) * (np.sqrt(np.sum((V_t.detach().numpy() - u_ex) ** 2)))
        error_nt_mean = np.sqrt(2*L / sp_st) * (np.sqrt(np.sum((V_nt.detach().numpy() - u_ex) ** 2)))
        error_nt_max = np.max(np.absolute(u_ex - V_nt.detach().numpy()))
        error_t_max = np.max(np.absolute(u_ex - V_t.detach().numpy()))
        err_nt_max_vec[j] = error_nt_max
        err_t_max_vec[j] = error_t_max
        err_nt_mean_vec[j] = error_nt_mean
        err_t_mean_vec[j] = error_t_mean
        plt.figure(j + 1)
        plt.plot(S, V_nt, S, V_t, S, u_ex)
    else:
        params_main = problem_main.params
        u_ex = u_exs[j][:,-1]
        L = params_main['L']
        sp_st = problem_main.space_steps
        error_t_mean = np.sqrt(2 * L / sp_st) * (np.sqrt(np.sum((V_t.detach().numpy() - u_ex.detach().numpy()) ** 2)))
        error_nt_mean = np.sqrt(2 * L / sp_st) * (np.sqrt(np.sum((V_nt.detach().numpy() - u_ex.detach().numpy()) ** 2)))
        error_nt_max = np.max(np.absolute(u_ex.detach().numpy() - V_nt.detach().numpy()))
        error_t_max = np.max(np.absolute(u_ex.detach().numpy() - V_t.detach().numpy()))
        err_nt_max_vec[j] = error_nt_max
        err_t_max_vec[j] = error_t_max
        err_nt_mean_vec[j] = error_nt_mean
        err_t_mean_vec[j] = error_t_mean
        plt.figure(j + 1)
        plt.plot(S, V_nt, S, V_t, S, u_ex)

err_mat = np.zeros((4,rng))
err_mat[0,:] = err_nt_max_vec
err_mat[1,:] = err_t_max_vec
err_mat[2,:] = err_nt_mean_vec
err_mat[3,:] = err_t_mean_vec

ratio_max = err_mat[0,:]/err_mat[1,:]
ratio_l2 = err_mat[2,:]/err_mat[3,:]

# plt.figure(2)
# plt.plot(S,V_nt, 'o')
# plt.plot(S,V_nt, S, u_ex)
# plt.plot(S,V_nt, S, V_t, S, u_ex)

# t = problem_main.time
# UU = uu
# X, Y = np.meshgrid(S, t, indexing="ij")
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(X, Y, UU, cmap=cm.viridis)

V_nt = V_nt.detach().numpy()
V_t = V_t.detach().numpy()

# fig, ax = plt.subplots()
# ax.plot(S, V_nt, color='blue') #, marker='o')
# ax.plot(S, V_t, color='red', marker='x')
# ax.plot(S, u_ex, color='black')
# ax.legend(('WENO-Z', 'WENO-DS', 'ref. sol.'), loc=1)
# ax.set_xlabel('x')
# ax.set_ylabel('u')
# axins = inset_axes(ax, width=1.5, height=1.5, loc=6)
# axins.plot(S, V_nt, color='blue')
# axins.plot(S, V_t, color='red', marker='x')
# axins.plot(S, u_ex, color='black')
# axins.set_xlim(-4.55, -4.25)  # Limit the region for zoom
# axins.set_ylim(-0.01, 0.03)
# plt.xticks(visible=False)  # Not present ticks
# plt.yticks(visible=False)
# axins2 = inset_axes(ax, width=1.5, height=1.5, loc=7)
# axins2.plot(S, V_nt, color='blue') #, marker='o')
# axins2.plot(S, V_t, color='red', marker='x')
# axins2.plot(S, u_ex, color='black')
# axins2.set_xlim(4.25, 4.55)  # Limit the region for zoom
# axins2.set_ylim(-0.01, 0.03)
# plt.xticks(visible=False)  # Not present ticks
# plt.yticks(visible=False)
# mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.5")
# mark_inset(ax, axins2, loc1=3, loc2=4, fc="none", ec="0.5")
# plt.draw()
# plt.show()
#
# fig, (ax1, ax2) = plt.subplots(2)
# fig.suptitle('Vertically stacked subplots')
# ax1.plot(S, V_nt, color='blue')
# ax1.plot(S, V_t, color='red', marker='x')
# ax1.plot(S, u_ex, color='black')
# ax1.set_xlim(-4.55, -4.25)  # Limit the region for zoom
# ax1.set_ylim(-0.001, 0.022)
# ax2.plot(S, V_nt, color='blue')
# ax2.plot(S, V_t, color='red', marker='x')
# ax2.plot(S, u_ex, color='black')
# ax2.set_xlim(4.25, 4.55)  # Limit the region for zoom
# ax2.set_ylim(-0.001, 0.022)

