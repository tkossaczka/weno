import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from define_WENO_Network_2 import WENONetwork_2
from scipy.stats import norm
from define_problem_PME import PME
from initial_condition_generator import init_PME

torch.set_default_dtype(torch.float64)

def validation_problems_barenblatt_2d(j):
    params_vld = []
    params_vld.append({'T': 2, 'e': 1e-13, 'L': 10, 'power': 2, 'd': 2})
    params_vld.append({'T': 2, 'e': 1e-13, 'L': 10, 'power': 3, 'd': 2})
    params_vld.append({'T': 2, 'e': 1e-13, 'L': 10, 'power': 4, 'd': 2})
    params_vld.append({'T': 2, 'e': 1e-13, 'L': 10, 'power': 5, 'd': 2})
    params_vld.append({'T': 2, 'e': 1e-13, 'L': 10, 'power': 6, 'd': 2})
    params_vld.append({'T': 2, 'e': 1e-13, 'L': 10, 'power': 7, 'd': 2})
    params_vld.append({'T': 2, 'e': 1e-13, 'L': 10, 'power': 8, 'd': 2})
    return params_vld[j]

train_model = WENONetwork_2()
# train_model = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/Models/Model_47/999.pt")
train_model = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/Models_2d/Model_5/400.pt")   # 4/200 # 5/400
problem= PME
example = "Barenblatt_2d"
rng = 7
err_nt_max_vec = np.zeros(rng)
err_nt_mean_vec = np.zeros(rng)
err_t_max_vec = np.zeros(rng)
err_t_mean_vec = np.zeros(rng)

for j in range(rng):
    if example == "boxes_2d":
        params = {'T': 0.5, 'power': 2, 'd': 2, 'L': 10, 'e': 1e-13}
        problem_main = problem(sample_id=None, example=example, space_steps=80, time_steps=None, params=params)
        print(problem_main.params)
        u_init, nn = train_model.init_run_weno(problem_main, vectorized=True, just_one_time_step=False, dim=2)
        u_nt = u_init
        for k in range(nn):
            u_nt = train_model.run_weno_2d(problem_main, u_nt, mweno=True, mapped=False, vectorized=True, trainable=False, k=k)
        x = problem_main.x
        with torch.no_grad():
            u_t = u_init
            for k in range(nn):
                u_t = train_model.run_weno_2d(problem_main, u_t, mweno=True, mapped=False, vectorized=True, trainable=True, k=k)
    elif example == "Barenblatt_2d":
        params = validation_problems_barenblatt_2d(j)
        problem_main = problem(sample_id=None, example=example, space_steps=64, time_steps=None, params=params)
        print(problem_main.params)
        u_init, nn = train_model.init_run_weno(problem_main, vectorized=True, just_one_time_step=False, dim=2)
        u_nt = u_init
        for k in range(nn):
            u_nt = train_model.run_weno_2d(problem_main, u_nt, mweno=True, mapped=False, vectorized=True, trainable=False, k=k)
        x = problem_main.x
        with torch.no_grad():
            u_t = u_init
            for k in range(nn):
                u_t = train_model.run_weno_2d(problem_main, u_t, mweno=True, mapped=False, vectorized=True, trainable=True, k=k)
        u_ex = problem_main.exact(problem_main.params["T"])
        space_steps = problem_main.space_steps
        error_nt_max = np.max(np.max(np.abs(u_ex - u_nt.detach().numpy())))
        error_t_max = np.max(np.max(np.abs(u_ex - u_t.detach().numpy())))
        error_nt_mean = (1 / space_steps) * (np.sqrt(np.sum((u_ex - u_nt.detach().numpy()) ** 2)))
        error_t_mean = (1 / space_steps) * (np.sqrt(np.sum((u_ex - u_t.detach().numpy()) ** 2)))
        err_nt_max_vec[j] = error_nt_max
        err_t_max_vec[j] = error_t_max
        err_nt_mean_vec[j] = error_nt_mean
        err_t_mean_vec[j] = error_t_mean

err_mat = np.zeros((4,rng))
err_mat[0,:] = err_nt_max_vec
err_mat[1,:] = err_t_max_vec
err_mat[2,:] = err_nt_mean_vec
err_mat[3,:] = err_t_mean_vec

ratio_max = err_mat[0,:]/err_mat[1,:]
ratio_l2 = err_mat[2,:]/err_mat[3,:]

# UU = u_nt.detach().numpy()
# X, Y = np.meshgrid(x, x, indexing="ij")
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(X, Y, UU, cmap=cm.viridis)
# plt.figure(2)
# plt.contour(X,Y,UU, 20)
#
# UU2 = u_t.detach().numpy()
# X, Y = np.meshgrid(x, x, indexing="ij")
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(X, Y, UU2, cmap=cm.viridis)
# plt.figure(4)
# plt.contour(X,Y,UU2, 20)