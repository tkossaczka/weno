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
train_model = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/Models/Model_7/29.pt")

def validation_problems(j):
    params_vld = []
    params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 2, 'd': 1})
    params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 3, 'd': 1})
    params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 4, 'd': 1})
    params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 5, 'd': 1})
    return params_vld[j]

problem= PME
type = "Barenblatt"
rng = 4
err_nt_max_vec = np.zeros(rng)
err_nt_mean_vec = np.zeros(rng)
err_t_max_vec = np.zeros(rng)
err_t_mean_vec = np.zeros(rng)

for j in range(rng):
    print(j)
    params = validation_problems(j)
    # params = {'T': 2, 'e': 1e-13, 'L': 6, 'power': 8, 'd': 1}
    #params = None
    problem_main = problem(type=type, space_steps=40, time_steps=None, params = params)
    params = problem_main.get_params()
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
    if type == "Barenblatt":
        # parameters needed for the computation of exact solution
        params_main = problem_main.params
        T = params_main['T']
        L = params_main['L']
        sp_st = problem_main.space_steps
        u_ex = problem_main.exact(T,type)
        error_t_mean = np.sqrt(2*L / sp_st) * (np.sqrt(np.sum((u_t.detach().numpy() - u_ex) ** 2)))
        error_nt_mean = np.sqrt(2*L / sp_st) * (np.sqrt(np.sum((u_nt.detach().numpy() - u_ex) ** 2)))
        error_nt_max = np.max(np.absolute(u_ex - u_nt.detach().numpy()))
        error_t_max = np.max(np.absolute(u_ex - u_t.detach().numpy()))
        err_nt_max_vec[j] = error_nt_max
        err_t_max_vec[j] = error_t_max
        err_nt_mean_vec[j] = error_nt_mean
        err_t_mean_vec[j] = error_t_mean
        plt.figure(j + 1)
        plt.plot(S, V_nt, S, V_t, S, u_ex)
    else:
        plt.plot(S, V_nt,S, V_t)

if type == "Barenblatt":
    err_mat = np.zeros((4,rng))
    err_mat[0,:] = err_nt_max_vec
    err_mat[1,:] = err_t_max_vec
    err_mat[2,:] = err_nt_mean_vec
    err_mat[3,:] = err_t_mean_vec


# plt.figure(2)
# plt.plot(S,V_nt, 'o')
# plt.plot(S,V_nt, S, u_ex)
# plt.plot(S,V_nt, S, V_t, S, u_ex)

# VV = V.detach().numpy()
# X, Y = np.meshgrid(x, t, indexing="ij")
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(X, Y, VV, cmap=cm.viridis)


