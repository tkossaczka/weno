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

torch.set_default_dtype(torch.float64)

#train_model = WENONetwork_2()
# train_model = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/Models/Model_41/690.pt") #45/500 #46/650 # 47/999 # 41/690 for boxes
train_model = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/Models_boxes/Model_0/200.pt") #45/500 #46/650 # 47/999 # 41/690 for boxes

def validation_problems(j):
    params_vld = []
    params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 2, 'd': 1})
    params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 3, 'd': 1})
    params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 4, 'd': 1})
    params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 5, 'd': 1})
    params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 6, 'd': 1})
    params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 7, 'd': 1})
    params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 8, 'd': 1})
    return params_vld[j]

def validation_problems_boxes(j):
    params_vld = []
    params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': 2, 'd': 1})
    params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': 3, 'd': 1})
    params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': 4, 'd': 1})
    params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': 5, 'd': 1})
    return params_vld[j]
u_ex_0 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Basic_test_set/u_ex_0")
u_ex_1 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Basic_test_set/u_ex_1")
u_ex_2 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Basic_test_set/u_ex_2")
u_ex_3 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Basic_test_set/u_ex_3")
u_exs = [u_ex_0[0:1024 + 1:16, :], u_ex_1[0:1024 + 1:16, :], u_ex_2[0:1024 + 1:16, :], u_ex_3[0:1024 + 1:16, :]]

# df=pd.read_csv("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Test_set/parameters.txt")
# def validation_problems_boxes(j):
#     power = float(df[df.sample_id==j]["power"])
#     params_vld = []
#     params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': power, 'd': 1})
#     params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': power, 'd': 1})
#     params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': power, 'd': 1})
#     params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': power, 'd': 1})
#     params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': power, 'd': 1})
#     params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': power, 'd': 1})
#     params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': power, 'd': 1})
#     params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': power, 'd': 1})
#     return params_vld[j]
# u_exs = []
# for j in range(8):
#     u_exs.append(torch.Tensor(np.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Test_set/u_exact64_{}.npy".format(j))))

problem= PME
example = "boxes"
rng = 4
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
    with torch.no_grad():
        for k in range(nn):
            u_t = train_model.run_weno(problem_main, u_t, mweno=True, mapped=False, vectorized=True, trainable=True, k=k)
        V_t, _, _ = problem_main.transformation(u_t)
    u_nt = u_init
    for k in range(nn):
        u_nt = train_model.run_weno(problem_main, u_nt, mweno=True, mapped=False, vectorized=True, trainable=False, k=k)
    V_nt, S, _ = problem_main.transformation(u_nt)
    if example == "Barenblatt":
        params_main = problem_main.params
        T = params_main['T']
        L = params_main['L']
        sp_st = problem_main.space_steps
        u_ex = problem_main.exact(T)
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
        params_main = problem_main.params
        u_ex = u_exs[j][:,-1]
        L = params_main['L']
        sp_st = problem_main.space_steps
        error_t_mean = np.sqrt(2 * L / sp_st) * (np.sqrt(np.sum((u_t.detach().numpy() - u_ex.detach().numpy()) ** 2)))
        error_nt_mean = np.sqrt(2 * L / sp_st) * (np.sqrt(np.sum((u_nt.detach().numpy() - u_ex.detach().numpy()) ** 2)))
        error_nt_max = np.max(np.absolute(u_ex.detach().numpy() - u_nt.detach().numpy()))
        error_t_max = np.max(np.absolute(u_ex.detach().numpy() - u_t.detach().numpy()))
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


