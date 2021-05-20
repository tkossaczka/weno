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

torch.set_default_dtype(torch.float64)

train_model = WENONetwork_2()
train_model = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Models/Model_42/4800.pt")

problem= Buckley_Leverett

example = "gravity"
# example = "degenerate"

valid_problems = validation_problems.validation_problems_BL_5
_, rng, folder = valid_problems(0)
u_exs = validation_problems.exacts_validation_BL(folder)

err_nt_max_vec = np.zeros(rng)
err_nt_mean_vec = np.zeros(rng)
err_t_max_vec = np.zeros(rng)
err_t_mean_vec = np.zeros(rng)

for j in range(rng):
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

err_mat = np.zeros((4,rng))
err_mat[0,:] = err_nt_max_vec
err_mat[1,:] = err_t_max_vec
err_mat[2,:] = err_nt_mean_vec
err_mat[3,:] = err_t_mean_vec

ratio_max = err_mat[0,:]/err_mat[1,:]
ratio_l2 = err_mat[2,:]/err_mat[3,:]
