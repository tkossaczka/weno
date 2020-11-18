from initial_condition_generator import init_Euler
import numpy as np
import torch
from define_WENO_Euler import WENONetwork_Euler
from define_Euler_system import Euler_system
import matplotlib.pyplot as plt
from matplotlib import cm
import os, sys

train_model = WENONetwork_Euler()
torch.set_default_dtype(torch.float64)
params=None
problem = Euler_system
sp_st = 2048
init_cond = "Sod"
base_path = "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Euler_System_Test/Euler_System_Data/"

for j in range(3):
    print(j)
    problem_main = problem(space_steps=sp_st, init_cond=init_cond, time_steps=None, params=params, time_disc=None, init_mid=False, init_general=True)
    params = problem_main.get_params()
    gamma = params['gamma']
    tt = problem_main.t
    q_0, q_1, q_2, lamb, nn, h = train_model.init_Euler(problem_main, vectorized = True, just_one_time_step=False)
    p = problem_main.p
    u = problem_main.u
    rho = problem_main.rho
    _,x,t = problem_main.transformation(q_0)

    x_ex = np.linspace(0, 1, 2048+1)
    p_ex = torch.zeros((x_ex.shape[0],t.shape[0]))
    rho_ex = torch.zeros((x_ex.shape[0],t.shape[0]))
    u_ex = torch.zeros((x_ex.shape[0],t.shape[0]))

    for k in range(0,t.shape[0]):
        p_ex[:,k], rho_ex[:,k], u_ex[:,k], _,_ = problem_main.exact(x_ex, t[k])

    if not os.path.exists(base_path):
        os.mkdir(base_path)
    path_rho = os.path.join(base_path, "rho_ex_{}".format(j))
    path_u = os.path.join(base_path, "u_ex_{}".format(j))
    path_p = os.path.join(base_path, "p_ex_{}".format(j))
    torch.save(rho_ex, path_rho)
    torch.save(u_ex, path_u)
    torch.save(p_ex, path_p)

    if not os.path.exists(os.path.join(base_path, "parameters.txt")):
        with open(os.path.join(base_path, "parameters.txt"), "a") as f:
            f.write("{},{},{},{},{},{}\n".format("rho[0]","rho[1]","u[0]","u[1]","p[0]","p[1]"))
    with open(os.path.join(base_path, "parameters.txt"), "a") as f:
        f.write("{},{},{},{},{},{}\n".format(rho[0],rho[1],u[0],u[1],p[0],p[1]))

    # torch.save(rho_ex, "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Euler_System_Test/Euler_System_Data/rho_ex_{}".format(j))
    # torch.save(u_ex, "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Euler_System_Test/Euler_System_Data/u_ex_{}".format(j))
    # torch.save(p_ex, "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Euler_System_Test/Euler_System_Data/p_ex_{}".format(j))


# X, Y = np.meshgrid(x_ex, t, indexing="ij")
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(X, Y, rho_ex.detach().numpy(), cmap=cm.viridis)