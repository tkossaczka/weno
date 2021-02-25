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

train_model = WENONetwork_2()
train_model = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/Models/Model_46/650.pt")
problem= PME
example = "Barenblatt_2d"

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
    params = {'T': 1.4, 'power': 5, 'd': 2, 'L': 10, 'e': 1e-13}
    problem_main = problem(sample_id=None, example=example, space_steps=30, time_steps=None, params=params)
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
    error_nt_mean = (1 / space_steps) * (np.sqrt(np.sum(u_ex - u_nt.detach().numpy()) ** 2))
    error_t_mean = (1 / space_steps) * (np.sqrt(np.sum(u_ex - u_t.detach().numpy()) ** 2))

UU = u_nt.detach().numpy()
X, Y = np.meshgrid(x, x, indexing="ij")
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, UU, cmap=cm.viridis)
plt.figure(2)
plt.contour(X,Y,UU, 20)

UU2 = u_t.detach().numpy()
X, Y = np.meshgrid(x, x, indexing="ij")
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, UU2, cmap=cm.viridis)
plt.figure(4)
plt.contour(X,Y,UU2, 20)