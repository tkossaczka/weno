import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from define_WENO_Network_2 import WENONetwork_2
from scipy.stats import norm
from define_problem_Buckley_Leverett import Buckley_Leverett

torch.set_default_dtype(torch.float64)

train_model = WENONetwork_2()
train_model = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Models/Model_42/4800.pt")
problem= Buckley_Leverett
example = "gravity_2d"

if example == "gravity_2d":
    problem_main = problem(sample_id=None, example = example, space_steps=120, time_steps=None, params=None)
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

UU = u_nt.detach().numpy()
X, Y = np.meshgrid(x, x, indexing="ij")
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, UU, cmap=cm.viridis)
plt.figure(2)
plt.contour(X,Y,UU,20)

UU2 = u_t.detach().numpy()
X, Y = np.meshgrid(x, x, indexing="ij")
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, UU2, cmap=cm.viridis)
plt.figure(4)
plt.contour(X,Y,UU2, 20)

u_ex = np.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/BL_2d/u_ex_2.npy")
x_ex = np.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/BL_2d/x.npy")

space_steps_exact = 120
space_steps = 60
divider_space = space_steps_exact / space_steps
divider_space = int(divider_space)
u_exact_adjusted = u_ex[0:space_steps_exact+1:divider_space,0:space_steps_exact+1:divider_space]

error_nt_max = np.max(np.max(np.abs(u_exact_adjusted-UU)))
error_t_max = np.max(np.max(np.abs(u_exact_adjusted-UU2)))
error_nt_mean = (1 / space_steps)*(np.sqrt(np.sum((u_exact_adjusted-UU)** 2)))
error_t_mean = (1 / space_steps)*(np.sqrt(np.sum((u_exact_adjusted-UU2)** 2)))

X_ex, Y_ex = np.meshgrid(x_ex, x_ex, indexing="ij")
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X_ex, Y_ex, u_ex, cmap=cm.viridis)
plt.figure(6)
plt.contour(X_ex, Y_ex, u_ex, 20)

# np.save("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/BL_2d/u_ex_t.npy",UU2)
# np.save("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/BL_2d/x.npy",x)

# # 120 trained
# u_t = np.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/BL_2d/u_ex_t.npy")
# x_ex = np.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/BL_2d/x.npy")
# X_ex, Y_ex = np.meshgrid(x_ex, x_ex, indexing="ij")
# fig = plt.figure(figsize=(5.0, 5.0))
# ax = fig.gca(projection='3d')
# ax.plot_surface(X_ex, Y_ex, u_t, cmap=cm.viridis)
# plt.savefig("BL_2d.pdf", bbox_inches='tight')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# vmin = 0.1
# vmax = 1
# levels = np.linspace(vmin, vmax, 21)
# plt.figure(figsize=(5.0, 5.0))
# plt.contour(X_ex, Y_ex, u_t, 20,levels=levels, vmin=vmin, vmax=vmax)
# plt.savefig("BL_2d_cont.pdf", bbox_inches='tight')


