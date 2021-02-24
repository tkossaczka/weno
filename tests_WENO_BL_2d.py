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
problem= Buckley_Leverett
example = "gravity_2d"

if example == "gravity_2d":
    problem_main = problem(sample_id=None, example = example, space_steps=120, time_steps=None, params=None)
    print(problem_main.params)
    u_init, nn = train_model.init_run_weno(problem_main, vectorized=True, just_one_time_step=False,dim=2)
    u_nt = u_init
    for k in range(nn):
        u_nt = train_model.run_weno_2d(problem_main, u_nt, mweno=False, mapped=True, vectorized=True, trainable=False, k=k)
    x = problem_main.x

UU = u_nt.detach().numpy()
X, Y = np.meshgrid(x, x, indexing="ij")
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, UU, cmap=cm.viridis)

plt.figure(2)
plt.contour(X,Y,UU)