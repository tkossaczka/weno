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

problem_main = problem(space_steps=60, time_steps=None, params=None)
print(problem_main.params)

u_init, nn = train_model.init_run_weno(problem_main, vectorized=True, just_one_time_step=False)
u_nt = u_init
for k in range(nn):
    u_nt = train_model.run_weno(problem_main, u_nt, mweno=False, mapped=True, vectorized=True, trainable=False, k=k)
V_nt, S, _ = problem_main.transformation(u_nt)
plt.plot(S, V_nt)


