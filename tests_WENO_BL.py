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
train_model = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Models/Model_0/99.pt") #45/500 #46/650

problem= Buckley_Leverett

example = "gravity"
# example = "degenerate"

problem_main = problem(sample_id=None, example = example, space_steps=64, time_steps=None, params=None)
print(problem_main.params)

u_init, nn = train_model.init_run_weno(problem_main, vectorized=True, just_one_time_step=False)
u_nt = u_init
for k in range(nn):
    u_nt = train_model.run_weno(problem_main, u_nt, mweno=False, mapped=True, vectorized=True, trainable=False, k=k)
V_nt, S, _ = problem_main.transformation(u_nt)

with torch.no_grad():
    u_t = u_init
    for k in range(nn):
        u_t = train_model.run_weno(problem_main, u_t, mweno=False, mapped=True, vectorized=True, trainable=True, k=k)
    V_t, S, _ = problem_main.transformation(u_t)

plt.plot(S, V_nt, S, V_t)


