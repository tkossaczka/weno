import torch
import pandas as pd
import numpy as np
from define_WENO_Network_2 import WENONetwork_2
from define_problem_PME import PME
from define_problem_Buckley_Leverett import Buckley_Leverett
import random
import os, sys, argparse
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
problem = PME
train_model = WENONetwork_2()

for j in range(86):
    df = pd.read_csv("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/parameters.txt")
    u_ex = np.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/u_exact_{}.npy".format(j))
    power = float(df[df.sample_id == j]["power"])
    params = {'T': 0.5, 'e': 1e-13, 'L': 6, 'power': power, 'd': 1}
    problem_main = problem(sample_id=None, example="boxes", space_steps=64, time_steps=None, params=params)
    space_steps_exact = u_ex.shape[0]
    time_steps_exact = u_ex.shape[1]
    divider_space = space_steps_exact / problem_main.space_steps
    divider_time = time_steps_exact / problem_main.time_steps
    divider_space = int(divider_space)
    divider_time = int(divider_time)
    u_ex = u_ex[0:space_steps_exact + 1:divider_space, 0:time_steps_exact + 1:divider_time]
    save_path = "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/"
    np.save(os.path.join(save_path, "u_exact64_{}".format(j)), u_ex)


