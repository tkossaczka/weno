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
# problem = PME
problem = Buckley_Leverett
train_model = WENONetwork_2()

# def validation_problems(j):
#     params_vld = []
#     params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 1, 'G': 3})
#     params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.4, 'G': 0})
#     params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.5, 'G': 5})
#     params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.3, 'G': 3})
#     params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.25, 'G': 1})
#     return params_vld[j]

# def validation_problems(j):
#     params_vld = []
#     params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 1, 'G': 5})
#     params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 1, 'G': 0})
#     params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.5, 'G': 2})
#     params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.5, 'G': 1})
#     params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.25, 'G': 4})
#     return params_vld[j]

def validation_problems(j):
    params_vld = []
    params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 1, 'G': 7})
    params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 1, 'G': 6.5})
    params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.25, 'G': 7})
    params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.1, 'G': 8})
    params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.25, 'G': 9})
    return params_vld[j]

for j in range(5,6):

    # df = pd.read_csv("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Basic_test_set/parameters.txt")
    # u_ex = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Basic_test_set/u_ex_{}".format(j))
    # power = float(df[df.sample_id == j]["power"])
    # params = {'T': 0.5, 'e': 1e-13, 'L': 6, 'power': power, 'd': 1}
    # problem_main = problem(sample_id=None, example="boxes", space_steps=64, time_steps=None, params=params)

    # df = pd.read_csv("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/parameters.txt")
    # u_ex = np.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/u_exact_{}.npy".format(j))
    # C = float(df[df.sample_id == j]["C"])
    # G = float(df[df.sample_id == j]["G"])
    # params =  {'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': C, 'G': G}
    # problem_main = problem(sample_id=None, example="gravity", space_steps=128, time_steps=None, params=params)

    u_ex = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Basic_test_set_1/u_ex_{}".format(j))
    params_test = validation_problems(j)
    C = params_test["C"]
    G = params_test["G"]
    params =  {'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': C, 'G': G}
    # params = {'T': 0.7, 'L': -2, 'R': 2, 'e': 1e-13}
    problem_main = problem(sample_id=None, example="gravity", space_steps=128, time_steps=None, params=params)
    u_ex = u_ex.detach().numpy()

    space_steps_exact = u_ex.shape[0]
    time_steps_exact = u_ex.shape[1]
    divider_space = space_steps_exact / problem_main.space_steps
    divider_time = time_steps_exact / problem_main.time_steps
    divider_space = int(divider_space)
    divider_time = int(divider_time)
    u_ex = u_ex[0:space_steps_exact + 1:divider_space, 0:time_steps_exact + 1:divider_time]

    # save_path = "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Basic_test_set"
    # u_ex = u_ex.detach().numpy()
    # np.save(os.path.join(save_path, "u_exact64_{}".format(j)), u_ex)

    # save_path = "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024"
    # np.save(os.path.join(save_path, "u_exact128_{}.npy".format(j)), u_ex)

    u_ex = torch.Tensor(u_ex)
    save_path = "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Basic_test_set_1"
    torch.save(u_ex, os.path.join(save_path, "u_ex128_{}".format(j)))



