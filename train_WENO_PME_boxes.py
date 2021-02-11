from define_WENO_Network_2 import WENONetwork_2
import torch
from torch import optim
from define_problem_PME import PME
from initial_condition_generator import init_PME
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
import random

torch.set_default_dtype(torch.float64)

# TRAIN NETWORK
train_model = WENONetwork_2()

# DROP PROBLEM FOR TRAINING
problem_class = PME

def monotonicity_loss(u):
    monotonicity = torch.sum(torch.max(u[:-1]-u[1:], torch.Tensor([0.0])))
    loss = monotonicity
    return loss

def exact_loss(u, u_ex):
    error = train_model.compute_error(u, u_ex)
    loss = error
    return loss

#optimizer = optim.SGD(train_model.parameters(), lr=0.1)
optimizer = optim.Adam(train_model.parameters(), lr=0.001)

def validation_problems(j):
    params_vld = []
    params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': 2, 'd': 1})
    params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': 3, 'd': 1})
    params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': 4, 'd': 1})
    params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': 5, 'd': 1})
    return params_vld[j]

u_ex_0 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Basic_test_set/u_ex64_0")
u_ex_1 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Basic_test_set/u_ex64_1")
u_ex_2 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Basic_test_set/u_ex64_2")
u_ex_3 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Basic_test_set/u_ex64_3")
u_exs = [u_ex_0, u_ex_1, u_ex_2, u_ex_3]

all_loss_test = []
df=pd.read_csv("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024_2/parameters.txt")

for j in range(18):
    loss_test = []
    sample_id=j
    u_ex = np.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO//PME_Test/PME_Data_1024_2/u_exact64_{}.npy".format(sample_id))
    u_ex = torch.Tensor(u_ex)
    # Forward path
    # power = float(df[df.sample_id==sample_id]["power"])
    # params = {'T': 0.5, 'e': 1e-13, 'L': 6, 'power': power, 'd': 1}
    params = None
    height = float(df[df.sample_id==sample_id]["height"])
    problem_main = problem_class(type = "boxes", space_steps=64, time_steps=None, params=params)
    problem_main.initial_condition, _ = init_PME(problem_main.x, height)
    problem_main.initial_condition = torch.Tensor(problem_main.initial_condition)
    u_init, nn = train_model.init_run_weno(problem_main, vectorized=True, just_one_time_step=False)
    u_train = u_init
    params = problem_main.params
    # parameters needed for the computation of exact solution
    # params_main = problem_main.params
    # T = params_main['T']
    # x, time = problem_main.x, problem_main.time
    print(sample_id, height, params)
    for k in range(nn):
        # Forward path
        u_train = train_model.forward(problem_main,u_train,k)
        # Train model:
        optimizer.zero_grad()  # Clear gradients
        # Calculate loss
        params = problem_main.get_params()
        ex_loss = exact_loss(u_train,u_ex[:,k+1])
        loss = ex_loss
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize weights
        print(j, k, loss)
        u_train.detach_()
    base_path ="C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/Models/Model_8/"
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    path = os.path.join(base_path, "{}.pt".format(j))
    torch.save(train_model, path)
    # TEST IF LOSS IS DECREASING WITH THE NUMBER OF ITERATIONS INCREASING
    for kk in range(2,3):  # power = 4
        single_problem_loss_test = []
        params_test = validation_problems(kk)
        height_ex = 1
        problem_test = problem_class(type = "boxes", space_steps=64, time_steps=None, params=params_test)
        problem_test.initial_condition, _ = init_PME(problem_test.x, height_ex)
        problem_test.initial_condition = torch.Tensor(problem_test.initial_condition)
        with torch.no_grad():
            u_init, nn = train_model.init_run_weno(problem_test, vectorized=True, just_one_time_step=False)
            u_test = u_init
            for k in range(nn):
                u_test = train_model.run_weno(problem_test, u_test, mweno=True, mapped=False, trainable=True, vectorized=True, k=k)
            single_problem_loss_test.append(exact_loss(u_test,u_exs[kk][:, -1]))
        loss_test.append(single_problem_loss_test)
    all_loss_test.append(loss_test)

# print("number of parameters:", sum(p.numel() for p in train_model.parameters()))
# g=train_model.parameters()
# g.__next__()

all_loss_test = np.array(all_loss_test)
plt.plot(all_loss_test[:,:,0])
