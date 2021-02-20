from define_WENO_Network import WENONetwork
from define_WENO_Euler import WENONetwork_Euler
import torch
from torch import optim
from define_Euler_system import Euler_system
import numpy as np
import matplotlib.pyplot as plt
import torch.onnx
import os, sys
import random
import pandas as pd

torch.set_default_dtype(torch.float64)
problem = Euler_system
train_model = WENONetwork_Euler()

def exact_loss(u, u_ex):
    error = train_model.compute_error(u, u_ex)
    loss = error
    return loss

def overflows_loss(u, u_ex):
    u_max = torch.max(u_ex)
    u_min = torch.min(u_ex)
    overflows = torch.sum(torch.abs(torch.min(u, u_min)-u_min) + torch.max(u, u_max)-u_max )
    loss = overflows # peeks_left + peeks_right
    return loss

sp_st = 64 #*2*2*2 #*2*2*2
init_cond = "Sod"
time_disc = None
problem_main = problem(space_steps=sp_st, init_cond = init_cond, time_steps=None, params = None, time_disc=time_disc, init_mid=False, init_general=False)
params = problem_main.get_params()
gamma = params['gamma']
method = "char"
T = params['T']
x = problem_main.x

all_loss_test_exact = []
all_loss_test_overflows = []
for i in range(150):
    print(i)
    train_model = torch.load('C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Euler_System_Test/Models/Model_78/{}.pt'.format(i))
    loss_test_exact = []
    loss_test_overflows = []
    single_problem_loss_test_exact = []
    single_problem_loss_test_overflows = []
    q_0_test, q_1_test, q_2_test, lamb_test, nn, h = train_model.init_Euler(problem_main, vectorized=True,just_one_time_step=False)
    q_0_input, q_1_input, q_2_input = q_0_test, q_1_test, q_2_test
    with torch.no_grad():
        t_update = 0
        t = 0.9 * h / lamb_test
        while t_update < T:
            if (t_update + t) > T:
                t = T - t_update
            t_update = t_update + t
            q_0_test, q_1_test, q_2_test, lamb_test = train_model.run_weno(problem_main, mweno=True, mapped=False,method="char", q_0=q_0_input, q_1=q_1_input,q_2=q_2_input, lamb=lamb_test,vectorized=True, trainable=True, k=0, dt=t)
            t = 0.9 * h / lamb_test
            q_0_input = q_0_test.detach().numpy()
            q_1_input = q_1_test.detach().numpy()
            q_2_input = q_2_test.detach().numpy()
            q_0_input = torch.Tensor(q_0_input)
            q_1_input = torch.Tensor(q_1_input)
            q_2_input = torch.Tensor(q_2_input)
    rho_test = q_0_test
    u_test = q_1_test / rho_test
    E_test = q_2_test
    p_test = (gamma - 1) * (E_test - 0.5 * rho_test * u_test ** 2)
    p_ex, rho_ex, u_ex, _, _ = problem_main.exact(x, T)
    single_problem_loss_test_exact.append(exact_loss(rho_test, rho_ex))
    single_problem_loss_test_exact.append(exact_loss(p_test, p_ex))
    single_problem_loss_test_exact.append(exact_loss(u_test, u_ex))
    single_problem_loss_test_overflows.append(overflows_loss(u_test, u_ex))
    # for k in range(nn + 1):
    #     single_problem_loss_test_exact.append(exact_loss(u_test[:, k], u_ex[:, k]).detach().numpy().max())
    loss_test_exact.append(single_problem_loss_test_exact)
    loss_test_overflows.append(single_problem_loss_test_overflows)
    all_loss_test_exact.append(loss_test_exact)
    all_loss_test_overflows.append(loss_test_overflows)

all_loss_test_exact = np.array(all_loss_test_exact) #shape (training_steps, num_valid_problems, time_steps)
all_loss_test_overflows = np.array(all_loss_test_overflows) #shape (training_steps, num_valid_problems, time_steps)
plt.figure(1)
plt.plot(all_loss_test_overflows[:,:,-1])
plt.figure(2)
plt.plot(all_loss_test_exact[:,:,-1])