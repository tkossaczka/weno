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

# TRAIN NETWORK
train_model = WENONetwork_Euler()

# DROP PROBLEM FOR TRAINING
#params = None
problem_class = Euler_system

def monotonicity_loss(u):
    monotonicity = torch.sum(torch.abs(torch.min(u[:-1]-u[1:], torch.Tensor([0.0]))))
    loss = monotonicity
    return loss

def monotonicity_loss_mid(u, x):
    monotonicity = torch.zeros(x.shape[0])
    for k in range(x.shape[0]-1):
        if x[k] <= 0.5:
            monotonicity[k] = (torch.abs(torch.max(u[:-1]-u[1:], torch.Tensor([0.0]))))[k]
        elif x[k] > 0.58:
            monotonicity[k] = (torch.abs(torch.min(u[:-1]-u[1:], torch.Tensor([0.0]))))[k]
    loss = torch.sum(monotonicity)
    return loss

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

def create_init_cond(df,x,row):
    m = x.shape[0]
    r0 = torch.zeros(m)
    u0 = torch.zeros(m)
    p0 = torch.zeros(m)
    x_mid=0.5
    r0[x <= x_mid] = df.loc[row]["rho[0]"]
    r0[x > x_mid] = df.loc[row]["rho[1]"]
    u0[x <= x_mid] = df.loc[row]["u[0]"]
    u0[x > x_mid] = df.loc[row]["u[1]"]
    p0[x <= x_mid] = df.loc[row]["p[0]"]
    p0[x > x_mid] = df.loc[row]["p[1]"]
    a0 = torch.sqrt(gamma * p0 / r0)
    E0 = p0 / (gamma - 1) + 0.5 * r0 * u0 ** 2
    q0 = torch.stack([r0, r0 * u0, E0]).T
    return q0, u0, a0

#optimizer = optim.SGD(train_model.parameters(), lr=0.1)
optimizer = optim.Adam(train_model.parameters(), lr=0.01)

# rho_ex=torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Euler_System_Test/Shock_entropy_exact/rho_ex")
# u_ex=torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Euler_System_Test/Shock_entropy_exact/u_ex")
# p_ex=torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Euler_System_Test/Shock_entropy_exact/p_ex")
# divider = 8
# rho_ex_s=rho_ex[0:2048 + 1:divider, 0:2048 + 1:divider]
# u_ex_s=u_ex[0:2048 + 1:divider, 0:2048 + 1:divider]
# p_ex_s=p_ex[0:2048 + 1:divider, 0:2048 + 1:divider]

all_loss_test = []
losses = []
method = "char"
time_disc = None
for j in range(60):
    #init_id = 0
    #print(j)
    # Forward path
    params = None
    sp_st = 64
    init_cond = "Sod"
    problem_main = problem_class(space_steps=sp_st, init_cond=init_cond, time_steps=None, params=params, time_disc=time_disc, init_mid=False, init_general=True)
    gamma = problem_main.params['gamma']
    T = problem_main.params["T"]
    tt = problem_main.t
    time_st = problem_main.time.shape[0]
    x = problem_main.x
    time = problem_main.time
    q_0, q_1, q_2, lamb, nn, h = train_model.init_Euler(problem_main, vectorized=True, just_one_time_step=True)
    init_id = random.randint(0,time_st-2)
    print(init_id)
    p_ex_0, rho_ex_0, u_ex_0, _, _ = problem_main.exact(x, time[init_id])
    # MACH test
    # c = np.sqrt(gamma * p_ex_0 / rho_ex_0)
    # MA = u_ex_0 / c
    # if not all(i < 1 for i in MA):
    #     print('mach error')
    #     exit()
    q_0 = rho_ex_0 # rho
    q_1 = u_ex_0*rho_ex_0  #rho*u
    q_2 = p_ex_0/(gamma-1) +0.5*rho_ex_0*u_ex_0**2 # E
    lamb = float(torch.max(torch.abs(u_ex_0 + (gamma*p_ex_0/rho_ex_0)**(1/2))))
    q_0_train = q_0
    q_1_train = q_1
    q_2_train = q_2
    single_problem_losses = []
    loss_test = []
    for k in range(nn):
        q_0_train_out, q_1_train_out, q_2_train_out, lamb = train_model.forward(problem_main, method, q_0_train, q_1_train, q_2_train, lamb, k, dt=None, mweno=True, mapped=False)
        rho = q_0_train_out
        u = q_1_train_out / rho
        E = q_2_train_out
        p = (gamma - 1) * (E - 0.5 * rho * u ** 2)
        q_0_train = q_0_train_out
        q_1_train = q_1_train_out
        q_2_train = q_2_train_out
        #print(k)
    p_ex_1, rho_ex_1, u_ex_1, _, _ = problem_main.exact(x, time[init_id+1])
    # p_ex_1, rho_ex_1, u_ex_1, _, _ = problem_main.exact(x, time[k+1])
    q_0_ex = rho_ex_1
    q_1_ex = u_ex_1*rho_ex_1
    q_2_ex = p_ex_1/(gamma-1) + 0.5*rho_ex_1*u_ex_1**2
    # p_ex_1, rho_ex_1, u_ex_1 = p_ex_s[:,init_id+1], rho_ex_s[:,init_id+1], u_ex_s[:,init_id+1]
    # Train model:
    optimizer.zero_grad()  # Clear gradients
    # Calculate loss
    #loss_0 = monotonicity_loss(rho)
    loss_00 = exact_loss(rho, rho_ex_1)
    loss_0 = overflows_loss(rho, rho_ex_1)
    # loss_00 = exact_loss(q_0_train, q_0_ex)
    #loss_1 = monotonicity_loss_mid(u, x)
    loss_11 = exact_loss(u, u_ex_1)
    loss_1 = overflows_loss(u, u_ex_1)
    # loss_11 = exact_loss(q_1_train, q_1_ex)
    #loss_2 = monotonicity_loss(p)
    loss_22 = exact_loss(p, p_ex_1)
    loss_2 = overflows_loss(p, p_ex_1)
    # loss_22 = exact_loss(q_2_train, q_2_ex)
    #loss_3 = overflows_loss(u,u_ex_1)
    loss =  loss_00 + loss_11 + loss_22 #+ loss_0 + loss_2 + loss_1
    if np.isnan(loss.detach().numpy())== True:
        exit()
    loss.backward()  # Backward pass
    optimizer.step()  # Optimize weights
    print(j, k, loss)
    single_problem_losses.append(loss.detach().numpy().max())
    q_0_train = q_0_train.detach()
    q_1_train = q_1_train.detach()
    q_2_train = q_2_train.detach()
    #lamb = lamb.detach()
    base_path ="C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Euler_System_Test/Models/Model_70/"
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    path = os.path.join(base_path, "{}.pt".format(j))
    torch.save(train_model, path)
    losses.append(single_problem_losses)
    # TEST IF LOSS IS DECREASING WITH THE NUMBER OF ITERATIONS INCREASING
    single_problem_loss_test = []
    problem_test = problem_class(space_steps=sp_st, init_cond=init_cond, time_steps=None, params=params, time_disc=time_disc, init_mid=False, init_general=False)
    q_0_test, q_1_test, q_2_test, lamb_test, nn, h = train_model.init_Euler(problem_test, vectorized=True, just_one_time_step=False)
    q_0_input, q_1_input, q_2_input = q_0_test, q_1_test, q_2_test
    with torch.no_grad():
        t_update = 0
        t = 0.9 * h / lamb_test
        while t_update < T:
            if (t_update + t) > T:
                t = T - t_update
            t_update = t_update + t
            q_0_test, q_1_test, q_2_test, lamb_test = train_model.run_weno(problem_main, mweno=True, mapped=False, method="char", q_0=q_0_input, q_1=q_1_input, q_2=q_2_input,lamb=lamb_test, vectorized=True, trainable=True, k=0, dt=t)
            t = 0.9 * h / lamb_test
            q_0_input = q_0_test.detach().numpy()
            q_1_input = q_1_test.detach().numpy()
            q_2_input = q_2_test.detach().numpy()
            q_0_input = torch.Tensor(q_0_input)
            q_1_input = torch.Tensor(q_1_input)
            q_2_input = torch.Tensor(q_2_input)
    # with torch.no_grad():
    #     for k in range(nn):
    #         q_0_test, q_1_test, q_2_test, lamb_test = train_model.run_weno(problem_test, mweno = True, mapped = False, method="char", q_0=q_0_input, q_1=q_1_input, q_2=q_2_input, lamb=lamb_test, vectorized=True, trainable=True, k=k, dt=None)
    #         q_0_input = q_0_test.detach().numpy()
    #         q_1_input = q_1_test.detach().numpy()
    #         q_2_input = q_2_test.detach().numpy()
    #         q_0_input = torch.Tensor(q_0_input)
    #         q_1_input = torch.Tensor(q_1_input)
    #         q_2_input = torch.Tensor(q_2_input)
    rho_test = q_0_test
    u_test = q_1_test / rho_test
    E_test = q_2_test
    p_test = (gamma - 1) * (E_test - 0.5 * rho_test * u_test ** 2)
    p_ex_test, rho_ex_test, u_ex_test, _, _ = problem_test.exact(x, T)
    if np.isnan(exact_loss(rho_test, rho_ex_test).detach().numpy())==True:
        exit()
    single_problem_loss_test.append(exact_loss(rho_test, rho_ex_test))
    single_problem_loss_test.append(exact_loss(p_test, p_ex_test))
    single_problem_loss_test.append(exact_loss(u_test, u_ex_test))
    loss_test.append(single_problem_loss_test)
    all_loss_test.append(loss_test)

losses = np.array(losses)
#plt.plot(losses[:,-1])
all_loss_test = np.array(all_loss_test)
plt.figure(1)
plt.plot(all_loss_test[:,:,0])
plt.figure(2)
plt.plot(all_loss_test[:,:,1])
plt.figure(3)
plt.plot(all_loss_test[:,:,2])

#plt.plot(S, V_train.detach().numpy())
# print("number of parameters:", sum(p.numel() for p in train_model.parameters()))
# g=train_model.parameters()
# g.__next__()

