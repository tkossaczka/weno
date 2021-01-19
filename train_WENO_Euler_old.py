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

#optimizer = optim.SGD(train_model.parameters(), lr=0.1)
optimizer = optim.Adam(train_model.parameters(), lr=0.01)

rho_ex=torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Euler_System_Test/rho_ex")
u_ex=torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Euler_System_Test/u_ex")
p_ex=torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Euler_System_Test/p_ex")
divider = 32
rho_ex_0=rho_ex[0:2048 + 1:divider, 0:416 + 1:divider]
u_ex_0=u_ex[0:2048 + 1:divider, 0:416 + 1:divider]
p_ex_0=p_ex[0:2048 + 1:divider, 0:416 + 1:divider]

losses = []
method = "char"
time_disc = None
for j in range(40):
    # Forward path
    params = None
    sp_st = 64
    time_st = p_ex_0.shape[1]
    init_cond = "Sod"
    problem_main = problem_class(space_steps=sp_st, init_cond=init_cond, time_steps=None, params=params, time_disc=time_disc, init_mid=True)
    #print(k, problem_main.time_steps)
    gamma = problem_main.params['gamma']
    T = problem_main.params["T"]
    tt = problem_main.t
    q_0, q_1, q_2, lamb, nn, h = train_model.init_Euler(problem_main, vectorized=True, just_one_time_step=True)
    init_id = random.randint(0,time_st-2)
    print(init_id)
    # init_id = 60
    q_0 = rho_ex_0[:,init_id] # rho
    q_1 = u_ex_0[:,init_id]*rho_ex_0[:,init_id]   #rho*u
    q_2 = p_ex_0[:,init_id]/(gamma-1) +0.5*rho_ex_0[:,init_id]*u_ex_0[:,init_id]*2 # E
    lamb = float(torch.max(torch.abs(u_ex_0 + (gamma*p_ex_0/rho_ex_0)**(1/2))))
    _, x, t = problem_main.transformation(q_0)
    # x_ex = torch.linspace(0, 1, sp_st + 1)
    # p_ex = torch.zeros((x_ex.shape[0], t.shape[0]))
    # rho_ex = torch.zeros((x_ex.shape[0], t.shape[0]))
    # u_ex = torch.zeros((x_ex.shape[0], t.shape[0]))
    # rho_ex[:,0] = q_0
    # u_ex[:,0] = q_1/q_0
    # p_ex[:,0] = q_2
    q_0_train = q_0
    q_1_train = q_1
    q_2_train = q_2
    single_problem_losses = []
    # t = 0.5 * h / lamb
    # q_0_train, q_1_train, q_2_train, lamb = train_model.forward(problem_main, method, q_0_train, q_1_train, q_2_train, lamb, k=0, dt=t, mweno=True, mapped=False)
    # rho = q_0_train
    # u = q_1_train / rho
    # E = q_2_train
    # p = (gamma - 1) * (E - 0.5 * rho * u ** 2)
    # k=0
    # p_ex, rho_ex, u_ex, _, _ = problem_main.exact(x_ex, t)
    # p_ex, rho_ex, u_ex, _, _ = problem_main.exact(x, T / 2 + tt)
    #nn=30
    for k in range(nn):
        q_0_train_out, q_1_train_out, q_2_train_out, lamb = train_model.forward(problem_main, method, q_0_train, q_1_train, q_2_train, lamb, k, dt=None, mweno=True, mapped=False)
        rho = q_0_train_out
        u = q_1_train_out / rho
        E = q_2_train_out
        p = (gamma - 1) * (E - 0.5 * rho * u ** 2)
        #p_ex, rho_ex, u_ex, _, _ = problem_main.exact(x, t[k+1])
        q_0_train = q_0_train_out
        q_1_train = q_1_train_out
        q_2_train = q_2_train_out
        print(k)
    rho_ex_1 = rho_ex_0[:,init_id + 1]
    u_ex_1 = u_ex_0[:,init_id + 1]
    p_ex_1 = p_ex_0[:, init_id + 1]
    # Train model:
    optimizer.zero_grad()  # Clear gradients
    # Calculate loss
    #loss_0 = monotonicity_loss(rho)
    loss_00 = exact_loss(rho, rho_ex_1)
    # loss_00 = exact_loss(rho, rho_ex)
    #loss_1 = monotonicity_loss_mid(u, x)
    loss_11 = exact_loss(u, u_ex_1)
    # loss_11 = exact_loss(u, u_ex)
    #loss_2 = monotonicity_loss(p)
    loss_22 = exact_loss(p, p_ex_1)
    # loss_22 = exact_loss(p, p_ex)
    loss =  loss_00 + loss_11 + loss_22 #+ loss_00 + loss_22 + loss_11
    loss.backward()  # Backward pass
    optimizer.step()  # Optimize weights
    print(j, k, loss)
    single_problem_losses.append(loss.detach().numpy().max())
    q_0_train = q_0_train.detach()
    q_1_train = q_1_train.detach()
    q_2_train = q_2_train.detach()
    #lamb = lamb.detach()
    base_path ="C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Euler_System_Test/Models/Model_28/"
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    path = os.path.join(base_path, "{}.pt".format(j))
    torch.save(train_model, path)
    # onnx_path = os.path.join(base_path, "{}.onnx".format(j))
    # torch.onnx.export(train_model, (problem_main, method, q_0_train, q_1_train, q_2_train, lamb, k, None, True, False),
    #                   onnx_path, export_params=True,)
    losses.append(single_problem_losses)

losses = np.array(losses)
plt.plot(losses[:,-1])

#plt.plot(S, V_train.detach().numpy())
#print("number of parameters:", sum(p.numel() for p in train_model.parameters()))
# g=train_model.parameters()
# g.__next__()

