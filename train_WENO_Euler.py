from define_WENO_Network import WENONetwork
from define_WENO_Euler import WENONetwork_Euler
import torch
from torch import optim
from define_Euler_system import Euler_system
import numpy as np
import matplotlib.pyplot as plt

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
        elif x[k] > 0.5:
            monotonicity[k] = (torch.abs(torch.min(u[:-1]-u[1:], torch.Tensor([0.0]))))[k]
    loss = torch.sum(monotonicity)
    return loss

def exact_loss(u, u_ex):
    error = train_model.compute_error(u, u_ex)
    loss = error
    return loss

#optimizer = optim.SGD(train_model.parameters(), lr=0.1)
optimizer = optim.Adam(train_model.parameters())

losses = []

for j in range(15):
    # Forward path
    params = None
    sp_st = 64*2*2
    init_cond = "Sod"
    problem_main = problem_class(space_steps=sp_st, init_cond=init_cond, time_steps=None, params=params)
    #print(k, problem_main.time_steps)
    gamma = problem_main.params['gamma']
    q_0, q_1, q_2, lamb, nn = train_model.init_Euler(problem_main, vectorized=True, just_one_time_step=False)
    _, x, t = problem_main.transformation(q_0)
    x_ex = torch.linspace(0, 1, sp_st + 1)
    p_ex = torch.zeros((x_ex.shape[0], t.shape[0]))
    rho_ex = torch.zeros((x_ex.shape[0], t.shape[0]))
    u_ex = torch.zeros((x_ex.shape[0], t.shape[0]))
    rho_ex[:,0] = q_0
    u_ex[:,0] = q_1/q_0
    p_ex[:,0] = q_2
    q_0_train = q_0
    q_1_train = q_1
    q_2_train = q_2
    single_problem_losses = []
    for k in range(nn):
        q_0_train, q_1_train, q_2_train, lamb = train_model.forward(problem_main, q_0_train, q_1_train, q_2_train, lamb, k)
        rho = q_0_train
        u = q_1_train / rho
        E = q_2_train
        p = (gamma - 1) * (E - 0.5 * rho * u ** 2)
        p_ex[:, k+1], rho_ex[:, k+1], u_ex[:, k+1], _, _ = problem_main.exact(x_ex, t[k+1])
        # Train model:
        optimizer.zero_grad()  # Clear gradients
        # Calculate loss
        loss_0 = monotonicity_loss(rho) #, rho_ex[:, k+1])
        loss_00 = exact_loss(rho, rho_ex[:, k+1])
        loss_1 = monotonicity_loss_mid(u, x)
        loss_11 = exact_loss(u, u_ex[:, k + 1])
        loss_2 = monotonicity_loss(p)
        loss_22 = exact_loss(p, p_ex[:, k + 1])
        # loss = loss_0 + 1*loss_00 + loss_2 + 1*loss_22 + loss_1 + 1*loss_11
        loss = loss_00 + loss_22 + loss_11
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize weights
        print(j, k, loss)
        single_problem_losses.append(loss.detach().numpy().max())
        q_0_train = q_0_train.detach()
        q_1_train = q_1_train.detach()
        q_2_train = q_2_train.detach()
        lamb = lamb.detach()
    path = "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Euler_System_Test/Models/Model_07/{}".format(j)
    torch.save(train_model, path)
    losses.append(single_problem_losses)

losses = np.array(losses)
plt.plot(losses[:,-1])

#plt.plot(S, V_train.detach().numpy())
#print("number of parameters:", sum(p.numel() for p in train_model.parameters()))
# g=train_model.parameters()
# g.__next__()

