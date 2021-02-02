from define_WENO_Network_2 import WENONetwork_2
import torch
from torch import optim
from define_problem_Digital import Digital_option
from define_problem_heat_eq import heat_equation
from define_problem_Call import Call_option
from define_problem_Buckley_Leverett import Buckley_Leverett
from define_problem_PME import PME
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

torch.set_default_dtype(torch.float64)

# TRAIN NETWORK
train_model = WENONetwork_2()

# DROP PROBLEM FOR TRAINING
#params = None
#problem_class = Buckley_Leverett
problem_class = Digital_option

def monotonicity_loss(u):
    monotonicity = torch.sum(torch.max(u[:-1]-u[1:], torch.Tensor([0.0])))
    loss = monotonicity
    return loss

def exact_loss(u, u_ex):
    error = train_model.compute_error(u, u_ex)
    loss = error
    return loss

#optimizer = optim.SGD(train_model.parameters(), lr=0.1)
optimizer = optim.Adam(train_model.parameters())

params_test = {'sigma': 0.3, 'rate': 0.1, 'E': 50, 'T': 1, 'e': 1e-13, 'xl': -6, 'xr': 1.5}
all_loss_test = []

for j in range(500):
    loss_test = []
    # Forward path
    params = None
    problem_main = problem_class(space_steps=100, time_steps=None, params=params)
    u_exact, exact = problem_main.exact(first_step=False)
    u_exact = torch.Tensor(u_exact)
    exact = torch.Tensor(exact)
    u_init, nn = train_model.init_run_weno(problem_main, vectorized=True, just_one_time_step=False)
    u_train = u_init
    # parameters needed for the computation of exact solution
    params_main = problem_main.params
    rate = params_main['rate']
    sigma = params_main['sigma']
    T = params_main['T']
    E = params_main['E']
    x, time = problem_main.x, problem_main.time
    tt = T - time
    S = E * np.exp(x)
    for k in range(nn):
        exact = np.exp(-rate * (T - tt[k+1])) * norm.cdf((np.log(S / E) + (rate - (sigma ** 2) / 2) * (T - tt[k+1])) / (sigma * np.sqrt(T - tt[k+1])))
        exact = torch.Tensor(exact)
        # Forward path
        u_train = train_model.forward(problem_main,u_train,k)
        V_train, _, _ = problem_main.transformation(u_train)
        # Train model:
        optimizer.zero_grad()  # Clear gradients
        # Calculate loss
        params = problem_main.get_params()
        mon_loss = monotonicity_loss(V_train)
        ex_loss = exact_loss(V_train,exact)
        loss = mon_loss + ex_loss
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize weights
        print(j, k, loss)
        u_train.detach_()
    base_path ="C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Digital_Option_Test/Models/Model_12/"
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    path = os.path.join(base_path, "{}.pt".format(j))
    torch.save(train_model, path)
    # TEST IF LOSS IS DECREASING WITH THE NUMBER OF ITERATIONS INCREASING
    single_problem_loss_test = []
    problem_test = problem_class(space_steps=100, time_steps=None, params=params_test)
    with torch.no_grad():
        u_init, nn = train_model.init_run_weno(problem_test, vectorized=True, just_one_time_step=True)
        u_test = u_init
        for k in range(nn):
            u_test = train_model.run_weno(problem_test, u_test, mweno=True, mapped=False, trainable=True, vectorized=True, k=k)
        V_test, _, _ = problem_test.transformation(u_test)
        for k in range(nn + 1):
            single_problem_loss_test.append(monotonicity_loss(V_test))
    loss_test.append(single_problem_loss_test)
    all_loss_test.append(loss_test)

#plt.plot(S, V_train.detach().numpy())
# print("number of parameters:", sum(p.numel() for p in train_model.parameters()))
# g=train_model.parameters()
# g.__next__()

all_loss_test = np.array(all_loss_test)
plt.plot(all_loss_test[:,:,0])


# torch.save(train_model, "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Digital_Option_Test/Models/Model_1")