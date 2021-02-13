from define_WENO_Network_2 import WENONetwork_2
import torch
from torch import optim
from define_problem_PME import PME
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

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
    params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 2, 'd': 1})
    params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 3, 'd': 1})
    params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 4, 'd': 1})
    params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 5, 'd': 1})
    return params_vld[j]

all_loss_test = []

for j in range(100):
    loss_test = []
    # Forward path
    params = None
    #params = {'T': 2, 'e': 1e-13, 'L': 6, 'power': 4, 'd': 1}
    problem_main = problem_class(type = "Barenblatt", space_steps=64, time_steps=None, params=params)
    u_init, nn = train_model.init_run_weno(problem_main, vectorized=True, just_one_time_step=False)
    u_train = u_init
    # parameters needed for the computation of exact solution
    params_main = problem_main.params
    T = params_main['T']
    x, time = problem_main.x, problem_main.time
    power = problem_main.params['power']
    for k in range(nn):
        # Forward path
        u_train = train_model.forward(problem_main,u_train,k)
        u_exact = problem_main.exact(time[k+1], power, "Barenblatt")
        u_exact = torch.Tensor(u_exact)
        # Train model:
        optimizer.zero_grad()  # Clear gradients
        # Calculate loss
        params = problem_main.get_params()
        ex_loss = exact_loss(u_train,u_exact)
        loss = ex_loss
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize weights
        print(j, k, loss)
        u_train.detach_()
    base_path ="C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/Models/Model_26/"
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    path = os.path.join(base_path, "{}.pt".format(j))
    torch.save(train_model, path)
    # TEST IF LOSS IS DECREASING WITH THE NUMBER OF ITERATIONS INCREASING
    for kk in range(2,3):
        single_problem_loss_test = []
        params_test = validation_problems(kk)
        problem_test = problem_class(type = "Barenblatt", space_steps=64, time_steps=None, params=params_test)
        T_test = problem_test.params['T']
        power_test = problem_test.params['power']
        with torch.no_grad():
            u_init, nn = train_model.init_run_weno(problem_test, vectorized=True, just_one_time_step=False)
            u_test = u_init
            for k in range(nn):
                u_test = train_model.run_weno(problem_test, u_test, mweno=True, mapped=False, trainable=True, vectorized=True, k=k)
            u_exact_test = problem_test.exact(T_test, power_test, "Barenblatt")
            u_exact_test = torch.Tensor(u_exact_test)
            single_problem_loss_test.append(exact_loss(u_test,u_exact_test))
        loss_test.append(single_problem_loss_test)
    all_loss_test.append(loss_test)

# print("number of parameters:", sum(p.numel() for p in train_model.parameters()))
# g=train_model.parameters()
# g.__next__()

all_loss_test = np.array(all_loss_test)
plt.plot(all_loss_test[:,:,0])
