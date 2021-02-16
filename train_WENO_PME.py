from define_WENO_Network_2 import WENONetwork_2
from utils.problem_handler import ProblemHandler
import torch
from torch import optim
from define_problem_PME import PME
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import torch.nn as nn
import torch.nn.functional as F

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
    loss = 10e4*error
    return loss

#optimizer = optim.SGD(train_model.parameters(), lr=0.1)
# optimizer = optim.Adam(train_model.parameters(), lr=0.1)
optimizer = optim.Adam(train_model.parameters(), lr=0.1, weight_decay=0.0001)

def validation_problems(j):
    params_vld = []
    params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 2, 'd': 1})
    params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 3, 'd': 1})
    params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 4, 'd': 1})
    params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 5, 'd': 1})
    return params_vld[j]

all_loss_test = []

current_problem_classes = [
  (PME, {"example": "Barenblatt", "space_steps": 64, "time_steps": None, "params": None}),
]

phandler = ProblemHandler(problem_classes = current_problem_classes,
                          max_num_open_problems=200)
test_modulo=100
for j in range(400):
    loss_test = []
    # Forward path
    # problem_main = problem_class(type = "Barenblatt", space_steps=64, time_steps=None, params=params)
    problem_specs, problem_id = phandler.get_random_problem(0.1)
    problem = problem_specs["problem"]
    params = problem.params
    step = problem_specs["step"]
    u_last = problem_specs["last_solution"]
    u_new = train_model.forward(problem,u_last,step)
    u_exact = problem.exact(problem.time[step+1])
    u_exact = torch.Tensor(u_exact)
    optimizer.zero_grad()
    loss = exact_loss(u_new,u_exact)
    loss.backward()  # Backward pass
    optimizer.step()  # Optimize weights
    u_new.detach_()
    phandler.update_problem(problem_id, u_new)
    base_path = "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/Models/Model_29/"
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    path = os.path.join(base_path, "{}.pt".format(j))
    torch.save(train_model, path)
    # TEST IF LOSS IS DECREASING WITH THE NUMBER OF ITERATIONS INCREASING
    if not (j%test_modulo):
      print("TESTING ON VALIDATION PROBLEMS")
      for kk in range(4):
        single_problem_loss_test = []
        params_test = validation_problems(kk)
        problem_test = problem_class(example = "Barenblatt", space_steps=64, time_steps=None, params=params_test)
        T_test = problem_test.params['T']
        power_test = problem_test.params['power']
        with torch.no_grad():
          u_init, tt = train_model.init_run_weno(problem_test, vectorized=True, just_one_time_step=False)
          u_test = u_init
          for k in range(tt):
            u_test = train_model.run_weno(problem_test, u_test, mweno=True, mapped=False, trainable=True, vectorized=True, k=k)
          u_exact_test = problem_test.exact(T_test)
          u_exact_test = torch.Tensor(u_exact_test)
          single_problem_loss_test.append(exact_loss(u_test,u_exact_test))
        loss_test.append(single_problem_loss_test)
      print(loss_test)
      all_loss_test.append(loss_test)

# print("number of parameters:", sum(p.numel() for p in train_model.parameters()))
# g=train_model.parameters()
# g.__next__()

all_loss_test = np.array(all_loss_test)
norm_losses=all_loss_test[:,:,0]/all_loss_test[:,:,0].max(axis=0)[None, :]
print("trained:", all_loss_test[:,:,0].min(axis=0))
plt.plot(norm_losses)
plt.show()
