from define_WENO_Network_2 import WENONetwork_2
from sub_WENO_Network import sub_WENO
from utils.problem_handler import ProblemHandler
from validation_problems import validation_problems
import torch
from torch import optim
from define_problem_PME import PME
from define_problem_Buckley_Leverett import Buckley_Leverett
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import torch.nn as nn
import torch.nn.functional as F
import random

torch.set_default_dtype(torch.float64)
current_problem_classes = [(PME, {"sample_id": None, "example": "Barenblatt", "space_steps": 64, "time_steps": None, "params": None})]
example = "Barenblatt"
_, rng = validation_problems.validation_problems_barenblatt(1)
phandler = ProblemHandler(problem_classes = current_problem_classes, max_num_open_problems=200)
problem_class = PME

def exact_compare_loss(u, u_nt, u_ex):
    error_t = torch.mean((u_ex - u)**2)
    error_nt = torch.mean((u_ex - u_nt)**2)
    if error_t > error_nt:
        loss = error_t-error_nt
    else:
        loss = error_t-error_t
    return loss

def exact_loss(u, u_ex):
    error = train_model.compute_error(u, u_ex)
    # loss = error # PME boxes
    # if loss > 0.001:
    #     loss = loss/10
    loss = 10e2*error # PME Barenblatt
    if loss > 0.01:
        loss = torch.sqrt(loss)
    # loss = error
    return loss

model_basic = 13
model_id_basic = 40
base_path = "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/Models/Model_{}/".format(model_basic)
all_loss_test_2 = []
train_model = torch.load(os.path.join(base_path,"{}.pt".format(model_id_basic)))
train_model.train_with_coeff = True
optimizer = optim.Adam(train_model.m_nn.parameters(), lr=0.001)

model_new = 14
test_modulo=10
for j in range(100):
    loss_test = []
    problem_specs, problem_id = phandler.get_random_problem(0.1)
    problem = problem_specs["problem"]
    params = problem.params
    step = problem_specs["step"]
    u_last = problem_specs["last_solution"]
    u_new = train_model.forward(problem, u_last, step, mweno=True, mapped=False)
    # u_new = train_model.run_weno(problem, u_last, mweno=True, mapped=False, vectorized=True, trainable=True, k=step)
    u_new[u_new<0]=0
    # u_new_nt = train_model.run_weno(problem, u_last, mweno=True, mapped=False, vectorized=True, trainable=False, k=step)
    # u_new_nt[u_new_nt<0]=0
    u_exact = problem.exact(problem.time[step+1])
    u_exact = torch.Tensor(u_exact)
    optimizer.zero_grad()
    # loss = exact_compare_loss(u_new,u_new_nt,u_exact)
    loss = exact_loss(u_new, u_exact)
    print(loss)
    loss.backward()
    optimizer.step()
    u_new.detach_()
    phandler.update_problem(problem_id, u_new)
    if not (j % test_modulo):
        base_path = "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/Models/Model_{}/".format(model_new)
        if not os.path.exists(base_path):
            os.mkdir(base_path)
        path = os.path.join(base_path, "0{}.pt".format(j))
        torch.save(train_model, path)
    # TEST IF LOSS IS DECREASING WITH THE NUMBER OF ITERATIONS INCREASING
    if not (j % test_modulo):
        print("TESTING ON VALIDATION PROBLEMS")
        for kk in range(rng):
            single_problem_loss_test = []
            params_test,_ = validation_problems.validation_problems_barenblatt(kk)
            problem_test = problem_class(sample_id=None, example="Barenblatt", space_steps=64, time_steps=None, params=params_test)
            with torch.no_grad():
                u_init, tt = train_model.init_run_weno(problem_test, vectorized=True, just_one_time_step=False)
                u_test = u_init
                for k in range(tt):
                    u_test = train_model.run_weno(problem_test, u_test, mweno=True, mapped=False, trainable=True, vectorized=True, k=k)
                    u_test[u_test < 0] = 0
            T_test = problem_test.params['T']
            power_test = problem_test.params['power']
            u_exact_test = problem_test.exact(T_test)
            u_exact_test = torch.Tensor(u_exact_test)
            single_problem_loss_test.append(exact_loss(u_test, u_exact_test))
            loss_test.append(single_problem_loss_test)
        print(loss_test)
        all_loss_test_2.append(loss_test)

all_loss_test_2 = np.array(all_loss_test_2)
norm_losses_2=all_loss_test_2[:,:,0]/all_loss_test_2[:,:,0].max(axis=0)[None, :]
print("trained:", all_loss_test_2[:,:,0].min(axis=0))
plt.figure(2)
plt.plot(norm_losses_2)
plt.legend(['2.157', '3.012', '3.697', '3.987', '4.158', '4.572', '4.723', '5.041', '5.568', '6.087', '6.284', '7.124', '7.958'])

