from define_WENO_Network_2 import WENONetwork_2
from utils.problem_handler import ProblemHandler
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

torch.set_default_dtype(torch.float64)

train_model = WENONetwork_2()

def monotonicity_loss(u):
    monotonicity = torch.sum(torch.max(u[:-1]-u[1:], torch.Tensor([0.0])))
    loss = monotonicity
    return loss

def exact_loss(u, u_ex):
    error = train_model.compute_error(u, u_ex)
    loss = 10e1*error # PME boxes
    # loss = 10e4*error # PME Barenblatt
    # loss = error
    return loss

def exact_loss_2d(u, u_ex):
    error = torch.mean((u_ex - u) ** 2)
    loss = 10e4*error
    return loss

# optimizer = optim.Adam(train_model.parameters(), lr=0.0001)   # Buckley-Leverett
optimizer = optim.Adam(train_model.parameters(), lr=0.01, weight_decay=0.001)  # PME boxes
# optimizer = optim.Adam(train_model.parameters(), lr=0.1, weight_decay=0.0001) # PME Barenblatt
#optimizer = optim.SGD(train_model.parameters(), lr=0.01, weight_decay=0.00001)

def validation_problems_barenblatt(j):
    params_vld = []
    params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 2, 'd': 1})
    params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 3, 'd': 1})
    params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 4, 'd': 1})
    params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 5, 'd': 1})
    return params_vld[j]

def validation_problems_barenblatt_2d(j):
    params_vld = []
    params_vld.append({'T': 2, 'e': 1e-13, 'L': 10, 'power': 2, 'd': 2})
    params_vld.append({'T': 2, 'e': 1e-13, 'L': 10, 'power': 3, 'd': 2})
    params_vld.append({'T': 2, 'e': 1e-13, 'L': 10, 'power': 4, 'd': 2})
    params_vld.append({'T': 2, 'e': 1e-13, 'L': 10, 'power': 5, 'd': 2})
    params_vld.append({'T': 2, 'e': 1e-13, 'L': 10, 'power': 6, 'd': 2})
    params_vld.append({'T': 2, 'e': 1e-13, 'L': 10, 'power': 7, 'd': 2})
    params_vld.append({'T': 2, 'e': 1e-13, 'L': 10, 'power': 8, 'd': 2})
    return params_vld[j]

def validation_problems_boxes(j):
    params_vld = []
    params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': 2, 'd': 1})
    params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': 3, 'd': 1})
    params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': 4, 'd': 1})
    params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': 5, 'd': 1})
    params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': 6, 'd': 1})
    return params_vld[j]

def validation_problems_BL(j):
    params_vld = []
    params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 1, 'G': 5})
    params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 1, 'G': 0})
    params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.5, 'G': 2})
    params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.5, 'G': 1})
    params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.25, 'G': 4})
    return params_vld[j]
def validation_problems_BL_2(j):
    params_vld = []
    params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 1, 'G': 3})
    params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.4, 'G': 0})
    params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.5, 'G': 5})
    params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.3, 'G': 3})
    params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.25, 'G': 1})
    return params_vld[j]
def validation_problems_BL_3(j):
    params_vld = []
    params_vld.append({'T': 0.2, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 1, 'G': 5})
    params_vld.append({'T': 0.2, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 1, 'G': 0})
    params_vld.append({'T': 0.2, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.5, 'G': 2})
    params_vld.append({'T': 0.2, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.5, 'G': 1})
    params_vld.append({'T': 0.2, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.25, 'G': 4})
    return params_vld[j]
all_loss_test = []

problem_class = PME

# current_problem_classes = [(PME, {"sample_id": None, "example": "Barenblatt_2d", "space_steps": 32, "time_steps": None, "params": None})]
# example = "Barenblatt_2d"
# rng = 4

# current_problem_classes = [(PME, {"sample_id": None, "example": "Barenblatt", "space_steps": 64, "time_steps": None, "params": None})]
# example = "Barenblatt"
# rng = 4

current_problem_classes = [(PME, {"sample_id": 0, "example": "boxes", "space_steps": 64, "time_steps": None, "params": 0})]
example = "boxes"
u_ex_0 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Basic_test_set/u_ex_0")
u_ex_1 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Basic_test_set/u_ex_1")
u_ex_2 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Basic_test_set/u_ex_2")
u_ex_3 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Basic_test_set/u_ex_3")
u_ex_4 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Basic_test_set/u_ex_4")
u_exs = [u_ex_0, u_ex_1, u_ex_2, u_ex_3, u_ex_4]
rng = 5

# problem_class = Buckley_Leverett
# current_problem_classes = [(Buckley_Leverett, {"sample_id": 1, "example": "gravity", "space_steps": 64, "time_steps": None, "params": 0})]
# example = "gravity"
# folder = 1
# u_ex_0 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Basic_test_set_{}/u_ex64_0".format(folder))
# u_ex_1 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Basic_test_set_{}/u_ex64_1".format(folder))
# u_ex_2 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Basic_test_set_{}/u_ex64_2".format(folder))
# u_ex_3 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Basic_test_set_{}/u_ex64_3".format(folder))
# u_ex_4 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Basic_test_set_{}/u_ex64_4".format(folder))
# u_exs = [u_ex_0, u_ex_1, u_ex_2, u_ex_3, u_ex_4]

phandler = ProblemHandler(problem_classes = current_problem_classes, max_num_open_problems=200)
test_modulo=50
for j in range(2000):
    loss_test = []
    problem_specs, problem_id = phandler.get_random_problem(0.1)
    problem = problem_specs["problem"]
    #print(problem.sample_id)
    params = problem.params
    step = problem_specs["step"]
    u_last = problem_specs["last_solution"]
    if example == "Barenblatt_2d":
        u_new = train_model.forward(problem, u_last, step, mweno = True, mapped = False, dim =2)
    else:
        u_new = train_model.forward(problem, u_last, step, mweno = True, mapped = False)
    if example == "Barenblatt" or example == "Barenblatt_2d":
        u_exact = problem.exact(problem.time[step+1])
    elif example == "boxes" or example == "gravity":
        u_exact = problem.exact(step + 1)
    u_exact = torch.Tensor(u_exact)
    optimizer.zero_grad()
    if example == "Barenblatt_2d":
        loss = exact_loss_2d(u_new,u_exact)
    else:
        loss = exact_loss(u_new,u_exact)
    minibatch_size=25
    loss = loss/minibatch_size
    loss.backward()  # Backward pass
    # if j%minibatch_size == 0:
    #     print("optimizer_step")
    optimizer.step()  # Optimize weights
        # optimizer.zero_grad()
    u_new.detach_()
    phandler.update_problem(problem_id, u_new)
    if example == "Barenblatt":
        base_path = "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/Models/Model_50/"
    if example == "boxes":
        base_path = "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/Models_boxes/Model_12/"
    elif example == "Barenblatt_2d":
        base_path = "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/Models_2d/Model_7/"
    elif example == "gravity":
        base_path = "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Models/Model_14/"
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    path = os.path.join(base_path, "{}.pt".format(j))
    torch.save(train_model, path)
    # TEST IF LOSS IS DECREASING WITH THE NUMBER OF ITERATIONS INCREASING
    if not (j % test_modulo):
        print("TESTING ON VALIDATION PROBLEMS")
        for kk in range(rng):
            single_problem_loss_test = []
            if example == "Barenblatt":
                params_test = validation_problems_barenblatt(kk)
                problem_test = problem_class(sample_id=None, example="Barenblatt", space_steps=64, time_steps=None, params=params_test)
            elif example == "Barenblatt_2d":
                params_test = validation_problems_barenblatt_2d(kk)
                problem_test = problem_class(sample_id=None, example="Barenblatt_2d", space_steps=32, time_steps=None, params=params_test)
            elif example == "boxes":
                params_test = validation_problems_boxes(kk)
                problem_test = problem_class(sample_id=None, example="boxes", space_steps=64, time_steps=None, params=params_test)
            elif example == "gravity":
                params_test = validation_problems_BL(kk)
                problem_test = problem_class(sample_id=None, example="gravity", space_steps=64, time_steps=None, params=params_test)
            with torch.no_grad():
                if example == "Barenblatt_2d":
                    u_init, tt = train_model.init_run_weno(problem_test, vectorized=True, just_one_time_step=False, dim=2)
                    u_test = u_init
                    for k in range(tt):
                        u_test = train_model.run_weno_2d(problem_test, u_test, mweno=True, mapped=False, trainable=True, vectorized=True, k=k)
                else:
                    u_init, tt = train_model.init_run_weno(problem_test, vectorized=True, just_one_time_step=False)
                    u_test = u_init
                    for k in range(tt):
                        u_test = train_model.run_weno(problem_test, u_test, mweno=True, mapped=False, trainable=True, vectorized=True, k=k)
            if example == "Barenblatt":
                T_test = problem_test.params['T']
                power_test = problem_test.params['power']
                u_exact_test = problem_test.exact(T_test)
                u_exact_test = torch.Tensor(u_exact_test)
                single_problem_loss_test.append(exact_loss(u_test, u_exact_test))
            elif example == "boxes":
                single_problem_loss_test.append(exact_loss(u_test, u_exs[kk][0:1024 + 1:16, -1]))
            elif example == "gravity":
                single_problem_loss_test.append(exact_loss(u_test, u_exs[kk][:, -1]))
            elif example == "Barenblatt_2d":
                T_test = problem_test.params['T']
                power_test = problem_test.params['power']
                u_exact_test = problem_test.exact(T_test)
                u_exact_test = torch.Tensor(u_exact_test)
                single_problem_loss_test.append(exact_loss_2d(u_test, u_exact_test))
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

plt.figure(2)
plt.plot(all_loss_test[:,:,0])
# np.save("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Models/Model_8/all_loss_test.npy",all_loss_test)