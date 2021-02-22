from define_WENO_Network_2 import WENONetwork_2
from utils.problem_handler import ProblemHandler
import torch
from torch import optim
from define_problem_PME import PME
from define_problem_PME_boxes import PME_boxes
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
    # loss = 10e1*error
    # loss = 10e4*error
    loss = error
    return loss

#optimizer = optim.SGD(train_model.parameters(), lr=0.1)
# optimizer = optim.Adam(train_model.parameters(), lr=0.1)
optimizer = optim.Adam(train_model.parameters(), lr=0.001, weight_decay=0.0001)
# optimizer = optim.Adam(train_model.parameters(), lr=0.01, weight_decay=0.001)

def validation_problems(j):
    params_vld = []
    params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 2, 'd': 1})
    params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 3, 'd': 1})
    params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 4, 'd': 1})
    params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 5, 'd': 1})
    return params_vld[j]

def validation_problems_boxes(j):
    params_vld = []
    params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': 2, 'd': 1})
    params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': 3, 'd': 1})
    params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': 4, 'd': 1})
    params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': 5, 'd': 1})
    return params_vld[j]

def validation_problems_BL(j):
    params_vld = []
    params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 1, 'G': 5})
    params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 1, 'G': 0})
    params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.5, 'G': 2})
    params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.5, 'G': 1})
    params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.25, 'G': 4})
    return params_vld[j]

all_loss_test = []

# problem_class = PME
# current_problem_classes = [(PME, {"example": "Barenblatt", "space_steps": 64, "time_steps": None, "params": None})]
# example = "Barenblatt"

# problem_class = PME_boxes
# current_problem_classes = [(PME_boxes, {"sample_id": 1, "example": "boxes", "space_steps": 64, "time_steps": None, "params": 0})]
# example = "boxes"
# u_ex_0 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Basic_test_set/u_ex64_0")
# u_ex_1 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Basic_test_set/u_ex64_1")
# u_ex_2 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Basic_test_set/u_ex64_2")
# u_ex_3 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Basic_test_set/u_ex64_3")
# u_exs = [u_ex_0, u_ex_1, u_ex_2, u_ex_3]

problem_class = Buckley_Leverett
current_problem_classes = [(Buckley_Leverett, {"sample_id": 1, "example": "gravity", "space_steps": 64, "time_steps": None, "params": 0})]
example = "gravity"
u_ex_0 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Basic_test_set/u_ex64_0")
u_ex_1 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Basic_test_set/u_ex64_1")
u_ex_2 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Basic_test_set/u_ex64_2")
u_ex_3 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Basic_test_set/u_ex64_3")
u_ex_4 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Basic_test_set/u_ex64_4")
u_exs = [u_ex_0, u_ex_1, u_ex_2, u_ex_3, u_ex_4]

phandler = ProblemHandler(problem_classes = current_problem_classes, max_num_open_problems=200)
test_modulo=20
for j in range(100):
    loss_test = []
    problem_specs, problem_id = phandler.get_random_problem(0.1)
    problem = problem_specs["problem"]
    params = problem.params
    step = problem_specs["step"]
    u_last = problem_specs["last_solution"]
    u_new = train_model.forward(problem,u_last,step)
    if example == "Barenblatt":
        u_exact = problem.exact(problem.time[step+1])
    elif example == "boxes" or example == "gravity":
        u_exact = problem.exact(step + 1)
    u_exact = torch.Tensor(u_exact)
    optimizer.zero_grad()
    loss = exact_loss(u_new,u_exact)
    loss.backward()  # Backward pass
    optimizer.step()  # Optimize weights
    u_new.detach_()
    phandler.update_problem(problem_id, u_new)
    base_path = "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Models/Model_0/"
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    path = os.path.join(base_path, "{}.pt".format(j))
    torch.save(train_model, path)
    # TEST IF LOSS IS DECREASING WITH THE NUMBER OF ITERATIONS INCREASING
    if not (j%test_modulo):
      print("TESTING ON VALIDATION PROBLEMS")
      for kk in range(5):
        single_problem_loss_test = []
        if example == "Barenblatt":
            params_test = validation_problems(kk)
            problem_test = problem_class(example="Barenblatt", space_steps=64, time_steps=None, params=params_test)
        elif example == "boxes":
            params_test = validation_problems_boxes(kk)
            problem_test = problem_class(sample_id = None, example = "boxes", space_steps=64, time_steps=None, params=params_test)
        elif example == "gravity":
            params_test = validation_problems_BL(kk)
            problem_test = problem_class(sample_id=None, example="gravity", space_steps=64, time_steps=None, params=params_test)
        with torch.no_grad():
          u_init, tt = train_model.init_run_weno(problem_test, vectorized=True, just_one_time_step=False)
          u_test = u_init
          for k in range(tt):
            u_test = train_model.run_weno(problem_test, u_test, mweno=True, mapped=False, trainable=True, vectorized=True, k=k)
          if example == "Barenblatt":
              T_test = problem_test.params['T']
              power_test = problem_test.params['power']
              u_exact_test = problem_test.exact(T_test)
              u_exact_test = torch.Tensor(u_exact_test)
              single_problem_loss_test.append(exact_loss(u_test,u_exact_test))
          elif example == "boxes":
              single_problem_loss_test.append(exact_loss(u_test, u_exs[kk][:, -1]))
          elif example == "gravity":
              single_problem_loss_test.append(exact_loss(u_test, u_exs[kk][:, -1]))
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

# plt.figure(2)
# plt.plot(all_loss_test[:,:,0])
# np.save("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/Models/Model_0/all_loss_test.npy",all_loss_test)