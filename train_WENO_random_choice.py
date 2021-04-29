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

train_model = WENONetwork_2()
train_model = sub_WENO(train_with_coeff=False)

def monotonicity_loss(u):
    monotonicity = torch.sum(torch.max(u[:-1]-u[1:], torch.Tensor([0.0])))
    loss = monotonicity
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

def overflows_loss(u):
    u_min = torch.Tensor([0.0])
    overflows = torch.sum(torch.abs(torch.min(u, u_min) - u_min))
    loss = 10e4*overflows
    return loss

def exact_loss_2d(u, u_ex):
    error = torch.mean((u_ex - u) ** 2)
    loss = 10e4*error
    return loss

# optimizer = optim.Adam(train_model.parameters(), lr=0.0001)   # Buckley-Leverett
# optimizer = optim.Adam(train_model.parameters(), lr=0.0001) #, weight_decay=0.001)  # PME boxes
optimizer = optim.Adam(train_model.inner_nn_weno6.parameters(), lr=0.1) #, weight_decay=0.1) # PME Barenblatt   # todo je lepsi lr 0.01?
# optimizer = optim.Adam([{'params': train_model.parameters(), 'lr': 0.1}, {'params': train_model2.parameters(), 'lr': 0.001}] ) #, weight_decay=0.1) # PME Barenblatt   # todo je lepsi lr 0.01?
# optimizer = optim.SGD(train_model.parameters(), lr=0.01, weight_decay=0.00001)

all_loss_test = []
#all_loss_test_2 = []

problem_class = PME

# current_problem_classes = [(PME, {"sample_id": None, "example": "Barenblatt_2d", "space_steps": 32, "time_steps": None, "params": None})]
# example = "Barenblatt_2d"
# rng = 4
#
current_problem_classes = [(PME, {"sample_id": None, "example": "Barenblatt", "space_steps": 64, "time_steps": None, "params": None})]
example = "Barenblatt"
model = 13
_, rng = validation_problems.validation_problems_barenblatt(1)

# current_problem_classes = [(PME, {"sample_id": 0, "example": "boxes", "space_steps": 64, "time_steps": None, "params": 0})]
# example = "boxes"
# u_ex_0 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Basic_test_set/u_ex_0")
# u_ex_1 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Basic_test_set/u_ex_1")
# u_ex_2 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Basic_test_set/u_ex_2")
# u_ex_3 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Basic_test_set/u_ex_3")
# u_ex_4 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Basic_test_set/u_ex_4")
# u_exs = [u_ex_0, u_ex_1, u_ex_2, u_ex_3, u_ex_4]
# rng = 4

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
# rng = 5

phandler = ProblemHandler(problem_classes = current_problem_classes, max_num_open_problems=200)
test_modulo=10
for j in range(200):
    loss_test = []
    #loss_test_2 = []
    problem_specs, problem_id = phandler.get_random_problem(0.1)
    problem = problem_specs["problem"]
    #print(problem.sample_id, problem.power)
    params = problem.params
    step = problem_specs["step"]
    u_last = problem_specs["last_solution"]
    if example == "Barenblatt_2d":
        u_new = train_model.forward(problem, u_last, step, mweno = True, mapped = False, dim =2)
    else:
        u_new = train_model.forward(problem, u_last, step, mweno = True, mapped = False)
        # u_new = train_model2.forward(problem, u_last, step, mweno = True, mapped = False)
        # u_new_nt = train_model.run_weno(problem, u_last, mweno=True, mapped=False, vectorized=True, trainable=False, k=step)
        if example == 'Barenblatt':
            u_new[u_new<0]=0
            # u_new_nt[u_new_nt<0]=0
    if example == "Barenblatt" or example == "Barenblatt_2d":
        u_exact = problem.exact(problem.time[step+1])
    elif example == "boxes" or example == "gravity":
        u_exact = problem.exact(step + 1)
    u_exact = torch.Tensor(u_exact)
    optimizer.zero_grad()
    if example == "Barenblatt_2d":
        loss = exact_loss_2d(u_new,u_exact)
    else:
        loss_exact = exact_loss(u_new,u_exact)
        #loss_overflows = overflows_loss(u_new)
        print(loss_exact) #, loss_overflows)
        loss = loss_exact #+ loss_overflows
    # print(loss)
    # minibatch_size=25
    # loss = loss/minibatch_size
    loss.backward()  # Backward pass
    # if j%minibatch_size == 0:
    #     print("optimizer_step")
    optimizer.step()  # Optimize weights
        # optimizer.zero_grad()
    ##############################
    u_new.detach_()
    phandler.update_problem(problem_id, u_new)
    if not (j % test_modulo):
        if example == "Barenblatt":
            base_path = "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/Models/Model_{}/".format(model)
        elif example == "boxes":
            base_path = "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/Models_boxes/Model_19/"  # TODO model 18 je uz obsadeny!!!!!
        elif example == "Barenblatt_2d":
            base_path = "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/Models_2d/Model_8/"
        elif example == "gravity":
            base_path = "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Models/Model_17/"
        if not os.path.exists(base_path):
            os.mkdir(base_path)
        path = os.path.join(base_path, "{}.pt".format(j))
        torch.save(train_model, path)
    # TEST IF LOSS IS DECREASING WITH THE NUMBER OF ITERATIONS INCREASING
    if not (j % test_modulo):
        print("TESTING ON VALIDATION PROBLEMS")
        for kk in range(rng):
            single_problem_loss_test = []
            #single_problem_loss_test_2 = []
            if example == "Barenblatt":
                params_test,_ = validation_problems.validation_problems_barenblatt(kk)
                problem_test = problem_class(sample_id=None, example="Barenblatt", space_steps=64, time_steps=None, params=params_test)
            elif example == "Barenblatt_2d":
                params_test,_  = validation_problems.validation_problems_barenblatt_2d(kk)
                problem_test = problem_class(sample_id=None, example="Barenblatt_2d", space_steps=32, time_steps=None, params=params_test)
            elif example == "boxes":
                params_test,_  = validation_problems.validation_problems_boxes(kk)
                problem_test = problem_class(sample_id=None, example="boxes", space_steps=64, time_steps=None, params=params_test)
            elif example == "gravity":
                params_test,_  = validation_problems.validation_problems_BL(kk)
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
                        if example == 'Barenblatt':
                            u_test[u_test < 0] = 0
            if example == "Barenblatt":
                T_test = problem_test.params['T']
                power_test = problem_test.params['power']
                u_exact_test = problem_test.exact(T_test)
                u_exact_test = torch.Tensor(u_exact_test)
                single_problem_loss_test.append(exact_loss(u_test, u_exact_test))
                #single_problem_loss_test_2.append(overflows_loss(u_test))
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
            #loss_test_2.append(single_problem_loss_test_2)
        print(loss_test) #, loss_test_2)
        all_loss_test.append(loss_test)
        #all_loss_test_2.append(loss_test_2)
        # if np.max(np.array(all_loss_test)[:, :, 0][0,:] / np.array(all_loss_test)[:, :, 0][-1,:]) > bound:
        #     print("lr will be updated")
        #     optimizer.defaults['lr'] = optimizer.defaults['lr'] * 0.5
        #     bound = bound+0.05

# print("number of parameters:", sum(p.numel() for p in train_model.parameters()))
# g=train_model.parameters()
# g.__next__()

all_loss_test = np.array(all_loss_test)
norm_losses=all_loss_test[:,:,0]/all_loss_test[:,:,0].max(axis=0)[None, :]
print("trained:", all_loss_test[:,:,0].min(axis=0))
plt.plot(norm_losses)
plt.legend(['2.157', '3.012', '3.697', '3.987', '4.158', '4.572', '4.723', '5.041', '5.568', '6.087', '6.284', '7.124', '7.958'])


# plt.figure(2)
# all_loss_test_2 = np.array(all_loss_test_2)
# norm_losses=all_loss_test_2[:,:,0]/all_loss_test_2[:,:,0].max(axis=0)[None, :]
# print("trained:", all_loss_test_2[:,:,0].min(axis=0))
# plt.plot(norm_losses)
# plt.show()

# plt.figure(2)
# plt.plot(all_loss_test[:,:,0])

# np.save("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Models/Model_8/all_loss_test.npy",all_loss_test)

# def exact_compare_loss(u, u_nt, u_ex):
#     error_t = torch.mean((u_ex - u)**2)
#     error_nt = torch.mean((u_ex - u_nt)**2)
#     if error_t > error_nt:
#         loss = error_t-error_nt
#     else:
#         loss = error_t-error_t
#     return loss
#
# all_loss_test_2 = []
# train_model = torch.load(os.path.join(base_path,"{}.pt".format(j-test_modulo+1)))
# train_model.train_with_coeff = True
# optimizer = optim.Adam(train_model.m_nn.parameters(), lr=0.001)
# train_model.train_with_coeff = True
# test_modulo=10
# for j in range(100):
#     loss_test = []
#     problem_specs, problem_id = phandler.get_random_problem(0.1)
#     problem = problem_specs["problem"]
#     params = problem.params
#     step = problem_specs["step"]
#     u_last = problem_specs["last_solution"]
#     u_new = train_model.forward(problem, u_last, step, mweno=True, mapped=False)
#     # u_new = train_model.run_weno(problem, u_last, mweno=True, mapped=False, vectorized=True, trainable=True, k=step)
#     u_new[u_new<0]=0
#     # u_new_nt = train_model.run_weno(problem, u_last, mweno=True, mapped=False, vectorized=True, trainable=False, k=step)
#     # u_new_nt[u_new_nt<0]=0
#     u_exact = problem.exact(problem.time[step+1])
#     u_exact = torch.Tensor(u_exact)
#     optimizer.zero_grad()
#     # loss = exact_compare_loss(u_new,u_new_nt,u_exact)
#     loss = exact_loss(u_new, u_exact)
#     print(loss)
#     loss.backward()
#     optimizer.step()
#     u_new.detach_()
#     phandler.update_problem(problem_id, u_new)
#     if not (j % test_modulo):
#         base_path = "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/Models/Model_{}/".format(model)
#         if not os.path.exists(base_path):
#             os.mkdir(base_path)
#         path = os.path.join(base_path, "0{}.pt".format(j))
#         torch.save(train_model, path)
#     # TEST IF LOSS IS DECREASING WITH THE NUMBER OF ITERATIONS INCREASING
#     if not (j % test_modulo):
#         print("TESTING ON VALIDATION PROBLEMS")
#         for kk in range(rng):
#             single_problem_loss_test = []
#             params_test,_ = validation_problems.validation_problems_barenblatt(kk)
#             problem_test = problem_class(sample_id=None, example="Barenblatt", space_steps=64, time_steps=None, params=params_test)
#             with torch.no_grad():
#                 u_init, tt = train_model.init_run_weno(problem_test, vectorized=True, just_one_time_step=False)
#                 u_test = u_init
#                 for k in range(tt):
#                     u_test = train_model.run_weno(problem_test, u_test, mweno=True, mapped=False, trainable=True, vectorized=True, k=k)
#                     u_test[u_test < 0] = 0
#             T_test = problem_test.params['T']
#             power_test = problem_test.params['power']
#             u_exact_test = problem_test.exact(T_test)
#             u_exact_test = torch.Tensor(u_exact_test)
#             single_problem_loss_test.append(exact_loss(u_test, u_exact_test))
#             loss_test.append(single_problem_loss_test)
#         print(loss_test)
#         all_loss_test_2.append(loss_test)
#
# all_loss_test_2 = np.array(all_loss_test_2)
# norm_losses_2=all_loss_test_2[:,:,0]/all_loss_test_2[:,:,0].max(axis=0)[None, :]
# print("trained:", all_loss_test_2[:,:,0].min(axis=0))
# plt.figure(2)
# plt.plot(norm_losses_2)
# plt.legend(['2.157', '3.012', '3.697', '3.987', '4.158', '4.572', '4.723', '5.041', '5.568', '6.087', '6.284', '7.124', '7.958'])
#
