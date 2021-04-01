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

problem_class = Buckley_Leverett
rng = 5

# problem_class = Digital_option

# problem_class = PME
# example = "boxes"
# example = "Barenblatt"
# rng = 4

def monotonicity_loss(u):
    monotonicity = torch.sum(torch.max(u[:-1]-u[1:], torch.Tensor([0.0])))
    loss = monotonicity
    return loss

def exact_loss(u, u_ex):
    error = train_model.compute_error(u, u_ex)
    # loss = 10e1*error # PME boxes
    # loss = 10e4*error # PME Barenblatt
    loss = error # Buckley-Leverett
    return loss

#optimizer = optim.SGD(train_model.parameters(), lr=0.1)
optimizer = optim.Adam(train_model.parameters(), lr=0.0001)   # Buckley-Leverett
# optimizer = optim.Adam(train_model.parameters(), lr=0.001, weight_decay=0.00001) # PME boxes
# optimizer = optim.Adam(train_model.parameters(), lr=0.1, weight_decay=0.0001) # PME Barenblatt

def validation_problems_barenblatt(j):
    params_vld = []
    params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 2, 'd': 1})
    params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 3, 'd': 1})
    params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 4, 'd': 1})
    params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 5, 'd': 1})
    return params_vld[j]

def validation_problems_digital(j):
    params_vld = []
    params_vld.append({'sigma': 0.3, 'rate': 0.1, 'E': 50, 'T': 1, 'e': 1e-13, 'xl': -6, 'xr': 1.5})
    params_vld.append({'sigma': 0.25, 'rate': 0.1, 'E': 50, 'T': 1, 'e': 1e-13, 'xl': -6, 'xr': 1.5})
    params_vld.append({'sigma': 0.2, 'rate': 0.08, 'E': 50, 'T': 1, 'e': 1e-13, 'xl': -6, 'xr': 1.5})
    return params_vld[j]

def validation_problems_BL(j):
    params_vld = []
    params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 1, 'G': 5})
    params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 1, 'G': 0})
    params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.5, 'G': 2})
    params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.5, 'G': 1})
    params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.25, 'G': 4})
    return params_vld[j]

def validation_problems_boxes(j):
    params_vld = []
    params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': 2, 'd': 1})
    params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': 3, 'd': 1})
    params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': 4, 'd': 1})
    params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': 5, 'd': 1})
    params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': 6, 'd': 1})
    return params_vld[j]

if problem_class == Digital_option:
    rng = 3

if problem_class == Buckley_Leverett:
    example = "gravity"
    folder = 1
    u_ex_0 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Basic_test_set_{}/u_ex64_0".format(folder))
    u_ex_1 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Basic_test_set_{}/u_ex64_1".format(folder))
    u_ex_2 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Basic_test_set_{}/u_ex64_2".format(folder))
    u_ex_3 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Basic_test_set_{}/u_ex64_3".format(folder))
    u_ex_4 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Basic_test_set_{}/u_ex64_4".format(folder))
    u_exs = [u_ex_0, u_ex_1, u_ex_2, u_ex_3, u_ex_4]
    rng = 5

if problem_class == PME and example == "boxes":
    u_ex_0 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Basic_test_set/u_ex_0")
    u_ex_1 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Basic_test_set/u_ex_1")
    u_ex_2 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Basic_test_set/u_ex_2")
    u_ex_3 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Basic_test_set/u_ex_3")
    u_ex_4 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Basic_test_set/u_ex_4")
    u_exs = [u_ex_0, u_ex_1, u_ex_2, u_ex_3, u_ex_4]
    rng = 4

all_loss_test = []
save_id = 0

for j in range(200):
    loss_test = []
    # Forward path
    if problem_class == Digital_option:
        params = None
        problem_main = problem_class(space_steps=64, time_steps=None, params=params)
        u_init, nn = train_model.init_run_weno(problem_main, vectorized=True, just_one_time_step=True)
        # # parameters needed for the computation of exact solution
        # params_main = problem_main.params
        # rate = params_main['rate']
        # sigma = params_main['sigma']
        # T = params_main['T']
        # E = params_main['E']
        # x, time = problem_main.x, problem_main.time
        # tt = T - time
        # S = E * np.exp(x)
    elif problem_class == Buckley_Leverett:
        params = 0
        problem_main = problem_class(sample_id=j, example = "gravity", space_steps = 64, time_steps = None, params = params)
        u_init, nn = train_model.init_run_weno(problem_main, vectorized=True, just_one_time_step=False)
    elif problem_class == PME:
        if example == "boxes":
            params = 0
        elif example == "Barenblatt":
            params = None
        problem_main = problem_class(sample_id=0, example = example, space_steps=64, time_steps=None, params=params)
        u_init, nn = train_model.init_run_weno(problem_main, vectorized=True, just_one_time_step=False)
    u_train = u_init
    print(problem_main.params)
    for k in range(nn):
        #exact = np.exp(-rate * (T - tt[k+1])) * norm.cdf((np.log(S / E) + (rate - (sigma ** 2) / 2) * (T - tt[k+1])) / (sigma * np.sqrt(T - tt[k+1])))
        #exact = torch.Tensor(exact)
        # Forward path
        u_train = train_model.forward(problem_main,u_train,k, mweno = True, mapped = False)
        V_train, _, _ = problem_main.transformation(u_train)
        # Train model:
        optimizer.zero_grad()  # Clear gradients
        # Calculate loss
        if problem_class == Digital_option:
            mon_loss = monotonicity_loss(V_train)
            loss = mon_loss
        elif problem_class == Buckley_Leverett:
            exact = problem_main.exact(k + 1)
            ex_loss = exact_loss(V_train,exact)
            loss = ex_loss
        elif problem_class == PME and example == "boxes":
            exact = problem_main.exact(k + 1)
            ex_loss = exact_loss(V_train, exact)
            loss = ex_loss
        elif problem_class == PME and example == "Barenblatt":
            exact = torch.Tensor(problem_main.exact(problem_main.time[k + 1]))
            ex_loss = exact_loss(V_train, exact)
            loss = ex_loss
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize weights
        print(j, k, loss)
        u_train.detach_()
    if problem_class == Digital_option:
        base_path ="C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Digital_Option_Test/Models/Model_17/"
    elif problem_class == Buckley_Leverett:
        base_path = "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Models/Model_16/"
    elif problem_class == PME and example == "boxes":
        base_path = "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/Models_boxes/Model_16/"
    elif problem_class == PME and example == "Barenblatt":
        base_path = "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/Models/Model_51/"
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    path = os.path.join(base_path, "{}.pt".format(save_id))
    torch.save(train_model, path)
    # TEST IF LOSS IS DECREASING WITH THE NUMBER OF ITERATIONS INCREASING
    for kk in range(rng):
        single_problem_loss_test = []
        if problem_class == Digital_option:
            params_test = validation_problems_digital(kk)
            problem_test = problem_class(space_steps=64, time_steps=None, params=params_test)
        elif problem_class == Buckley_Leverett:
            params_test = validation_problems_BL(kk)
            problem_test = problem_class(sample_id=None, example="gravity", space_steps=64, time_steps=None, params=params_test)
        elif problem_class == PME and example == "boxes":
            params_test = validation_problems_boxes(kk)
            problem_test = problem_class(sample_id=None, example="boxes", space_steps=64, time_steps=None, params=params_test)
        elif problem_class == PME and example == "Barenblatt":
            params_test = validation_problems_barenblatt(kk)
            problem_test = problem_class(sample_id=None, example="Barenblatt", space_steps=64, time_steps=None, params=params_test)
        with torch.no_grad():
            if problem_class == Digital_option:
                u_init, nn = train_model.init_run_weno(problem_test, vectorized=True, just_one_time_step=True)
            elif problem_class == Buckley_Leverett or problem_class == PME:
                u_init, nn = train_model.init_run_weno(problem_test, vectorized=True, just_one_time_step=False)
            u_test = u_init
            for k in range(nn):
                u_test = train_model.run_weno(problem_test, u_test, mweno=True, mapped=False, trainable=True, vectorized=True, k=k)
            V_test, _, _ = problem_test.transformation(u_test)
            if problem_class == Digital_option:
                single_problem_loss_test.append(monotonicity_loss(V_test))
            elif problem_class == Buckley_Leverett:
                u_ex = u_exs[kk][:, -1]
                single_problem_loss_test.append(exact_loss(V_test,u_ex))
            elif problem_class == PME and example == "boxes":
                u_ex = u_exs[kk][0:1024 + 1:16, -1]
                single_problem_loss_test.append(exact_loss(V_test, u_ex))
            elif problem_class == PME and example == "Barenblatt":
                T_test = problem_test.params['T']
                power_test = problem_test.params['power']
                u_exact_test = problem_test.exact(T_test)
                u_exact_test = torch.Tensor(u_exact_test)
                single_problem_loss_test.append(exact_loss(u_test, u_exact_test))
        loss_test.append(single_problem_loss_test)
    all_loss_test.append(loss_test)
    save_id = save_id + 1

#plt.plot(S, V_train.detach().numpy())
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


# torch.save(train_model, "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Digital_Option_Test/Models/Model_1")