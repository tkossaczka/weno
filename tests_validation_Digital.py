import torch
import matplotlib.pyplot as plt
import numpy as np
from define_problem_Digital import Digital_option
from define_WENO_Network_2 import WENONetwork_2

train_model = WENONetwork_2()
torch.set_default_dtype(torch.float64)

def monotonicity_loss(u):
    monotonicity = torch.sum(torch.max(u[:-1]-u[1:], torch.Tensor([0.0])))
    loss = monotonicity
    return loss

def exact_loss(u, u_ex):
    error = train_model.compute_error(u, u_ex)
    loss = error
    return loss

def validation_problems(j):
    params_vld = []
    params_vld.append({'sigma': 0.3, 'rate': 0.1, 'E': 50, 'T': 1, 'e': 1e-13, 'xl': -6, 'xr': 1.5})
    params_vld.append({'sigma': 0.25, 'rate': 0.1, 'E': 50, 'T': 1, 'e': 1e-13, 'xl': -6, 'xr': 1.5})
    params_vld.append({'sigma': 0.2, 'rate': 0.08, 'E': 50, 'T': 1, 'e': 1e-13, 'xl': -6, 'xr': 1.5})
    return params_vld[j]

problem = Digital_option
all_loss_test = []

for i in range(7000):
    print(i)
    train_model = torch.load('C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Digital_Option_Test/Models/Model_17/{}.pt'.format(i))
    loss_test = []
    single_problem_loss_test = []
    for kk in range(3):
        single_problem_loss_test = []
        params_test = validation_problems(kk)
        problem_test = problem(space_steps=100, time_steps=None, params=params_test)
        u_init, nn = train_model.init_run_weno(problem_test, vectorized=True, just_one_time_step=True)
        with torch.no_grad():
            u_init, nn = train_model.init_run_weno(problem_test, vectorized=True, just_one_time_step=True)
            u_test = u_init
            for k in range(nn):
                u_test = train_model.run_weno(problem_test, u_test, mweno=True, mapped=False, trainable=True,
                                              vectorized=True, k=k)
            V_test, _, _ = problem_test.transformation(u_test)
            for k in range(nn + 1):
                single_problem_loss_test.append(monotonicity_loss(V_test))
        loss_test.append(single_problem_loss_test)
    all_loss_test.append(loss_test)

all_loss_test = np.array(all_loss_test)
plt.plot(all_loss_test[:,:,0])

