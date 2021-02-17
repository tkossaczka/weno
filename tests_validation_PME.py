import torch
import matplotlib.pyplot as plt
import numpy as np
from define_problem_PME import PME
from define_problem_PME_boxes import PME_boxes
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

u_ex_0 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Basic_test_set/u_ex64_0")
u_ex_1 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Basic_test_set/u_ex64_1")
u_ex_2 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Basic_test_set/u_ex64_2")
u_ex_3 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Basic_test_set/u_ex64_3")
u_exs = [u_ex_0, u_ex_1, u_ex_2, u_ex_3]

problem = PME_boxes
all_loss_test = []
example = "boxes"

for i in range(800):
    print(i)
    train_model = torch.load('C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/Models/Model_38/{}.pt'.format(i))
    loss_test = []
    for kk in range(4):
        single_problem_loss_test = []
        if example == "Barenblatt":
            params_test = validation_problems(kk)
        else:
            params_test = validation_problems_boxes(kk)
        problem_test = problem(sample_id = None, example = example, space_steps=64, time_steps=None, params=params_test)
        T = problem_test.params['T']
        power = problem_test.params['power']
        u_init, nn = train_model.init_run_weno(problem_test, vectorized=True, just_one_time_step=False)
        with torch.no_grad():
            u_test = u_init
            for k in range(nn):
                u_test = train_model.run_weno(problem_test, u_test, mweno=True, mapped=False, trainable=True, vectorized=True, k=k)
            if example == "Barenblatt":
                u_ex = problem_test.exact(T)
                u_ex = torch.Tensor(u_ex)
                single_problem_loss_test.append(exact_loss(u_test, u_ex))
            else:
                single_problem_loss_test.append(exact_loss(u_test,u_exs[kk][:, -1]))
        loss_test.append(single_problem_loss_test)
    all_loss_test.append(loss_test)

all_loss_test = np.array(all_loss_test)
# plt.plot(all_loss_test[:,:,0])

norm_losses=all_loss_test[:,:,0]/all_loss_test[:,:,0].max(axis=0)[None, :]
print("trained:", all_loss_test[:,:,0].min(axis=0))
plt.plot(norm_losses)
plt.show()

# np.save("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/Models/Model_37/all_loss_test.npy",all_loss_test)
