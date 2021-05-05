import torch
import matplotlib.pyplot as plt
import numpy as np
from define_problem_PME import PME
from define_WENO_Network_2 import WENONetwork_2
from validation_problems import validation_problems

train_model = WENONetwork_2()
torch.set_default_dtype(torch.float64)

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

valid_problems = validation_problems.validation_problems_barenblatt
_, rng = valid_problems(1)

# def validation_problems(j):
#     params_vld = []
#     params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 2, 'd': 1})
#     params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 3, 'd': 1})
#     params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 4, 'd': 1})
#     params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 5, 'd': 1})
#     return params_vld[j]
#
# def validation_problems_boxes(j):
#     params_vld = []
#     params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': 2, 'd': 1})
#     params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': 3, 'd': 1})
#     params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': 4, 'd': 1})
#     params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': 5, 'd': 1})
#     return params_vld[j]
#
# u_ex_0 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Basic_test_set/u_ex64_0")
# u_ex_1 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Basic_test_set/u_ex64_1")
# u_ex_2 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Basic_test_set/u_ex64_2")
# u_ex_3 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Basic_test_set/u_ex64_3")
# u_exs = [u_ex_0, u_ex_1, u_ex_2, u_ex_3]

problem = PME
all_loss_test = []
example = "Barenblatt"

test_modulo=5
for i in range(200):
    if not (i % test_modulo):
        print(i)
        train_model = torch.load('C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/Models/Model_58/{}.pt'.format(i))
        loss_test = []
        for kk in range(rng):
            single_problem_loss_test = []
            if example == "Barenblatt":
                params_test, _ = valid_problems(kk)
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
plt.figure(figsize=(20.0, 10.0))
plt.xlabel('number of training steps')
plt.ylabel('LOSS')
my_xticks = [0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190]
my_xticks = [0,20,40,60,80,100,120,140,160,180]
plt.xticks(my_xticks)
plt.plot(my_xticks, norm_losses)
# plt.savefig("PME_validation.pdf", bbox_inches='tight')

# np.save("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/Models/Model_37/all_loss_test.npy",all_loss_test)
