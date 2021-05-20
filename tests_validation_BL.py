import torch
import matplotlib.pyplot as plt
import numpy as np
from define_problem_Buckley_Leverett import Buckley_Leverett
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
    loss = error
    return loss

problem = Buckley_Leverett
all_loss_test = []
example = "gravity"

valid_problems = validation_problems.validation_problems_BL_2
_, rng, folder = valid_problems(0)
u_exs = validation_problems.exacts_test_BL(folder)

test_modulo=400
for i in range(8000):
    if not (i % test_modulo):
        print(i)
        train_model = torch.load('C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Models/Model_38/{}.pt'.format(i))
        loss_test = []
        for kk in range(rng):
            single_problem_loss_test = []
            params_test, _, _ = valid_problems(kk)
            problem_test = problem(sample_id = None, example = example, space_steps=128, time_steps=None, params=params_test)
            with torch.no_grad():
                u_init, nn = train_model.init_run_weno(problem_test, vectorized=True, just_one_time_step=False)
                u_test = u_init
                for k in range(nn):
                    u_test = train_model.run_weno(problem_test, u_test, mweno=True, mapped=False, trainable=True, vectorized=True, k=k)
                single_problem_loss_test.append(exact_loss(u_test, u_exs[kk][:, -1]))
            loss_test.append(single_problem_loss_test)
        all_loss_test.append(loss_test)

all_loss_test = np.array(all_loss_test)
# plt.plot(all_loss_test[:,:,0])

norm_losses=all_loss_test[:,:,0]/all_loss_test[:,:,0].max(axis=0)[None, :]
print("trained:", all_loss_test[:,:,0].min(axis=0))
plt.figure(figsize=(20.0, 10.0))
plt.xlabel('number of training steps')
plt.ylabel('LOSS')
plt.plot(norm_losses)
# my_xticks = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160,165,170,175,180,185,190,195]
# my_xticks = [0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190]
# my_xticks = [0,20,40,60,80,100,120,140,160,180]
# plt.xticks(my_xticks)
# plt.plot(my_xticks, norm_losses)
# plt.savefig("PME_2d_validation.pdf", bbox_inches='tight')

# np.save("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/Models/Model_37/all_loss_test.npy",all_loss_test)
