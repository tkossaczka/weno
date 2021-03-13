import torch
from torch import optim
import os
import numpy as np
import matplotlib.pyplot as plt

from network.define_WENO_Network_2 import WENONetwork_2
from utils.problem_handler import ProblemHandler
from network.losses import exact_loss, exact_loss_2d, monotonicity_loss
from define_problem_PME import PME

torch.set_default_dtype(torch.float64)

train_model = WENONetwork_2()

optimizer = optim.Adam(train_model.parameters(), lr=0.001, weight_decay=0.00001)  # PME boxes

def validation_problems_boxes(j):
    params_vld = []
    params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': 2, 'd': 1})
    params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': 3, 'd': 1})
    params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': 4, 'd': 1})
    params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': 5, 'd': 1})
    return params_vld[j]
all_loss_test = []

problem_class = PME

current_problem_classes = [(PME, {"sample_id": 0, "example": "boxes", "space_steps": 64, "time_steps": None, "params": 0})]
example = "boxes"
u_ex_0 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Basic_test_set/u_ex_0")
u_ex_1 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Basic_test_set/u_ex_1")
u_ex_2 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Basic_test_set/u_ex_2")
u_ex_3 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Basic_test_set/u_ex_3")
u_exs = [u_ex_0, u_ex_1, u_ex_2, u_ex_3]
rng = 4


phandler = ProblemHandler(problem_classes = current_problem_classes, max_num_open_problems=200)
test_modulo=50
for j in range(200):
    loss_test = []
    problem_specs, problem_id = phandler.get_random_problem(0.1)
    problem = problem_specs["problem"]
    #print(problem.sample_id)
    params = problem.params
    step = problem_specs["step"]
    u_last = problem_specs["last_solution"]
    u_new = train_model.forward(problem, u_last, step, mweno=True, mapped=False)
    u_exact = problem.exact(step + 1)
    u_exact = torch.Tensor(u_exact)
    optimizer.zero_grad()
    minibatch_size=25
    loss = loss/minibatch_size
    loss.backward()  # Backward pass
    optimizer.step()  # Optimize weights
    u_new.detach_()
    phandler.update_problem(problem_id, u_new)
    base_path = "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/Models_boxes/Model_11/"

    if not os.path.exists(base_path):
        os.mkdir(base_path)
    path = os.path.join(base_path, "{}.pt".format(j))
    torch.save(train_model, path)
    # TEST IF LOSS IS DECREASING WITH THE NUMBER OF ITERATIONS INCREASING
    if not (j % test_modulo):
        print("TESTING ON VALIDATION PROBLEMS")
        for kk in range(rng):
            params_test = validation_problems_boxes(kk)
            problem_test = problem_class(sample_id=None, example="boxes", space_steps=64, time_steps=None,
                                         params=params_test)

            with torch.no_grad():
                u_init, tt = train_model.init_run_weno(problem_test, vectorized=True, just_one_time_step=False)
                u_test = u_init
                for k in range(tt):
                    u_test = train_model.run_weno(problem_test, u_test, mweno=True, mapped=False, trainable=True, vectorized=True, k=k)
            single_problem_loss_test = []
            single_problem_loss_test.append(exact_loss(u_test, u_exs[kk][0:1024 + 1:16, -1]))
            loss_test.append(single_problem_loss_test)
        print(loss_test)
        all_loss_test.append(loss_test)

all_loss_test = np.array(all_loss_test)
norm_losses=all_loss_test[:,:,0]/all_loss_test[:,:,0].max(axis=0)[None, :]
print("trained:", all_loss_test[:,:,0].min(axis=0))
plt.plot(norm_losses)
plt.show()

plt.figure(2)
plt.plot(all_loss_test[:,:,0])
# np.save("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Models/Model_8/all_loss_test.npy",all_loss_test)