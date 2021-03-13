import torch
import matplotlib.pyplot as plt
import numpy as np
from define_WENO_Network import WENONetwork
from define_Euler_system import Euler_system
torch.set_default_dtype(torch.float64)

# TRAIN NETWORK
train_model = WENONetwork()

# DROP PROBLEM FOR TRAINING
problem_main = Euler_system

def exact_loss(u, u_ex):
    error = train_model.compute_error(u, u_ex)
    loss = error
    return loss

def overflows_loss(u, u_ex):
    u_max = torch.max(u_ex)
    u_min = torch.min(u_ex)
    overflows = torch.sum(torch.abs(torch.min(u, u_min)-u_min) + torch.max(u, u_max)-u_max )
    loss = overflows # peeks_left + peeks_right
    return loss

all_loss_test = []
for i in range(30):
    print(i)
    train_model = torch.load('C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Euler_System_Test/Models/Model_54/{}.pt'.format(i))
    loss_test = []
    for j in range(1):
        print(j)
        single_problem_loss_test = []
        params = None
        sp_st = 64
        init_cond = "Sod"
        problem_test = problem_main(space_steps=sp_st, init_cond=init_cond, time_steps=None, params=params,time_disc=None, init_mid=False, init_general=False)
        T = problem_test.params["T"]
        x = problem_test.x
        q_0_test, q_1_test, q_2_test, lamb_test, nn, h = train_model.init_Euler(problem_test, vectorized=True, just_one_time_step=False)
        q_0_input, q_1_input, q_2_input = q_0_test, q_1_test, q_2_test
        with torch.no_grad():
            t_update = 0
            t = 0.9 * h / lamb_test
            while t_update < T:
                if (t_update + t) > T:
                    t = T - t_update
                t_update = t_update + t
                q_0_test, q_1_test, q_2_test, lamb_test = train_model.run_weno(problem_test, mweno=True, mapped=False, method="char", q_0=q_0_input, q_1=q_1_input,q_2=q_2_input, lamb=lamb_test,vectorized=True, trainable=True, k=0, dt=t)
                t = 0.9 * h / lamb_test
                q_0_input = q_0_test.detach().numpy()
                q_1_input = q_1_test.detach().numpy()
                q_2_input = q_2_test.detach().numpy()
                q_0_input = torch.Tensor(q_0_input)
                q_1_input = torch.Tensor(q_1_input)
                q_2_input = torch.Tensor(q_2_input)
        rho_test = q_0_test
        u_test = q_1_test / rho_test
        E_test = q_2_test
        p_test = (1.4 - 1) * (E_test - 0.5 * rho_test * u_test ** 2)
        p_ex_test, rho_ex_test, u_ex_test, _, _ = problem_test.exact(x, T)
        single_problem_loss_test.append(overflows_loss(u_test, u_ex_test).detach().numpy().max())
        loss_test.append(single_problem_loss_test)
    all_loss_test.append(loss_test)

all_loss_test = np.array(all_loss_test) #shape (training_steps, num_valid_problems, time_steps)
plt.plot(all_loss_test[:,:,-1])