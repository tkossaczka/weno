from define_WENO_Network import WENONetwork
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

torch.set_default_dtype(torch.float64)

# TRAIN NETWORK
train_model = WENONetwork()

# DROP PROBLEM FOR TRAINING
#params = None
#problem_class = Buckley_Leverett
problem_class = Digital_option

def monotonicity_loss(u, problem_class, params, problem_main):
    # _, exact = problem_main.exact(first_step=True)
    # exact = torch.Tensor(exact)
    # error = torch.max(torch.abs(u-exact))
    monotonicity = torch.sum(torch.max(u[:-1]-u[1:], torch.Tensor([0.0])))
    loss = monotonicity #+ error

    # u_left = torch.zeros()
    # u_right = torch.zeros()
    # for k in range(0,len(S)):
    #     if S[k]<-0.25:
    #         u_left[k] = u[k]
    #     else:
    #         u_right[k] = u[k]
    #
    # peeks_left = torch.sum(torch.max(u_left[:-1]-u_left[1:], torch.Tensor([0.0])))
    # peeks_right = torch.sum(torch.abs(torch.min(u_right[:-1] - u_right[1:], torch.Tensor([0.0]))))

    # overflows = torch.sum(torch.abs(torch.min(u, torch.Tensor([0.0])) +
    #                                 (torch.max(u, torch.Tensor([1.0]))-torch.Tensor([1.0])))) # *(torch.max(x, torch.Tensor([1.0])) != 1)))
    # problem_ex = problem_class(space_steps=100*2*2*2*2*2*2*2, time_steps=50*4*4*4*4*4*4*4, params=params)
    # _, u_ex = train_model.compute_exact(problem_class, problem_ex, 100, 50,
    #                                                       just_one_time_step=True, trainable=False)
    # error = train_model.compute_error(u, u_ex)
    # loss = overflows + error # peeks_left + peeks_right
    return loss


#optimizer = optim.SGD(train_model.parameters(), lr=0.1)
optimizer = optim.Adam(train_model.parameters())

params_test = {'sigma': 0.3, 'rate': 0.1, 'E': 50, 'T': 1, 'e': 1e-13, 'xl': -6, 'xr': 1.5}
all_loss_test = []

for k in range(4000):
    loss_test = []
    # Forward path
    params = None
    problem_main = problem_class(space_steps=100, time_steps=None, params=params)
    V_train = train_model.forward(problem_main)
    # Train model:
    optimizer.zero_grad()  # Clear gradients
    # Calculate loss
    params = problem_main.get_params()
    loss = monotonicity_loss(V_train[:,1], problem_class, params, problem_main)  # Digital
    loss.backward()  # Backward pass
    optimizer.step()  # Optimize weights
    print(k, loss)
    base_path ="C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Digital_Option_Test/Models/Model_7/"
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    path = os.path.join(base_path, "{}.pt".format(k))
    torch.save(train_model, path)
    # TEST IF LOSS IS DECREASING WITH THE NUMBER OF ITERATIONS INCREASING
    single_problem_loss_test = []
    problem_test = problem_class(space_steps=100, time_steps=None, params=params_test)
    with torch.no_grad():
        u_test = train_model.run_weno(problem_test, trainable=True, vectorized=True, just_one_time_step=True)
        V_test, _, _ = problem_test.transformation(u_test)
    single_problem_loss_test.append(monotonicity_loss(V_test[:,1], problem_class, params_test, problem_test))
    loss_test.append(single_problem_loss_test)
    all_loss_test.append(loss_test)

#plt.plot(S, V_train.detach().numpy())
print("number of parameters:", sum(p.numel() for p in train_model.parameters()))
# g=train_model.parameters()
# g.__next__()

all_loss_test = np.array(all_loss_test)
plt.plot(all_loss_test[:,:,0])


# torch.save(train_model, "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Digital_Option_Test/Models/Model_1")