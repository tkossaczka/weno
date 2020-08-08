from define_WENO_Network import WENONetwork
import torch
from torch import optim
from define_problem_Digital import Digital_option
from define_problem_heat_eq import heat_equation
from define_problem_Call import Call_option
from define_problem_Buckley_Leverett import Buckley_Leverett
from define_problem_PME import PME

torch.set_default_dtype(torch.float64)

# TRAIN NETWORK
train_model = WENONetwork()

# DROP PROBLEM FOR TRAINING
params = None
#problem_class = Buckley_Leverett
problem_class = Digital_option

def monotonicity_loss(u, problem_class, params):
    loss = torch.sum(torch.max(u[:-1]-u[1:], torch.Tensor([0.0])))

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
    # loss = overflows # + error # peeks_left + peeks_right
    #print(error)
    #print(overflows)
    return loss


#optimizer = optim.SGD(train_model.parameters(), lr=0.1)
optimizer = optim.Adam(train_model.parameters())

for k in range(100):
    # Forward path
    params = None
    #params = {'sigma': 0.3, 'rate': 0.1, 'E': 50, 'T': 1, 'e': 1e-13, 'xl': -6, 'xr': 1.5}
    #my_problem = Digital_option(space_steps=160, time_steps=1,params=params)
    #my_problem = heat_equation(space_steps=160, time_steps=1, params=None)
    #my_problem = Buckley_Leverett(space_steps=200, time_steps=1,params=params)

    #problem_main = problem_class(space_steps=100, time_steps=50, params=params)
    problem_main = problem_class(space_steps=160, time_steps=None, params=params)

    V_train = train_model.forward(problem_main)
    # Train model:
    optimizer.zero_grad()  # Clear gradients
    # Calculate loss
    params = problem_main.get_params()
    #loss = monotonicity_loss(V_train[:,1])
    loss = monotonicity_loss(V_train, problem_class, params=params)
    loss.backward()  # Backward pass
    optimizer.step()  # Optimize weights
    print(k, loss)
    #print(params)

#plt.plot(S, V_train.detach().numpy())
print("number of parameters:", sum(p.numel() for p in train_model.parameters()))
# g=train_model.parameters()
# g.__next__()

torch.save(train_model, "model")