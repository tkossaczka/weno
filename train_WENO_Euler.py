from define_WENO_Network import WENONetwork
import torch
from torch import optim
from define_Euler_system import Euler_system

torch.set_default_dtype(torch.float64)

# TRAIN NETWORK
train_model = WENONetwork()

# DROP PROBLEM FOR TRAINING
#params = None
problem_class = Euler_system

def monotonicity_loss(u):
    monotonicity = torch.sum(torch.max(u[:-1]-u[1:], torch.Tensor([0.0])))
    loss = monotonicity
    return loss

def exact_loss(u, u_ex):
    error = train_model.compute_error(u, u_ex)
    loss = error
    return loss

#optimizer = optim.SGD(train_model.parameters(), lr=0.1)
optimizer = optim.Adam(train_model.parameters())

for k in range(10):
    # Forward path
    params = None
    problem_main = problem_class(space_steps=128*2, time_steps=None, params=params)
    V_train = train_model.forward_Euler(problem_main)
    # Train model:
    optimizer.zero_grad()  # Clear gradients
    # Calculate loss
    params = problem_main.get_params()
    loss = exact_loss(V_train[:,1], problem_class, params, problem_main)  # Digital
    #loss = monotonicity_loss(V_train, problem_class, params=params)  # Buckley
    loss.backward()  # Backward pass
    optimizer.step()  # Optimize weights
    print(k, loss)
    #print(params)

#plt.plot(S, V_train.detach().numpy())
print("number of parameters:", sum(p.numel() for p in train_model.parameters()))
# g=train_model.parameters()
# g.__next__()

torch.save(train_model, "model4")