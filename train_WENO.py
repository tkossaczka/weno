from define_WENO_Network import WENONetwork
import torch
from torch import optim
from define_problem_Digital import Digital_option
from define_problem_heat_eq import heat_equation

#train_model=WENONetwork()
#V=train_model.forward()

torch.set_default_dtype(torch.float64)

# TRAIN NETWORK
train_model = WENONetwork()

def monotonicity_loss(x):
    return torch.sum(torch.max(x[:-1]-x[1:], torch.Tensor([0.0])))

#optimizer = optim.SGD(train_model.parameters(), lr=0.001)
optimizer = optim.Adam(train_model.parameters())

for k in range(1000):
    # Forward path
    my_problem = Digital_option(space_steps=160, time_steps=1, params=None)
    #my_problem = heat_equation(space_steps=160, time_steps=1, params=None)
    V_train = train_model.forward(my_problem)
    # Train model:
    optimizer.zero_grad()  # Clear gradients
    loss = monotonicity_loss(V_train[:,1]) # Calculate loss
    loss.backward()  # Backward pass
    optimizer.step()  # Optimize weights
    print(k, loss)

#plt.plot(S, V_train.detach().numpy())
print("number of parameters:", sum(p.numel() for p in train_model.parameters()))
# g=train_model.parameters()
# g.__next__()

#torch.save(train_model, "model")