import numpy as np
import matplotlib.pyplot as plt
from BS_WENO import BS_WENO
from compute_omegas5 import compute_omegas5
from compute_omegas6 import compute_omegas6
from initial_condition import initial_condition
import torch
from torch import nn, optim # Sets of preset layers and optimizers
import torch.nn.functional as F # Sets of functions such as ReLU
from torchvision import datasets, transforms # Popular datasets, architectures and common


class WENONetwork(nn.Module):
    def __init__(self, pre_omegas5, pre_omegas6):
        super().__init__()
        self.omegas5 = nn.Parameter(pre_omegas5, requires_grad=True)
        self.omegas6 = nn.Parameter(pre_omegas6, requires_grad=True)

    def forward(self):
        sigma = 0.3;
        rate = 0.1;
        E = 50;
        T = 1;
        e = 10 ** (-13);
        xl = -6
        xr = 1.5
        m = 80;
        omegas5=self.omegas5
        omegas6=self.omegas6
        V, S, tt = BS_WENO(sigma, rate, E, T, e, xl, xr, m, omegas5, omegas6)
        return V

    def return_S_tt(self):
        sigma = 0.3;
        rate = 0.1;
        E = 50;
        T = 1;
        e = 10 ** (-13);
        xl = -6
        xr = 1.5
        m = 80;
        omegas5 = self.omegas5
        omegas6 = self.omegas6
        V, S, tt = BS_WENO(sigma, rate, E, T, e, xl, xr, m, omegas5, omegas6)
        return S, tt


m = 80;
e = 10 ** (-13);
E = 50;
xl = -6
xr = 1.5
u=initial_condition(E,xl,xr,m)

pre_omegas6=compute_omegas6(u,m,1,e,None)
pre_omegas5=compute_omegas5(u,m,1,e,None)

train_model=WENONetwork(pre_omegas5, pre_omegas6)

V=train_model.forward()

def my_loss(input):
    loss = torch.max((input)**2)
    return loss

#criterion = nn.MSELoss()
optimizer = optim.SGD(train_model.parameters(), lr=0.001)

S,tt=train_model.return_S_tt()
plt.plot(S, V.detach().numpy())
V

for k in range(10):
    V_train = train_model.forward()

    # Train model:
    optimizer.zero_grad()  # Clear gradients
    # loss = criterion(V_train, V_zeroflux) # Calculate loss
    difference=torch.min(V_train[1:-1]-V_train[0:-2],0)
    loss = my_loss(difference)
    loss.backward()  # Backward pass
    optimizer.step()  # Optimize weights
    print(k, loss, train_model.omegas5, train_model.omegas6)

S,tt=train_model.return_S_tt()
plt.plot(S, V_train.detach().numpy())