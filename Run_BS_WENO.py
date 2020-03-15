import numpy as np
import matplotlib.pyplot as plt
from BS_WENO import BS_WENO
from compute_omegas5 import compute_omegas5
from compute_omegas6 import compute_omegas6
import torch
from torch import nn, optim # Sets of preset layers and optimizers
import torch.nn.functional as F # Sets of functions such as ReLU
from torchvision import datasets, transforms # Popular datasets, architectures and common


class WENONetwork(nn.Module):
    def __init__(self):
        super().__init__()
        params = self.get_params()
        u=self.initial_condition()
        self.pre_omegas6=compute_omegas6(u,params['m'],1,params['e'])
        self.pre_omegas5=compute_omegas5(u,params['m'],1,params['e'])
        self.omegas5 = nn.Parameter(self.pre_omegas5, requires_grad=True)
        self.omegas6 = nn.Parameter(self.pre_omegas6, requires_grad=True)

    def get_params(self):
        params = dict()
        params["sigma"] = 0.3;
        params["rate"] = 0.1;
        params["E"] = 50;
        params["T"] = 1;
        params["e"] = 10 ** (-13);
        params["xl"] = -6
        params["xr"] = 1.5
        params["m"] = 80;
        return params

    def initial_condition(self):
        params = self.get_params()
        E=params['E']
        m=params['m']
        xl=params['xl']
        xr=params['xr']
        Smin = np.exp(xl) * E;
        Smax = np.exp(xr) * E;
        G = np.log(Smin / E);
        L = np.log(Smax / E);
        x = np.linspace(G, L, m + 1)
        u = torch.zeros((x.shape[0]))[:, None]
        for k in range(0, m + 1):
            if x[k] > 0:
                u[k, 0] = 1 / E;
            else:
                u[k, 0] = 0;
        return u

    def forward(self):
        params = self.get_params()
        V, S, tt = BS_WENO(params["sigma"], params["rate"], params["E"], params["T"], params["e"], params["xl"],
                           params["xr"], params["m"], None)
        return V

    def return_S_tt(self):
        params = self.get_params()
        V, S, tt = BS_WENO(params["sigma"], params["rate"], params["E"], params["T"], params["e"], params["xl"],
                           params["xr"], params["m"], None)
        return S, tt


train_model=WENONetwork()

V=train_model.forward()

def my_loss(input):
    loss = torch.max((input)**2)
    return loss

#criterion = nn.MSELoss()
optimizer = optim.SGD(train_model.parameters(), lr=0.001)

S,tt=train_model.return_S_tt()
plt.plot(S, V.detach().numpy())
plt.show()
V

# for k in range(10):
#     V_train = train_model.forward()
#
#     # Train model:
#     optimizer.zero_grad()  # Clear gradients
#     # loss = criterion(V_train, V_zeroflux) # Calculate loss
#     difference=torch.min(V_train[1:-1]-V_train[0:-2],0)
#     loss = my_loss(difference)
#     loss.backward()  # Backward pass
#     optimizer.step()  # Optimize weights
#     print(k, loss, train_model.omegas5, train_model.omegas6)
#
# S,tt=train_model.return_S_tt()
# plt.plot(S, V_train.detach().numpy())