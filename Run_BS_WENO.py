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

        # self.omegas5 = nn.Parameter(self.pre_omegas5, requires_grad=True)
        # self.omegas6 = nn.Parameter(self.pre_omegas6, requires_grad=True)
        # torch.randn(2, requires_grad=True)
        self.lag = 6
        self.pairs= int((self.lag+1) * (self.lag + 2) /2)
        self.weights5 = nn.Parameter(torch.zeros([self.lag,6]))
        self.weights6 = nn.Parameter(torch.zeros([self.lag+4,6]))
        # self.weights = nn.Parameter(torch.randn([self.pairs, 12]))

    def get_params(self):
        params = dict()
        params["sigma"] = 0.3 + 0.1 * np.random.randn();
        params["rate"] = 0.2 + max(0.1 * np.random.randn(), -0.2);
        params["E"] = 50;
        params["T"] = 1;
        params["e"] = 10 ** (-13);
        params["xl"] = -6
        params["xr"] = 1.5
        params["m"] = 160;
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
                           params["xr"], params["m"], self.weights5, self.weights6)
        print(params["sigma"], params["rate"])
        return V

    def return_S_tt(self):
        params = self.get_params()
        V, S, tt = BS_WENO(params["sigma"], params["rate"], params["E"], params["T"], params["e"], params["xl"],
                           params["xr"], params["m"], self.weights5, self.weights6)
        return S, tt


train_model=WENONetwork()

V=train_model.forward()

def monotonicity_loss(x):
    return torch.sum(torch.max(x[:-1]-x[1:], torch.Tensor([0.0])))

optimizer = optim.SGD(train_model.parameters(), lr=0.0001)
# optimizer = optim.Adam(train_model.parameters())

S,tt=train_model.return_S_tt()
plt.plot(S, V.detach().numpy())
plt.show()
V

for k in range(1000):
    # Forward path
    V_train = train_model.forward()
    # Train model:
    optimizer.zero_grad()  # Clear gradients
    loss = monotonicity_loss(V_train) # Calculate loss
    loss.backward()  # Backward pass
    optimizer.step()  # Optimize weights
    w5 = train_model.weights5.detach().numpy()
    w6 = train_model.weights6.detach().numpy()
    print(k, loss)

S,tt = train_model.return_S_tt()
plt.plot(S, V_train.detach().numpy())
w5=train_model.weights5.detach().numpy()
w6=train_model.weights6.detach().numpy()