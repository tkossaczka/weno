import torch
import numpy as np

def initial_condition(E,xl,xr,m):
    Smin = np.exp(xl) * E;
    Smax = np.exp(xr) * E;

    G = np.log(Smin / E);
    L = np.log(Smax / E);

    x = np.linspace(G, L, m + 1)

    u = torch.zeros((x.shape[0]))[:, None]

    for k in range(0, m + 1):
        if x[k] > 0:
            u[k,0] = 1 / E;
        else:
            u[k,0] = 0;

    return u
