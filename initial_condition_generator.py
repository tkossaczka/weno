import numpy as np
import torch
import random

def init_Euler(x):
    m = x.shape[0]
    r0 = torch.zeros(m)
    u0 = torch.zeros(m)
    p0 = torch.zeros(m)
    sw = random.randint(0,1)
    p = np.array([1.0, 0.1])
    u = np.array([0.0, 0.0])
    rho = np.array([1.0, 0.125])
    if sw==0:
        p[0] = random.uniform(0.5,10) * p[0] + random.uniform(-0.05, 0.05)
        p[1] = (1/random.uniform(5,10)) * p[0] #+ random.uniform(-0.05, 0.05)
        rho[0] = p[0]
        rho[1] = p[1] + random.uniform(-0.05, 0.05)
        u[0] = random.uniform(0.0,1.0)
    if sw==1:
        rho[0] = random.uniform(1.0, 3.0) * rho[0]
        rho[1] = (1 / 10) * rho[0]  + random.uniform(-0.05, 0.05)
        u[0] = random.uniform(0.0,1.0)
    # if sw==0: # change rho
    #     rho[0] = random.uniform(1.0,2.0)
    #     rho[1] = rho[0]/10 + random.uniform(-0.05, 0.05)
    # elif sw==1: # change u
    #     u[0] = random.uniform(0.0,1.0)
    # elif sw==2: #change p and rho
    #     rho[0] = random.uniform(0.5, 10.0)
    #     p[0]=rho[0]
    #     rho[1] = rho[0]*(1/random.uniform(5,10))
    #     p[1] = rho[1]
    x_mid = 0.5
    r0[x <= x_mid] = rho[0]
    r0[x > x_mid] = rho[1]
    u0[x <= x_mid] = u[0]
    u0[x > x_mid] = u[1]
    p0[x <= x_mid] = p[0]
    p0[x > x_mid] = p[1]

    return r0, u0, p0, rho, u, p