import numpy as np
import torch
from scipy.stats import norm

class Call_option():
    def __init__(self, space_steps, time_steps=None, params=None, w5_minus='Lax-Friedrichs'):
        """
        Atributes needed to be initialized to make WENO network functional
        space_steps, time_steps, initial_condition, boundary_condition, x, time, h, n
        """

        self.params = params
        if params is None:
            self.init_params()
        self.space_steps = space_steps
        self.time_steps = time_steps
        n, self.t, self.h, self.x, self.time = self.__compute_n_t_h_x_time()
        if time_steps is None:
            self.time_steps = n
        self.initial_condition = self.compute_initial_condition()
        self.boundary_condition = self.compute_boundary_condition()
        self.w5_minus = w5_minus

    def init_params(self):
        params = dict()
        params["sigma"] = 0.31 + max(0.1 * np.random.randn(), -0.3)
        params["rate"] = 0.21 + max(0.1 * np.random.randn(), -0.2)
        params["E"] = 50
        params["T"] = 1
        params["e"] = 10 ** (-13)
        params["xl"] = -6
        params["xr"] = 1.5
        self.params = params

    def get_params(self):
        return self.params

    def __compute_n_t_h_x_time(self):
        sigma = self.params["sigma"]
        E = self.params["E"]
        T = self.params["T"]
        xl = self.params["xl"]
        xr = self.params["xr"]
        m = self.space_steps
        Smin = np.exp(xl) * E
        Smax = np.exp(xr) * E
        G = np.log(Smin / E)
        L = np.log(Smax / E)
        theta = T
        h = (-G + L) / m
        n = np.ceil((theta * sigma ** 2) / (0.8 * (h ** 2)))
        n = int(n)
        t = theta / n
        x = np.linspace(G, L, m + 1)
        time = np.linspace(0, theta, n + 1)
        return n, t, h, x, time

    def compute_initial_condition(self):
        x = self.x
        m = self.space_steps
        u_init = np.maximum(np.exp(x)-1,np.zeros(m+1))
        u_init = torch.Tensor(u_init)
        return u_init


    def compute_boundary_condition(self):
        rate = self.params["rate"]
        time = self.time
        x = self.x
        time = torch.Tensor(time)
        x = torch.Tensor(x)
        n = self.time_steps
        m = self.space_steps
        t = self.t
        time = time[0:n+1]

        u_bc_l = torch.zeros((3, n+1))
        u_bc_r = torch.zeros((3, n+1))
        for k in range(m-2,m+1):
            u_bc_r[k-(m-2),:] = torch.exp(x[k]) - torch.exp(-rate * time)

        u1_bc_l = torch.zeros((3, n+1))
        u1_bc_r = torch.zeros((3, n+1))
        for k in range(m-2,m+1):
            u1_bc_r[k-(m-2),:] = torch.exp(x[k]) - torch.exp(-rate * time) + t * rate * torch.exp(-rate * time)

        u2_bc_l = torch.zeros((3, n+1))
        u2_bc_r = torch.zeros((3, n+1))
        for k in range(m-2,m+1):
            u2_bc_r[k-(m-2),:] = torch.exp(x[k]) -  torch.exp(-rate * time) + 0.5 * t * rate * torch.exp(-rate * time) - 0.25 * (t ** 2) * (
                rate ** 2) * torch.exp(-rate * time)

        return u_bc_l, u_bc_r, u1_bc_l, u1_bc_r, u2_bc_l, u2_bc_r

    def der_2(self):
        sigma = self.params["sigma"]
        term_2 = 0.5*sigma**2
        return term_2

    def der_1(self):
        sigma = self.params["sigma"]
        rate = self.params["rate"]
        term_1 = rate - 0.5*sigma**2
        return term_1

    def der_0(self):
        rate = self.params["rate"]
        term_0 = -rate
        return term_0

    def der_const(self):
        term_const=0
        return term_const

    def funct_convection(self, u):
        u = -u
        return u

    def funct_diffusion(self, u):
        return u

    def funct_derivative(self,u):
        u_der = u**0
        return u_der

    def exact(self):
        m = self.space_steps
        n,_, _,_,_ = self.__compute_n_t_h_x_time()
        sigma = self.params["sigma"]
        rate = self.params["rate"]
        T = self.params["T"]
        E = self.params["E"]
        x, time = self.x, self.time
        S = E * np.exp(x)
        tt = T - time
        Call = np.zeros((m + 1, n + 1))
        for k in range(0, n + 1):
            for j in range(0, m + 1):
                Call[j, k] = S[j] * norm.cdf(
                    (np.log(S[j] / E) + (rate + (sigma ** 2) / 2) * (T - tt[k])) / (sigma * np.sqrt(T - tt[k]))) - E*np.exp(-rate * (T - tt[k])) * norm.cdf(
                    (np.log(S[j] / E) + (rate - (sigma ** 2) / 2) * (T - tt[k])) / (sigma * np.sqrt(T - tt[k])))
        u_Call = Call[:, n] / E
        return u_Call

    def err(self, u_last):
        u_Call = self.exact()
        u_last = u_last.detach().numpy()
        xerr = np.absolute([u_Call - u_last])
        xmaxerr = np.max([xerr])
        return xmaxerr

    def transformation(self, u):
        m = self.space_steps
        n = self.time_steps
        T = self.params["T"]
        E = self.params["E"]
        tt = T - self.time
        S = E * np.exp(self.x)
        V = torch.zeros((m + 1, n+1))
        for k in range(0, m + 1):
            V[k, :] = E * u[k, :]
        return V, S, tt



