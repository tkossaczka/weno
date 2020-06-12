import numpy as np
import torch
from scipy.stats import norm

class transport_equation():
    def __init__(self, space_steps, time_steps=None, params=None, w5_minus=False):
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
        self.initial_condition = self.__compute_initial_condition()
        self.boundary_condition = self.__compute_boundary_condition()
        self.w5_minus = w5_minus

    def init_params(self):
        params = dict()
        params["T"] = 1
        params["e"] = 10 ** (-13)
        params["L"] = 1
        self.params = params

    def get_params(self):
        return self.params

    def __compute_n_t_h_x_time(self):
        T = self.params["T"]
        L= self.params["L"]
        m = self.space_steps
        h = 2 * L / m
        n = np.ceil(T / (0.4 * h**(5/3)))
        n = int(n)
        t = T / n
        x = np.linspace(-L, L, m + 1)
        time = np.linspace(0, T, n + 1)
        return n, t, h, x, time

    def __compute_initial_condition(self):
        m = self.space_steps
        x = self.x
        u_init = torch.zeros(m+1)
        for k in range(0, m + 1):
            u_init[k] = np.sin(np.pi*x[k])
        return u_init

    def __compute_boundary_condition(self):
        time = self.time
        time = torch.Tensor(time)
        n = self.time_steps
        m = self.space_steps
        t = self.t
        x = self.x
        time = time[0:n+1]

        u_bc_l = torch.zeros((3, n+1))
        u_bc_r = torch.zeros((3, n + 1))
        for k in range(0,3):
            u_bc_l[k,:] = np.sin(np.pi*(x[k]-time))
        for k in range(m-2,m+1):
            u_bc_r[k-(m-2),:] = np.sin(np.pi*(x[k]-time))

        u1_bc_l = torch.zeros((3, n+1))
        u1_bc_r = torch.zeros((3, n+1))
        for k in range(0,3):
            u1_bc_l[k,:] = np.sin(np.pi*(x[k]-time)) - t* np.pi*(np.cos(np.pi*(x[k]-time)))
        for k in range(m-2,m+1):
            u1_bc_r[k-(m-2),:] = np.sin(np.pi*(x[k]-time)) - t* np.pi*(np.cos(np.pi*(x[k]-time)))

        u2_bc_l = torch.zeros((3, n+1))
        u2_bc_r = torch.zeros((3, n+1))
        for k in range(0,3):
            u2_bc_l[k, :] = np.sin(np.pi*(x[k]-time)) - 0.5* t* np.pi*(np.cos(np.pi*(x[k]-time)))  - 0.25* np.pi**2 * t**2 *(np.sin(np.pi*(x[k]-time)))
        for k in range(m-2,m+1):
            u2_bc_r[k-(m-2), :] = np.sin(np.pi*(x[k]-time)) - 0.5* t* np.pi*(np.cos(np.pi*(x[k]-time))) - 0.25* np.pi**2 * t**2 *(np.sin(np.pi*(x[k]-time)))

        return u_bc_l, u_bc_r, u1_bc_l, u1_bc_r, u2_bc_l, u2_bc_r

    def der_2(self):
        term_2 = 0
        return term_2

    def der_1(self):
        term_1 = -1
        return term_1

    def der_0(self):
        term_0 = 0
        return term_0

    def der_const(self):
        term_const=0
        return term_const

    def funct(self, u):
        return u

    def exact(self):
        m = self.space_steps
        n,_, _,_,_ = self.__compute_n_t_h_x_time()
        x, time = self.x, self.time
        uex = np.zeros((m + 1, n + 1))
        for k in range(0, n + 1):
            for j in range(0, m + 1):
                uex[j, k] = np.sin(np.pi*(x[j]-time[k]))
        u_ex = uex[:, n]
        return u_ex

    def err(self, u_last):
        u_ex = self.exact()
        u_last = u_last.detach().numpy()
        xerr = np.absolute(u_ex - u_last)
        xmaxerr = np.max(xerr)
        return xmaxerr

    def transformation(self, u):
        u = u
        t = self.time
        x = self.x
        return u, x, t