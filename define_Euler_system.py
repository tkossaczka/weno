import numpy as np
import torch

class Euler_system():
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
        self.initial_condition, self.u0, self.a0 = self.__compute_initial_condition()
        self.boundary_condition = self.__compute_boundary_condition()
        self.w5_minus = w5_minus

    def init_params(self):
        params = dict()
        params["T"] = 0.1 #5 #1
        params["e"] = 10 ** (-13)
        params["L"] = 0 #0 # -1
        params["R"] = 1 #2 # 1
        params["gamma"] = 1.4
        self.params = params

    def get_params(self):
        return self.params

    def __compute_n_t_h_x_time(self):
        T = self.params["T"]
        L= self.params["L"]
        R = self.params["R"]
        m = self.space_steps
        h = (np.abs(L) + np.abs(R)) / m
        n = np.ceil(T / (1.0 * h**(5/3)))  #0.4
        n = int(n)
        t = T / n
        x = np.linspace(L, R, m + 1)
        time = np.linspace(0, T, n + 1)
        return n, t, h, x, time

    def __compute_initial_condition(self):
        m = self.space_steps
        x = self.x
        gamma = self.params["gamma"]
        p = np.array([1.0, 0.1])
        u = np.array([0.0, 0.0])
        rho = np.array([1.0, 0.125])
        x_mid = 0.5
        r0 = np.zeros(m+1)
        u0 = np.zeros(m+1)
        p0 = np.zeros(m+1)
        r0[x<=x_mid] = rho[0]
        r0[x>x_mid] = rho[1]
        u0[x <= x_mid] = u[0]
        u0[x > x_mid] = u[1]
        p0[x <= x_mid] = p[0]
        p0[x > x_mid] = p[1]
        a0 = np.sqrt(gamma*p0/r0)
        E0 = p0/(gamma-1) +0.5*r0*u0**2
        q0 = np.array([r0, r0*u0, E0]).T
        q0 = torch.Tensor(q0)
        a0 = torch.Tensor(a0)
        u0 = torch.Tensor(u0)
        return q0, u0, a0

    def __compute_boundary_condition(self):
        time = self.time
        time = torch.Tensor(time)
        n = self.time_steps
        m = self.space_steps
        t = self.t
        x = self.x
        time = time[0:n+1]

        u_bc_l = torch.zeros(3, 3)
        u_bc_r = torch.zeros(3, 3)
        u_bc_l[:, 0] = 1
        u_bc_l[:, 1] = 0
        u_bc_l[:, 2] = 2.5
        u_bc_r[:, 0] = 0.125
        u_bc_r[:, 1] = 0
        u_bc_r[:, 2] = 0.25

        u1_bc_l = torch.zeros(3, 3)
        u1_bc_r = torch.zeros(3, 3)
        u1_bc_l[:, 0] = 1
        u1_bc_l[:, 1] = 0
        u1_bc_l[:, 2] = 2.5
        u1_bc_r[:, 0] = 0.125
        u1_bc_r[:, 1] = 0
        u1_bc_r[:, 2] = 0.25

        u2_bc_l = torch.zeros(3, 3)
        u2_bc_r = torch.zeros(3, 3)
        u2_bc_l[:, 0] = 1
        u2_bc_l[:, 1] = 0
        u2_bc_l[:, 2] = 2.5
        u2_bc_r[:, 0] = 0.125
        u2_bc_r[:, 1] = 0
        u2_bc_r[:, 2] = 0.25

        return u_bc_l, u_bc_r, u1_bc_l, u1_bc_r, u2_bc_l, u2_bc_r

    def der_2(self):
        term_2 = 0
        return term_2

    def der_1(self):
        term_1 = 1
        return term_1

    def der_0(self):
        term_0 = 0
        return term_0

    def der_const(self):
        term_const=0
        return term_const

    def funct_convection(self, u):
        return u

    def funct_diffusion(self, u):
        return u

    def funct_derivative(self, u):
        u_der = u ** 0
        return u_der

    def exact(self):
        m = self.space_steps
        n,_, _,_,_ = self.__compute_n_t_h_x_time()
        x, time = self.x, self.time
        uex = np.zeros((m + 1, n + 1))
        for k in range(0, n + 1):
            for j in range(0, m + 1):
                # uex[j, k] = np.sin(np.pi*(x[j]-time[k]))
                uex[j, k] = (x[j]-time[k])**3 + np.cos((x[j]-time[k]))
        u_ex = uex[:, n]
        return u_ex

    def err(self, u_last):
        u_ex = self.exact()
        u_last = u_last.detach().numpy()
        xerr = np.max(np.absolute(u_ex - u_last))
        #xerr = np.mean((u_ex - u_last)**2)
        return xerr

    def transformation(self, u):
        u = u
        t = self.time
        x = self.x
        return u, x, t