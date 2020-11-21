import numpy as np
import torch
from scipy.optimize import root_scalar
from initial_condition_generator import init_Euler

class Euler_system():
    def __init__(self, space_steps, init_cond, time_steps=None, params=None, time_disc=None, init_mid=False, init_general=False, w5_minus='Lax-Friedrichs'):
        """
        Atributes needed to be initialized to make WENO network functional
        space_steps, time_steps, initial_condition, boundary_condition, x, time, h, n
        """
        self.params = params
        if params is None:
            self.init_params()
        self.space_steps = space_steps
        self.time_steps = time_steps
        n, self.t, self.h, self.x, self.time = self.__compute_n_t_h_x_time(time_disc)
        if time_steps is None:
            self.time_steps = n
        self.initial_condition, self.u0, self.a0 = self.__compute_initial_condition(init_cond, init_mid, init_general)
        #self.boundary_condition = self.__compute_boundary_condition()
        self.w5_minus = w5_minus

    def init_params(self):
        params = dict()
        params["T"] = 0.1 #0.1 #2 #0.1 #0.1 #5 #1
        params["e"] = 10 ** (-6)
        params["L"] = 0 #-5 #0 #0 #0 # -1
        params["R"] = 1 #5 #1 #1 #2 # 1
        params["gamma"] = 1.4
        self.params = params

    def get_params(self):
        return self.params

    def __compute_n_t_h_x_time(self, time_disc):
        T = self.params["T"]
        L= self.params["L"]
        R = self.params["R"]
        m = self.space_steps
        h = (np.abs(L) + np.abs(R)) / m
        if time_disc == "adaptive":
            n = None
            t = None
            time = None
        elif time_disc == None:
            n = np.ceil(T / (0.5 * h))  # 10 sod # 1 lax
            #n = 416 #Sod for 2048 space steps
            #n = 2048 #shock_entropy for 2048 space steps
            n = int(n)
            t = T / n
            time = np.linspace(0, T, n + 1)
        x = np.linspace(L, R, m + 1)
        return n, t, h, x, time

    def __compute_initial_condition(self, init_cond, init_mid, init_general):
        m = self.space_steps
        x = self.x
        gamma = self.params["gamma"]
        T = self.params["T"]
        r0 = torch.zeros(m + 1)
        u0 = torch.zeros(m + 1)
        p0 = torch.zeros(m + 1)
        if init_cond == "Sod":
            if init_mid == True:
                self.p = np.array([1.0, 0.1])
                self.u = np.array([0.0, 0.0])
                self.rho = np.array([1.0, 0.125])
                self.p = torch.Tensor(self.p)
                self.u = torch.Tensor(self.u)
                self.rho = torch.Tensor(self.rho)
                x_ex = np.linspace(0, 1, m + 1)
                p0, r0, u0, _, _ = self.exact(x_ex, T / 2)
            elif init_general == True:
                r0, u0, p0, self.rho, self.u, self.p = init_Euler(x)
            else:
                self.p = np.array([1.0, 0.1])
                self.u = np.array([0.0, 0.0])
                self.rho = np.array([1.0, 0.125])
                x_mid = 0.5
                r0[x <= x_mid] = self.rho[0]
                r0[x > x_mid] = self.rho[1]
                u0[x <= x_mid] = self.u[0]
                u0[x > x_mid] = self.u[1]
                p0[x <= x_mid] = self.p[0]
                p0[x > x_mid] = self.p[1]
        elif init_cond == "Lax":
            self.p = np.array([3.528, 0.571])
            self.u = np.array([0.698, 0.0])
            self.rho = np.array([0.445, 0.5])
            x_mid = 0.5
            r0[x <= x_mid] = self.rho[0]
            r0[x > x_mid] = self.rho[1]
            u0[x <= x_mid] = self.u[0]
            u0[x > x_mid] = self.u[1]
            p0[x <= x_mid] = self.p[0]
            p0[x > x_mid] = self.p[1]
        elif init_cond == "shock_entropy":
            self.p = np.array([31/3, 1.0])
            self.u = np.array([(4*np.sqrt(35))/9, 0.0])
            self.rho = 0
            x_mid = -4
            r0[x <= x_mid] = 27/7
            r0[x > x_mid] = 1+0.2*torch.sin(5*torch.Tensor(x[x > x_mid]))
            u0[x <= x_mid] = self.u[0]
            u0[x > x_mid] = self.u[1]
            p0[x <= x_mid] = self.p[0]
            p0[x > x_mid] = self.p[1]
        elif init_cond == "blast_waves":
            self.p = np.array([1000.0, 0.01, 100.0])
            self.u = np.array([0.0, 0.0, 0.0])
            self.rho = np.array([1.0, 1.0, 1.0])
            x_mid_1 = 0.1
            x_mid_2 = 0.9
            r0[x <= x_mid_1] = self.rho[0]
            r0[(x > x_mid_1) & (x <= x_mid_2)]  = self.rho[1]
            r0[x > x_mid_2] = self.rho[2]
            u0[x <= x_mid_1] = self.u[0]
            u0[(x > x_mid_1) & (x <= x_mid_2)] = self.u[1]
            u0[x > x_mid_2] = self.u[2]
            p0[x <= x_mid_1] = self.p[0]
            p0[(x > x_mid_1) & (x <= x_mid_2)] = self.p[1]
            p0[x > x_mid_2] = self.p[2]
        self.p = torch.Tensor(self.p)
        self.u = torch.Tensor(self.u)
        self.rho = torch.Tensor(self.rho)
        a0 = torch.sqrt(gamma*p0/r0)
        E0 = p0/(gamma-1) +0.5*r0*u0**2
        q0 = torch.stack([r0, r0*u0, E0]).T
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

    def exact(self, x, T):
        rho1 = self.rho[0]
        rho4 = self.rho[1]
        u1 = self.u[0]
        u4 = self.u[1]
        p1 = self.p[0]
        p4 = self.p[1]
        gamma = self.params["gamma"]
        alph = (gamma+1)/(gamma-1)
        PRL = p4/p1
        c1 = torch.sqrt((gamma*p1)/rho1)
        c4 = torch.sqrt((gamma*p4)/rho4)
        def func(P,gamma,u1,u4,c1,c4):
            return (1/P) * (1 + (gamma-1)/2 * (u1-u4)/c1 - ((gamma-1)*c4*(P-1))/(c1*np.sqrt(2*gamma*(gamma-1+(gamma+1)*P))) ) ** ((2*gamma)/(gamma-1)) - PRL
        sol = root_scalar(func, args=(gamma,u1,u4,c1,c4), method='bisect', bracket=[1,10], x0=3)
        P = sol.root
        p3 = P*p4
        rho3 = rho4*((1+alph*P)/(alph+P))
        rho2 = rho1*(P*p4/p1)**(1/gamma)
        u2 = u1 - u4 + 2/(gamma-1) * c1 * (1 - (P*p4/p1)**((gamma-1)/(2*gamma)))
        c2 = np.sqrt(gamma*p3/rho2)
        c3 = np.sqrt(gamma*p3/rho3)

        x0 = 0.5
        pos1 = x0 + (u1 - c1)*T
        pos2 = x0 + (u2 + u4 - c2)*T
        pos3 = x0 + (u2 + u4)*T
        pos4 = x0 + T*(c4 * np.sqrt( (gamma-1)/(2*gamma) + (gamma+1)*P/(2*gamma) ) + u4)

        p = torch.zeros(x.shape[0])
        u = torch.zeros(x.shape[0])
        rho = torch.zeros(x.shape[0])
        mach = torch.zeros(x.shape[0])
        c = torch.zeros(x.shape[0])

        for k in range(x.shape[0]):
            if x[k] <= pos1:
                p[k] = p1
                u[k] = u1
                rho[k] = rho1
                c[k] = c1
                mach[k] = u1/c1
            elif x[k] <= pos2:
                p[k] = p1 * (1 + (pos1 - x[k])/(c1*alph*T)) ** (2*gamma/(gamma-1))
                rho[k] = rho1 * (1 + (pos1 - x[k])/(c1*alph*T)) ** (2/(gamma-1))
                u[k] = u1 + (2/(gamma+1)) * ((x[k]-pos1)/T)
                c[k] = torch.sqrt(gamma*p[k]/rho[k])
                mach[k] = u[k]/c[k]
            elif x[k] <= pos3:
                p[k] = p3
                rho[k] = rho2
                u[k] = u2+u4
                c[k] = c2
                mach[k] = (u2+u4)/c2
            elif x[k] <= pos4:
                p[k] = p3
                rho[k] = rho3
                u[k] = u2 + u4
                c[k] = c3
                mach[k] = (u2 + u4) / c3
            else:
                p[k] = p4
                rho[k] = rho4
                u[k] = u4
                c[k] = c4
                mach[k] = u4 / c4

        return p, rho, u, c, mach

    def err(self, u_last):
        u_ex = self.exact()
        u_last = u_last.detach().numpy()
        #xerr = np.max(np.absolute(u_ex - u_last))
        xerr = np.mean((u_ex - u_last)**2)
        return xerr

    def transformation(self, u):
        u = u
        t = self.time
        x = self.x
        return u, x, t