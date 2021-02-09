import numpy as np
import torch
import random
from initial_condition_generator import init_PME

class PME():
    def __init__(self, type, space_steps, time_steps=None, params=None, w5_minus="no"):
        """
        Atributes needed to be initialized to make WENO network functional
        space_steps, time_steps, initial_condition, boundary_condition, x, time, h, n
        """

        self.params = params
        if params is None:
            self.init_params()
        self.space_steps = space_steps
        self.time_steps = time_steps
        n, self.t, self.h, self.x, self.time = self.__compute_n_t_h_x_time(type)
        if time_steps is None:
            self.time_steps = n
        self.initial_condition = self.__compute_initial_condition(type)
        self.boundary_condition = self.__compute_boundary_condition()
        self.w5_minus = w5_minus

    def init_params(self):
        params = dict()
        params["T"] = 0.5
        params["e"] = 10 ** (-13)
        params["L"] = 6
        params["power"] = random.uniform(2,5)
        params["d"] = 1
        self.params = params

    def get_params(self):
        return self.params

    def __compute_n_t_h_x_time(self, type):
        if type == "Barenblatt":
            T = self.params["T"]
            L= self.params["L"]
            m = self.space_steps
            h = 2 * L / m
            n = np.ceil(9*(T-1)/(h**2)) #9 pre m=2,3,4,5; 17 pre m=8
            n = int(n)
            t = (T-1) / n
            x = np.linspace(-L, L, m + 1)
            time = np.linspace(1, T, n + 1)
        if type == "boxes":
            T = self.params["T"]
            L = self.params["L"]
            m = self.space_steps
            h = 2 * L / m
            n = np.ceil(15 * (T) / (h ** 2))  # 15 pre rovnaku vysku a m=2,3,4,5,6; 20 pre rovnaku vysku a m=7,8; 180 pre roznu vysku
            n = int(n)
            t = (T) / n
            x = np.linspace(-L, L, m + 1)
            time = np.linspace(0, T, n + 1)
        return n, t, h, x, time

    def __compute_initial_condition(self,type):
        x = self.x
        if type == "Barenblatt":
            mm = self.params["power"]
            d = self.params["d"]
            m = self.space_steps
            alpha = d / ((mm-1)*d+2)
            kk = (alpha*(mm-1))/(2*mm*d)
            #kk = 1/(mm+1)
            u_init = np.zeros(m+1)
            for k in range(0, m + 1):
                u_init[k] = (np.maximum(1 - kk*np.abs(x[k])**2, 0)) ** (1 / (mm - 1))
                #u_init[k] = (np.maximum(1-(kk*(mm-1))/(2*mm)*np.abs(x[k])**2,0))**(1/(mm-1))
        if type == "boxes":
            u_init = init_PME(x)
        u_init = torch.Tensor(u_init)
        return u_init

    def __compute_boundary_condition(self):
        n = self.time_steps

        u_bc_l = torch.zeros((3, n+1))
        u_bc_r = torch.zeros((3, n + 1))

        u1_bc_l = torch.zeros((3, n+1))
        u1_bc_r = torch.zeros((3, n+1))

        u2_bc_l = torch.zeros((3, n+1))
        u2_bc_r = torch.zeros((3, n+1))

        return u_bc_l, u_bc_r, u1_bc_l, u1_bc_r, u2_bc_l, u2_bc_r

    def der_2(self):
        term_2 = 1
        return term_2

    def der_1(self):
        term_1 = 0
        return term_1

    def der_0(self):
        term_0 = 0
        return term_0

    def der_const(self):
        term_const=0
        return term_const

    def funct_diffusion(self,u):
        power = self.params["power"]
        u_diff = torch.abs(u) ** power
        return u_diff

    def funct_convection(self, u):
        return torch.abs(u)

    def funct_derivative(self,u):
        u_der =u**0
        return u_der


    def funct(self, u):
        power = self.params["power"]
        u = u**power
        return u

    # def exact(self, t, type):
    #     #T = self.params["T"]
    #     mm = self.params["power"]
    #     d = self.params["d"]
    #     alpha = d / ((mm - 1) * d + 2)
    #     kk = (alpha * (mm - 1)) / (2 * mm * d)
    #     n,_, _,_,_ = self.__compute_n_t_h_x_time(type)
    #     x, time = self.x, self.time
    #     #kk = 1/(mm+1)
    #     #u_ex = (1/T**kk) * (np.maximum(1-((kk*(mm-1))/(2*mm))*((np.abs(x)**2)/T**(2*kk)),0))**(1/(mm - 1))
    #     u_ex = (t**(-alpha))*(np.maximum(1-kk*((np.abs(x))**2)*t**(-2*alpha/d), 0)) ** (1/(mm-1))
    #     return u_ex

    def err(self, u_last):
        u_ex = self.exact()
        u_last = u_last.detach().numpy()
        xerr = np.absolute(u_ex - u_last)
        xmaxerr = np.max(xerr)
        return xmaxerr

    def whole_exact(self, type):
        mm = self.params["power"]
        d = self.params["d"]
        alpha = d / ((mm - 1) * d + 2)
        kk = (alpha * (mm - 1)) / (2 * mm * d)
        n, _, _, _, _ = self.__compute_n_t_h_x_time(type)
        x, time = self.x, self.time
        m = self.space_steps
        # kk = 1/(mm+1)
        # u_ex = (1/T**kk) * (np.maximum(1-((kk*(mm-1))/(2*mm))*((np.abs(x)**2)/T**(2*kk)),0))**(1/(mm - 1))
        u_whole_ex = np.zeros((m+1,n+1))
        for k in range(0,n+1):
            u_whole_ex[:,k] = (time[k] ** (-alpha)) * (np.maximum(1 - kk * ((np.abs(x)) ** 2) * time[k] ** (-2 * alpha / d), 0)) ** (1 / (mm - 1))
        return u_whole_ex

    def transformation(self, u):
        u = u
        t = self.time
        x = self.x
        return u, x, t



