import numpy as np
import torch
import random
from initial_condition_generator import init_PME
import pandas as pd

class PME():
    def __init__(self, sample_id, example, space_steps, time_steps=None, params=None):
        if example == "boxes":
            if sample_id != None:
                self.sample_id = random.randint(0,373) #sample_id + 1
                self.df = pd.read_csv("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/parameters.txt")
                self.u_ex = np.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/u_exact64_{}.npy".format(self.sample_id))
                self.u_ex = torch.Tensor(self.u_ex)
                power = float(self.df[self.df.sample_id == self.sample_id]["power"])
                self.params = {'T': 0.5, 'e': 1e-13, 'L': 6, 'power': power, 'd': 1}
            else:
                self.params = params
                self.sample_id = sample_id
        elif example == "Barenblatt" or example == "boxes_2d" or example == "Barenblatt_2d":
            self.params = params
        self.example = example
        if params is None:
            self.init_params()
        self.space_steps = space_steps
        self.time_steps = time_steps
        self.h, self.x = self.__compute_h_x()
        self.initial_condition = self.__compute_initial_condition()
        n, self.t, self.time = self.__compute_n_t_time()
        if time_steps is None:
            self.time_steps = n
        self.boundary_condition = self.compute_boundary_condition()
        self.w5_minus = "no"

    def init_params(self):
        params = dict()
        example = self.example
        if example == "Barenblatt":
            params["T"] = 1.4 #2 #1.4
            params["power"] = random.uniform(2, 8)  # random.uniform(2,5) #random.uniform(2,8)
            params["d"] = 1
            params["L"] = 6
        elif example == "boxes":
            params["T"] = 0.5  # 2 #1.4
            params["power"] = random.uniform(2, 8)  # random.uniform(2,5) #random.uniform(2,8)
            params["d"] = 1
            params["L"] = 6
        elif example == "boxes_2d":
            params["T"] = 0.5  # 2 #1.4
            params["power"] = random.uniform(2, 5)
            params["d"] = 2
            params["L"] = 10
        elif example == "Barenblatt_2d":
            params["T"] = 2
            params["power"] = random.uniform(2, 8)
            params["d"] = 2
            params["L"] = 10
        params["e"] = 10 ** (-13)
        self.params = params

    def get_params(self):
        return self.params

    def __compute_h_x(self):
        example = self.example
        T = self.params["T"]
        L = self.params["L"]
        m = self.space_steps
        h = 2 * L / m
        x = np.linspace(-L, L, m + 1)
        return h, x

    def __compute_n_t_time(self):
        example = self.example
        uu = self.initial_condition
        T = self.params["T"]
        h = self.h
        power = self.params["power"]
        if example == "Barenblatt":
            dif = (power*uu**(power-1))
            CFL = torch.max(dif)/0.4
            n = np.ceil(CFL*(T-1)/(h**2)) #10 pre m=2,3,4,5; 17 pre m=8
            n = int(n)
            t = (T-1) / n
            time = np.linspace(1, T, n + 1)
        elif example == "Barenblatt_2d":
            n = np.ceil(25 * (T-1) / (h ** 2)) #15 pre m=2,3,4,5; 25 pre m = 6,7,8
            n = int(n)
            t = (T-1) / n
            time = np.linspace(1, T, n + 1)
        elif example == "boxes":
            dif = (power*uu**(power-1))
            CFL = torch.max(dif)/0.4
            n = np.ceil(CFL * (T) / (h ** 2))  # 15 pre rovnaku vysku a m=2,3,4,5,6; 20 pre rovnaku vysku a m=7,8; 180 pre roznu vysku
            n = int(n)
            t = (T) / n
            time = np.linspace(0, T, n + 1)
        elif example == "boxes_2d":
            n = np.ceil(8 * (T) / (h ** 2))
            n = int(n)
            t = (T) / n
            time = np.linspace(0, T, n + 1)
        return n, t, time

    def __compute_initial_condition(self):
        x = self.x
        example = self.example
        if example == "Barenblatt":
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
        elif example == "Barenblatt_2d":
            y = self.x
            mm = self.params["power"]
            d = self.params["d"]
            m = self.space_steps
            alpha = d / ((mm - 1) * d + 2)
            kk = (alpha * (mm - 1)) / (2 * mm * d)
            u_init = np.zeros((m + 1, m+1))
            for k in range(0, m + 1):
                for j in range(m+1):
                    u_init[j,k] = (np.maximum(1 - kk * (np.abs(x[k])**2 + np.abs(y[j])**2) , 0)) ** (1 / (mm - 1))
        elif example == "boxes":
            u_init, self.height = init_PME(x)
        elif example == "boxes_2d":
            m = self.space_steps
            u_init = np.zeros((m + 1, m+1))
            y = self.x
            for k in range(m+1):
                for j in range(m+1):
                    if (x[k]-2)**2 + (y[j]+2)**2 < 6:
                        u_init[j,k] = np.exp(-1/(6-(x[k]-2)**2-(y[j]+2)**2))
                    elif (x[k]+2)**2 + (y[j]-2)**2 < 6:
                        u_init[j,k] = np.exp(-1/(6-(x[k]+2)**2-(y[j]-2)**2))
                    else:
                        u_init[j,k] = 0
        u_init = torch.Tensor(u_init)
        return u_init

    def compute_boundary_condition(self):
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

    def exact(self, t):
        example = self.example
        if example == "Barenblatt":
            mm = self.params['power']
            d = self.params["d"]
            alpha = d / ((mm - 1) * d + 2)
            kk = (alpha * (mm - 1)) / (2 * mm * d)
            x = self.x
            u_ex = (t**(-alpha))*(np.maximum(1-kk*((np.abs(x))**2)*t**(-2*alpha/d), 0)) ** (1/(mm-1))
        elif example == "Barenblatt_2d":
            m = self.space_steps
            mm = self.params['power']
            d = self.params["d"]
            alpha = d / ((mm - 1) * d + 2)
            kk = (alpha * (mm - 1)) / (2 * mm * d)
            x = self.x
            y = self.x
            u_ex = np.zeros((m + 1, m+1))
            for k in range(0, m + 1):
                for j in range(m+1):
                    u_ex[j,k] = (t ** (-alpha)) * (np.maximum(1 - kk * (np.abs(x[k])**2 + np.abs(y[j])**2)  * t ** (-2 * alpha / d), 0)) ** (1 / (mm - 1))
        elif example == "boxes":
            u_ex = self.u_ex[:, t]
        return u_ex

    def err(self, u_last):
        u_ex = self.exact()
        u_last = u_last.detach().numpy()
        xerr = np.absolute(u_ex - u_last)
        xmaxerr = np.max(xerr)
        return xmaxerr

    def whole_exact(self):
        mm = self.params["power"]
        d = self.params["d"]
        alpha = d / ((mm - 1) * d + 2)
        kk = (alpha * (mm - 1)) / (2 * mm * d)
        n, _, _, _, _ = self.__compute_n_t_h_x_time()
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



