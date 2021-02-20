import numpy as np
import torch
import random
import pandas as pd
import matplotlib.pyplot as plt

class Buckley_Leverett():
    def __init__(self, sample_id, example, space_steps, time_steps=None, params=None):
        if sample_id == None:
            self.params = params
            self.sample_id = sample_id
        else:
            self.sample_id = random.randint(0,5)
            self.df = pd.read_csv("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/parameters.txt")
            self.u_ex = np.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/u_exact64_{}.npy".format(self.sample_id))
            self.u_ex = torch.Tensor(self.u_ex)
            C = float(self.df[self.df.sample_id == self.sample_id]["C"])
            G = float(self.df[self.df.sample_id == self.sample_id]["G"])
            self.params = {'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': C, 'G': G}
        self.example = example
        if self.params is None:
            self.init_params()
        self.space_steps = space_steps
        self.time_steps = time_steps
        n, self.t, self.h, self.x, self.time = self.__compute_n_t_h_x_time()
        if time_steps is None:
            self.time_steps = n
        self.initial_condition = self.__compute_initial_condition()
        self.boundary_condition = self.__compute_boundary_condition()
        self.w5_minus = 'Lax-Friedrichs'


    def init_params(self):
        params = dict()
        params["T"] = 0.1 #0.5 #0.2
        params["e"] = 10 ** (-13)
        params["L"] = 0 #-2 # 0
        params["R"] = 1 #2 #1
        params["C"] = random.uniform(0.1, 0.95)
        params["G"] = random.uniform(0,5)
        self.params = params

    def __compute_n_t_h_x_time(self):
        T = self.params["T"]
        L= self.params["L"]
        R= self.params["R"]
        m = self.space_steps
        h = (np.abs(L) + np.abs(R)) / m
        n = np.ceil(0.1*T / (h ** 2))
        # n=1000
        n = int(n)
        t = T / n
        x = np.linspace(L, R, m + 1)
        time = np.linspace(0, T, n + 1)
        return n, t, h, x, time

    def __compute_initial_condition(self):
        m = self.space_steps
        x = self.x
        example = self.example
        u_init = np.zeros(m+1)
        # for k in range(0, m + 1):
        #     if x[k] >= 1-1/np.sqrt(2) and x[k] <= 1:
        #         u_init[k] = 1
        #     else:
        #         u_init[k] = 0
        # for k in range(0, m + 1):
        #     if x[k] > -1/np.sqrt(2)-0.4 and x[k] < -1/np.sqrt(2)+0.4:
        #         u_init[k] = 1
        #     elif x[k] > 1/np.sqrt(2)-0.4 and x[k] < 1/np.sqrt(2)+0.4:
        #         u_init[k] = -1
        #     else:
        #         u_init[k] = 0
        if example == "gravity":
            for k in range(0, m + 1):
                if x[k] >= 0 and x[k] < 1-1/np.sqrt(2):
                    u_init[k] = 0
                else:
                    u_init[k] = 1
        u_init = torch.Tensor(u_init)
        return u_init

    def __compute_boundary_condition(self):
        n = self.time_steps
        example = self.example

        u_bc_l = torch.zeros((3, n+1))
        # u_bc_r = torch.zeros((3, n + 1))
        if example == "gravity":
            u_bc_r = torch.ones((3, n + 1))

        u1_bc_l = torch.zeros((3, n+1))
        # u1_bc_r = torch.zeros((3, n + 1))
        if example == "gravity":
            u1_bc_r = torch.ones((3, n+1))

        u2_bc_l = torch.zeros((3, n+1))
        # u2_bc_r = torch.zeros((3, n + 1))
        if example == "gravity":
            u2_bc_r = torch.ones((3, n+1))

        return u_bc_l, u_bc_r, u1_bc_l, u1_bc_r, u2_bc_l, u2_bc_r

    def der_2(self):
        example = self. example
        if example == "gravity":
            term_2 = 0.01
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

    def funct_diffusion(self,u):
        m = self.space_steps
        example = self.example
        ep = 0.00000001
        u_diff = torch.zeros(m+1)
        # u = u.detach().numpy()
        if example == "gravity":
            u_diff =  (2*u**2 - (4/3)*u**3)  #*((u >= 0-ep) & (u<=1+ep))
        # u_diff = (u) * ((torch.abs(u) > 0.25))
        #u_diff = (u) * ((np.abs(u) > 0.25))
        # for k in range(0, m + 1):
        #     if u[k] >= -0.25  and u[k] <= 0.25 :
        #         u_diff[k] = 0
        #     else:
        #         u_diff[k] = u[k]
        # for k in range(0, m + 1):
        #     if u[k] < -0.25  or u[k] > 0.25 :
        #         u_diff[k] = u[k]
        #     else:
        #         u_diff[k] = 0
        return u_diff

    def funct_convection(self, u):
        C = self.params["C"]
        G = self.params["G"]
        example = self.example
        # u_conv = (u**2)/(u**2+C*(1-u)**2)
        #u_conv = (4*u**2)/(4*u**2+(1-u)**2)
        # u_conv = (u**2)*(1-G*(1-u)**2)/(u**2+(1-u)**2)
        if example == "gravity":
            u_conv = (u ** 2) * (1 - G * (1 - u) ** 2) / (u ** 2 + C*(1 - u) ** 2)
        #u_conv = (u ** 2)  / (u ** 2 + (1 - u) ** 2)
        # u_conv = u**2
        return u_conv

    def funct_derivative(self,u):
        C = self.params["C"]
        G = self.params["G"]
        example = self.example
        # u_der = 2*C*u*(1-u)/(u**2+C*(1-u)**2)**2
        # u_der =  8*u*(1-u)/(5*u**2-2*u+1)**2
        # u_der = (-20*u**5+50*u**4-60*u**3+38*u**2-8*u) / (2 * u ** 2 - 2 * u + 1) ** 2
        if example == "gravity":
            u_der = - ( (2*C+2)*G*u**5 + (-8*C-2)*G*u**4 + 12*C*G*u**3 + (2*C-8*C*G)*u**2 + (2*C*G-2*C)*u )/( (u**2+C*(1-u)**2)**2 )
        #u_der = 2 * u * (1 - u) / (2 * u ** 2 - 2 * u + 1) ** 2
        # u_der = 2*u
        return u_der



    # def exact(self):
    #
    #     return
    #
    # def err(self, u_last):
    #     u_ex = self.exact()
    #     u_last = u_last.detach().numpy()
    #     xerr = np.absolute(u_ex - u_last)
    #     xmaxerr = np.max(xerr)
    #     return xmaxerr

    def transformation(self, u):
        u = u
        t = self.time
        x = self.x
        return u, x, t



