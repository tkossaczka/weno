import numpy as np
import torch
import random
from initial_condition_generator import init_PME
from define_problem_PME import PME
import pandas as pd

class PME_boxes(PME):
    def __init__(self, sample_id, example, space_steps, time_steps=None, params=None, w5_minus="no"):
        if sample_id == None:
            self.params = params
            self.sample_id = sample_id
        else:
            self.sample_id = random.randint(0,143)
            self.df = pd.read_csv("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/parameters.txt")
            self.u_ex = np.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/u_exact64_{}.npy".format(self.sample_id))
            self.u_ex = torch.Tensor(self.u_ex)
            power = float(self.df[self.df.sample_id == self.sample_id]["power"])
            self.params = {'T': 0.5, 'e': 1e-13, 'L': 6, 'power': power, 'd': 1}
        # self.params = params
        self.example = example
        if params is None:
            self.init_params()
        self.space_steps = space_steps
        self.time_steps = time_steps
        n, self.t, self.h, self.x, self.time = self.__compute_n_t_h_x_time()
        if time_steps is None:
            self.time_steps = n
        self.initial_condition = self.__compute_initial_condition()
        self.boundary_condition = self.compute_boundary_condition()
        self.w5_minus = w5_minus

    def init_params(self):
        params = dict()
        params["T"] = 0.5 #2 #1.4
        params["e"] = 10 ** (-13)
        params["L"] = 6
        params["power"] = random.uniform(2,6) #random.uniform(2,5) #random.uniform(2,8)
        params["d"] = 1
        self.params = params

    def __compute_n_t_h_x_time(self):
        example = self.example
        if example == "boxes":
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

    def __compute_initial_condition(self):
        x = self.x
        example = self.example
        if example == "boxes":
            u_init, self.height = init_PME(x)
        u_init = torch.Tensor(u_init)
        return u_init

    def exact(self, k):
        u_ex = self.u_ex[:,k]
        return u_ex