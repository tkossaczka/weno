from define_problem_Digital import Digital_option
import numpy as np
import torch
from scipy.stats import norm

class Digital_option_GS(Digital_option):
    def __init__(self, space_steps, time_steps=None, params=None, w5_minus='both'):
        """
        Atributes needed to be initialized to make WENO network functional
        space_steps, time_steps, initial_condition, boundary_condition, x, time, h, n
        """

        self.params = params
        if params is None:
            self.init_params()
        self.space_steps = space_steps
        self.time_steps = time_steps
        n, self.t, self.h, self.x, self.time,_,_ = self.__compute_n_t_h_x_time()
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
        params["psi"] = 20
        self.params = params

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
        h = 1/m
        kappa = np.log(1)
        psi = self.params["psi"]
        c1 = np.arcsinh(psi * (G - kappa))
        c2 = np.arcsinh(psi * (L - kappa))
        ymin = (np.arcsinh(psi * (G - kappa)) - c1) / (c2 - c1)
        ymax = (np.arcsinh(psi * (L - kappa)) - c1) / (c2 - c1)
        y = np.linspace(ymin, ymax, m+1)
        xx = ((c2 - c1) / psi) * np.cosh(c2 * y + c1 * (1 - y))
        xxx = (((c2 - c1)**2) / psi) * np.sinh(c2 * y + c1 * (1 - y))
        x = (np.sinh(c2 * y + c1 * (1 - y))) / psi + kappa
        n = np.ceil(np.max((theta * sigma**2) / (0.83 * (h**2) * xx**2)))
        n = int(n)
        t = theta / n
        time = np.linspace(0, theta, n + 1)
        return n, t, h, x, time, xx, xxx

    def der_2(self):
        sigma = self.params["sigma"]
        _,_,_,_,_,xx,_ = self.__compute_n_t_h_x_time()
        term_2 = (0.5*sigma**2)/xx**2
        term_2 = torch.Tensor(term_2)
        return term_2

    def der_1(self):
        sigma = self.params["sigma"]
        rate = self.params["rate"]
        _,_,_,_,_,xx,xxx = self.__compute_n_t_h_x_time()
        term_1 = (rate - 0.5*sigma**2)/xx -  (0.5*sigma**2)*(xxx/xx**3)
        term_1 = torch.Tensor(term_1)
        return term_1

    def exact(self):
        m = self.space_steps
        n, _, _, _, _,_,_ = self.__compute_n_t_h_x_time()
        sigma = self.params["sigma"]
        rate = self.params["rate"]
        T = self.params["T"]
        E = self.params["E"]
        x, time = self.x, self.time
        tt = T - time
        S = E * np.exp(x)
        Digital = np.zeros((m + 1, n + 1))
        for k in range(0, n + 1):
            for j in range(0, m + 1):
                Digital[j, k] = np.exp(-rate * (T - tt[k])) * norm.cdf(
                    (np.log(S[j] / E) + (rate - (sigma ** 2) / 2) * (T - tt[k])) / (sigma * np.sqrt(T - tt[k])))
        u_Digital = Digital[:, n] / E
        return u_Digital