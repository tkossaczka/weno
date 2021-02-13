import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import torch
from torch import nn
from define_WENO_Network import WENONetwork

class WENONetwork_2(WENONetwork):
    # def get_inner_nn_weno5(self):
    #     net = nn.Sequential(
    #         nn.Conv1d(2, 5, kernel_size=5, stride=1, padding=2),
    #         nn.ELU(),
    #         nn.Conv1d(5, 5, kernel_size=5, stride=1, padding=2),
    #         nn.ELU(),
    #         # nn.Conv1d(40, 80, kernel_size=1, stride=1, padding=0),
    #         # nn.ELU(),
    #         # nn.Conv1d(80, 40, kernel_size=1, stride=1, padding=0),
    #         # nn.ELU(),
    #         # nn.Conv1d(40, 20, kernel_size=3, stride=1, padding=1),
    #         # nn.ELU(),
    #         nn.Conv1d(5, 1, kernel_size=1, stride=1, padding=0),
    #         nn.Sigmoid())
    #     return net

    def get_inner_nn_weno6(self):
        net = nn.Sequential(
            nn.Conv1d(2, 5, kernel_size=5, stride=1, padding=2),
            nn.ELU(),
            nn.Conv1d(5, 5, kernel_size=5, stride=1, padding=2),
            nn.ELU(),
            # nn.Conv1d(20, 40, kernel_size=1, stride=1, padding=0),
            # nn.ELU(),
            # nn.Conv1d(80, 40, kernel_size=1, stride=1, padding=0),
            # nn.ELU(),
            # nn.Conv1d(40, 20, kernel_size=3, stride=1, padding=1),
            # nn.ELU(),
            nn.Conv1d(5, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid())
        return net

    def init_run_weno(self, problem, vectorized, just_one_time_step):
        m = problem.space_steps
        n, t, h = problem.time_steps, problem.t, problem.h
        # x, time = problem.x, problem.time
        # w5_minus = problem.w5_minus

        if vectorized:
            u = problem.initial_condition
        else:
            u = torch.zeros((m + 1, n + 1))
            u[:, 0] = problem.initial_condition

        if just_one_time_step is True:
            nn = 1
        else:
            nn = n
        return u, nn

    def run_weno(self, problem, u, mweno, mapped, vectorized, trainable, k):
        e = problem.params['e']
        n, t, h = problem.time_steps, problem.t, problem.h
        term_2 = problem.der_2()
        term_1 = problem.der_1()
        term_0 = problem.der_0()
        term_const = problem.der_const()
        u_bc_l, u_bc_r, u1_bc_l, u1_bc_r, u2_bc_l, u2_bc_r = problem.boundary_condition
        w5_minus = problem.w5_minus

        if vectorized:
            uu = u
            #ll=1
        else:
            uu = u[:,k]
            #ll=k

        uu_conv = problem.funct_convection(uu)
        uu_diff = problem.funct_diffusion(uu)
        u1 = torch.zeros(uu.shape[0])
        RHSd = self.WENO6(uu_diff, e, mweno=mweno, mapped=mapped, trainable=trainable)
        if w5_minus=='both':
            RHSc_p = self.WENO5(uu_conv, e, w5_minus=False, mweno=mweno, mapped=mapped, trainable=trainable)
            RHSc_n= self.WENO5(uu_conv, e, w5_minus=True, mweno=mweno, mapped=mapped, trainable=trainable)
            u1[3:-3] = uu[3:-3] + t * ((term_2[3:-3] / h ** 2) * RHSd + ((term_1>=0)[3:-3])*(term_1[3:-3] / h) * RHSc_n
                                              + ((term_1<0)[3:-3])*(term_1[3:-3] / h) * RHSc_p + term_0 * uu[3:-3])
        elif w5_minus=='Lax-Friedrichs':
            max_der = torch.max(torch.abs(problem.funct_derivative(uu)))
            RHSc_p = self.WENO5(0.5*(uu_conv+max_der*uu), e, w5_minus=False, mweno=mweno, mapped=mapped, trainable=trainable)
            RHSc_n = self.WENO5(0.5*(uu_conv-max_der*uu), e, w5_minus=True, mweno=mweno, mapped=mapped, trainable=trainable)
            RHSc = RHSc_p + RHSc_n
            u1[3:-3] = uu[3:-3] + t * ((term_2 / h ** 2) * RHSd - (term_1 / h) * RHSc + term_0 * uu[3:-3])
        elif w5_minus == 'no':
            u1[3:-3] = uu[3:-3] + t * ((term_2 / h ** 2) * RHSd + term_0 * uu[3:-3])
        else:
            RHSc = self.WENO5(uu_conv, e, w5_minus=w5_minus, mweno=mweno, mapped=mapped, trainable=trainable)
            u1[3:-3] = uu[3:-3] + t * ((term_2 / h ** 2) * RHSd - (term_1 / h) * RHSc + term_0 * uu[3:-3])

        u1[0:3] = u1_bc_l[:,k]
        u1[-3:] = u1_bc_r[:,k]

        uu1_conv = problem.funct_convection(u1)
        uu1_diff = problem.funct_diffusion(u1)
        u2 = torch.zeros(uu.shape[0])
        RHS1d = self.WENO6(uu1_diff, e, mweno=mweno, mapped=mapped, trainable=trainable)
        if w5_minus=='both':
            RHS1c_p = self.WENO5(uu1_conv, e, w5_minus=False, mweno=mweno, mapped=mapped, trainable=trainable)
            RHS1c_n = self.WENO5(uu1_conv, e, w5_minus=True, mweno=mweno, mapped=mapped, trainable=trainable)
            u2[3:-3] = 0.75*uu[3:-3]+0.25*u1[3:-3]+0.25*t*((term_2[3:-3]/h ** 2)*RHS1d+ (term_1>=0)[3:-3]*(term_1[3:-3] / h)*RHS1c_n
                                                               + (term_1<0)[3:-3]*(term_1[3:-3] / h) * RHS1c_p +term_0*u1[3:-3])
        elif w5_minus=='Lax-Friedrichs':
            max_der = torch.max(torch.abs(problem.funct_derivative(u1)))
            RHS1c_p = self.WENO5(0.5*(uu1_conv+max_der*u1), e, w5_minus=False, mweno=mweno, mapped=mapped, trainable=trainable)
            RHS1c_n = self.WENO5(0.5*(uu1_conv-max_der*u1), e, w5_minus=True, mweno=mweno, mapped=mapped, trainable=trainable)
            RHS1c = RHS1c_p + RHS1c_n
            u2[3:-3] = 0.75*uu[3:-3]+0.25*u1[3:-3]+0.25*t*((term_2/h ** 2)*RHS1d-(term_1/h)*RHS1c+term_0*u1[3:-3])
        elif w5_minus == 'no':
            u2[3:-3] = 0.75*uu[3:-3]+0.25*u1[3:-3]+0.25*t*((term_2/h ** 2)*RHS1d+term_0*u1[3:-3])
        else:
            RHS1c = self.WENO5(uu1_conv, e, w5_minus=w5_minus, mweno=mweno, mapped=mapped, trainable=trainable)
            u2[3:-3] = 0.75*uu[3:-3]+0.25*u1[3:-3]+0.25*t*((term_2/h ** 2)*RHS1d-(term_1/h)*RHS1c+term_0*u1[3:-3])

        u2[0:3] = u2_bc_l[:,k]
        u2[-3:] = u2_bc_r[:,k]

        uu2_conv = problem.funct_convection(u2)
        uu2_diff = problem.funct_diffusion(u2)
        u_ret = torch.zeros(uu.shape[0])
        RHS2d = self.WENO6(uu2_diff, e, mweno=mweno, mapped=mapped, trainable=trainable)
        if w5_minus=='both':
            RHS2c_p = self.WENO5(uu2_conv, e, w5_minus=False, mweno=mweno, mapped=mapped, trainable=trainable)
            RHS2c_n = self.WENO5(uu2_conv, e, w5_minus=True, mweno=mweno, mapped=mapped, trainable=trainable)
            if vectorized:
                u_ret[3:-3] = (1 / 3) * uu[3:-3] + (2 / 3) * u2[3:-3] + (2 / 3) * t * (
                        (term_2[3:-3] / h ** 2) * RHS2d + (term_1>=0)[3:-3]*(term_1[3:-3] / h) * RHS2c_n
                        + (term_1<0)[3:-3]*(term_1[3:-3] / h) * RHS2c_p + term_0 * u2[3:-3])
                u_ret[0:3] = u_bc_l[:, k+1]
                u_ret[-3:] = u_bc_r[:, k+1]
            else:
                u[3:-3, k+1] = (1 / 3) * uu[3:-3] + (2 / 3) * u2[3:-3] + (2 / 3) * t * (
                        (term_2[3:-3] / h ** 2) * RHS2d + (term_1>=0)[3:-3]*(term_1[3:-3] / h) * RHS2c_n
                        + (term_1<0)[3:-3]*(term_1[3:-3] / h) * RHS2c_p + term_0 * u2[3:-3])
                u[0:3, k+1] = u_bc_l[:, k+1]
                u[-3:, k+1] = u_bc_r[:, k+1]
        elif w5_minus == 'Lax-Friedrichs':
            max_der = torch.max(torch.abs(problem.funct_derivative(u2)))
            RHS2c_p = self.WENO5(0.5 * (uu2_conv + max_der * u2), e, w5_minus=False, mweno=mweno, mapped=mapped,
                                 trainable=trainable)
            RHS2c_n = self.WENO5(0.5 * (uu2_conv - max_der * u2), e, w5_minus=True, mweno=mweno, mapped=mapped,
                                 trainable=trainable)
            RHS2c = RHS2c_p + RHS2c_n
            if vectorized:
                u_ret[3:-3] = (1 / 3) * uu[3:-3] + (2 / 3) * u2[3:-3] + (2 / 3) * t * (
                        (term_2 / h ** 2) * RHS2d - (term_1 / h) * RHS2c + term_0 * u2[3:-3])
                u_ret[0:3] = u_bc_l[:, k+1]
                u_ret[-3:] = u_bc_r[:, k+1]
            else:
                u[3:-3, k+1] = (1 / 3) * uu[3:-3] + (2 / 3) * u2[3:-3] + (2 / 3) * t * (
                        (term_2 / h ** 2) * RHS2d - (term_1 / h) * RHS2c + term_0 * u2[3:-3])
                u[0:3, k+1] = u_bc_l[:, k+1]
                u[-3:, k+1] = u_bc_r[:, k+1]
                u_ret = u[:, k + 1]
        elif w5_minus == 'no':
            if vectorized:
                u_ret[3:-3] = (1 / 3) * uu[3:-3] + (2 / 3) * u2[3:-3] + (2 / 3) * t * (
                        (term_2 / h ** 2) * RHS2d + term_0 * u2[3:-3])
                u_ret[0:3] = u_bc_l[:, k+1]
                u_ret[-3:] = u_bc_r[:, k+1]
            else:
                u[3:-3, k+1] = (1 / 3) * uu[3:-3] + (2 / 3) * u2[3:-3] + (2 / 3) * t * (
                        (term_2 / h ** 2) * RHS2d + term_0 * u2[3:-3])
                u[0:3, k+1] = u_bc_l[:, k+1]
                u[-3:, k+1] = u_bc_r[:, k+1]
                u_ret = u[:, k + 1]
        else:
            RHS2c = self.WENO5(uu2_conv, e, w5_minus=w5_minus, mweno=mweno, mapped=mapped, trainable=trainable)
            if vectorized:
                u_ret[3:-3] = (1 / 3) * uu[3:-3] + (2 / 3) * u2[3:-3] + (2 / 3) * t * (
                        (term_2 / h ** 2) * RHS2d - (term_1 / h) * RHS2c + term_0 * u2[3:-3])
                u_ret[0:3] = u_bc_l[:, k+1]
                u_ret[-3:] = u_bc_r[:, k+1]
            else:
                u[3:-3, k+1] = (1 / 3) * uu[3:-3] + (2 / 3) * u2[3:-3] + (2 / 3) * t * (
                        (term_2 / h ** 2) * RHS2d - (term_1 / h) * RHS2c + term_0 * u2[3:-3])
                u[0:3, k+1] = u_bc_l[:, k+1]
                u[-3:, k+1] = u_bc_r[:, k+1]
                u_ret = u[:, k + 1]
        return u_ret

    def forward(self, problem, u_ret, k):
        u = self.run_weno(problem,u_ret,mweno=True,mapped=False,vectorized=True,trainable=True,k=k)
        # V,_,_ = problem.transformation(u)
        return u

    def compare_wenos(self, problem):
        u_init, nn = self.init_run_weno(problem, vectorized=True, just_one_time_step=True)
        u_trained = u_init
        for k in range(nn):
            u_trained = self.run_weno(problem, u_trained, mweno=True, mapped=False, trainable=True, vectorized=True, k=k)
        V_trained, S, tt = problem.transformation(u_trained)
        u_classic = u_init
        for k in range(nn):
            u_classic = self.run_weno(problem, u_classic, mweno=True, mapped=False, trainable=False, vectorized=True, k=k)
        V_classic, S, tt = problem.transformation(u_classic)
        plt.plot(S, V_classic.detach().numpy(), S, V_trained.detach().numpy())
        plt.show()

    def order_compute(self, iterations, initial_space_steps, initial_time_steps, params, problem_class, trainable):
        problem = problem_class(space_steps=initial_space_steps, time_steps=initial_time_steps, params=params)
        vecerr = np.zeros((iterations))[:, None]
        order = np.zeros((iterations - 1))[:, None]
        u_init, nn = self.init_run_weno(problem, vectorized=True, just_one_time_step=False)
        u = u_init
        for k in range(nn):
            u = self.run_weno(problem,u,mweno=True, mapped=False, trainable=trainable, vectorized=True,k=k)
        u_last = u
        xmaxerr = problem.err(u_last) #, first_step=False)
        vecerr[0] = xmaxerr
        print(problem.space_steps, problem.time_steps)
        for i in range(1, iterations):
            if initial_time_steps is None:
                spec_time_steps = None
            else:
                spec_time_steps = problem.time_steps*4
            problem = problem_class(space_steps=problem.space_steps * 2, time_steps=spec_time_steps, params=params)
            u, nn = self.init_run_weno(problem, vectorized=True, just_one_time_step=False)
            for k in range(nn):
                u = self.run_weno(problem,u,mweno=True, mapped=False, trainable=trainable, vectorized=True, k=k)
            u_last = u
            xmaxerr = problem.err(u_last) #, first_step=False)
            vecerr[i] = xmaxerr
            order[i - 1] = np.log(vecerr[i - 1] / vecerr[i]) / np.log(2)
            print(problem.space_steps, problem.time_steps)
        return vecerr, order

    def compute_exact(self,problem_class, problem, space_steps, time_steps, just_one_time_step, trainable):
        if hasattr(problem_class, 'exact'):
            print('nic netreba')
        else:
            u, nn = self.init_run_weno(problem, vectorized=False, just_one_time_step=just_one_time_step)
            for k in range(nn):
                u[:, k + 1] = self.run_weno(problem, u, mweno=True, mapped=False, vectorized=False, trainable=trainable, k=k)
        u_exact = u
        space_steps_exact = problem.space_steps
        time_steps_exact = problem.time_steps
        divider_space = space_steps_exact / space_steps
        divider_time = time_steps_exact / time_steps
        divider_space = int(divider_space)
        divider_time = int(divider_time)
        u_exact_adjusted = u_exact[0:space_steps_exact+1:divider_space,0:time_steps_exact+1:divider_time]
        return u_exact, u_exact_adjusted
