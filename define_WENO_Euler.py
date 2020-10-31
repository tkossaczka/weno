import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import torch
from torch import nn
from define_WENO_Network import WENONetwork

class WENONetwork_Euler(WENONetwork):
    def get_inner_nn_weno5(self):
        net = nn.Sequential(
            nn.Conv1d(3, 20, kernel_size=5, stride=1, padding=2),
            nn.ELU(),
            nn.Conv1d(20, 40, kernel_size=5, stride=1, padding=2),
            nn.ELU(),
            nn.Conv1d(40, 80, kernel_size=1, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(80, 40, kernel_size=1, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(40, 20, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Conv1d(20, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid())
        return net

    def prepare_dif(self, dif):
        return dif.T[None, :, :]

    def flux_func(self, problem, q):
        gamma = problem.params['gamma']
        rho = q[:,0]
        u = q[:,1]/rho
        E = q[:,2]
        p = (gamma-1)*(E-0.5*rho*u**2)
        flux = torch.stack([rho*u,rho*u**2 + p,u*(E+p)]).T
        return flux

    def init_Euler(self,problem, vectorized, just_one_time_step):
        m = problem.space_steps
        n, t, h = problem.time_steps, problem.t, problem.h
        init_cond = problem.initial_condition
        a0 = problem.a0
        u0 = problem.u0
        lambda0 = torch.max(torch.abs(u0 + a0))

        if vectorized:
            q0 = init_cond
            q0_0 = q0[:,0]
            q0_1 = q0[:,1]
            q0_2 = q0[:,2]
        else:
            q0_0 = torch.zeros((m + 1, n + 1))
            q0_1 = torch.zeros((m + 1, n + 1))
            q0_2 = torch.zeros((m + 1, n + 1))
            q0_0[:, 0] = init_cond[:, 0]  # rho
            q0_1[:, 0] = init_cond[:, 1]  # rho * u
            q0_2[:, 0] = init_cond[:, 2]  # E

        if just_one_time_step is True:
            nn = 1
        else:
            nn = n

        lamb = lambda0
        q_0 = q0_0
        q_1 = q0_1
        q_2 = q0_2

        return q_0, q_1, q_2, lamb, nn

    def run_weno(self, problem, mweno, mapped, q_0, q_1, q_2, lamb, trainable, vectorized, k):
        gamma = problem.params['gamma']
        n, t, h = problem.time_steps, problem.t, problem.h
        e = problem.params['e']
        if vectorized:
            q = torch.stack([q_0, q_1, q_2]).T
            q0 = q
            rho = q[:,0]
            u = q[:,1]/rho
            E = q[:,2]
            p = (gamma - 1)*(E-0.5*rho*u**2)

            u1 = torch.zeros(q_0.shape[0],3)
            flux = self.flux_func(problem, q)
            RHSc_p = self.WENO5(0.5 * (flux + lamb * q), e, w5_minus=False, mweno=mweno, mapped=mapped, trainable=trainable)
            RHSc_n = self.WENO5(0.5 * (flux - lamb * q), e, w5_minus=True, mweno=mweno, mapped=mapped, trainable=trainable)
            RHSc = RHSc_p + RHSc_n
            u1[3:-3, :] = q[3:-3, :] + t * (-(1 / h) * RHSc)
            u1[0:3, :] = q0[0:3, :]
            u1[-3:, :] = q0[-3:, :]

            u2 = torch.zeros(q_0.shape[0],3)
            flux = self.flux_func(problem, u1)
            RHS1c_p = self.WENO5(0.5 * (flux + lamb * u1), e, w5_minus=False, mweno=mweno, mapped=mapped,trainable=trainable)
            RHS1c_n = self.WENO5(0.5 * (flux - lamb * u1), e, w5_minus=True, mweno=mweno, mapped=mapped,trainable=trainable)
            RHS1c = RHS1c_p + RHS1c_n
            u2[3:-3, :] = 0.75 * q[3:-3, :] + 0.25 * u1[3:-3, :] + 0.25 * t * ( - (1/ h) * RHS1c)
            u2[0:3, :] = q0[0:3, :]
            u2[-3:, :] = q0[-3:, :]

            q_ret = torch.zeros(q_0.shape[0], 3)
            flux = self.flux_func(problem, u2)
            RHS2c_p = self.WENO5(0.5 * (flux + lamb * u2), e, w5_minus=False, mweno=mweno, mapped=mapped,trainable=trainable)
            RHS2c_n = self.WENO5(0.5 * (flux - lamb * u2), e, w5_minus=True, mweno=mweno, mapped=mapped,trainable=trainable)
            RHS2c = RHS2c_p + RHS2c_n
            q_ret[3:-3, :] = (1 / 3) * q[3:-3, :] + (2 / 3) * u2[3:-3, :] + (2 / 3)* t * (- (1 / h) * RHS2c)
            q_ret[0:3, :] = q0[0:3, :]
            q_ret[-3:, :] = q0[-3:, :]

            rho = q_ret[:,0]
            u = q_ret[:,1]/rho
            E = q_ret[:,2]
            p = (gamma-1)*(E-0.5*rho*u**2)
            a = (gamma*p/rho)**(1/2)
            lamb_ret = torch.max(torch.abs(u)+a)

            q_0_ret = q_ret[:,0]
            q_1_ret = q_ret[:,1]
            q_2_ret = q_ret[:,2]
        else:
            q0_0 = q_0[:,k]
            q0_1 = q_1[:,k]
            q0_2 = q_2[:,k]
            q = torch.stack([q0_0,q0_1,q0_2]).T
            rho = q0_0
            u = q0_1 / rho
            E = q0_2
            p = (gamma - 1) * (E - 0.5 * rho * u ** 2)

            u1 = torch.zeros(q_0.shape[0], 3)
            flux = self.flux_func(problem, q)
            RHSc_p = self.WENO5(0.5 * (flux + lamb * q), e, w5_minus=False, mweno=mweno, mapped=mapped,
                                trainable=trainable)
            RHSc_n = self.WENO5(0.5 * (flux - lamb * q), e, w5_minus=True, mweno=mweno, mapped=mapped,
                                trainable=trainable)
            RHSc = RHSc_p + RHSc_n
            u1[3:-3, :] = q[3:-3, :] + t * (-(1 / h) * RHSc)
            u1[0:3, :] = q[0:3, :]
            u1[-3:, :] = q[-3:, :]

            u2 = torch.zeros(q_0.shape[0], 3)
            flux = self.flux_func(problem, u1)
            RHS1c_p = self.WENO5(0.5 * (flux + lamb * u1), e, w5_minus=False, mweno=mweno, mapped=mapped,
                                 trainable=trainable)
            RHS1c_n = self.WENO5(0.5 * (flux - lamb * u1), e, w5_minus=True, mweno=mweno, mapped=mapped,
                                 trainable=trainable)
            RHS1c = RHS1c_p + RHS1c_n
            u2[3:-3, :] = 0.75 * q[3:-3, :] + 0.25 * u1[3:-3, :] + 0.25 * t * (- (1 / h) * RHS1c)
            u2[0:3, :] = q[0:3, :]
            u2[-3:, :] = q[-3:, :]

            flux = self.flux_func(problem, u2)
            RHS2c_p = self.WENO5(0.5 * (flux + lamb * u2), e, w5_minus=False, mweno=mweno, mapped=mapped,
                                 trainable=trainable)
            RHS2c_n = self.WENO5(0.5 * (flux - lamb * u2), e, w5_minus=True, mweno=mweno, mapped=mapped,
                                 trainable=trainable)
            RHS2c = RHS2c_p + RHS2c_n
            q[3:-3, :] = (1 / 3) * q[3:-3, :] + (2 / 3) * u2[3:-3, :] + (2 / 3) * t * (- (1 / h) * RHS2c)
            q[0:3, :] = q[0:3, :]
            q[-3:, :] = q[-3:, :]

            rho = q[:, 0]
            u = q[:, 1] / rho
            E = q[:, 2]
            p = (gamma - 1) * (E - 0.5 * rho * u ** 2)
            a = (gamma * p / rho) ** (1 / 2)
            lamb = torch.max(torch.abs(u) + a)

            q_0[:,k+1] = q[:,0]
            q_1[:,k+1] = q[:,1]
            q_2[:,k+1] = q[:,2]
            rho = q_0
            u = q_1 / rho
            E = q_2
            p = (gamma - 1) * (E - 0.5 * rho * u ** 2)

        return q_0_ret, q_1_ret, q_2_ret, lamb_ret

    def forward(self, problem, q_0, q_1, q_2, lamb, k):
        q_0, q_1, q_2, lamb = self.run_weno(problem, mweno=True, mapped=False, q_0=q_0, q_1=q_1, q_2=q_2, lamb=lamb, trainable=True, vectorized=True, k=k)
        return q_0, q_1, q_2, lamb