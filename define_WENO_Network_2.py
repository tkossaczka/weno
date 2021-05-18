import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import torch
import torch.nn as nn
import torch.nn.functional as F
from define_WENO_Network import WENONetwork

class FancyNet(nn.Module):
    # STRUCTURE FOR BUCKLEY-LEVERETT
    def __init__(self):
        self.num_inner_convs = 1
        super(FancyNet, self).__init__()
        self.conv0 = nn.Conv1d(2, 10, kernel_size=5, stride=1, padding=2)
        self.convs = nn.ModuleList([nn.Conv1d(10, 10, kernel_size=5, stride=1, padding=2) for k in range(self.num_inner_convs)])
        # self.conv1 = nn.Conv1d(10, 10, kernel_size=5, stride=1, padding=2)
        self.conv_out = nn.Conv1d(10, 1, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        x = F.elu(self.conv0(x))
        for k in range(self.num_inner_convs):
            x = F.elu(self.convs[k](x)) #+ x
        x = torch.sigmoid(self.conv_out(x))
        return x

# class FancyNet(nn.Module):  # structure for PME
#     def __init__(self):
#         self.num_inner_convs = 1
#         super(FancyNet, self).__init__()
#         self.conv0 = nn.Conv1d(2, 5, kernel_size=5, stride=1, padding=2)
#         self.convs = nn.ModuleList([nn.Conv1d(5, 5, kernel_size=5, stride=1, padding=2) for k in range(self.num_inner_convs)])
#         self.conv_out = nn.Conv1d(5, 1, kernel_size=1, stride=1, padding=0)
#
#     def forward(self, x):
#         x = F.elu(self.conv0(x))
#         for k in range(self.num_inner_convs):
#             x = F.elu(self.convs[k](x)) + x
#         # x = torch.sigmoid(self.conv_out(x))
#         x = (self.conv_out(x))
#         return x

# class MNet(nn.Module):
#     def __init__(self):
#         self.num_inner_lins = 1
#         super(MNet, self).__init__()
#         self.lin0 = nn.Linear(1,32)
#         self.lins = nn.ModuleList([nn.Linear(32, 32) for k in range(self.num_inner_lins)])
#         self.lin_out = nn.Linear(32, 1)
#
#     def forward(self, x):
#         x = F.elu(self.lin0(x))
#         for k in range(self.num_inner_lins):
#             x = F.elu(self.lins[k](x)) + x
#         x = torch.sigmoid(self.lin_out(x))
#         return x

class WENONetwork_2(WENONetwork):
    # def __init__(self, train_with_coeff=True):
    #     super().__init__()
    #     self.train_with_coeff = train_with_coeff
    #     if train_with_coeff == True:
    #         self.m_nn = self.get_m_nn()

    def get_inner_nn_weno6(self):
        return FancyNet()

    def get_inner_nn_weno5(self):
        return FancyNet()

    # def get_m_nn(self):
    #     return MNet()

    # def parse_mnet_input(self, problem): # z problemu extrahujem input do MNetu
    #     power = problem.params['power']
    #     power = torch.as_tensor(power)
    #     return power

    # def WENO6(self, problem, uu, e, mweno, mapped, trainable=True):
    #     uu_left = uu[:-1]
    #     uu_right = uu[1:]
    #
    #     def get_fluxes(uu):
    #         flux0 = (uu[0:-5] - 3 * uu[1:-4] - 9 * uu[2:-3] + 11 * uu[3:-2]) / 12
    #         flux1 = (uu[1:-4] - 15 * uu[2:-3] + 15 * uu[3:-2] - uu[4:-1]) / 12
    #         flux2 = (-11 * uu[2:-3] + 9 * uu[3:-2] + 3 * uu[4:-1] - uu[5:]) / 12
    #         return flux0, flux1, flux2
    #
    #     fluxp0, fluxp1, fluxp2 = get_fluxes(uu_right)
    #     fluxn0, fluxn1, fluxn2 = get_fluxes(uu_left)
    #
    #     def get_betas(uu):
    #         beta0 = 13 / 12 * (uu[0:-5] - 3 * uu[1:-4] + 3 * uu[2:-3] - uu[3:-2]) ** 2 + 1 / 4 * (
    #                 uu[0:-5] - 5 * uu[1:-4] + 7 * uu[2:-3] - 3 * uu[3:-2]) ** 2
    #         beta1 = 13 / 12 * (uu[1:-4] - 3 * uu[2:-3] + 3 * uu[3:-2] - uu[4:-1]) ** 2 + 1 / 4 * (
    #                 uu[1:-4] - uu[2:-3] - uu[3:-2] + uu[4:-1]) ** 2
    #         beta2 = 13 / 12 * (uu[2:-3] - 3 * uu[3:-2] + 3 * uu[4:-1] - uu[5:]) ** 2 + 1 / 4 * (
    #                 -3 * uu[2:-3] + 7 * uu[3:-2] - 5 * uu[4:-1] + uu[5:]) ** 2
    #         return beta0, beta1, beta2
    #
    #     betap0, betap1, betap2 = get_betas(uu_right)
    #     betan0, betan1, betan2 = get_betas(uu_left)
    #
    #     old_betas_p = [betap0, betap1, betap2]
    #     old_betas_n = [betan0, betan1, betan2]
    #
    #     if trainable:
    #         dif = self.get_average_diff(uu)
    #         dif2 = self.get_average_diff2(uu)
    #         power = self.parse_mnet_input(problem)
    #         dif12 = torch.stack([dif, dif2])
    #         dif12 = self.prepare_dif(dif12)
    #         beta_multiplicators = self.inner_nn_weno6(dif12)[0, 0, :] + self.weno6_mult_bias
    #         # if self.train_with_coeff == True:
    #         #     train_coefficient = self.m_nn((power[None]-5.0)/50)
    #         #     beta_multiplicators = torch.abs(beta_multiplicators)**train_coefficient
    #             #print(train_coefficient)
    #
    #         betap_corrected_list = []
    #         betan_corrected_list = []
    #
    #         mult_shifts_p = [-1, 0, 1]
    #         for k, beta in enumerate([betap0, betap1, betap2]):
    #             shift = mult_shifts_p[k]  # k-1 #(3-k)-1
    #             betap_corrected_list.append(beta * beta_multiplicators[3 + shift:-3 + shift])
    #         mult_shifts_n = [-1, 0, 1]  # [-2, -1, 0]
    #         # mult_shifts_n = [-2, -1, 0]
    #         for k, beta in enumerate([betan0, betan1, betan2]):
    #             shift = mult_shifts_n[k]  # k #3-k
    #             betan_corrected_list.append(beta * beta_multiplicators[3 + shift:-3 + shift])
    #
    #         [betap0, betap1, betap2] = betap_corrected_list
    #         [betan0, betan1, betan2] = betan_corrected_list
    #
    #     gamap0 = 1 / 21
    #     gamap1 = 19 / 21
    #     gamap2 = 1 / 21
    #     gaman0 = 4 / 27
    #     gaman1 = 19 / 27
    #     gaman2 = 4 / 27
    #     sigmap = 42 / 15
    #     sigman = 27 / 15
    #
    #     def get_omegas_mweno(betas, gamas, old_betas):
    #         beta_range_square = (old_betas[2] - old_betas[0]) ** 2
    #         return [gama / (e + beta) ** 2 * (beta_range_square + (e + beta) ** 2) for beta, gama in zip(betas, gamas)]
    #
    #     def get_omegas_weno(betas, gamas, old_betas):
    #         return [gama / (e + beta) ** 2 for beta, gama in zip(betas, gamas)]
    #
    #     omegas_func_dict = {0: get_omegas_weno, 1: get_omegas_mweno}
    #     [omegapp_0, omegapp_1, omegapp_2] = omegas_func_dict[int(mweno)]([betap0, betap1, betap2],
    #                                                                      [gamap0, gamap1, gamap2], old_betas_p)
    #     [omeganp_0, omeganp_1, omeganp_2] = omegas_func_dict[int(mweno)]([betap0, betap1, betap2],
    #                                                                      [gaman0, gaman1, gaman2], old_betas_p)
    #     [omegapn_0, omegapn_1, omegapn_2] = omegas_func_dict[int(mweno)]([betan0, betan1, betan2],
    #                                                                      [gamap0, gamap1, gamap2], old_betas_n)
    #     [omegann_0, omegann_1, omegann_2] = omegas_func_dict[int(mweno)]([betan0, betan1, betan2],
    #                                                                      [gaman0, gaman1, gaman2], old_betas_n)
    #
    #     def normalize(tensor_list):
    #         sum_ = sum(tensor_list)  # note, that inbuilt sum applies __add__ iteratively therefore its overloaded-
    #         return [tensor / sum_ for tensor in tensor_list]
    #
    #     [omegapp0, omegapp1, omegapp2] = normalize([omegapp_0, omegapp_1, omegapp_2])
    #     [omeganp0, omeganp1, omeganp2] = normalize([omeganp_0, omeganp_1, omeganp_2])
    #     [omegapn0, omegapn1, omegapn2] = normalize([omegapn_0, omegapn_1, omegapn_2])
    #     [omegann0, omegann1, omegann2] = normalize([omegann_0, omegann_1, omegann_2])
    #
    #     omegaps = [omegapp0, omegapp1, omegapp2, omegapn0, omegapn1, omegapn2]
    #     omegans = [omeganp0, omeganp1, omeganp2, omegann0, omegann1, omegann2]
    #
    #     [omegap0, omegap1, omegap2, omegan0, omegan1, omegan2] = [sigmap * omegap - sigman * omegan
    #                                                               for omegap, omegan in zip(omegaps, omegans)]
    #
    #     if mapped:
    #         d0 = -2 / 15
    #         d1 = 19 / 15
    #         d2 = -2 / 15
    #
    #         def get_alpha(omega, d):
    #             return (omega * (d + d ** 2 - 3 * d * omega + omega ** 2)) / (d ** 2 + omega * (1 - 2 * d))
    #
    #         [alphap0, alphap1, alphap2] = [get_alpha(omega, d) for omega, d in zip([omegap0, omegap1, omegap2],
    #                                                                                [d0, d1, d2])]
    #         [alphan0, alphan1, alphan2] = [get_alpha(omega, d) for omega, d in zip([omegan0, omegan1, omegan2],
    #                                                                                [d0, d1, d2])]
    #         [omegap0, omegap1, omegap2] = normalize([alphap0, alphap1, alphap2])
    #         [omegan0, omegan1, omegan2] = normalize([alphan0, alphan1, alphan2])
    #
    #     fluxp = omegap0 * fluxp0 + omegap1 * fluxp1 + omegap2 * fluxp2
    #     fluxn = omegan0 * fluxn0 + omegan1 * fluxn1 + omegan2 * fluxn2
    #
    #     RHS = (fluxp - fluxn)
    #
    #     return RHS

    def init_run_weno(self, problem, vectorized, just_one_time_step, dim=1):
        m = problem.space_steps
        n, t, h = problem.time_steps, problem.t, problem.h
        if dim == 1:
            if vectorized:
                u = problem.initial_condition
            else:
                u = torch.zeros((m + 1, n + 1))
                u[:, 0] = problem.initial_condition
            if just_one_time_step is True:
                nn = 1
            else:
                nn = n
        elif dim == 2:
            u = problem.initial_condition
            if just_one_time_step is True:
                nn = 1
            else:
                nn = n
        return u, nn

    def run_weno(self, problem, u, mweno, mapped, vectorized, trainable, k):
        e = problem.params['e']
        #power = problem.params['power']
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
        RHSd = self.WENO6( uu_diff, e, mweno=mweno, mapped=mapped, trainable=trainable)
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
        RHS1d = self.WENO6( uu1_diff, e, mweno=mweno, mapped=mapped, trainable=trainable)
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
        RHS2d = self.WENO6( uu2_diff, e, mweno=mweno, mapped=mapped, trainable=trainable)
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

    def run_weno_2d(self, problem, u, mweno, mapped, vectorized, trainable, k):
        e = problem.params['e']
        n, t, h = problem.time_steps, problem.t, problem.h
        m = problem.space_steps
        term_2 = problem.der_2()
        term_1 = problem.der_1()
        term_0 = problem.der_0()
        term_const = problem.der_const()
        # u_bc_l_x, u_bc_r_x, u1_bc_l_x, u1_bc_r_x, u2_bc_l_x, u2_bc_r_x = problem.boundary_condition_2d
        w5_minus = problem.w5_minus
        uu = u

        uu_diff = problem.funct_diffusion(uu)
        u1 = torch.zeros(m+1,m+1)
        RHSd_x = torch.zeros(m+1,m+1)
        RHSd_y = torch.zeros(m+1,m+1)
        RHSc_p_x = torch.zeros(m+1,m+1)
        RHSc_n_x = torch.zeros(m+1,m+1)
        RHSc_p_y = torch.zeros(m+1,m+1)
        RHSc_n_y = torch.zeros(m+1,m+1)

        uu_diff_x = torch.reshape(uu_diff.T, (1,(m+1)*(m+1)))
        uu_diff_y = torch.reshape(uu_diff, (1,(m+1)*(m+1)))
        RHSd_xr = self.WENO6(uu_diff_x.squeeze(0), e, mweno=mweno, mapped=mapped, trainable=trainable)
        RHSd_yr = self.WENO6(uu_diff_y.squeeze(0), e, mweno=mweno, mapped=mapped, trainable=trainable)
        RHSd_y2 = torch.zeros((m+1)**2)
        RHSd_y2[3:-3] = RHSd_yr
        RHSd_x2 = torch.zeros((m+1)**2)
        RHSd_x2[3:-3] = RHSd_xr
        RHSd_y = RHSd_y2.reshape(m+1,m+1)
        RHSd_x = (RHSd_x2.reshape(m+1,m+1)).T

        # for j in range(m+1):
        #     RHSd_x[3:-3,j] = self.WENO6(uu_diff[:,j], e, mweno=mweno, mapped=mapped, trainable=trainable)
        #     RHSd_y[j,3:-3] = self.WENO6(uu_diff[j,:], e, mweno=mweno, mapped=mapped, trainable=trainable)
        if w5_minus=='Lax-Friedrichs':
            uu_conv_x = problem.funct_convection_x(uu)
            uu_conv_y = problem.funct_convection_y(uu)
            for j in range(m + 1):
                max_der_x = torch.max(torch.abs(problem.funct_derivative_x(uu[:,j])))
                RHSc_p_x[3:-3,j] = self.WENO5(0.5 * (uu_conv_x[:,j] + max_der_x * uu[:,j]), e, w5_minus=False, mweno=mweno, mapped=mapped, trainable=trainable)
                RHSc_n_x[3:-3,j] = self.WENO5(0.5 * (uu_conv_x[:,j] - max_der_x * uu[:,j]), e, w5_minus=True, mweno=mweno, mapped=mapped, trainable=trainable)
                max_der_y = torch.max(torch.abs(problem.funct_derivative_y(uu[j,:])))
                RHSc_p_y[j,3:-3] = self.WENO5(0.5 * (uu_conv_y[j,:] + max_der_y * uu[j,:]), e, w5_minus=False, mweno=mweno, mapped=mapped, trainable=trainable)
                RHSc_n_y[j,3:-3] = self.WENO5(0.5 * (uu_conv_y[j,:] - max_der_y * uu[j,:]), e, w5_minus=True, mweno=mweno, mapped=mapped, trainable=trainable)
            RHSc_x = RHSc_p_x + RHSc_n_x
            RHSc_y = RHSc_p_y + RHSc_n_y
            u1 = uu + t * ((term_2 / h ** 2) * RHSd_x + (term_2 / h ** 2) * RHSd_y - (term_1 / h) * RHSc_x - (term_1 / h) * RHSc_y + term_0 * uu)
        if w5_minus == "no":
            u1 = uu + t * ((term_2 / h ** 2) * RHSd_x + (term_2 / h ** 2) * RHSd_y + term_0 * uu)

        # u1[0:3,:] = u1_bc_l_x[0:3,:]
        # u1[-3:,:] = u1_bc_r_x[0:3,:]
        # u1[:,0:3] = u1_bc_l_x[0:3,:]
        # u1[:,-3:] = u1_bc_r_x[0:3,:]

        uu1_diff = problem.funct_diffusion(u1)
        u2 = torch.zeros(m+1,m+1)
        RHS1d_x = torch.zeros(m+1,m+1)
        RHS1d_y = torch.zeros(m+1,m+1)
        RHS1c_p_x = torch.zeros(m+1,m+1)
        RHS1c_n_x = torch.zeros(m+1,m+1)
        RHS1c_p_y = torch.zeros(m+1,m+1)
        RHS1c_n_y = torch.zeros(m+1,m+1)

        uu1_diff_x = torch.reshape(uu1_diff.T, (1, (m + 1) * (m + 1)))
        uu1_diff_y = torch.reshape(uu1_diff, (1, (m + 1) * (m + 1)))
        RHS1d_xr = self.WENO6(uu1_diff_x.squeeze(0), e, mweno=mweno, mapped=mapped, trainable=trainable)
        RHS1d_yr = self.WENO6(uu1_diff_y.squeeze(0), e, mweno=mweno, mapped=mapped, trainable=trainable)
        RHS1d_y2 = torch.zeros((m + 1) ** 2)
        RHS1d_y2[3:-3] = RHS1d_yr
        RHS1d_x2 = torch.zeros((m + 1) ** 2)
        RHS1d_x2[3:-3] = RHS1d_xr
        RHS1d_y = RHS1d_y2.reshape(m + 1, m + 1)
        RHS1d_x = (RHS1d_x2.reshape(m + 1, m + 1)).T

        # for j in range(m + 1):
        #     RHS1d_x[3:-3, j] = self.WENO6(uu1_diff[:, j], e, mweno=mweno, mapped=mapped, trainable=trainable)
        #     RHS1d_y[j, 3:-3] = self.WENO6(uu1_diff[j, :], e, mweno=mweno, mapped=mapped, trainable=trainable)
        if w5_minus=='Lax-Friedrichs':
            uu1_conv_x = problem.funct_convection_x(u1)
            uu1_conv_y = problem.funct_convection_y(u1)
            for j in range(m + 1):
                max_der_x = torch.max(torch.abs(problem.funct_derivative_x(u1[:,j])))
                RHS1c_p_x[3:-3,j] = self.WENO5(0.5 * (uu1_conv_x[:,j] + max_der_x * u1[:,j]), e, w5_minus=False, mweno=mweno, mapped=mapped, trainable=trainable)
                RHS1c_n_x[3:-3,j] = self.WENO5(0.5 * (uu1_conv_x[:,j] - max_der_x * u1[:,j]), e, w5_minus=True, mweno=mweno, mapped=mapped, trainable=trainable)
                max_der_y = torch.max(torch.abs(problem.funct_derivative_y(u1[j,:])))
                RHS1c_p_y[j,3:-3] = self.WENO5(0.5 * (uu1_conv_y[j,:] + max_der_y * u1[j,:]), e, w5_minus=False, mweno=mweno, mapped=mapped, trainable=trainable)
                RHS1c_n_y[j,3:-3] = self.WENO5(0.5 * (uu1_conv_y[j,:] - max_der_y * u1[j,:]), e, w5_minus=True, mweno=mweno, mapped=mapped, trainable=trainable)
            RHS1c_x = RHS1c_p_x + RHS1c_n_x
            RHS1c_y = RHS1c_p_y + RHS1c_n_y
            u2 = 0.75*uu+0.25*u1+0.25*t*((term_2/h ** 2)*RHS1d_x +(term_2/h ** 2)*RHS1d_y - (term_1/h)*RHS1c_x - (term_1/h)*RHS1c_y + term_0*u1)
        if w5_minus == "no":
            u2 = 0.75 * uu + 0.25 * u1 + 0.25 * t * ((term_2 / h ** 2) * RHS1d_x + (term_2 / h ** 2) * RHS1d_y  + term_0 * u1)

        # u2[0:3,:] = u2_bc_l_x
        # u2[-3:,:] = u2_bc_r_x
        # u2[:,0:3] = u2_bc_l_x
        # u2[:,-3:] = u2_bc_r_x

        uu2_diff = problem.funct_diffusion(u2)
        u_ret = torch.zeros(m+1,m+1)
        RHS2d_x = torch.zeros(m+1,m+1)
        RHS2d_y = torch.zeros(m+1,m+1)
        RHS2c_p_x = torch.zeros(m+1,m+1)
        RHS2c_n_x = torch.zeros(m+1,m+1)
        RHS2c_p_y = torch.zeros(m+1,m+1)
        RHS2c_n_y = torch.zeros(m+1,m+1)

        uu2_diff_x = torch.reshape(uu2_diff.T, (1, (m + 1) * (m + 1)))
        uu2_diff_y = torch.reshape(uu2_diff, (1, (m + 1) * (m + 1)))
        RHS2d_xr = self.WENO6(uu2_diff_x.squeeze(0), e, mweno=mweno, mapped=mapped, trainable=trainable)
        RHS2d_yr = self.WENO6(uu2_diff_y.squeeze(0), e, mweno=mweno, mapped=mapped, trainable=trainable)
        RHS2d_y2 = torch.zeros((m + 1) ** 2)
        RHS2d_y2[3:-3] = RHS2d_yr
        RHS2d_x2 = torch.zeros((m + 1) ** 2)
        RHS2d_x2[3:-3] = RHS2d_xr
        RHS2d_y = RHS2d_y2.reshape(m + 1, m + 1)
        RHS2d_x = (RHS2d_x2.reshape(m + 1, m + 1)).T

        # for j in range(m + 1):
        #     RHS2d_x[3:-3, j] = self.WENO6(uu2_diff[:, j], e, mweno=mweno, mapped=mapped, trainable=trainable)
        #     RHS2d_y[j, 3:-3] = self.WENO6(uu2_diff[j, :], e, mweno=mweno, mapped=mapped, trainable=trainable)
        if w5_minus == 'Lax-Friedrichs':
            uu2_conv_x = problem.funct_convection_x(u2)
            uu2_conv_y = problem.funct_convection_y(u2)
            for j in range(m + 1):
                max_der_x = torch.max(torch.abs(problem.funct_derivative_x(u2[:,j])))
                RHS2c_p_x[3:-3,j] = self.WENO5(0.5 * (uu2_conv_x[:,j] + max_der_x * u2[:,j]), e, w5_minus=False, mweno=mweno, mapped=mapped, trainable=trainable)
                RHS2c_n_x[3:-3,j] = self.WENO5(0.5 * (uu2_conv_x[:,j] - max_der_x * u2[:,j]), e, w5_minus=True, mweno=mweno, mapped=mapped, trainable=trainable)
                max_der_y = torch.max(torch.abs(problem.funct_derivative_y(u2[j,:])))
                RHS2c_p_y[j,3:-3] = self.WENO5(0.5 * (uu2_conv_y[j,:] + max_der_y * u2[j,:]), e, w5_minus=False, mweno=mweno, mapped=mapped, trainable=trainable)
                RHS2c_n_y[j,3:-3] = self.WENO5(0.5 * (uu2_conv_y[j,:] - max_der_y * u2[j,:]), e, w5_minus=True, mweno=mweno, mapped=mapped, trainable=trainable)
            RHS2c_x = RHS2c_p_x + RHS2c_n_x
            RHS2c_y = RHS2c_p_y + RHS2c_n_y
            u_ret = (1 / 3) * uu + (2 / 3) * u2 + (2 / 3) * t * ( (term_2 / h ** 2) * RHS2d_x + (term_2 / h ** 2) * RHS2d_y - (term_1 / h) * RHS2c_x - (term_1 / h) * RHS2c_y + term_0 * u2)
        if w5_minus == "no":
            u_ret = (1 / 3) * uu + (2 / 3) * u2 + (2 / 3) * t * ( (term_2 / h ** 2) * RHS2d_x + (term_2 / h ** 2) * RHS2d_y + term_0 * u2)

        # u_ret[0:3, :] = u_bc_l_x
            # u_ret[-3:, :] = u_bc_r_x
            # u_ret[:, 0:3] = u_bc_l_x
            # u_ret[:, -3:] = u_bc_r_x

        return u_ret

    def forward(self, problem, u_ret, k, mweno, mapped, dim = 1):
        if dim == 2:
            u = self.run_weno_2d(problem,u_ret,mweno=mweno,mapped=mapped,vectorized=True,trainable=True,k=k)
        else:
            u = self.run_weno(problem,u_ret,mweno=mweno,mapped=mapped,vectorized=True,trainable=True,k=k)
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
        return V_classic, V_trained

    def order_compute(self, iterations, initial_space_steps, initial_time_steps, params, problem_class, trainable):
        problem = problem_class(space_steps=initial_space_steps, time_steps=initial_time_steps, params=params)
        vecerr_max = np.zeros((iterations))[:, None]
        vecerr_l2 = np.zeros((iterations))[:, None]
        order_max = np.zeros((iterations - 1))[:, None]
        order_l2 = np.zeros((iterations - 1))[:, None]
        u_init, nn = self.init_run_weno(problem, vectorized=True, just_one_time_step=False)
        u = u_init
        for k in range(nn):
            u = self.run_weno(problem,u,mweno=True, mapped=False, trainable=trainable, vectorized=True,k=k)
        u_last = u
        xmaxerr = problem.err(u_last) #, first_step=False)
        l2_err = problem.err_l2(u_last)
        vecerr_max[0] = xmaxerr
        vecerr_l2[0] = l2_err
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
            vecerr_max[i] = xmaxerr
            order_max[i - 1] = np.log(vecerr_max[i - 1] / vecerr_max[i]) / np.log(2)
            l2_err = problem.err_l2(u_last)
            vecerr_l2[i] = l2_err
            order_l2[i - 1] = np.log(vecerr_l2[i - 1] / vecerr_l2[i]) / np.log(2)
            print(problem.space_steps, problem.time_steps)
        return vecerr_max, order_max, vecerr_l2, order_l2

    def compute_exact(self, problem_class, problem, space_steps, time_steps, just_one_time_step, trainable):
        if hasattr(problem_class, 'exact'):
            print('nic netreba')
        else:
            u, nn = self.init_run_weno(problem, vectorized=False, just_one_time_step=just_one_time_step)
            for k in range(nn):
                u[:, k + 1] = self.run_weno(problem, u, mweno=True, mapped=False, vectorized=False, trainable=trainable, k=k)
        u_exact = u
        #space_steps_exact = problem.space_steps
        #time_steps_exact = problem.time_steps
        #divider_space = space_steps_exact / space_steps
        #divider_time = time_steps_exact / time_steps
        #divider_space = int(divider_space)
        #divider_time = int(divider_time)
        #u_exact_adjusted = u_exact[0:space_steps_exact+1:divider_space,0:time_steps_exact+1:divider_time]
        return u_exact #, u_exact_adjusted
