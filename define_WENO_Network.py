import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import torch
from torch import nn, optim # Sets of preset layers and optimizers
from scipy.stats import norm
import torch.nn.functional as F # Sets of functions such as ReLU
from torchvision import datasets, transforms # Popular datasets, architectures and common


class WENONetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner_nn_weno5_plus = self.get_inner_nn_weno5()
        self.inner_nn_weno5_minus = self.get_inner_nn_weno5()
        self.inner_nn_weno6 = self.get_inner_nn_weno6()
        self.weno5_mult_bias, self.weno6_mult_bias = self.get_multiplicator_biases()

    # def get_inner_nn_weno5(self):
    #     net = nn.Sequential(
    #         nn.Conv1d(2, 20, kernel_size=5, stride=1, padding=2),
    #         nn.ELU(),
    #         nn.Conv1d(20, 20, kernel_size=5, stride=1, padding=2),
    #         nn.ELU(),
    #         # nn.Conv1d(10, 10, kernel_size=1, stride=1, padding=0),
    #         # nn.ELU(),
    #         # nn.Conv1d(80, 40, kernel_size=1, stride=1, padding=0),
    #         # nn.ELU(),
    #         # nn.Conv1d(40, 20, kernel_size=3, stride=1, padding=1),
    #         # nn.ELU(),
    #         nn.Conv1d(20, 1, kernel_size=1, stride=1, padding=0),
    #         nn.Sigmoid()
    #         )
    #     return net
    #
    # def get_inner_nn_weno6(self):
    #     net = nn.Sequential(
    #         nn.Conv1d(2, 10, kernel_size=5, stride=1, padding=2),
    #         nn.ELU(),
    #         nn.Conv1d(10, 10, kernel_size=5, stride=1, padding=2),
    #         nn.ELU(),
    #         # nn.Conv1d(40, 80, kernel_size=1, stride=1, padding=0),
    #         # nn.ELU(),
    #         # nn.Conv1d(80, 40, kernel_size=1, stride=1, padding=0),
    #         # nn.ELU(),
    #         # nn.Conv1d(40, 20, kernel_size=3, stride=1, padding=1),
    #         # nn.ELU(),
    #         nn.Conv1d(10, 1, kernel_size=1, stride=1, padding=0),
    #         nn.Sigmoid()
    #         )
    #     return net

    def get_multiplicator_biases(self):
        # first for weno 5, second for weno 6
        return 0.1, 0.1

    def prepare_dif(self, dif):
        return dif[None, :, :]

    def WENO5(self, uu, e, w5_minus, mweno, mapped, trainable=True):
        uu_left = uu[:-1]
        uu_right = uu[1:]

        def get_fluxes(uu):
            if w5_minus is True:
                flux0 = (11 * uu[3:-2] - 7 * uu[4:-1] + 2 * uu[5:]) / 6
                flux1 = (2 * uu[2:-3] + 5 * uu[3:-2] - uu[4:-1]) / 6
                flux2 = (-uu[1:-4] + 5 * uu[2:-3] + 2 * uu[3:-2]) / 6
            else:
                flux0 = (2 * uu[0:-5] - 7 * uu[1:-4] + 11 * uu[2:-3]) / 6
                flux1 = (- uu[1:-4] + 5 * uu[2:-3] + 2* uu[3:-2]) / 6
                flux2 = (2*uu[2:-3] + 5 * uu[3:-2] - uu[4:-1]) / 6
            return flux0, flux1, flux2

        fluxp0, fluxp1, fluxp2 = get_fluxes(uu_right)
        fluxn0, fluxn1, fluxn2 = get_fluxes(uu_left)

        def get_betas(uu):
            if w5_minus is True:
                beta0 = 13 / 12 * (uu[3:-2] - 2 * uu[4:-1] + uu[5:]) ** 2 + 1 / 4 * (
                            3 * uu[3:-2] - 4 * uu[4:-1] + uu[5:]) ** 2
                beta1 = 13 / 12 * (uu[2:-3] - 2 * uu[3:-2] + uu[4:-1]) ** 2 + 1 / 4 * (uu[2:-3] - uu[4:-1]) ** 2
                beta2 = 13 / 12 * (uu[1:-4] - 2 * uu[2:-3] + uu[3:-2]) ** 2 + 1 / 4 * (
                            uu[1:-4] - 4 * uu[2:-3] + 3 * uu[3:-2]) ** 2
            else:
                beta0 = 13 / 12 * (uu[0:-5] - 2 * uu[1:-4] + uu[2:-3]) ** 2 + 1 / 4 * (
                        uu[0:-5] - 4 * uu[1:-4] + 3*uu[2:-3]) ** 2
                beta1 = 13 / 12 * (uu[1:-4] - 2 * uu[2:-3] + uu[3:-2]) ** 2 + 1 / 4 * (uu[1:-4] - uu[3:-2]) ** 2
                beta2 = 13 / 12 * (uu[2:-3] - 2 * uu[3:-2] + uu[4:-1]) ** 2 + 1 / 4 * (
                        3*uu[2:-3] - 4 * uu[3:-2] + uu[4:-1]) ** 2
            return beta0, beta1, beta2

        betap0, betap1, betap2 = get_betas(uu_right)
        betan0, betan1, betan2 = get_betas(uu_left)

        old_betas_p = [betap0, betap1, betap2]
        old_betas_n = [betan0, betan1, betan2]

        if trainable:
            dif = self.get_average_diff(uu)
            dif2 = self.get_average_diff2(uu)
            dif12 = torch.stack([dif, dif2 ])
            # dif12 = torch.stack([dif, dif2, dif**power, dif2**power]) #,dif**3,dif2**3,dif**2*dif2,dif2**2*dif])
            dif12 = self.prepare_dif(dif12)

            if w5_minus:
                beta_multiplicators = self.inner_nn_weno5_minus(dif12)[0, 0, :] + self.weno5_mult_bias
            else:
                beta_multiplicators = self.inner_nn_weno5_plus(dif12)[0, 0, :] + self.weno5_mult_bias

            # beta_multiplicators = self.inner_nn_weno5(dif)[0, :, :].T + self.weno5_mult_bias
            # beta_multiplicators_left = beta_multiplicators[:-1]
            # beta_multiplicators_right = beta_multiplicators[1:]

            betap_corrected_list = []
            betan_corrected_list = []

            if w5_minus is True:
                mult_shifts_p = [1, 0, -1]  # [2, 1, 0]
                # mult_shifts_p = [2, 1, 0]
                for k, beta in enumerate([betap0, betap1, betap2]):
                    shift = mult_shifts_p[k]  # k-1 #(3-k)-1
                    betap_corrected_list.append(beta * beta_multiplicators[3+shift:-3+shift])
                mult_shifts_n = [1, 0, -1]
                # mult_shifts_p = [2, 1, 0]
                for k, beta in enumerate([betan0, betan1, betan2]):
                    shift = mult_shifts_n[k]  # k #3-k
                    betan_corrected_list.append(beta * beta_multiplicators[3+shift:-3+shift])
            else:
                mult_shifts_p = [-1, 0, 1]
                for k, beta in enumerate([betap0, betap1, betap2]):
                    shift = mult_shifts_p[k]  # k-1 #(3-k)-1
                    betap_corrected_list.append(beta * beta_multiplicators[3+shift:-3+shift])
                mult_shifts_n = [-1, 0, 1]  # [-2, -1, 0]
                # mult_shifts_n = [-2, -1, 0]
                for k, beta in enumerate([betan0, betan1, betan2]):
                    shift = mult_shifts_n[k]  # k #3-k
                    betan_corrected_list.append(beta * beta_multiplicators[3+shift:-3+shift])

            # for k, beta in enumerate([betap0, betap1, betap2]):
            #     shift = k -1
            #     betap_corrected_list.append(beta * (beta_multiplicators[3+shift:-3+shift]))
            # for k, beta in enumerate([betan0, betan1, betan2]):
            #     shift = k - 1
            #     betan_corrected_list.append(beta * (beta_multiplicators[3+shift:-3+shift]))

            [betap0, betap1, betap2] = betap_corrected_list
            [betan0, betan1, betan2] = betan_corrected_list

        d0 = 1 / 10
        d1 = 6 / 10
        d2 = 3 / 10

        def get_omegas_mweno(betas, ds, old_betas):
            beta_range_square = (old_betas[2] - old_betas[0]) ** 2
            return [d / (e + beta) ** 2 * (beta_range_square + (e + beta) ** 2) for beta, d in zip(betas, ds)]

        def get_omegas_weno(betas, ds, old_betas):
            return [d / (e + beta) ** 2 for beta, d in zip(betas, ds)]

        omegas_func_dict = {0: get_omegas_weno, 1: get_omegas_mweno}
        [omegap_0, omegap_1, omegap_2] = omegas_func_dict[int(mweno)]([betap0, betap1, betap2], [d0, d1, d2], old_betas_p)
        [omegan_0, omegan_1, omegan_2] = omegas_func_dict[int(mweno)]([betan0, betan1, betan2], [d0, d1, d2], old_betas_n)

        def normalize(tensor_list):
            sum_ = sum(tensor_list)  # note, that inbuilt sum applies __add__ iteratively therefore its overloaded-
            return [tensor / sum_ for tensor in tensor_list]

        [omegap0, omegap1, omegap2] = normalize([omegap_0, omegap_1, omegap_2])
        [omegan0, omegan1, omegan2] = normalize([omegan_0, omegan_1, omegan_2])

        if mapped:
            def get_alpha(omega, d):
                return (omega * (d + d ** 2 - 3 * d * omega + omega ** 2)) / (d ** 2 + omega * (1 - 2 * d))

            [alphap0, alphap1, alphap2] = [get_alpha(omega, d) for omega, d in zip([omegap0, omegap1, omegap2],
                                                                                   [d0, d1, d2])]
            [alphan0, alphan1, alphan2] = [get_alpha(omega, d) for omega, d in zip([omegan0, omegan1, omegan2],
                                                                                   [d0, d1, d2])]

            [omegap0, omegap1, omegap2] = normalize([alphap0, alphap1, alphap2])
            [omegan0, omegan1, omegan2] = normalize([alphan0, alphan1, alphan2])

        fluxp = omegap0 * fluxp0 + omegap1 * fluxp1 + omegap2 * fluxp2
        fluxn = omegan0 * fluxn0 + omegan1 * fluxn1 + omegan2 * fluxn2

        RHS = (fluxp - fluxn)

        return RHS

    def WENO6(self, uu, e, mweno, mapped, trainable=True):
        uu_left = uu[:-1]
        uu_right = uu[1:]

        def get_fluxes(uu):
            flux0 = (uu[0:-5] - 3 * uu[1:-4] - 9 * uu[2:-3] + 11 * uu[3:-2]) / 12
            flux1 = (uu[1:-4] - 15 * uu[2:-3] + 15 * uu[3:-2] - uu[4:-1]) / 12
            flux2 = (-11 * uu[2:-3] + 9 * uu[3:-2] + 3 * uu[4:-1] - uu[5:]) / 12
            return flux0, flux1, flux2

        fluxp0, fluxp1, fluxp2 = get_fluxes(uu_right)
        fluxn0, fluxn1, fluxn2 = get_fluxes(uu_left)

        def get_betas(uu):
            beta0 = 13 / 12 * (uu[0:-5] - 3 * uu[1:-4] + 3 * uu[2:-3] - uu[3:-2]) ** 2 + 1 / 4 * (
                        uu[0:-5] - 5 * uu[1:-4] + 7 * uu[2:-3] - 3 * uu[3:-2]) ** 2
            beta1 = 13 / 12 * (uu[1:-4] - 3 * uu[2:-3] + 3 * uu[3:-2] - uu[4:-1]) ** 2 + 1 / 4 * (
                        uu[1:-4] - uu[2:-3] - uu[3:-2] + uu[4:-1]) ** 2
            beta2 = 13 / 12 * (uu[2:-3] - 3 * uu[3:-2] + 3 * uu[4:-1] - uu[5:]) ** 2 + 1 / 4 * (
                        -3 * uu[2:-3] + 7 * uu[3:-2] - 5 * uu[4:-1] + uu[5:]) ** 2
            return beta0, beta1, beta2

        betap0, betap1, betap2 = get_betas(uu_right)
        betan0, betan1, betan2 = get_betas(uu_left)

        old_betas_p = [betap0, betap1, betap2]
        old_betas_n = [betan0, betan1, betan2]

        if trainable:
            dif = self.get_average_diff(uu)
            dif2 = self.get_average_diff2(uu)
            dif12 = torch.stack([dif, dif2])
            dif12 = self.prepare_dif(dif12)
            beta_multiplicators = self.inner_nn_weno6(dif12)[0, 0, :] + self.weno6_mult_bias

            betap_corrected_list = []
            betan_corrected_list = []

            mult_shifts_p = [-1, 0, 1]
            for k, beta in enumerate([betap0, betap1, betap2]):
                shift = mult_shifts_p[k]  # k-1 #(3-k)-1
                betap_corrected_list.append(beta * beta_multiplicators[3+shift:-3+shift])
            mult_shifts_n = [-1, 0, 1]  # [-2, -1, 0]
            # mult_shifts_n = [-2, -1, 0]
            for k, beta in enumerate([betan0, betan1, betan2]):
                shift = mult_shifts_n[k]  # k #3-k
                betan_corrected_list.append(beta * beta_multiplicators[3+shift:-3+shift])

            # for k, beta in enumerate([betap0, betap1, betap2]):
            #     shift = k -1
            #     betap_corrected_list.append(beta * (beta_multiplicators[3+shift:-3+shift]))
            # for k, beta in enumerate([betan0, betan1, betan2]):
            #     shift = k - 1
            #     betan_corrected_list.append(beta * (beta_multiplicators[3+shift:-3+shift]))

            [betap0, betap1, betap2] = betap_corrected_list
            [betan0, betan1, betan2] = betan_corrected_list

        gamap0 = 1 / 21
        gamap1 = 19 / 21
        gamap2 = 1 / 21
        gaman0 = 4 / 27
        gaman1 = 19 / 27
        gaman2 = 4 / 27
        sigmap = 42 / 15
        sigman = 27 / 15

        def get_omegas_mweno(betas, gamas, old_betas):
            beta_range_square = (old_betas[2] - old_betas[0]) ** 2
            return [gama / (e + beta) ** 2 * (beta_range_square + (e + beta) ** 2) for beta, gama in zip(betas, gamas)]

        def get_omegas_weno(betas, gamas, old_betas):
            return [gama / (e + beta) ** 2 for beta, gama in zip(betas, gamas)]

        omegas_func_dict = {0: get_omegas_weno, 1: get_omegas_mweno}
        [omegapp_0, omegapp_1, omegapp_2] = omegas_func_dict[int(mweno)]([betap0, betap1, betap2],
                                                                         [gamap0, gamap1, gamap2], old_betas_p)
        [omeganp_0, omeganp_1, omeganp_2] = omegas_func_dict[int(mweno)]([betap0, betap1, betap2],
                                                                         [gaman0, gaman1, gaman2], old_betas_p)
        [omegapn_0, omegapn_1, omegapn_2] = omegas_func_dict[int(mweno)]([betan0, betan1, betan2],
                                                                         [gamap0, gamap1, gamap2], old_betas_n)
        [omegann_0, omegann_1, omegann_2] = omegas_func_dict[int(mweno)]([betan0, betan1, betan2],
                                                                         [gaman0, gaman1, gaman2], old_betas_n)

        def normalize(tensor_list):
            sum_ = sum(tensor_list)  # note, that inbuilt sum applies __add__ iteratively therefore its overloaded-
            return [tensor / sum_ for tensor in tensor_list]

        [omegapp0, omegapp1, omegapp2] = normalize([omegapp_0, omegapp_1, omegapp_2])
        [omeganp0, omeganp1, omeganp2] = normalize([omeganp_0, omeganp_1, omeganp_2])
        [omegapn0, omegapn1, omegapn2] = normalize([omegapn_0, omegapn_1, omegapn_2])
        [omegann0, omegann1, omegann2] = normalize([omegann_0, omegann_1, omegann_2])

        omegaps = [omegapp0, omegapp1, omegapp2, omegapn0, omegapn1, omegapn2]
        omegans = [omeganp0, omeganp1, omeganp2, omegann0, omegann1, omegann2]

        [omegap0, omegap1, omegap2, omegan0, omegan1, omegan2] = [sigmap * omegap - sigman * omegan
                                                                  for omegap, omegan in zip(omegaps, omegans)]

        if mapped:
            d0 = -2 / 15
            d1 = 19 / 15
            d2 = -2 / 15

            def get_alpha(omega, d):
                return (omega * (d + d ** 2 - 3 * d * omega + omega ** 2)) / (d ** 2 + omega * (1 - 2 * d))

            [alphap0, alphap1, alphap2] = [get_alpha(omega, d) for omega, d in zip([omegap0, omegap1, omegap2],
                                                                                   [d0, d1, d2])]
            [alphan0, alphan1, alphan2] = [get_alpha(omega, d) for omega, d in zip([omegan0, omegan1, omegan2],
                                                                                   [d0, d1, d2])]
            [omegap0, omegap1, omegap2] = normalize([alphap0, alphap1, alphap2])
            [omegan0, omegan1, omegan2] = normalize([alphan0, alphan1, alphan2])

        fluxp = omegap0 * fluxp0 + omegap1 * fluxp1 + omegap2 * fluxp2
        fluxn = omegan0 * fluxn0 + omegan1 * fluxn1 + omegan2 * fluxn2

        RHS = (fluxp - fluxn)

        return RHS

    def get_average_diff(self, uu):
        dif = uu[1:] - uu[:-1]
        dif_left = torch.zeros_like(uu)
        dif_right = torch.zeros_like(uu)
        dif_left[:-1] = dif
        dif_left[-1] = dif[-1]
        dif_right[1:] = dif
        dif_right[0] = dif[0]
        dif_final = 0.5 * dif_left + 0.5 * dif_right
        return dif_final

    def get_average_diff2(self, uu):
        dif = uu[1:] - uu[:-1]
        dif_left = torch.zeros_like(uu)
        dif_right = torch.zeros_like(uu)
        dif_left[:-1] = dif
        dif_left[-1] = dif[-1]
        dif_right[1:] = dif
        dif_right[0] = dif[0]
        dif_final2 = dif_left - 2 * uu + dif_right
        return dif_final2

    def run_weno(self, problem, trainable, vectorized, just_one_time_step):
        mweno = True
        mapped = False
        m = problem.space_steps
        e = problem.params['e']
        n, t, h = problem.time_steps, problem.t, problem.h
        x, time = problem.x, problem.time
        term_2 = problem.der_2()
        term_1 = problem.der_1()
        term_0 = problem.der_0()
        term_const = problem.der_const()
        u_bc_l, u_bc_r, u1_bc_l, u1_bc_r, u2_bc_l, u2_bc_r = problem.boundary_condition
        w5_minus = problem.w5_minus

        if vectorized:
            u = torch.zeros(m+1,1)
        else:
            u = torch.zeros((m+1, n+1))

        u[:,0] = problem.initial_condition

        if just_one_time_step is True:
            nn = 1
        else:
            nn = n

        for l in range(1, nn+1):
            u = torch.Tensor(u)
            if vectorized:
                ll=1
            else:
                ll=l

            uu_conv = problem.funct_convection(u[:, ll - 1])
            uu_diff = problem.funct_diffusion(u[:, ll - 1])
            u1 = torch.zeros(x.shape[0])
            RHSd = self.WENO6(uu_diff, e, mweno=mweno, mapped=mapped, trainable=trainable)
            if w5_minus=='both':
                RHSc_p = self.WENO5(uu_conv, e, w5_minus=False, mweno=mweno, mapped=mapped, trainable=trainable)
                RHSc_n= self.WENO5(uu_conv, e, w5_minus=True, mweno=mweno, mapped=mapped, trainable=trainable)
                u1[3:-3] = u[3:-3, ll - 1] + t * ((term_2[3:-3] / h ** 2) * RHSd + ((term_1>=0)[3:-3])*(term_1[3:-3] / h) * RHSc_n
                                                  + ((term_1<0)[3:-3])*(term_1[3:-3] / h) * RHSc_p + term_0 * u[3:-3, ll - 1])
            elif w5_minus=='Lax-Friedrichs':
                max_der = torch.max(torch.abs(problem.funct_derivative(u[:, ll - 1])))
                RHSc_p = self.WENO5(0.5*(uu_conv+max_der*u[:, ll - 1]), e, w5_minus=False, mweno=mweno, mapped=mapped, trainable=trainable)
                RHSc_n = self.WENO5(0.5*(uu_conv-max_der*u[:, ll - 1]), e, w5_minus=True, mweno=mweno, mapped=mapped, trainable=trainable)
                RHSc = RHSc_p + RHSc_n
                u1[3:-3] = u[3:-3, ll - 1] + t * ((term_2 / h ** 2) * RHSd - (term_1 / h) * RHSc + term_0 * u[3:-3, ll - 1])
            else:
                RHSc = self.WENO5(uu_conv, e, w5_minus=w5_minus, mweno=mweno, mapped=mapped, trainable=trainable)
                u1[3:-3] = u[3:-3, ll - 1] + t * ((term_2 / h ** 2) * RHSd - (term_1 / h) * RHSc + term_0 * u[3:-3, ll - 1])

            u1[0:3] = u1_bc_l[:,l - 1]
            u1[m - 2:] = u1_bc_r[:,l - 1]

            uu1_conv = problem.funct_convection(u1)
            uu1_diff = problem.funct_diffusion(u1)
            u2 = torch.zeros(x.shape[0])
            RHS1d = self.WENO6(uu1_diff, e, mweno=mweno, mapped=mapped, trainable=trainable)
            if w5_minus=='both':
                RHS1c_p = self.WENO5(uu1_conv, e, w5_minus=False, mweno=mweno, mapped=mapped, trainable=trainable)
                RHS1c_n = self.WENO5(uu1_conv, e, w5_minus=True, mweno=mweno, mapped=mapped, trainable=trainable)
                u2[3:-3] = 0.75*u[3:-3,ll-1]+0.25*u1[3:-3]+0.25*t*((term_2[3:-3]/h ** 2)*RHS1d+ (term_1>=0)[3:-3]*(term_1[3:-3] / h)*RHS1c_n
                                                                   + (term_1<0)[3:-3]*(term_1[3:-3] / h) * RHS1c_p +term_0*u1[3:-3])
            elif w5_minus=='Lax-Friedrichs':
                max_der = torch.max(torch.abs(problem.funct_derivative(u1)))
                RHS1c_p = self.WENO5(0.5*(uu1_conv+max_der*u1), e, w5_minus=False, mweno=mweno, mapped=mapped, trainable=trainable)
                RHS1c_n = self.WENO5(0.5*(uu1_conv-max_der*u1), e, w5_minus=True, mweno=mweno, mapped=mapped, trainable=trainable)
                RHS1c = RHS1c_p + RHS1c_n
                u2[3:-3] = 0.75*u[3:-3,ll-1]+0.25*u1[3:-3]+0.25*t*((term_2/h ** 2)*RHS1d-(term_1/h)*RHS1c+term_0*u1[3:-3])
            else:
                RHS1c = self.WENO5(uu1_conv, e, w5_minus=w5_minus, mweno=mweno, mapped=mapped, trainable=trainable)
                u2[3:-3] = 0.75*u[3:-3,ll-1]+0.25*u1[3:-3]+0.25*t*((term_2/h ** 2)*RHS1d-(term_1/h)*RHS1c+term_0*u1[3:-3])

            u2[0:3] = u2_bc_l[:,l - 1]
            u2[m - 2:] = u2_bc_r[:,l - 1]

            uu2_conv = problem.funct_convection(u2)
            uu2_diff = problem.funct_diffusion(u2)
            RHS2d = self.WENO6(uu2_diff, e, mweno=mweno, mapped=mapped, trainable=trainable)
            if w5_minus=='both':
                RHS2c_p = self.WENO5(uu2_conv, e, w5_minus=False, mweno=mweno, mapped=mapped, trainable=trainable)
                RHS2c_n = self.WENO5(uu2_conv, e, w5_minus=True, mweno=mweno, mapped=mapped, trainable=trainable)
                if vectorized:
                    u[3:-3, 0] = (1 / 3) * u[3:-3, ll - 1] + (2 / 3) * u2[3:-3] + (2 / 3) * t * (
                            (term_2[3:-3] / h ** 2) * RHS2d + (term_1>=0)[3:-3]*(term_1[3:-3] / h) * RHS2c_n
                            + (term_1<0)[3:-3]*(term_1[3:-3] / h) * RHS2c_p + term_0 * u2[3:-3])
                    u[0:3, 0] = u_bc_l[:, l]
                    u[m - 2:, 0] = u_bc_r[:, l]
                else:
                    u[3:-3, l] = (1 / 3) * u[3:-3, ll - 1] + (2 / 3) * u2[3:-3] + (2 / 3) * t * (
                            (term_2[3:-3] / h ** 2) * RHS2d + (term_1>=0)[3:-3]*(term_1[3:-3] / h) * RHS2c_n
                            + (term_1<0)[3:-3]*(term_1[3:-3] / h) * RHS2c_p + term_0 * u2[3:-3])
                    u[0:3, l] = u_bc_l[:, l]
                    u[m - 2:, l] = u_bc_r[:, l]
            elif w5_minus == 'Lax-Friedrichs':
                max_der = torch.max(torch.abs(problem.funct_derivative(u2)))
                RHS2c_p = self.WENO5(0.5 * (uu2_conv + max_der * u2), e, w5_minus=False, mweno=mweno, mapped=mapped,
                                     trainable=trainable)
                RHS2c_n = self.WENO5(0.5 * (uu2_conv - max_der * u2), e, w5_minus=True, mweno=mweno, mapped=mapped,
                                     trainable=trainable)
                RHS2c = RHS2c_p + RHS2c_n
                if vectorized:
                    u[3:-3, 0] = (1 / 3) * u[3:-3, ll - 1] + (2 / 3) * u2[3:-3] + (2 / 3) * t * (
                            (term_2 / h ** 2) * RHS2d - (term_1 / h) * RHS2c + term_0 * u2[3:-3])
                    u[0:3, 0] = u_bc_l[:, l]
                    u[m - 2:, 0] = u_bc_r[:, l]
                else:
                    u[3:-3, l] = (1 / 3) * u[3:-3, ll - 1] + (2 / 3) * u2[3:-3] + (2 / 3) * t * (
                            (term_2 / h ** 2) * RHS2d - (term_1 / h) * RHS2c + term_0 * u2[3:-3])
                    u[0:3, l] = u_bc_l[:, l]
                    u[m - 2:, l] = u_bc_r[:, l]
            else:
                RHS2c = self.WENO5(uu2_conv, e, w5_minus=w5_minus, mweno=mweno, mapped=mapped, trainable=trainable)
                if vectorized:
                    u[3:-3, 0] = (1 / 3) * u[3:-3, ll - 1] + (2 / 3) * u2[3:-3] + (2 / 3) * t * (
                            (term_2 / h ** 2) * RHS2d - (term_1 / h) * RHS2c + term_0 * u2[3:-3])
                    u[0:3, 0] = u_bc_l[:, l]
                    u[m - 2:, 0] = u_bc_r[:, l]
                else:
                    u[3:-3, l] = (1 / 3) * u[3:-3, ll - 1] + (2 / 3) * u2[3:-3] + (2 / 3) * t * (
                            (term_2 / h ** 2) * RHS2d - (term_1 / h) * RHS2c + term_0 * u2[3:-3])
                    u[0:3, l] = u_bc_l[:, l]
                    u[m - 2:, l] = u_bc_r[:, l]
        return u

    def forward(self, problem):
        u = self.run_weno(problem, trainable=True, vectorized=True, just_one_time_step = True)
        V,_,_ = problem.transformation(u)
        return V

    def get_axes(self, problem, u):
        _, S, t = problem.transformation(u)
        return S, t

    def full_WENO(self, problem, trainable, plot=True, vectorized=False):
        u = self.run_weno(problem, trainable=trainable, vectorized=vectorized, just_one_time_step=False)
        V, S, tt = problem.transformation(u)
        V = V.detach().numpy()
        if plot:
            X, Y = np.meshgrid(S, tt, indexing="ij")
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.plot_surface(X, Y, V, cmap=cm.viridis)
        return V, S, tt, u

    def compare_wenos(self, problem):
        u_trained = self.run_weno(problem, trainable=True, vectorized=False, just_one_time_step=True)
        V_trained, S, tt = problem.transformation(u_trained)
        u_classic = self.run_weno(problem, trainable=False, vectorized=False, just_one_time_step=True)
        V_classic, S, tt = problem.transformation(u_classic)
        plt.plot(S, V_classic.detach().numpy()[:,1], S, V_trained.detach().numpy()[:,1])
        plt.show()

    def compute_exact(self,problem_class, problem, space_steps, time_steps, just_one_time_step, trainable):
        if hasattr(problem_class, 'exact'):
            print('nic netreba')
        else:
            u_exact = self.run_weno(problem, trainable=trainable, vectorized=True , just_one_time_step=just_one_time_step)
        space_steps_exact = problem.space_steps
        time_steps_exact = problem.time_steps
        divider_space = space_steps_exact / space_steps
        divider_time = time_steps_exact / time_steps
        divider_space = int(divider_space)
        divider_time = int(divider_time)
        u_exact_adjusted = u_exact[0:space_steps_exact+1:divider_space,0:time_steps_exact+1:divider_time]
        return u_exact, u_exact_adjusted

    def compute_error(self, u, u_ex):
        u_last = u
        u_ex_last = u_ex
        err = torch.mean((u_ex_last - u_last)**2)
        #err = torch.max(torch.abs(u_ex_last - u_last))
        return err

    def order_compute(self, iterations, initial_space_steps, initial_time_steps, params, problem_class, trainable):
        problem = problem_class(space_steps=initial_space_steps, time_steps=initial_time_steps, params=params)
        vecerr = np.zeros((iterations))[:, None]
        order = np.zeros((iterations - 1))[:, None]
        if hasattr(problem_class,'exact'):
            u = self.run_weno(problem, trainable=trainable, vectorized=True, just_one_time_step=False)
            u_last = u[:,-1]
            xmaxerr = problem.err(u_last) #, first_step=False)
            vecerr[0] = xmaxerr
            print(problem.space_steps, problem.time_steps)
            for i in range(1, iterations):
                if initial_time_steps is None:
                    spec_time_steps = None
                else:
                    spec_time_steps = problem.time_steps*4
                problem = problem_class(space_steps=problem.space_steps * 2, time_steps=spec_time_steps, params=params)
                u = self.run_weno(problem, trainable=trainable, vectorized=True, just_one_time_step=False)
                u_last = u[:, -1]
                xmaxerr = problem.err(u_last) #, first_step=False)
                vecerr[i] = xmaxerr
                order[i - 1] = np.log(vecerr[i - 1] / vecerr[i]) / np.log(2)
                print(problem.space_steps, problem.time_steps)
        else:
            u = self.run_weno(problem, trainable=trainable, vectorized=True, just_one_time_step=False)
            u_last = u[:, -1]
            u_last = u_last.detach().numpy()
            fine_space_steps = initial_space_steps*2*2*2*2*2
            if initial_time_steps is None:
                fine_time_steps = None
            else:
                fine_time_steps = initial_time_steps*4*4*4*4*4
            problem_fine = problem_class(space_steps=fine_space_steps, time_steps=fine_time_steps, params=params)
            m = problem.space_steps
            u_ex = self.run_weno(problem_fine, trainable=False, vectorized=True, just_one_time_step=False)
            u_ex_last = u_ex[:,-1]
            u_ex_last = u_ex_last.detach().numpy()
            divider = fine_space_steps/m
            divider = int(divider)
            u_ex_short = u_ex_last[0:fine_space_steps+1:divider]
            xerr = np.absolute(u_ex_short - u_last)
            xmaxerr = np.max(xerr)
            vecerr[0] = xmaxerr
            print(problem.space_steps, problem.time_steps)
            for i in range(1, iterations):
                if initial_time_steps is None:
                    spec_time_steps = None
                else:
                    spec_time_steps = problem.time_steps*4
                problem = problem_class(space_steps=problem.space_steps * 2, time_steps=spec_time_steps, params=params)
                m = problem.space_steps
                u = self.run_weno(problem, trainable=trainable, vectorized=True, just_one_time_step=False)
                u_last = u[:, -1]
                u_last = u_last.detach().numpy()
                divider = fine_space_steps / m
                divider = int(divider)
                u_ex_short = u_ex_last[0:fine_space_steps+1:divider]
                xerr = np.absolute(u_ex_short - u_last)
                xmaxerr = np.max(xerr)
                vecerr[i] = xmaxerr
                order[i - 1] = np.log(vecerr[i - 1] / vecerr[i]) / np.log(2)
                print(problem.space_steps, problem.time_steps)
        return vecerr, order




