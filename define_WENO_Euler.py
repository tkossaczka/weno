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

    def comp_eigenvectors_matrix(self, problem, q):
        gamma = problem.params['gamma']
        R_right = torch.zeros(q.shape[0]-5,3,3)
        R_left = torch.zeros(q.shape[0]-5,3,3)
        rho_left = q[1:-4,0]
        rho_right = q[2:-3,0]
        rho_sqrt_left = torch.sqrt(q[1:-4,0])
        rho_sqrt_right = torch.sqrt(q[2:-3,0])
        E_left = q[1:-4,2]
        E_right = q[2:-3,2]
        rho_u_left = q[1:-4,1]
        rho_u_right = q[2:-3,1]
        p_left = (gamma-1)*(E_left - 0.5*(rho_u_left**2)/rho_left)
        p_right = (gamma - 1) * (E_right - 0.5 * (rho_u_right ** 2) / rho_right)
        u = (rho_u_left/rho_sqrt_left + rho_u_right/rho_sqrt_right) / (rho_sqrt_left + rho_sqrt_right)
        H = ((E_left+p_left)/rho_sqrt_left + (E_right+p_right)/rho_sqrt_right) / (rho_sqrt_left + rho_sqrt_right)
        c = torch.sqrt( (gamma-1)*(H-0.5*u**2) )
        b1 = (gamma-1)/c**2
        b2 = 0.5*(u**2)*b1
        for i in range(0,q.shape[0]-5):
            R_right[i,:,:] = torch.Tensor([[1., 1., 1.],[u[i]-c[i], u[i], u[i]+c[i]],[H[i]-u[i]*c[i], 0.5*u[i]**2, H[i]+u[i]*c[i]]])
            R_left[i,:,:] = 0.5*torch.Tensor([[b2[i]+u[i]/c[i], -(b1[i]*u[i]+1/c[i]), b1[i]],[2*(1-b2[i]), 2*b1[i]*u[i], -2*b1[i]],[b2[i]-u[i]/c[i], -(b1[i]*u[i]-1/c[i]), b1[i]]])
        return R_left, R_right

    def WENO5_char(self, q, R_left, R_right, flux, lamb, e, w5_minus, mweno=True, mapped=False, trainable=True):
        RHS = torch.zeros((q.shape[0]-6,3))
        for i in range(q.shape[0]-7):
            q_l_p = torch.matmul(R_left[i+1,:,:],q[i+1:i + 8, :].T)
            flux_l_p = torch.matmul(R_left[i+1,:,:],flux[i+1:i+8,:].T)
            q_l_n = torch.matmul(R_left[i, :, :], q[i:i + 7, :].T)
            flux_l_n = torch.matmul(R_left[i, :, :], flux[i:i + 7, :].T)
            if w5_minus is True:
                uu_p = 0.5 * (flux_l_p - lamb * q_l_p)
                uu_n = 0.5 * (flux_l_n - lamb * q_l_n)
            else:
                uu_p = 0.5 * (flux_l_p + lamb * q_l_p)
                uu_n = 0.5 * (flux_l_n + lamb * q_l_n)
            uu_p = uu_p.T
            uu_n = uu_n.T
            uu_p_left = uu_p[:-1]
            uu_p_right = uu_p[1:]
            uu_n_left = uu_n[:-1]
            uu_n_right = uu_n[1:]

            def get_fluxes(uu):
                if w5_minus is True:
                    flux0 = (11 * uu[3] - 7 * uu[4] + 2 * uu[5]) / 6
                    flux1 = (2 * uu[2] + 5 * uu[3] - uu[4]) / 6
                    flux2 = (-uu[1] + 5 * uu[2] + 2 * uu[3]) / 6
                else:
                    flux0 = (2 * uu[0] - 7 * uu[1] + 11 * uu[2]) / 6
                    flux1 = (- uu[1] + 5 * uu[2] + 2* uu[3]) / 6
                    flux2 = (2*uu[2] + 5 * uu[3] - uu[4]) / 6
                return flux0, flux1, flux2

            fluxp0, fluxp1, fluxp2 = get_fluxes(uu_p_left)
            fluxn0, fluxn1, fluxn2 = get_fluxes(uu_n_left)

            def get_betas(uu):
                if w5_minus is True:
                    beta0 = 13 / 12 * (uu[3] - 2 * uu[4] + uu[5]) ** 2 + 1 / 4 * (
                                3 * uu[3] - 4 * uu[4] + uu[5]) ** 2
                    beta1 = 13 / 12 * (uu[2] - 2 * uu[3] + uu[4]) ** 2 + 1 / 4 * (uu[2] - uu[4]) ** 2
                    beta2 = 13 / 12 * (uu[1] - 2 * uu[2] + uu[3]) ** 2 + 1 / 4 * (
                                uu[1] - 4 * uu[2] + 3 * uu[3]) ** 2
                else:
                    beta0 = 13 / 12 * (uu[0] - 2 * uu[1] + uu[2]) ** 2 + 1 / 4 * (
                            uu[0] - 4 * uu[1] + 3*uu[2]) ** 2
                    beta1 = 13 / 12 * (uu[1] - 2 * uu[2] + uu[3]) ** 2 + 1 / 4 * (uu[1] - uu[3]) ** 2
                    beta2 = 13 / 12 * (uu[2] - 2 * uu[3] + uu[4]) ** 2 + 1 / 4 * (
                            3*uu[2] - 4 * uu[3] + uu[4]) ** 2
                return beta0, beta1, beta2

            betap0, betap1, betap2 = get_betas(uu_p_left)
            betan0, betan1, betan2 = get_betas(uu_n_left)

            # if trainable:
            #     dif = self.__get_average_diff(uu)
            #     dif = self.prepare_dif(dif)
            #     beta_multiplicators = self.inner_nn_weno5(dif)[0, :, :].T + self.weno5_mult_bias
            #     # beta_multiplicators_left = beta_multiplicators[:-1]
            #     # beta_multiplicators_right = beta_multiplicators[1:]
            #
            #     betap_corrected_list = []
            #     betan_corrected_list = []
            #     for k, beta in enumerate([betap0, betap1, betap2]):
            #         shift = k -1
            #         betap_corrected_list.append(beta * (beta_multiplicators[3+shift:-3+shift]))
            #     for k, beta in enumerate([betan0, betan1, betan2]):
            #         shift = k - 1
            #         betan_corrected_list.append(beta * (beta_multiplicators[3+shift:-3+shift]))
            #     [betap0, betap1, betap2] = betap_corrected_list
            #     [betan0, betan1, betan2] = betan_corrected_list

            d0 = 1 / 10
            d1 = 6 / 10
            d2 = 3 / 10

            def get_omegas_mweno(betas, ds):
                beta_range_square = (betas[2] - betas[0]) ** 2
                return [d / (e + beta) ** 2 * (beta_range_square + (e + beta) ** 2) for beta, d in zip(betas, ds)]

            def get_omegas_weno(betas, ds):
                return [d / (e + beta) ** 2 for beta, d in zip(betas, ds)]

            omegas_func_dict = {0: get_omegas_weno, 1: get_omegas_mweno}
            [omegap_0, omegap_1, omegap_2] = omegas_func_dict[int(mweno)]([betap0, betap1, betap2], [d0, d1, d2])
            [omegan_0, omegan_1, omegan_2] = omegas_func_dict[int(mweno)]([betan0, betan1, betan2], [d0, d1, d2])

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
            fluxp_r = torch.matmul(R_right[i+1, :, :], fluxp.T)
            fluxn_r = torch.matmul(R_right[i, :, :], fluxn.T)

            RHS[i,:] = fluxp_r - fluxn_r

        return RHS

    def init_Euler(self,problem, vectorized, just_one_time_step):
        m = problem.space_steps
        h = problem.h
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

        return q_0, q_1, q_2, lamb, nn, h

    def run_weno(self, problem, mweno, mapped, method, q_0, q_1, q_2, lamb, trainable, vectorized, k, dt):
        gamma = problem.params['gamma']
        n, t, h = problem.time_steps, problem.t, problem.h
        if dt == None:
            t=t
        else:
            t=dt
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
            if method == "char":
                R_left, R_right = self.comp_eigenvectors_matrix(problem, q)
                RHSc_p = self.WENO5_char(q, R_left, R_right, flux, lamb, e, w5_minus=False, mweno=mweno, mapped=mapped,trainable=trainable)
                RHSc_n = self.WENO5_char(q, R_left, R_right, flux, lamb, e, w5_minus=True, mweno=mweno, mapped=mapped,trainable=trainable)
                RHSc = RHSc_p + RHSc_n
            elif method == "comp":
                RHSc_p = self.WENO5(0.5 * (flux + lamb * q), e, w5_minus=False, mweno=mweno, mapped=mapped, trainable=trainable)
                RHSc_n = self.WENO5(0.5 * (flux - lamb * q), e, w5_minus=True, mweno=mweno, mapped=mapped, trainable=trainable)
                RHSc = RHSc_p + RHSc_n
            u1[3:-3, :] = q[3:-3, :] + t * (-(1 / h) * RHSc)
            u1[0:3, :] = q0[0:3, :]
            u1[-3:, :] = q0[-3:, :]

            u2 = torch.zeros(q_0.shape[0],3)
            flux = self.flux_func(problem, u1)  # NEVADI TU PREPISOVANIE? OVERIT!
            if method == "char":
                R_left, R_right = self.comp_eigenvectors_matrix(problem, u1) # ANI TU TO NEVADI???
                RHS1c_p = self.WENO5_char(u1, R_left, R_right, flux, lamb, e, w5_minus=False, mweno=mweno, mapped=mapped, trainable=trainable)
                RHS1c_n = self.WENO5_char(u1, R_left, R_right, flux, lamb, e, w5_minus=True, mweno=mweno, mapped=mapped, trainable=trainable)
                RHS1c = RHS1c_p + RHS1c_n
            elif method == "comp":
                RHS1c_p = self.WENO5(0.5 * (flux + lamb * u1), e, w5_minus=False, mweno=mweno, mapped=mapped,trainable=trainable)
                RHS1c_n = self.WENO5(0.5 * (flux - lamb * u1), e, w5_minus=True, mweno=mweno, mapped=mapped,trainable=trainable)
                RHS1c = RHS1c_p + RHS1c_n
            u2[3:-3, :] = 0.75 * q[3:-3, :] + 0.25 * u1[3:-3, :] + 0.25 * t * ( - (1/ h) * RHS1c)
            u2[0:3, :] = q0[0:3, :]
            u2[-3:, :] = q0[-3:, :]

            q_ret = torch.zeros(q_0.shape[0], 3)
            flux = self.flux_func(problem, u2)
            if method == "char":
                R_left, R_right = self.comp_eigenvectors_matrix(problem, u2)
                RHS2c_p = self.WENO5_char(u2, R_left, R_right, flux, lamb, e, w5_minus=False, mweno=mweno, mapped=mapped, trainable=trainable)
                RHS2c_n = self.WENO5_char(u2, R_left, R_right, flux, lamb, e, w5_minus=True, mweno=mweno, mapped=mapped, trainable=trainable)
                RHS2c = RHS2c_p + RHS2c_n
            elif method == "comp":
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