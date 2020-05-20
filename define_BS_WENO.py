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
        #params = self.get_params()
        #u_init=self.initial_condition()

        self.inner_nn_weno5 = nn.Sequential(
            nn.Conv1d(1, 6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(6, 3, kernel_size=5, stride=1, padding=2),
            nn.Sigmoid())

        self.inner_nn_weno6 = nn.Sequential(
            nn.Conv1d(1, 6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(6, 3, kernel_size=5, stride=1, padding=2),
            nn.Sigmoid())

    def get_params(self):
        params = dict()
        params["sigma"] = 0.31 + max(0.1 * np.random.randn(), -0.3)
        params["rate"] = 0.21 + max(0.1 * np.random.randn(), -0.2)
        params["E"] = 50
        params["T"] = 1
        params["e"] = 10 ** (-13)
        params["xl"] = -6
        params["xr"] = 1.5
        params["m"] = 160
        return params

    def initial_condition(self):
        params = self.get_params()
        sigma = params['sigma']
        rate = params['rate']
        E = params['E']
        T = params['T']
        e = params['e']
        xl = params['xl']
        xr = params['xr']
        m = params['m']
        Smin = np.exp(xl) * E
        Smax = np.exp(xr) * E
        G = np.log(Smin / E)
        L = np.log(Smax / E)
        theta = T
        h = (-G + L) / m
        n = np.ceil((theta * sigma ** 2) / (0.8 * (h ** 2)))
        n = int(n)
        t = theta / n
        x = np.linspace(G, L, m + 1)
        time = np.linspace(0, theta, n + 1)
        u = torch.zeros((x.shape[0],time.shape[0]))
        for k in range(0, m+1):
            if x[k] > 0:
                u[k, 0] = 1/E
            else:
                u[k, 0] = 0

        u_init = u[:,0]

        for j in range(0, n+1):
            u[0, j] = 0
            u[1, j] = 0
            u[2, j] = 0
            u[m, j] = np.exp(-rate * time[j]) / E
            u[m - 1, j] = np.exp(-rate * time[j]) / E
            u[m - 2, j] = np.exp(-rate * time[j]) / E

        d1 = np.exp(-rate * time) / E - t * rate * np.exp(-rate * time) / E
        d2 = np.exp(-rate * time) / E - t * rate * np.exp(-rate * time) / E
        d3 = np.exp(-rate * time) / E - t * rate * np.exp(-rate * time) / E

        c1 = np.exp(-rate * time) / E - 0.5 * t * rate * np.exp(-rate * time) / E + 0.25 * (t ** 2) * (
                rate ** 2) * np.exp(-rate * time) / E
        c2 = np.exp(-rate * time) / E - 0.5 * t * rate * np.exp(-rate * time) / E + 0.25 * (t ** 2) * (
                rate ** 2) * np.exp(-rate * time) / E
        c3 = np.exp(-rate * time) / E - 0.5 * t * rate * np.exp(-rate * time) / E + 0.25 * (t ** 2) * (
                rate ** 2) * np.exp(-rate * time) / E

        boundary = [d1, d2, d3, c1, c2, c3]

        return params, u, boundary, h, t, x, time, n

    def forward(self):
        params = self.get_params()
        V, S, tt,_ = self.BS_WENO(params["sigma"], params["rate"], params["E"], params["T"], params["e"], params["xl"],
                           params["xr"], params["m"], trainable=True)
        print(params["sigma"], params["rate"])
        return V

    def compare_wenos(self, params=None):
        if params is None:
            params = self.get_params()
        V_trained, S, tt,_ = self.BS_WENO(params["sigma"], params["rate"], params["E"], params["T"], params["e"], params["xl"],
                           params["xr"], params["m"], trainable=True)
        V_classic, S, tt,_ = self.BS_WENO(params["sigma"], params["rate"], params["E"], params["T"], params["e"], params["xl"],
                           params["xr"], params["m"], trainable=False)

        plt.plot(S, V_classic.detach().numpy()[:,1], S, V_trained.detach().numpy()[:,1])

    def full_WENO(self, trainable=True, params=None, plot=True):
        if params is None:
            params = self.get_params()
        V_trained, S, tt,_ = self.BS_WENO(params["sigma"], params["rate"], params["E"], params["T"], params["e"],
                                        params["xl"],params["xr"], params["m"], trainable=trainable, max_steps=None)
        V = V_trained.detach().numpy()
        if plot:
            X, Y = np.meshgrid(S, tt, indexing="ij")
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.plot_surface(X, Y, V, cmap=cm.viridis)

        return V, S, tt, params

    def return_S_tt(self):
        params = self.get_params()
        V, S, tt,_ = self.BS_WENO(params["sigma"], params["rate"], params["E"], params["T"], params["e"], params["xl"],
                           params["xr"], params["m"], trainable=True)
        return S, tt

    def WENO5_minus(self, u, l, e, mweno=True, mapped=False, trainable=True):
        uu = u[:, l - 1]
        uu_left = uu[:-1]
        uu_right = uu[1:]

        def get_fluxes(uu):
            flux0 = (11 * uu[3:-2] - 7 * uu[4:-1] + 2 * uu[5:]) / 6
            flux1 = (2 * uu[2:-3] + 5 * uu[3:-2] - uu[4:-1]) / 6
            flux2 = (-uu[1:-4] + 5 * uu[2:-3] + 2 * uu[3:-2]) / 6
            return flux0, flux1, flux2

        fluxp0, fluxp1, fluxp2 = get_fluxes(uu_right)
        fluxn0, fluxn1, fluxn2 = get_fluxes(uu_left)

        def get_betas(uu):
            beta0 = 13 / 12 * (uu[3:-2] - 2 * uu[4:-1] + uu[5:]) ** 2 + 1 / 4 * (
                        3 * uu[3:-2] - 4 * uu[4:-1] + uu[5:]) ** 2
            beta1 = 13 / 12 * (uu[2:-3] - 2 * uu[3:-2] + uu[4:-1]) ** 2 + 1 / 4 * (uu[2:-3] - uu[4:-1]) ** 2
            beta2 = 13 / 12 * (uu[1:-4] - 2 * uu[2:-3] + uu[3:-2]) ** 2 + 1 / 4 * (
                        uu[1:-4] - 4 * uu[2:-3] + 3 * uu[3:-2]) ** 2
            return beta0, beta1, beta2

        betap0, betap1, betap2 = get_betas(uu_right)
        betan0, betan1, betan2 = get_betas(uu_left)

        if trainable:
            beta_multiplicators_left = self.inner_nn_weno5(uu_left[None, None, :])[0,:,:]
            beta_multiplicators_right = self.inner_nn_weno5(uu_right[None,None,:])[0,:,:]
            # beta_multiplicators = torch.cat([beta_multiplicators_right, beta_multiplicators_left], 0)

            betap_corrected_list = []
            betan_corrected_list = []
            for k, beta in enumerate([betap0, betap1, betap2]):
                betap_corrected_list.append(beta * beta_multiplicators_right[k, 3:-2])
            for k, beta in enumerate([betan0, betan1, betan2]):
                betan_corrected_list.append(beta * beta_multiplicators_left[k, 2:-3])
            [betap0, betap1, betap2] = betap_corrected_list
            [betan0, betan1, betan2] = betan_corrected_list

        d0 = 1 / 10;
        d1 = 6 / 10;
        d2 = 3 / 10;

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
                return (omega * (d + d ^ 2 - 3 * d * omega + omega ** 2)) / (d ^ 2 + omega * (1 - 2 * d))

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

    def WENO6(self, u, l, e, mweno=True, mapped=False, trainable=True):
        uu = u[:, l - 1]
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

        if trainable:
            beta_multiplicators_left = self.inner_nn_weno6(uu_left[None, None, :])[0,:,:]
            beta_multiplicators_right = self.inner_nn_weno6(uu_right[None,None,:])[0,:,:]
            # beta_multiplicators = torch.cat([beta_multiplicators_right, beta_multiplicators_left], 0)

            betap_corrected_list = []
            betan_corrected_list = []
            for k, beta in enumerate([betap0, betap1, betap2]):
                betap_corrected_list.append(beta * beta_multiplicators_right[k, 3:-2])
            for k, beta in enumerate([betan0, betan1, betan2]):
                betan_corrected_list.append(beta * beta_multiplicators_left[k, 2:-3])
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

        def get_omegas_mweno(betas, gamas):
            beta_range_square = (betas[2] - betas[0]) ** 2
            return [gama / (e + beta) ** 2 * (beta_range_square + (e + beta) ** 2) for beta, gama in zip(betas, gamas)]

        def get_omegas_weno(betas, gamas):
            return [gama / (e + beta) ** 2 for beta, gama in zip(betas, gamas)]

        omegas_func_dict = {0: get_omegas_weno, 1: get_omegas_mweno}
        [omegapp_0, omegapp_1, omegapp_2] = omegas_func_dict[int(mweno)]([betap0, betap1, betap2],
                                                                         [gamap0, gamap1, gamap2])
        [omeganp_0, omeganp_1, omeganp_2] = omegas_func_dict[int(mweno)]([betap0, betap1, betap2],
                                                                         [gaman0, gaman1, gaman2])
        [omegapn_0, omegapn_1, omegapn_2] = omegas_func_dict[int(mweno)]([betan0, betan1, betan2],
                                                                         [gamap0, gamap1, gamap2])
        [omegann_0, omegann_1, omegann_2] = omegas_func_dict[int(mweno)]([betan0, betan1, betan2],
                                                                         [gaman0, gaman1, gaman2])

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
                return (omega * (d + d ^ 2 - 3 * d * omega + omega ** 2)) / (d ^ 2 + omega * (1 - 2 * d))

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

    def BS_WENO(self, sigma, rate, E, T, e, xl, xr, m, trainable, max_steps=1,comp_order=False):

        Smin = np.exp(xl) * E
        Smax = np.exp(xr) * E

        G = np.log(Smin / E)
        L = np.log(Smax / E)
        theta = T

        h = (-G + L) / m
        n = np.ceil((theta * sigma ** 2) / (0.8 * (h ** 2)))
        n = int(n)
        t = theta / n

        x = np.linspace(G, L, m + 1)
        time = np.linspace(0, theta, n + 1)

        if max_steps is None:
            max_steps = n

        steps = min(max_steps, n)+1

        u = np.zeros((x.shape[0],steps))

        for k in range(0, m + 1):
            if x[k] > 0:
                u[k, 0] = 1 / E
            else:
                u[k, 0] = 0

        for j in range(0, steps):
            u[0, j] = 0
            u[1, j] = 0
            u[2, j] = 0
            u[m, j] = np.exp(-rate * time[j]) / E
            u[m - 1, j] = np.exp(-rate * time[j]) / E
            u[m - 2, j] = np.exp(-rate * time[j]) / E

        a = 0

        d1 = np.exp(-rate * time) / E - t * rate * np.exp(-rate * time) / E
        d2 = np.exp(-rate * time) / E - t * rate * np.exp(-rate * time) / E
        d3 = np.exp(-rate * time) / E - t * rate * np.exp(-rate * time) / E

        c1 = np.exp(-rate * time) / E - 0.5 * t * rate * np.exp(-rate * time) / E + 0.25 * (t ** 2) * (
                    rate ** 2) * np.exp(-rate * time) / E
        c2 = np.exp(-rate * time) / E - 0.5 * t * rate * np.exp(-rate * time) / E + 0.25 * (t ** 2) * (
                    rate ** 2) * np.exp(-rate * time) / E
        c3 = np.exp(-rate * time) / E - 0.5 * t * rate * np.exp(-rate * time) / E + 0.25 * (t ** 2) * (
                    rate ** 2) * np.exp(-rate * time) / E


        for l in range(1, steps):
            u = torch.Tensor(u)

            RHSd = self.WENO6(u, l, e, mweno=True, mapped=False, trainable=trainable)
            RHSc = self.WENO5_minus(u, l, e, mweno=True, mapped=False, trainable=trainable)

            u1 = torch.zeros((x.shape[0]))[:, None]
            u1[3:-3, 0] = u[3:-3, l - 1] + t * (
                        (sigma ** 2) / (2 * h ** 2) * RHSd + ((rate - (sigma ** 2) / 2) / h) * RHSc - rate * u[3:-3,
                                                                                                             l - 1])

            u1[0:3, 0] = torch.Tensor([a, a, a])
            u1[m - 2:, 0] = torch.Tensor([d1[l - 1], d2[l - 1], d3[l - 1]])

            RHS1d = self.WENO6(u1, 1, e, mweno=True, mapped=False, trainable=trainable)
            RHS1c = self.WENO5_minus(u1, 1, e, mweno=True, mapped=False, trainable=trainable)

            u2 = torch.zeros((x.shape[0]))[:, None]
            u2[3:-3, 0] = 0.75 * u[3:-3, l - 1] + 0.25 * u1[3:-3, 0] + 0.25 * t * (
                        (sigma ** 2) / (2 * h ** 2) * RHS1d + ((rate - (sigma ** 2) / 2) / h) * RHS1c - rate * u1[3:-3, 0])

            u2[0:3, 0] = torch.Tensor([a, a, a])
            u2[m - 2:, 0] = torch.Tensor([c1[l - 1], c2[l - 1], c3[l - 1]])

            RHS2d = self.WENO6(u2, 1, e, mweno=True, mapped=False, trainable=trainable)
            RHS2c = self.WENO5_minus(u2, 1, e, mweno=True, mapped=False, trainable=trainable)

            u[3:-3, l] = ((1 / 3) * u[3:-3, l - 1] + (2 / 3) * u2[3:-3, 0] + (2 / 3) * t * (
                        (sigma ** 2) / (2 * h ** 2) * RHS2d + ((rate - (sigma ** 2) / 2) / h) * RHS2c)) - (
                                     2 / 3) * t * rate * u2[3:-3, 0]

        tt = T - time
        S = E * np.exp(x)
        V = torch.zeros((m + 1, steps))
        for k in range(0, m + 1):
            V[k, :] = E * u[k, :]

        if comp_order:
            Digital = np.zeros((m+1, n+1))
            for k in range(0, n+1):
                for j in range(0, m+1):
                    Digital[j, k] = np.exp(-rate * (T - tt[k])) * norm.cdf((np.log(S[j] / E) + (rate - (sigma**2) / 2) * (T - tt[k])) / (sigma * np.sqrt(T - tt[k])))
            uDigital = Digital[:, n]/E
            u_last = u[:, n ].detach().numpy()
            xerr = np.absolute([uDigital - u_last])
            xmaxerr = np.max([xerr])
        else:
            xmaxerr=0

        return V, S, tt, xmaxerr

    def order_compute(self, params, mm, trainable=True):
        order_numb=5
        vecerr = np.zeros((order_numb))[:, None]
        order = np.zeros((order_numb-1))[:, None]
        _,_,_,xmaxerr = self.BS_WENO(params["sigma"], params["rate"], params["E"], params["T"], params["e"],
                                        params["xl"], params["xr"], mm, trainable=trainable, max_steps=None, comp_order=True)
        vecerr[0] = xmaxerr
        for i in range(1,order_numb):
            mm=mm*2
            _,_,_, xmaxerr = self.BS_WENO(params["sigma"], params["rate"], params["E"], params["T"], params["e"],
                                                 params["xl"], params["xr"], mm, trainable=trainable, max_steps=None,  comp_order=True)
            vecerr[i] = xmaxerr
            order[i - 1] = np.log(vecerr[i - 1] / vecerr[i]) / np.log(2)
            print(mm)

        return vecerr, order



# train_model=WENONetwork()
#
# V=train_model.forward()
#
# def monotonicity_loss(x):
#     return torch.sum(torch.max(x[:-1]-x[1:], torch.Tensor([0.0])))
#
# #optimizer = optim.SGD(train_model.parameters(), lr=0.001)
# optimizer = optim.Adam(train_model.parameters())
#
# S,tt=train_model.return_S_tt()
#
# V
#
# for k in range(1500):
#     # Forward path
#     V_train = train_model.forward()
#     # Train model:
#     optimizer.zero_grad()  # Clear gradients
#     loss = monotonicity_loss(V_train[:,1]) # Calculate loss
#     loss.backward()  # Backward pass
#     optimizer.step()  # Optimize weights
#     print(k, loss)
#
# S,tt = train_model.return_S_tt()
# #plt.plot(S, V_train.detach().numpy())
# print("number of parameters:", sum(p.numel() for p in train_model.parameters()))
# train_model.compare_wenos()
# #_,_,_,params = train_model.full_WENO()
# #g=train_model.parameters()
# #g.__next__()
