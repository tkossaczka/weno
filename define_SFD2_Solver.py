import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

class SFD2_Solver():
    def __init__(self):
        return

    def SFD2(self, uu):
        RHSd = uu[:-2]-2*uu[1:-1]+uu[2:]
        return RHSd

    def SFD1(self, uu):
        RHSc = -0.5*uu[:-2]+0.5*uu[2:]
        return RHSc

    def run_sfd(self, problem, vectorized):
        m = problem.space_steps
        n, t, h = problem.time_steps, problem.t, problem.h
        x, time = problem.x, problem.time
        term_2 = problem.der_2()
        term_1 = problem.der_1()
        term_0 = problem.der_0()
        term_const = problem.der_const()
        u_bc_l, u_bc_r, u1_bc_l, u1_bc_r, u2_bc_l, u2_bc_r = problem.boundary_condition

        if vectorized:
            u = np.zeros((m+1,1))
        else:
            u = np.zeros((m+1, n+1))

        u[:,0] = problem.initial_condition

        for l in range(1, n+1):
            if vectorized:
                ll=1
            else:
                ll=l

            uu_conv = problem.funct_convection(u[:, ll - 1])
            uu_diff = problem.funct_diffusion(u[:, ll - 1])
            RHSd = self.SFD2(uu_diff)
            RHSc = self.SFD1(uu_conv)

            u1 = np.zeros(x.shape[0])
            u1[1:-1] = u[1:-1, ll - 1] + t * ((term_2 / h ** 2) * RHSd - (term_1 / h) * RHSc + term_0 * u[1:-1, ll - 1])

            u1[0:3] = u1_bc_l[:,l - 1]
            u1[m - 2:] = u1_bc_r[:,l - 1]

            uu1_conv = problem.funct_convection(u1)
            uu1_diff = problem.funct_diffusion(u1)
            RHS1d = self.SFD2(uu1_diff)
            RHS1c = self.SFD1(uu1_conv)

            u2 = np.zeros(x.shape[0])
            u2[1:-1] = 0.75*u[1:-1,ll-1]+0.25*u1[1:-1]+0.25*t*((term_2/h ** 2)*RHS1d-(term_1/h)*RHS1c+term_0*u1[1:-1])

            u2[0:3] = u2_bc_l[:,l - 1]
            u2[m - 2:] = u2_bc_r[:,l - 1]

            uu2_conv = problem.funct_convection(u2)
            uu2_diff = problem.funct_diffusion(u2)
            RHS2d = self.SFD2(uu2_diff)
            RHS2c = self.SFD1(uu2_conv)

            if vectorized:
                u[1:-1, 0] = (1 / 3) * u[1:-1, ll - 1] + (2 / 3) * u2[1:-1] + (2 / 3) * t * (
                        (term_2 / h ** 2) * RHS2d - (term_1 / h) * RHS2c + term_0 * u2[1:-1])
                u[0:3, 0] = u_bc_l[:, l]
                u[m - 2:, 0] = u_bc_r[:, l]
            else:
                u[1:-1, l] = (1 / 3) * u[1:-1, ll - 1] + (2 / 3) * u2[1:-1] + (2 / 3) * t * (
                        (term_2 / h ** 2) * RHS2d - (term_1 / h) * RHS2c + term_0 * u2[1:-1])
                u[0:3, l] = u_bc_l[:, l]
                u[m - 2:, l] = u_bc_r[:, l]

        return u

    def full_WENO(self, problem, plot=True, vectorized=False):
        u = self.run_weno(problem, vectorized=vectorized)
        V, S, tt = problem.transformation(u)
        V = V.detach().numpy()
        if plot:
            X, Y = np.meshgrid(S, tt, indexing="ij")
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.plot_surface(X, Y, V, cmap=cm.viridis)
        return V, S, tt

    def compare(self, problem):
        n = problem.time_steps
        u = self.run_sfd(problem, vectorized=False)
        V, S, tt = problem.transformation(u)
        #plt.plot(S, V[:,1])
        plt.plot(S, V[:, n])

    def order_compute(self, iterations, initial_space_steps, params, problem_class):
        problem = problem_class(space_steps=initial_space_steps, time_steps=None, params=params)
        vecerr = np.zeros((iterations))[:, None]
        order = np.zeros((iterations - 1))[:, None]
        u = self.run_sfd(problem, vectorized=False)
        u_last = u[:,-1]
        xmaxerr = problem.err(u_last)
        vecerr[0] = xmaxerr
        print(problem.space_steps)
        for i in range(1, iterations):
            problem = problem_class(space_steps=problem.space_steps * 2, time_steps=None, params=params)
            u = self.run_sfd(problem, vectorized=False)
            u_last = u[:, -1]
            xmaxerr = problem.err(u_last)
            vecerr[i] = xmaxerr
            order[i - 1] = np.log(vecerr[i - 1] / vecerr[i]) / np.log(2)
            print(problem.space_steps)
        return vecerr, order



