import torch

def WENO_loop(self, sigma, rate, E, T, e, xl, xr, m, trainable, max_steps=1, comp_order=False):
    for l in range(1, steps):
        u = torch.Tensor(u)

        RHSd = self.WENO6(u, l, e, mweno=True, mapped=False, trainable=trainable)
        RHSc = self.WENO5_minus(u, l, e, mweno=True, mapped=False, trainable=trainable)

        u1 = torch.zeros((x.shape[0]))[:, None]
        u1[3:-3, 0] = u[3:-3, l-1] + t*((problem.der_2(x, t)/h**2)*RHSd + (term_1/h)*RHSc + term_0* u[3:-3,l-1])

        u1[0:3, 0] = torch.Tensor([a1[l - 1], a2[l - 1], a3[l - 1]])
        u1[m - 2:, 0] = torch.Tensor([d1[l - 1], d2[l - 1], d3[l - 1]])

        RHS1d = self.WENO6(u1, 1, e, mweno=True, mapped=False, trainable=trainable)
        RHS1c = self.WENO5_minus(u1, 1, e, mweno=True, mapped=False, trainable=trainable)

        u2 = torch.zeros((x.shape[0]))[:, None]
        u2[3:-3, 0] = 0.75 * u[3:-3, l - 1] + 0.25 * u1[3:-3, 0] + 0.25 * t * (
            (term_2/h**2)*RHS1d + (term_1/h) * RHS1c + term_0* u1[3:-3, 0])

        u2[0:3, 0] = torch.Tensor([b1[l - 1], b2[l - 1], b3[l - 1]])
        u2[m - 2:, 0] = torch.Tensor([c1[l - 1], c2[l - 1], c3[l - 1]])

        RHS2d = self.WENO6(u2, 1, e, mweno=True, mapped=False, trainable=trainable)
        RHS2c = self.WENO5_minus(u2, 1, e, mweno=True, mapped=False, trainable=trainable)

        u[3:-3, l] = (1 / 3) * u[3:-3, l - 1] + (2 / 3) * u2[3:-3, 0] + (2 / 3) * t * (
                (term_2/h**2) * RHS2d + (term_1/h) * RHS2c + term_0* u2[3:-3, 0])

    return u
