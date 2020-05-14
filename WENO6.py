
def WENO6(u,l,e, corrections, mweno=True, mapped=False):
    uu=u[:,l-1]
    uu_left = uu[:-1]
    uu_right = uu[1:]

    def get_fluxes(uu):
        flux0 = (uu[0:-5] - 3 * uu[1:-4] - 9 * uu[2:-3] + 11 * uu[3:-2]) / 12;
        flux1 = (uu[1:-4] - 15 * uu[2:-3] + 15 * uu[3:-2] - uu[4:-1]) / 12;
        flux2 = (-11 * uu[2:-3] + 9 * uu[3:-2] + 3 * uu[4:-1] - uu[5:]) / 12;
        return flux0, flux1, flux2

    fluxp0, fluxp1, fluxp2 = get_fluxes(uu_right)
    fluxn0, fluxn1, fluxn2 = get_fluxes(uu_left)

    def get_betas(uu):
        beta0 = 13/12*(uu[0:-5] -3*uu[1:-4] +3*uu[2:-3]  -uu[3:-2]  )**2 + 1/4*( uu[0:-5] -5*uu[1:-4] +7*uu[2:-3]  -3*uu[3:-2])**2;
        beta1 = 13/12*(uu[1:-4]  -3*uu[2:-3] +3*uu[3:-2]  -uu[4:-1] )**2 + 1/4*( uu[1:-4]  -uu[2:-3]   -uu[3:-2]  +uu[4:-1] )**2;
        beta2 = 13/12*(uu[2:-3]  -3*uu[3:-2] +3*uu[4:-1] -uu[5:])**2 + 1/4*(-3*uu[2:-3]+7*uu[3:-2] -5*uu[4:-1] +uu[5:])**2;
        return beta0, beta1, beta2

    betap0, betap1, betap2 = get_betas(uu_right)
    betan0, betan1, betan2 = get_betas(uu_left)



    beta_corrected_list=[]
    for k, beta in enumerate([betap0, betap1, betap2, betan0, betan1, betan2]):
        beta_corrected_list.append(beta + corrections[k])
    [betap0, betap1, betap2, betan0, betan1, betan2] = beta_corrected_list

    gamap0 = 1 / 21;
    gamap1 = 19 / 21;
    gamap2 = 1 / 21;
    gaman0 = 4 / 27;
    gaman1 = 19 / 27;
    gaman2 = 4 / 27;
    sigmap = 42 / 15;
    sigman = 27 / 15;

    def get_omegas_mweno(betas, gamas):
        beta_range_square = (betas[2] - betas[0]) ** 2
        return [gama / (e + beta) ** 2 * (beta_range_square + (e + beta) ** 2) for beta, gama in zip(betas, gamas)]

    def get_omegas_weno(betas, gamas):
        return [gama / (e + beta) ** 2 for beta, gama in zip(betas, gamas)]

    omegas_func_dict = {0: get_omegas_weno, 1: get_omegas_mweno}
    [omegapp_0, omegapp_1, omegapp_2] = omegas_func_dict[int(mweno)]([betap0, betap1, betap2], [gamap0, gamap1, gamap2])
    [omeganp_0, omeganp_1, omeganp_2] = omegas_func_dict[int(mweno)]([betap0, betap1, betap2], [gaman0, gaman1, gaman2])
    [omegapn_0, omegapn_1, omegapn_2] = omegas_func_dict[int(mweno)]([betan0, betan1, betan2], [gamap0, gamap1, gamap2])
    [omegann_0, omegann_1, omegann_2] = omegas_func_dict[int(mweno)]([betan0, betan1, betan2], [gaman0, gaman1, gaman2])

    def normalize(tensor_list):
        sum_ = sum(tensor_list)  # note, that inbuilt sum applies __add__ iteratively therefore its overloaded-
        return [tensor / sum_ for tensor in tensor_list]

    [omegapp0, omegapp1, omegapp2] = normalize([omegapp_0, omegapp_1, omegapp_2])
    [omeganp0, omeganp1, omeganp2] = normalize([omeganp_0, omeganp_1, omeganp_2])
    [omegapn0, omegapn1, omegapn2] = normalize([omegapn_0, omegapn_1, omegapn_2])
    [omegann0, omegann1, omegann2] = normalize([omegann_0, omegann_1, omegann_2])

    omegaps = [omegapp0, omegapp1, omegapp2, omegapn0, omegapn1, omegapn2]
    omegans = [omeganp0, omeganp1, omeganp2, omegann0, omegann1, omegann2]

    [omegap0, omegap1, omegap2, omegan0, omegan1, omegan2] = [sigmap*omegap-sigman*omegan
                                                              for omegap, omegan in zip(omegaps, omegans)]

    if mapped:
        d0=-2/15;
        d1=19/15;
        d2=-2/15;
        def get_alpha(omega, d):
            return (omega * (d + d ^ 2 - 3 * d * omega + omega ** 2)) / (d ^ 2 + omega * (1 - 2 * d));

        [alphap0, alphap1, alphap2] = [get_alpha(omega, d) for omega, d in zip([omegap0, omegap1, omegap2],
                                                                               [d0, d1, d2])]
        [alphan0, alphan1, alphan2] = [get_alpha(omega, d) for omega, d in zip([omegan0, omegan1, omegan2],
                                                                               [d0, d1, d2])]
        [omegap0, omegap1, omegap2] = normalize([alphap0, alphap1, alphap2])
        [omegan0, omegan1, omegan2] = normalize([alphan0, alphan1, alphan2])

    fluxp=omegap0*fluxp0+omegap1*fluxp1+omegap2*fluxp2;
    fluxn=omegan0*fluxn0+omegan1*fluxn1+omegan2*fluxn2;

    RHS=(fluxp-fluxn); #/h^2;
    

    return RHS
