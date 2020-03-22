
def WENO5_minus(u, l, e, corrections, mweno=True, mapped=False):
    uu=u[:,l-1];
    uu_left = uu[:-1]
    uu_right = uu[1:]

    def get_fluxes(uu):
        flux0 = (11 * uu[3:-2] - 7 * uu[4:-1] + 2 * uu[5:]) / 6;
        flux1 = (2 * uu[2:-3] + 5 * uu[3:-2] - uu[4:-1]) / 6;
        flux2 = (-uu[1:-4] + 5 * uu[2:-3] + 2 * uu[3:-2]) / 6;
        return flux0, flux1, flux2

    fluxp0, fluxp1, fluxp2 = get_fluxes(uu_right)
    fluxn0, fluxn1, fluxn2 = get_fluxes(uu_left)

    def get_betas(uu):
        beta0 = 13/12*(uu[3:-2] -2*uu[4:-1] +uu[5:])**2 + 1/4*( 3*uu[3:-2] -4*uu[4:-1] +uu[5:])**2;
        beta1 = 13/12*(uu[2:-3] -2*uu[3:-2] +uu[4:-1])**2 + 1/4*( uu[2:-3] -uu[4:-1])**2;
        beta2 = 13/12*(uu[1:-4]  -2*uu[2:-3] +uu[3:-2])**2 + 1/4*(uu[1:-4] -4*uu[2:-3] +3*uu[3:-2])**2;
        return beta0, beta1, beta2

    betap0, betap1, betap2 = get_betas(uu_right)
    betan0, betan1, betan2 = get_betas(uu_left)

    beta_corrected_list=[]
    for k, beta in enumerate([betap0, betap1, betap2, betan0, betan1, betan2]):
        beta_corrected_list.append(beta + corrections[:,k])
    [betap0, betap1, betap2, betan0, betan1, betan2] = beta_corrected_list

    d0=1/10;
    d1=6/10;
    d2=3/10;

    def get_omegas_mweno(betas, ds):
        beta_range_square = (betas[2] - betas[0])**2
        return [d/(e+beta)**2*(beta_range_square + (e+beta)**2) for beta, d in zip(betas, ds)]

    def get_omegas_weno(betas, ds):
        return [d/(e+beta)**2 for beta, d in zip(betas, ds)]

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
            return (omega*(d+d^2-3*d*omega+omega**2))/(d^2+omega*(1-2*d));

        [alphap0, alphap1, alphap2] = [get_alpha(omega, d) for omega, d in zip([omegap0, omegap1, omegap2],
                                                                               [d0, d1, d2])]
        [alphan0, alphan1, alphan2] = [get_alpha(omega, d) for omega, d in zip([omegan0, omegan1, omegan2],
                                                                               [d0, d1, d2])]

        [omegap0, omegap1, omegap2] = normalize([alphap0, alphap1, alphap2])
        [omegan0, omegan1, omegan2] = normalize([alphan0, alphan1, alphan2])

    fluxp=omegap0*fluxp0+omegap1*fluxp1+omegap2*fluxp2;
    fluxn=omegan0*fluxn0+omegan1*fluxn1+omegan2*fluxn2;

    RHS=(fluxp-fluxn); #/h;

    return RHS
