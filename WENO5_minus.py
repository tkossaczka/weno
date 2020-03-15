import numpy as np

def WENO5_minus(u, m, l, e, mweno=True, mapped=False):
    uu=u[:,l-1];
    uu_left = uu[:-1]
    uu_right = uu[1:]

    def get_fluxes(uu):
        flux0 = (11 * uu[3:-2] - 7 * uu[4:-1] + 2 * uu[5:]) / 6;
        flux1 = (2 * uu[2:-3] + 5 * uu[3:-2] - uu[4:-1]) / 6;
        flux2 = (-uu[1:-4] + 5 * uu[2:-3] + 2 * uu[3:-2]) / 6;
        return flux0, flux1, flux2

    # fluxp0 = ( 11*uu[4:-2] - 7*uu[5:-1] + 2*uu[6:])/6;
    # fluxp1 = (   2*uu[3:-3] + 5*uu[4:-2] -uu[5:-1])/6;
    # fluxp2 = ( -uu[2:-4] + 5*uu[3:-3] +2*uu[4:-2])/6;
    #
    # fluxn0 = ( 11*uu[3:-3] - 7*uu[4:-2] + 2*uu[5:-1])/6;
    # fluxn1 = (   2*uu[2:-4] + 5*uu[3:-3] -uu[4:-2])/6;
    # fluxn2 = ( -uu[1:-5] + 5*uu[2:-4] +2*uu[3:-3])/6;

    fluxp0, fluxp1, fluxp2 = get_fluxes(uu_right)
    fluxn0, fluxn1, fluxn2 = get_fluxes(uu_left)

    def get_betas(uu):
        beta0 = 13/12*(uu[3:-2] -2*uu[4:-1] +uu[5:])**2 + 1/4*( 3*uu[3:-2] -4*uu[4:-1] +uu[5:])**2;
        beta1 = 13/12*(uu[2:-3] -2*uu[3:-2] +uu[4:-1])**2 + 1/4*( uu[2:-3] -uu[4:-1])**2;
        beta2 = 13/12*(uu[1:-4]  -2*uu[2:-3] +uu[3:-2])**2 + 1/4*(uu[1:-4] -4*uu[2:-3] +3*uu[3:-2])**2;
        return beta0, beta1, beta2

    # betap0 = 13/12*(uu[4:-2] -2*uu[5:-1] +uu[6:])**2 + 1/4*( 3*uu[4:-2] -4*uu[5:-1] +uu[6:])**2;
    # betap1 = 13/12*(uu[3:-3]  -2*uu[4:-2]  +uu[5:-1])**2 + 1/4*( uu[3:-3] -uu[5:-1])**2;
    # betap2 = 13/12*(uu[2:-4]   -2*uu[3:-3] +uu[4:-2])**2 + 1/4*(uu[2:-4] -4*uu[3:-3] +3*uu[4:-2])**2;
    #
    # betan0 = 13/12*(uu[3:-3] -2*uu[4:-2] +uu[5:-1])**2 + 1/4*( 3*uu[3:-3] -4*uu[4:-2] +uu[5:-1])**2;
    # betan1 = 13/12*(uu[2:-4]  -2*uu[3:-3]  +uu[4:-2])**2 + 1/4*( uu[2:-4] -uu[4:-2])**2;
    # betan2 = 13/12*(uu[1:-5]   -2*uu[2:-4] +uu[3:-3])**2 + 1/4*(uu[1:-5] -4*uu[2:-4] +3*uu[3:-3])**2;

    betap0, betap1, betap2 = get_betas(uu_right)
    betan0, betan1, betan2 = get_betas(uu_left)

    # betap0 = 13/12*(uu[1:-5] -2*uu[2:-4] +uu[3:-3])**2 + 1*( uu[1:-5] -3*uu[2:-4] +2*uu[3:-3])**2;
    # betap1 = 13/12*(uu[2:-4]  -2*uu[3:-3]  +uu[4:-2])**2 + 1*( -uu[2:-4] +uu[4:-2])**2;
    # betap2 = 13/12*(uu[3:-3]   -2*uu[4:-2] +uu[5:-1])**2 + 1*(-uu[3:-3] +uu[4:-2])**2;
    #
    # betan0 = 13/12*(uu[0:-6] -2*uu[1:-5] +uu[2:-4])**2 + 1*( uu[0:-6] -3*uu[1:-5] +2*uu[2:-4])**2;
    # betan1 = 13/12*(uu[1:-5]  -2*uu[2:-4]  +uu[3:-3])**2 + 1*( -uu[1:-5] +uu[3:-3])**2;
    # betan2 = 13/12*(uu[2:-4]   -2*uu[3:-3] +uu[4:-2])**2 + 1*(-uu[2:-4] +uu[3:-3])**2;

    d0=1/10;
    d1=6/10;
    d2=3/10;

    ## MWENO
    def get_omegas_mweno(betas, ds):
        beta_range_square = (betas[2] - betas[0])**2
        return [d/(e+beta)**2*(beta_range_square + (e+beta)**2) for beta, d in zip(betas, ds)]

    def get_omegas_weno(betas, ds):
        return [d/(e+beta)**2 for beta, d in zip(betas, ds)]

    omegas_func_dict = {0: get_omegas_weno, 1: get_omegas_mweno}
    [omegap_0, omegap_1, omegap_2] = omegas_func_dict[int(mweno)]([betap0, betap1, betap2], [d0, d1, d2])
    [omegan_0, omegan_1, omegan_2] = omegas_func_dict[int(mweno)]([betan0, betan1, betan2], [d0, d1, d2])
    # omegap_0=d0/(e+betap0)**2*((betap2-betap0)**2 + (e+betap0)**2);
    # omegap_1=d1/(e+betap1)**2*((betap2-betap0)**2 + (e+betap1)**2);
    # omegap_2=d2/(e+betap2)**2*((betap2-betap0)**2 + (e+betap2)**2);
    # 
    # omegan_0=d0/(e+betan0)**2*((betan2-betan0)**2 + (e+betan0)**2);
    # omegan_1=d1/(e+betan1)**2*((betan2-betan0)**2 + (e+betan1)**2);
    # omegan_2=d2/(e+betan2)**2*((betan2-betan0)**2 + (e+betan2)**2);

    ## WENO
    # omegap_0=d0/(e+betap0)**2; 
    # omegap_1=d1/(e+betap1)**2;
    # omegap_2=d2/(e+betap2)**2;
    # 
    # omegan_0=d0/(e+betan0)**2; 
    # omegan_1=d1/(e+betan1)**2;
    # omegan_2=d2/(e+betan2)**2; 

    ##
    def normalize(tensor_list):
        sum_ = sum(tensor_list)  # note, that inbuilt sum applies __add__ iteratively therefore its overloaded-
        return [tensor / sum_ for tensor in tensor_list]
    
    [omegap0, omegap1, omegap2] = normalize([omegap_0, omegap_1, omegap_2])
    [omegan0, omegan1, omegan2] = normalize([omegan_0, omegan_1, omegan_2])
    # omegap0=omegap_0/(omegap_0+omegap_1+omegap_2);
    # omegap1=omegap_1/(omegap_0+omegap_1+omegap_2);
    # omegap2=omegap_2/(omegap_0+omegap_1+omegap_2); 
    #     
    # omegan0=omegan_0/(omegan_0+omegan_1+omegan_2);
    # omegan1=omegan_1/(omegan_0+omegan_1+omegan_2);
    # omegan2=omegan_2/(omegan_0+omegan_1+omegan_2); 
    
    if mapped:
        # Mapped WENO
        def get_alpha(omega, d):
            return (omega*(d+d^2-3*d*omega+omega**2))/(d^2+omega*(1-2*d));

        [alphap0, alphap1, alphap2] = [get_alpha(omega, d) for omega, d in zip([omegap0, omegap1, omegap2],
                                                                               [d0, d1, d2])]
        [alphan0, alphan1, alphan2] = [get_alpha(omega, d) for omega, d in zip([omegan0, omegan1, omegan2],
                                                                               [d0, d1, d2])]
        
        # alphap0=(omegap0*(d0+d0^2-3*d0*omegap0+omegap0**2))/(d0^2+omegap0*(1-2*d0));
        # alphap1=(omegap1*(d1+d1^2-3*d1*omegap1+omegap1**2))/(d1^2+omegap1*(1-2*d1));
        # alphap2=(omegap2*(d2+d2^2-3*d2*omegap2+omegap2**2))/(d2^2+omegap2*(1-2*d2));
        #
        # alphan0=(omegan0*(d0+d0^2-3*d0*omegan0+omegan0**2))/(d0^2+omegan0*(1-2*d0));
        # alphan1=(omegan1*(d1+d1^2-3*d1*omegan1+omegan1**2))/(d1^2+omegan1*(1-2*d1));
        # alphan2=(omegan2*(d2+d2^2-3*d2*omegan2+omegan2**2))/(d2^2+omegan2*(1-2*d2));

        [omegap0, omegap1, omegap2] = normalize([alphap0, alphap1, alphap2])
        [omegan0, omegan1, omegan2] = normalize([alphan0, alphan1, alphan2])

        # omegap00=alphap0/(alphap0+alphap1+alphap2);
        # omegap11=alphap1/(alphap0+alphap1+alphap2);
        # omegap22=alphap2/(alphap0+alphap1+alphap2);
        #
        # omegan00=alphan0/(alphan0+alphan1+alphan2);
        # omegan11=alphan1/(alphan0+alphan1+alphan2);
        # omegan22=alphan2/(alphan0+alphan1+alphan2);

    fluxp=omegap0*fluxp0+omegap1*fluxp1+omegap2*fluxp2;
    fluxn=omegan0*fluxn0+omegan1*fluxn1+omegan2*fluxn2;

    RHS=(fluxp-fluxn); #/h;

    return RHS
