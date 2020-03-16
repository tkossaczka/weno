import numpy as np

def WENO6(u,m,l,e,mweno=True, mapped=False):
    uu=u[:,l-1]
    uu_left = uu[:-1]
    uu_right = uu[1:]

    def get_fluxes(uu):
        flux0 = (uu[0:-5] - 3 * uu[1:-4] - 9 * uu[2:-3] + 11 * uu[3:-2]) / 12;
        flux1 = (uu[1:-4] - 15 * uu[2:-3] + 15 * uu[3:-2] - uu[4:-1]) / 12;
        flux2 = (-11 * uu[2:-3] + 9 * uu[3:-2] + 3 * uu[4:-1] - uu[5:]) / 12;
        return flux0, flux1, flux2

    # fluxp0 = (  uu[1:-5] - 3*uu[2:-4] - 9*uu[3:-3]  + 11*uu[4:-2])/12;
    # fluxp1 = (   uu[2:-4] - 15*uu[3:-3]+ 15*uu[4:-2] - uu[5:-1]  )/12;
    # fluxp2 = (-11*uu[3:-3]+ 9*uu[4:-2] + 3*uu[5:-1] - uu[6:] )/12;
    #
    # fluxn0 = (  uu[0:-6] - 3*uu[1:-5] - 9*uu[2:-4] + 11*uu[3:-3])/12;
    # fluxn1 = (   uu[1:-5] - 15*uu[2:-4] + 15*uu[3:-3]- uu[4:-2]  )/12;
    # fluxn2 = (-11*uu[2:-4] + 9*uu[3:-3]  + 3*uu[4:-2] - uu[5:-1] )/12;

    fluxp0, fluxp1, fluxp2 = get_fluxes(uu_right)
    fluxn0, fluxn1, fluxn2 = get_fluxes(uu_left)

    def get_betas(uu):
        beta0 = 13/12*(uu[0:-5] -3*uu[1:-4] +3*uu[2:-3]  -uu[3:-2]  )**2 + 1/4*( uu[0:-5] -5*uu[1:-4] +7*uu[2:-3]  -3*uu[3:-2])**2;
        beta1 = 13/12*(uu[1:-4]  -3*uu[2:-3] +3*uu[3:-2]  -uu[4:-1] )**2 + 1/4*( uu[1:-4]  -uu[2:-3]   -uu[3:-2]  +uu[4:-1] )**2;
        beta2 = 13/12*(uu[2:-3]  -3*uu[3:-2] +3*uu[4:-1] -uu[5:])**2 + 1/4*(-3*uu[2:-3]+7*uu[3:-2] -5*uu[4:-1] +uu[5:])**2;
        return beta0, beta1, beta2

        # betap0 = 13/12*(uu[1:-5] -3*uu[2:-4] +3*uu[3:-3]  -uu[4:-2]  )**2 + 1/4*( uu[1:-5] -5*uu[2:-4] +7*uu[3:-3]  -3*uu[4:-2])**2;
        # betap1 = 13/12*(uu[2:-4]  -3*uu[3:-3] +3*uu[4:-2]  -uu[5:-1] )**2 + 1/4*( uu[2:-4]  -uu[3:-3]   -uu[4:-2]    +uu[5:-1] )**2;
        # betap2 = 13/12*(uu[3:-3]  -3*uu[4:-2] +3*uu[5:-1] -uu[6:])**2 + 1/4*(-3*uu[3:-3]+7*uu[4:-2] -5*uu[5:-1] +uu[6:])**2;
        #
        # betan0 = 13/12*(uu[0:-6] -3*uu[1:-5] +3*uu[2:-4] -uu[3:-3] )**2 + 1/4*( uu[0:-6] -5*uu[1:-5] +7*uu[2:-4] -3*uu[3:-3])**2;
        # betan1 = 13/12*(uu[1:-5]  -3*uu[2:-4]  +3*uu[3:-3] -uu[4:-2] )**2 + 1/4*( uu[1:-5]  -uu[2:-4]    -uu[3:-3]   +uu[4:-2] )**2;
        # betan2 = 13/12*(uu[2:-4]   -3*uu[3:-3]  +3*uu[4:-2] -uu[5:-1])**2 + 1/4*(-3*uu[2:-4] +7*uu[3:-3]  -5*uu[4:-2] +uu[5:-1])**2;

    betap0, betap1, betap2 = get_betas(uu_right)
    betan0, betan1, betan2 = get_betas(uu_left)

    gamap0 = 1 / 21;
    gamap1 = 19 / 21;
    gamap2 = 1 / 21;
    gaman0 = 4 / 27;
    gaman1 = 19 / 27;
    gaman2 = 4 / 27;
    sigmap = 42 / 15;
    sigman = 27 / 15;

        # MWENO

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

        # omegapp_0=(gamap0/(e+betap0)**2)*((betap2-betap0)**2 + (e+betap0)**2);
        # omegapp_1=(gamap1/(e+betap1)**2)*((betap2-betap0)**2 + (e+betap1)**2);
        # omegapp_2=(gamap2/(e+betap2)**2)*((betap2-betap0)**2 + (e+betap2)**2);
        # omeganp_0=(gaman0/(e+betap0)**2)*((betap2-betap0)**2 + (e+betap0)**2);
        # omeganp_1=(gaman1/(e+betap1)**2)*((betap2-betap0)**2 + (e+betap1)**2);
        # omeganp_2=(gaman2/(e+betap2)**2)*((betap2-betap0)**2 + (e+betap2)**2);
        #
        # omegapn_0=(gamap0/(e+betan0)**2)*((betan2-betan0)**2 + (e+betan0)**2);
        # omegapn_1=(gamap1/(e+betan1)**2)*((betan2-betan0)**2 + (e+betan1)**2);
        # omegapn_2=(gamap2/(e+betan2)**2)*((betan2-betan0)**2 + (e+betan2)**2);
        # omegann_0=(gaman0/(e+betan0)**2)*((betan2-betan0)**2 + (e+betan0)**2);
        # omegann_1=(gaman1/(e+betan1)**2)*((betan2-betan0)**2 + (e+betan1)**2);
        # omegann_2=(gaman2/(e+betan2)**2)*((betan2-betan0)**2 + (e+betan2)**2);

        # WENO
        # omegapp_0=(gamap0/(e+betap0)**2);
        # omegapp_1=(gamap1/(e+betap1)**2);
        # omegapp_2=(gamap2/(e+betap2)**2);
        # omeganp_0=(gaman0/(e+betap0)**2);
        # omeganp_1=(gaman1/(e+betap1)**2);
        # omeganp_2=(gaman2/(e+betap2)**2);
        # 
        # omegapn_0=(gamap0/(e+betan0)**2);
        # omegapn_1=(gamap1/(e+betan1)**2);
        # omegapn_2=(gamap2/(e+betan2)**2);
        # omegann_0=(gaman0/(e+betan0)**2);
        # omegann_1=(gaman1/(e+betan1)**2);
        # omegann_2=(gaman2/(e+betan2)**2);

        ##

    def normalize(tensor_list):
        sum_ = sum(tensor_list)  # note, that inbuilt sum applies __add__ iteratively therefore its overloaded-
        return [tensor / sum_ for tensor in tensor_list]

    [omegapp0, omegapp1, omegapp2] = normalize([omegapp_0, omegapp_1, omegapp_2])
    [omeganp0, omeganp1, omeganp2] = normalize([omeganp_0, omeganp_1, omeganp_2])
    [omegapn0, omegapn1, omegapn2] = normalize([omegapn_0, omegapn_1, omegapn_2])
    [omegann0, omegann1, omegann2] = normalize([omegann_0, omegann_1, omegann_2])

        # omegapp0=omegapp_0/(omegapp_0+omegapp_1+omegapp_2);
        # omegapp1=omegapp_1/(omegapp_0+omegapp_1+omegapp_2);
        # omegapp2=omegapp_2/(omegapp_0+omegapp_1+omegapp_2);
        # omeganp0=omeganp_0/(omeganp_0+omeganp_1+omeganp_2);
        # omeganp1=omeganp_1/(omeganp_0+omeganp_1+omeganp_2);
        # omeganp2=omeganp_2/(omeganp_0+omeganp_1+omeganp_2);
        #
        # omegapn0=omegapn_0/(omegapn_0+omegapn_1+omegapn_2);
        # omegapn1=omegapn_1/(omegapn_0+omegapn_1+omegapn_2);
        # omegapn2=omegapn_2/(omegapn_0+omegapn_1+omegapn_2);
        # omegann0=omegann_0/(omegann_0+omegann_1+omegann_2);
        # omegann1=omegann_1/(omegann_0+omegann_1+omegann_2);
        # omegann2=omegann_2/(omegann_0+omegann_1+omegann_2);

    omegaps = [omegapp0, omegapp1, omegapp2, omegapn0, omegapn1, omegapn2]
    omegans = [omeganp0, omeganp1, omeganp2, omegann0, omegann1, omegann2]

    [omegap0, omegap1, omegap2, omegan0, omegan1, omegan2] = [sigmap*omegap-sigman*omegan
                                                              for omegap, omegan in zip(omegaps, omegans)]
    # omegap0=sigmap*omegapp0-sigman*omeganp0;
    # omegap1=sigmap*omegapp1-sigman*omeganp1;
    # omegap2=sigmap*omegapp2-sigman*omeganp2;
    # omegan0=sigmap*omegapn0-sigman*omegann0;
    # omegan1=sigmap*omegapn1-sigman*omegann1;
    # omegan2=sigmap*omegapn2-sigman*omegann2;


        ## MWENO s povodnymi d-vahami
        # d0=-2/15;
        # d1=19/15;
        # d2=-2/15;
        # 
        # omegapp_0=d0/(e+betap0)**2*((betap2-betap0)**2 + (e+betap0)**2);
        # omegapp_1=d1/(e+betap1)**2*((betap2-betap0)**2 + (e+betap1)**2);
        # omegapp_2=d2/(e+betap2)**2*((betap2-betap0)**2 + (e+betap2)**2);
        # 
        # omegapn_0=d0/(e+betan0)**2*((betan2-betan0)**2 + (e+betan0)**2);
        # omegapn_1=d1/(e+betan1)**2*((betan2-betan0)**2 + (e+betan1)**2);
        # omegapn_2=d2/(e+betan2)**2*((betan2-betan0)**2 + (e+betan2)**2);
        #     
        # omegap0=omegapp_0/(omegapp_0+omegapp_1+omegapp_2);
        # omegap1=omegapp_1/(omegapp_0+omegapp_1+omegapp_2);
        # omegap2=omegapp_2/(omegapp_0+omegapp_1+omegapp_2); 
        #     
        # omegan0=omegapn_0/(omegapn_0+omegapn_1+omegapn_2);
        # omegan1=omegapn_1/(omegapn_0+omegapn_1+omegapn_2);
        # omegan2=omegapn_2/(omegapn_0+omegapn_1+omegapn_2); 

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

        ## Mapped WENO
        # d0=-2/15;
        # d1=19/15;
        # d2=-2/15;
        # 
        # alphap0=(omegap0*(d0+d0^2-3*d0*omegap0+omegap0**2))/(d0^2+omegap0*(1-2*d0));
        # alphap1=(omegap1*(d1+d1^2-3*d1*omegap1+omegap1**2))/(d1^2+omegap1*(1-2*d1));
        # alphap2=(omegap2*(d2+d2^2-3*d2*omegap2+omegap2**2))/(d2^2+omegap2*(1-2*d2));
        # 
        # alphan0=(omegan0*(d0+d0^2-3*d0*omegan0+omegan0**2))/(d0^2+omegan0*(1-2*d0));
        # alphan1=(omegan1*(d1+d1^2-3*d1*omegan1+omegan1**2))/(d1^2+omegan1*(1-2*d1));
        # alphan2=(omegan2*(d2+d2^2-3*d2*omegan2+omegan2**2))/(d2^2+omegan2*(1-2*d2));
        # 
        # omegap00=alphap0/(alphap0+alphap1+alphap2);
        # omegap11=alphap1/(alphap0+alphap1+alphap2);
        # omegap22=alphap2/(alphap0+alphap1+alphap2);
        #     
        # omegan00=alphan0/(alphan0+alphan1+alphan2);
        # omegan11=alphan1/(alphan0+alphan1+alphan2);
        # omegan22=alphan2/(alphan0+alphan1+alphan2);
        #       
        # fluxp=omegap00*fluxp0+omegap11*fluxp1+omegap22*fluxp2;
        # fluxn=omegan00*fluxn0+omegan11*fluxn1+omegan22*fluxn2;


    fluxp=omegap0*fluxp0+omegap1*fluxp1+omegap2*fluxp2;
    fluxn=omegan0*fluxn0+omegan1*fluxn1+omegan2*fluxn2;

    RHS=(fluxp-fluxn); #/h^2;
    

    return RHS
