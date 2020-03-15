import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

def compute_omegas5(u,m,l,e):
    uu=u[:,l-1];

    betap0 = 13/12*(uu[4:-2] -2*uu[5:-1] +uu[6:])**2 + 1/4*( 3*uu[4:-2] -4*uu[5:-1] +uu[6:])**2;
    betap1 = 13/12*(uu[3:-3]  -2*uu[4:-2]  +uu[5:-1])**2 + 1/4*( uu[3:-3] -uu[5:-1])**2;
    betap2 = 13/12*(uu[2:-4]   -2*uu[3:-3] +uu[4:-2])**2 + 1/4*(uu[2:-4] -4*uu[3:-3] +3*uu[4:-2])**2;

    betan0 = 13/12*(uu[3:-3] -2*uu[4:-2] +uu[5:-1])**2 + 1/4*( 3*uu[3:-3] -4*uu[4:-2] +uu[5:-1])**2;
    betan1 = 13/12*(uu[2:-4]  -2*uu[3:-3]  +uu[4:-2])**2 + 1/4*( uu[2:-4] -uu[4:-2])**2;
    betan2 = 13/12*(uu[1:-5]   -2*uu[2:-4] +uu[3:-3])**2 + 1/4*(uu[1:-5] -4*uu[2:-4] +3*uu[3:-3])**2;

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
    omegap_0=d0/(e+betap0)**2*((betap2-betap0)**2 + (e+betap0)**2);
    omegap_1=d1/(e+betap1)**2*((betap2-betap0)**2 + (e+betap1)**2);
    omegap_2=d2/(e+betap2)**2*((betap2-betap0)**2 + (e+betap2)**2);

    omegan_0=d0/(e+betan0)**2*((betan2-betan0)**2 + (e+betan0)**2);
    omegan_1=d1/(e+betan1)**2*((betan2-betan0)**2 + (e+betan1)**2);
    omegan_2=d2/(e+betan2)**2*((betan2-betan0)**2 + (e+betan2)**2);

    ## WENO
    # omegap_0=d0/(e+betap0)**2;
    # omegap_1=d1/(e+betap1)**2;
    # omegap_2=d2/(e+betap2)**2;
    #
    # omegan_0=d0/(e+betan0)**2;
    # omegan_1=d1/(e+betan1)**2;
    # omegan_2=d2/(e+betan2)**2;

    ##
    omegap0=omegap_0/(omegap_0+omegap_1+omegap_2);
    omegap1=omegap_1/(omegap_0+omegap_1+omegap_2);
    omegap2=omegap_2/(omegap_0+omegap_1+omegap_2);

    omegan0=omegan_0/(omegan_0+omegan_1+omegan_2);
    omegan1=omegan_1/(omegan_0+omegan_1+omegan_2);
    omegan2=omegan_2/(omegan_0+omegan_1+omegan_2);

    ## Mapped WENO
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

    #omegas = torch.zeros(m+1, 6)
    omegas=pad_sequence([omegap0,omegap1,omegap2,omegan0,omegan1,omegan2])

    return omegas
