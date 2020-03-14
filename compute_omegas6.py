import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

def compute_omegas6(u,m,l,e,omegas00):
    uu=u[:,l-1];

    if omegas00 is None:
    
        gamap0=1/21;
        gamap1=19/21;
        gamap2=1/21;
        gaman0=4/27;
        gaman1=19/27;
        gaman2=4/27;
        sigmap=42/15;
        sigman=27/15;

        betap0 = 13/12*(uu[1:m+1-5] -3*uu[2:m+1-4] +3*uu[3:m+1-3]  -uu[4:m+1-2]  )**2 + 1/4*( uu[1:m+1-5] -5*uu[2:m+1-4] +7*uu[3:m+1-3]  -3*uu[4:m+1-2])**2; 
        betap1 = 13/12*(uu[2:m+1-4]  -3*uu[3:m+1-3] +3*uu[4:m+1-2]  -uu[5:m+1-1] )**2 + 1/4*( uu[2:m+1-4]  -uu[3:m+1-3]   -uu[4:m+1-2]    +uu[5:m+1-1] )**2;
        betap2 = 13/12*(uu[3:m+1-3]  -3*uu[4:m+1-2] +3*uu[5:m+1-1] -uu[6:m+1-0])**2 + 1/4*(-3*uu[3:m+1-3]+7*uu[4:m+1-2] -5*uu[5:m+1-1] +uu[6:m+1-0])**2;

        betan0 = 13/12*(uu[0:m+1-6] -3*uu[1:m+1-5] +3*uu[2:m+1-4] -uu[3:m+1-3] )**2 + 1/4*( uu[0:m+1-6] -5*uu[1:m+1-5] +7*uu[2:m+1-4] -3*uu[3:m+1-3])**2; 
        betan1 = 13/12*(uu[1:m+1-5]  -3*uu[2:m+1-4]  +3*uu[3:m+1-3] -uu[4:m+1-2] )**2 + 1/4*( uu[1:m+1-5]  -uu[2:m+1-4]    -uu[3:m+1-3]   +uu[4:m+1-2] )**2;
        betan2 = 13/12*(uu[2:m+1-4]   -3*uu[3:m+1-3]  +3*uu[4:m+1-2] -uu[5:m+1-1])**2 + 1/4*(-3*uu[2:m+1-4] +7*uu[3:m+1-3]  -5*uu[4:m+1-2] +uu[5:m+1-1])**2;

        # MWENO
        omegapp_0=(gamap0/(e+betap0)**2)*((betap2-betap0)**2 + (e+betap0)**2);
        omegapp_1=(gamap1/(e+betap1)**2)*((betap2-betap0)**2 + (e+betap1)**2);
        omegapp_2=(gamap2/(e+betap2)**2)*((betap2-betap0)**2 + (e+betap2)**2);
        omeganp_0=(gaman0/(e+betap0)**2)*((betap2-betap0)**2 + (e+betap0)**2);
        omeganp_1=(gaman1/(e+betap1)**2)*((betap2-betap0)**2 + (e+betap1)**2);
        omeganp_2=(gaman2/(e+betap2)**2)*((betap2-betap0)**2 + (e+betap2)**2);

        omegapn_0=(gamap0/(e+betan0)**2)*((betan2-betan0)**2 + (e+betan0)**2);
        omegapn_1=(gamap1/(e+betan1)**2)*((betan2-betan0)**2 + (e+betan1)**2);
        omegapn_2=(gamap2/(e+betan2)**2)*((betan2-betan0)**2 + (e+betan2)**2);
        omegann_0=(gaman0/(e+betan0)**2)*((betan2-betan0)**2 + (e+betan0)**2);
        omegann_1=(gaman1/(e+betan1)**2)*((betan2-betan0)**2 + (e+betan1)**2);
        omegann_2=(gaman2/(e+betan2)**2)*((betan2-betan0)**2 + (e+betan2)**2);

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
        omegapp0=omegapp_0/(omegapp_0+omegapp_1+omegapp_2);
        omegapp1=omegapp_1/(omegapp_0+omegapp_1+omegapp_2);
        omegapp2=omegapp_2/(omegapp_0+omegapp_1+omegapp_2); 
        omeganp0=omeganp_0/(omeganp_0+omeganp_1+omeganp_2);
        omeganp1=omeganp_1/(omeganp_0+omeganp_1+omeganp_2);
        omeganp2=omeganp_2/(omeganp_0+omeganp_1+omeganp_2); 

        omegapn0=omegapn_0/(omegapn_0+omegapn_1+omegapn_2);
        omegapn1=omegapn_1/(omegapn_0+omegapn_1+omegapn_2);
        omegapn2=omegapn_2/(omegapn_0+omegapn_1+omegapn_2); 
        omegann0=omegann_0/(omegann_0+omegann_1+omegann_2);
        omegann1=omegann_1/(omegann_0+omegann_1+omegann_2);
        omegann2=omegann_2/(omegann_0+omegann_1+omegann_2); 

        omegap0=sigmap*omegapp0-sigman*omeganp0;
        omegap1=sigmap*omegapp1-sigman*omeganp1;
        omegap2=sigmap*omegapp2-sigman*omeganp2;
        omegan0=sigmap*omegapn0-sigman*omegann0;
        omegan1=sigmap*omegapn1-sigman*omegann1;
        omegan2=sigmap*omegapn2-sigman*omegann2;


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

        #omegas=torch.zeros((m+1,6))
        omegas=pad_sequence([omegap0,omegap1,omegap2,omegan0,omegan1,omegan2])

        # omegas[0] = omegap0
        # omegas[1] = omegap1
        # omegas[2] = omegap2
        # omegas[3] = omegan0
        # omegas[4] = omegan1
        # omegas[5] = omegan2

    else:
        omegap0 = omegas00[0]
        omegap1 = omegas00[1]
        omegap2 = omegas00[2]
        omegan0 = omegas00[3]
        omegan1 = omegas00[4]
        omegan2 = omegas00[5]
    
    return omegas
