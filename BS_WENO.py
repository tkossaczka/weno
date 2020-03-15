import numpy as np
from WENO5_minus import WENO5_minus
from WENO6 import WENO6
import torch

def BS_WENO(sigma,rate,E,T,e,xl,xr,m,omegas):
    #
    # TODO: omegas is [[omegas5_step1 ...], [omegas6_step1 ...]] or None. If list, standard weno is applied. if None, precomputed omegas are returned.

    Smin=np.exp(xl)*E; 
    Smax=np.exp(xr)*E; 

    G=np.log(Smin/E);
    L=np.log(Smax/E);
    theta=T;

    h=(-G+L)/m; 
    n=np.ceil((theta*sigma**2)/(0.8*(h**2))) 
    n=int(n)
    t=theta/n;

    x=np.linspace(G,L,m+1)
    time=np.linspace(0,theta,n+1); 

    u=np.zeros((x.shape[0],2));
    # TODO: use initial_condition method!

    for k in range(0,m+1):
        if x[k]>0:
            u[k,0]=1/E;  
        else:
            u[k,0]=0;

    for j in range(0,2):
        u[0,j]=0;
        u[1,j]=0;
        u[2,j]=0;
        u[m,j]=np.exp(-rate*time[j])/E;
        u[m-1,j]=np.exp(-rate*time[j])/E;
        u[m-2,j]=np.exp(-rate*time[j])/E;

    a=0;

    d1=np.exp(-rate*time)/E-t*rate*np.exp(-rate*time)/E;
    d2=np.exp(-rate*time)/E-t*rate*np.exp(-rate*time)/E;
    d3=np.exp(-rate*time)/E-t*rate*np.exp(-rate*time)/E;

    c1=np.exp(-rate*time)/E-0.5*t*rate*np.exp(-rate*time)/E+0.25*(t**2)*(rate**2)*np.exp(-rate*time)/E;
    c2=np.exp(-rate*time)/E-0.5*t*rate*np.exp(-rate*time)/E+0.25*(t**2)*(rate**2)*np.exp(-rate*time)/E;
    c3=np.exp(-rate*time)/E-0.5*t*rate*np.exp(-rate*time)/E+0.25*(t**2)*(rate**2)*np.exp(-rate*time)/E;

    l=1
    
    u = torch.Tensor(u)

    # TODO: if omegas is None, precompute omegas. else, use omegas from list
    RHSd=WENO6(u,m,l,e,omegas6_step1)
    RHSc=WENO5_minus(u,m,l,e,omegas5)

    u1=torch.zeros((x.shape[0]))[:, None]
    u1[3:m+1-3,0]=u[3:m+1-3,l-1]+t*((sigma**2)/(2*h**2)*RHSd+((rate-(sigma**2)/2)/h)*RHSc-rate*u[3:m+1-3,l-1]);

    u1[0:3,0]=torch.Tensor([a ,a ,a]);
    u1[m-2:m+1,0]=torch.Tensor([d1[l-1],d2[l-1] ,d3[l-1]]);
    # TODO: if omegas is None, precompute omegas. else, use omegas from list
    RHS1d=WENO6(u1,m,1,e,omegas6_step2)
    RHS1c=WENO5_minus(u1,m,1,e,omegas5)   
    
    u2=torch.zeros((x.shape[0]))[:, None]
    u2[3:m+1-3,0]=0.75*u[3:m+1-3,l-1]+0.25*u1[3:m+1-3,0]+0.25*t*((sigma**2)/(2*h**2)*RHS1d+((rate-(sigma**2)/2)/h)*RHS1c-rate*u1[3:m+1-3,0]);

    u2[0:3,0]=torch.Tensor([a, a ,a]);
    u2[m-2:m+1,0]=torch.Tensor([c1[l-1], c2[l-1] ,c3[l-1]]);
    # TODO: if omegas is None, precompute omegas. else, use omegas from list
    RHS2d=WENO6(u2,m,1,e,omegas6);
    RHS2c=WENO5_minus(u2,m,1,e,omegas5);

    u[3:m+1-3,l]=((1/3)*u[3:m+1-3,l-1]+(2/3)*u2[3:m+1-3,0]+(2/3)*t*((sigma**2)/(2*h**2)*RHS2d+((rate-(sigma**2)/2)/h)*RHS2c))-(2/3)*t*rate*u2[3:m+1-3,0];


    tt=T-time;
    S=E*np.exp(x);
    V=torch.zeros((m+1,2));
    for k in range(0,m+1):
        V[k,:]=E*u[k,:]

    # TODO: if omegas is None, return list of precomputed omegas. else, return V, S, tt
    return V[:,1],S,tt
