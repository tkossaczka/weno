import numpy as np
from WENO5_minus import WENO5_minus
from WENO6 import WENO6
import torch
from helper_functions import autocor_combinations
from helper import compute_combinations

def BS_WENO(sigma,rate,E,T,e,xl,xr,m, weights5, weights6):

    def compute_corrections(u, weights5, weights6):
        autocors5, autocors6 = compute_combinations(u, lag=6)
        bn5_0 = autocors5[0].matmul(weights5[:,0])
        bn5_1 = autocors5[1].matmul(weights5[:,1])
        bn5_2 = autocors5[2].matmul(weights5[:,2])
        bp5_0 = autocors5[3].matmul(weights5[:,3])
        bp5_1 = autocors5[4].matmul(weights5[:,4])
        bp5_2 = autocors5[5].matmul(weights5[:,5])

        bn6_0 = autocors6[0].matmul(weights6[:,0])
        bn6_1 = autocors6[1].matmul(weights6[:,1])
        bn6_2 = autocors6[2].matmul(weights6[:,2])
        bp6_0 = autocors6[3].matmul(weights6[:,3])
        bp6_1 = autocors6[4].matmul(weights6[:,4])
        bp6_2 = autocors6[5].matmul(weights6[:,5])

        multip5=[bn5_0, bn5_1, bn5_2, bp5_0, bp5_1,bp5_2]
        multip6 = [bn6_0, bn6_1, bn6_2, bp6_0, bp6_1, bp6_2]

        return multip5, multip6

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

    cor5, cor6 = compute_corrections(u[:,0], weights5,weights6)
    RHSd=WENO6(u,l,e, cor6, mweno=True, mapped=False)
    RHSc=WENO5_minus(u,l,e, cor5, mweno=True, mapped=False)

    u1=torch.zeros((x.shape[0]))[:, None]
    u1[3:-3,0]=u[3:-3,l-1]+t*((sigma**2)/(2*h**2)*RHSd+((rate-(sigma**2)/2)/h)*RHSc-rate*u[3:-3,l-1]);

    u1[0:3,0]=torch.Tensor([a ,a ,a]);
    u1[m-2:,0]=torch.Tensor([d1[l-1],d2[l-1] ,d3[l-1]]);

    cor5_1, cor6_1 = compute_corrections(u1[:, 0], weights5, weights6)
    RHS1d=WENO6(u1,1,e, cor6_1, mweno=True, mapped=False)
    RHS1c=WENO5_minus(u1,1,e, cor5_1, mweno=True, mapped=False)
    
    u2=torch.zeros((x.shape[0]))[:, None]
    u2[3:-3,0]=0.75*u[3:-3,l-1]+0.25*u1[3:-3,0]+0.25*t*((sigma**2)/(2*h**2)*RHS1d+((rate-(sigma**2)/2)/h)*RHS1c-rate*u1[3:-3,0]);

    u2[0:3,0]=torch.Tensor([a, a ,a]);
    u2[m-2:,0]=torch.Tensor([c1[l-1], c2[l-1] ,c3[l-1]]);

    cor5_2, cor6_2 = compute_corrections(u2[:, 0], weights5, weights6)
    RHS2d=WENO6(u2,1,e, cor6_2, mweno=True, mapped=False);
    RHS2c=WENO5_minus(u2,1,e, cor5_2, mweno=True, mapped=False);

    u[3:-3,l]=((1/3)*u[3:-3,l-1]+(2/3)*u2[3:-3,0]+(2/3)*t*((sigma**2)/(2*h**2)*RHS2d+((rate-(sigma**2)/2)/h)*RHS2c))-(2/3)*t*rate*u2[3:-3,0];


    tt=T-time;
    S=E*np.exp(x);
    V=torch.zeros((m+1,2));
    for k in range(0,m+1):
        V[k,:]=E*u[k,:]

    return V[:,1],S,tt
