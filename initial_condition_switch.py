import numpy as np
from scipy import signal

def init_cond(numb, x):
    #Lx = x[-1] - x[0]
    xmid = 1

    if numb==1: # Gaussian wave for
        u0 = np.exp(-20 * (x - xmid)**2)
    elif numb==2: # Sinusoidal wave
        u0 = np.sin(np.pi*x)
    elif numb==3: # Shifted sinusoidal wave
        u0 = 0.5 - np.sin(np.pi*x)
    elif numb==4: # Riemann problem
        u0 = np.zeros(x.shape[0])
        u0[x >= xmid] = 1
    elif numb==5: # Signal square
        u0 = signal.square(2 * np.pi * x)
    elif numb==6: # Square Jump
        u0 = np.zeros(x.shape[0])
        for k in range(0, x.shape[0]):
            if x[k] >= xmid - 0.3 and x[k] <= xmid + 0.3:
                u0[k] = 1
    elif numb==7: # Oleg's trapezoidal left
        u0 = np.zeros(x.shape[0])
        for k in range(0, x.shape[0] ):
            if x[k] >= xmid-0.3 and x[k] <= xmid + 0.3:
                u0[k] = np.exp(x[k])
    elif numb==8: # Oleg's trapezoidal right
        u0 = np.zeros(x.shape[0])
        for k in range(0, x.shape[0] ):
            if x[k] >= xmid-0.3 and x[k] <= xmid + 0.3:
                u0[k] = np.exp(-x[k])

    return u0