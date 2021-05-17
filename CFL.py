import numpy as np
import matplotlib.pyplot as plt


u = np.linspace(0, 1, 10000)
C=0.5
G=2

u_funct = - ( (2*C+2)*G*u**5 + (-8*C-2)*G*u**4 + 12*C*G*u**3 + (2*C-8*C*G)*u**2 + (2*C*G-2*C)*u )/( (u**2+C*(1-u)**2)**2 )
plt.plot(u,u_funct)
u_max = np.max(u_funct)

# u_funct = (-2* (C**2* (-1 + G* (-1 + u)**4 - 2* u)* (1 - u)**2 + G *u**6 + C* u**2* (3 - 2* u+ G *(1 - u)**2 *(-3 - 2* u+ 2* u**2))))/(C *(1 - u)**2 + u**2)**3
# plt.plot(u,u_funct)
# def find_nearest(array, value):
#     array = np.asarray(array)
#     idx = (np.abs(array - value)).argmin()
#     return array[idx]
# value = 0
# nearest = find_nearest(u_funct, value)
# pos = np.where(u_funct == nearest)
# u_pos = u[pos]
# u_max = - ( (2*C+2)*G*u_pos**5 + (-8*C-2)*G*u_pos**4 + 12*C*G*u_pos**3 + (2*C-8*C*G)*u_pos**2 + (2*C*G-2*C)*u_pos )/( (u_pos**2+C*(1-u_pos)**2)**2 )

