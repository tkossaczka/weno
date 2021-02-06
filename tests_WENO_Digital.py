import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from define_problem_Digital import Digital_option
from define_WENO_Network_2 import WENONetwork_2
from scipy.stats import norm
from define_problem_Call import Call_option
from define_problem_Call_GS import Call_option_GS
from define_problem_Digital_GS import Digital_option_GS

torch.set_default_dtype(torch.float64)

# train_model = WENONetwork_2()
train_model = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Digital_Option_Test/Models/Model_17/7999.pt")

params=None
#params = {'sigma': 0.3, 'rate': 0.02, 'E': 50, 'T': 1, 'e': 1e-13, 'xl': -6, 'xr': 1.5, 'psi':20}
#params = {'sigma': 0.3, 'rate': 0.25, 'E': 50, 'T': 1, 'e': 1e-13, 'xl': -1.5, 'xr': 2, 'psi':30}
# params = {'sigma': 0.3, 'rate': 0.1, 'E': 50, 'T': 1, 'e': 1e-13, 'xl': -6, 'xr': 1.5}
# params = {'sigma': 0.2, 'rate': 0.08, 'E': 50, 'T': 1, 'e': 1e-13, 'xl': -6, 'xr': 1.5}
# params = {'sigma': 0.33405, 'rate': 0.266078, 'E': 50, 'T': 1, 'e': 1e-13, 'xl': -6, 'xr': 1.5}
# params = {'sigma': 0.29235803798039667,'rate': 0.23532633811960602,'E': 50,'T': 0.5,'e': 1e-13,'xl': -6,'xr': 1.5} # 80 space steps

problem= Digital_option

problem_main = problem(space_steps=100, time_steps=None, params = params)
params = problem_main.get_params()

u_init, nn = train_model.init_run_weno(problem_main, vectorized=True, just_one_time_step=False)
# parameters needed for the computation of exact solution
params_main = problem_main.params
rate = params_main['rate']
sigma = params_main['sigma']
T = params_main['T']
E = params_main['E']
x, time = problem_main.x, problem_main.time
h = problem_main.h
tt = T - time
S = E * np.exp(x)
u_t = u_init
with torch.no_grad():
    for k in range(nn):
        u_t = train_model.run_weno(problem_main, u_t, mweno=True, mapped=False, vectorized=True, trainable=True, k=k)
    V_t, _, _ = problem_main.transformation(u_t)
u_nt = u_init
for k in range(nn):
    u_nt = train_model.run_weno(problem_main, u_nt, mweno=True, mapped=False, vectorized=True, trainable=False, k=k)
V_nt, _, _ = problem_main.transformation(u_nt)


V_ex = np.exp(-rate * (T - tt[nn])) * norm.cdf( (np.log(S / E) + (rate - (sigma ** 2) / 2) * (T - tt[nn])) / (sigma * np.sqrt(T - tt[nn])))
u_ex = V_ex/E

error_t_mean = np.sqrt(7.5 / 100) * (np.sqrt(np.sum((u_t.detach().numpy() - u_ex) ** 2)))
error_nt_mean = np.sqrt(7.5 / 100) * (np.sqrt(np.sum((u_nt.detach().numpy() - u_ex) ** 2)))
error_nt_max = np.max(np.absolute(u_ex - u_nt.detach().numpy()))
error_t_max = np.max(np.absolute(u_ex - u_t.detach().numpy()))

err_mat = np.zeros((4,1))
err_mat[0,:] = np.array(error_nt_max)
err_mat[1,:] = np.array(error_t_max)
err_mat[2,:] = np.array(error_nt_mean)
err_mat[3,:] = np.array(error_t_mean)

train_model.compare_wenos(problem_main)

plt.figure(2)
plt.plot(S,V_nt,S,V_t,S, V_ex)

# TEST GREEKS
e = problem_main.params['e']
der2_nt = train_model.WENO6(u_nt, e, mweno=True, mapped=False, trainable=False)
der1_nt = train_model.WENO5(u_nt, e, w5_minus='Lax-Friedrichs', mweno=True, mapped=False, trainable=False)
gamma_nt = E * (der2_nt / h**2 * (1 / S[3:-3]**2) - der1_nt / h * (1 / S[3:-3]**2))
plt.figure(3)
plt.plot(S[3:-3], gamma_nt)

der2_t = train_model.WENO6(u_t, e, mweno=True, mapped=False, trainable=True)
der1_t = train_model.WENO5(u_t, e, w5_minus='Lax-Friedrichs', mweno=True, mapped=False, trainable=True)
der2_t = der2_t.detach().numpy()
der1_t = der1_t.detach().numpy()
gamma_t = E * (der2_t / h**2 * (1 / S[3:-3]**2) - der1_t / h * (1 / S[3:-3]**2))
plt.plot(S[3:-3], gamma_t)


# VV = V.detach().numpy()
# X, Y = np.meshgrid(x, t, indexing="ij")
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(X, Y, VV, cmap=cm.viridis)


