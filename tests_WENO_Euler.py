import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from define_WENO_Network import WENONetwork
from define_WENO_Euler import WENONetwork_Euler
from define_Euler_system import Euler_system

train_model = WENONetwork_Euler()
train_model = torch.load('model7')
torch.set_default_dtype(torch.float64)
params=None
problem = Euler_system
problem_main = problem(space_steps=128, time_steps=None, params = params)
params = problem_main.get_params()
gamma = params['gamma']

q_0, q_1, q_2, lamb, nn = train_model.init_Euler(problem_main, vectorized = True, just_one_time_step=False)

for k in range(nn):
    q_0, q_1, q_2, lamb = train_model.run_weno(problem_main, mweno = True, mapped = False, q_0=q_0, q_1=q_1, q_2=q_2, lamb=lamb, vectorized=True, trainable = True, k=k)
_,x,t = problem_main.transformation(q_0)

q_0_np = q_0.detach().numpy()
q_1_np = q_1.detach().numpy()
q_2_np = q_2.detach().numpy()

# X, Y = np.meshgrid(x, t, indexing="ij")
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(X, Y, q_1_np/q_0_np, cmap=cm.viridis)

x_ex = np.linspace(0, 1, 128+1)
p_ex = torch.zeros((x_ex.shape[0],t.shape[0]))
rho_ex = torch.zeros((x_ex.shape[0],t.shape[0]))
u_ex = torch.zeros((x_ex.shape[0],t.shape[0]))
c_ex = torch.zeros((x_ex.shape[0],t.shape[0]))
mach_ex = torch.zeros((x_ex.shape[0],t.shape[0]))

for k in range(0,t.shape[0]):
    p_ex[:,k], rho_ex[:,k], u_ex[:,k], c_ex[:,k], mach_ex[:,k] = problem_main.exact(x_ex, t[k])

# X, Y = np.meshgrid(x_ex, t, indexing="ij")
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(X, Y, u_ex, cmap=cm.viridis)

rho = q_0
u = q_1/rho
E = q_2
p = (gamma - 1)*(E-0.5*rho*u**2)

plt.figure(1)
plt.plot(x,rho.detach().numpy(),x_ex,rho_ex[:,-1].detach().numpy())
plt.figure(2)
plt.plot(x,p.detach().numpy(),x_ex,p_ex[:,-1].detach().numpy())
plt.figure(3)
plt.plot(x,u.detach().numpy(),x_ex,u_ex[:,-1].detach().numpy())

