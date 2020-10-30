import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from define_WENO_Network import WENONetwork
from define_Euler_system import Euler_system

train_model = WENONetwork()
#train_model = torch.load('C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Models/Model_18/46')
torch.set_default_dtype(torch.float64)
params=None
problem = Euler_system
problem_main = problem(space_steps=100, time_steps=None, params = params)
params = problem_main.get_params()

q_0, q_1, q_2, rho, u, p = train_model.run_weno_Euler(problem_main, vectorized=True, trainable = False, just_one_time_step = False)
_,x,t = problem_main.transformation(u)
# plt.figure(1)
# plt.plot(x,r)
# plt.figure(2)
# plt.plot(x,u)
# plt.figure(3)
# plt.plot(x,p)

q_0_np = q_0.detach().numpy()
q_1_np = q_1.detach().numpy()
q_2_np = q_2.detach().numpy()

X, Y = np.meshgrid(x, t, indexing="ij")
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, q_1_np/q_0_np, cmap=cm.viridis)

x_ex, p_ex, rho_ex, u_ex, c_ex, mach_ex = problem_main.exact()

plt.figure(1)
plt.plot(x,rho,x_ex,rho_ex)
plt.figure(2)
plt.plot(x,p,x_ex,p_ex)
plt.figure(3)
plt.plot(x,u,x_ex,u_ex)

