import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from define_WENO_Euler import WENONetwork_Euler
from define_Euler_system import Euler_system

train_model = WENONetwork_Euler()
torch.set_default_dtype(torch.float64)
params=None
problem = Euler_system
sp_st = 2048
init_cond = "shock_entropy"
problem_main = problem(space_steps=sp_st, init_cond = init_cond, time_steps=None, params = params, time_disc=None, init_mid=False, init_general=False)
params = problem_main.get_params()
gamma = params['gamma']
tt = problem_main.t

#R_left, R_right = train_model.comp_eigenvectors_matrix(problem_main, problem_main.initial_condition)
q_0, q_1, q_2, lamb, nn, h = train_model.init_Euler(problem_main, vectorized = False, just_one_time_step=False)
q_0_nt, q_1_nt, q_2_nt, lamb_nt = q_0, q_1, q_2, lamb

for k in range(nn):
    q_0_nt[:,k+1], q_1_nt[:,k+1], q_2_nt[:,k+1], lamb_nt = train_model.run_weno(problem_main, mweno = True, mapped = False, method="char", q_0=q_0_nt[:,k], q_1=q_1_nt[:,k], q_2=q_2_nt[:,k], lamb=lamb_nt, vectorized=True, trainable=False, k=k, dt=None)

_,x,t = problem_main.transformation(q_0)

q_0_nt = q_0_nt.detach().numpy()
q_1_nt = q_1_nt.detach().numpy()
q_2_nt = q_2_nt.detach().numpy()

rho_nt = q_0_nt
u_nt = q_1_nt/rho_nt
E_nt = q_2_nt
p_nt = (gamma - 1)*(E_nt-0.5*rho_nt*u_nt**2)

rho_ex = torch.tensor(rho_nt)
u_ex = torch.tensor(u_nt)
E_ex = torch.tensor(E_nt)
p_ex = torch.tensor(p_nt)


plt.figure(1)
plt.plot(x,rho_nt[:,-1])
plt.figure(2)
plt.plot(x,p_nt[:,-1])
plt.figure(3)
plt.plot(x,u_nt[:,-1])

x_ex = np.linspace(0, 1, 2048+1)
p_ex = torch.zeros((x_ex.shape[0],t.shape[0]))
rho_ex = torch.zeros((x_ex.shape[0],t.shape[0]))
u_ex = torch.zeros((x_ex.shape[0],t.shape[0]))

for k in range(0,t.shape[0]):
    p_ex[:,k], rho_ex[:,k], u_ex[:,k], _,_ = problem_main.exact(x_ex, t[k])

X, Y = np.meshgrid(x, t, indexing="ij")
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, rho_nt, cmap=cm.viridis)

# for k in range(x.shape[0]):
#     plt.plot(x,u_ex_0[:,k])

rho_ex_s = rho_ex[0:2048 + 1:8]
u_ex_s = u_ex[0:2048 + 1:8]
p_ex_s = p_ex[0:2048 + 1:8]
# q0_ex = torch.Tensor(q0_ex)
# q1_ex = torch.Tensor(q1_ex)
# q2_ex = torch.Tensor(q2_ex)
torch.save(rho_ex, "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Euler_System_Test/Shock_entropy_exact/rho_ex")
torch.save(u_ex, "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Euler_System_Test/Shock_entropy_exact/u_ex")
torch.save(p_ex, "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Euler_System_Test/Shock_entropy_exact/p_ex")
torch.save(E_ex, "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Euler_System_Test/Shock_entropy_exact/E_ex")
torch.save(x, "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Euler_System_Test/Shock_entropy_exact/x_ex")
torch.save(t, "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Euler_System_Test/Shock_entropy_exact/t_ex")






