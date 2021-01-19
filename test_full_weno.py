import torch
import matplotlib.pyplot as plt
from define_problem_Digital import Digital_option
from define_WENO_Network import WENONetwork
from define_problem_heat_eq import heat_equation
from define_problem_Call import Call_option
from define_problem_transport_eq import transport_equation
from define_problem_PME import PME
from define_problem_Buckley_Leverett import Buckley_Leverett

torch.set_default_dtype(torch.float64)

train_model = torch.load('model')

params=None
#params = {'T': 0.4, 'e': 1e-13, 'L': 1, 'R': 1, 'C': 0.2}
params = {'sigma': 0.3, 'rate': 0.1, 'E': 50, 'T': 1, 'e': 1e-13, 'xl': -6, 'xr': 1.5}
#params = {'T': 2, 'e': 1e-13, 'L': 6, 'power' : 6}
problem = Digital_option
#problem = transport_equation
#problem = Buckley_Leverett
#problem = heat_equation
#problem = PME
my_problem = problem(space_steps=500, time_steps=None, params = params)
params = my_problem.params
V_t, S_t, tt_t, u_t = train_model.full_WENO(my_problem, trainable=True, plot=False, vectorized=False)
V_nt, S_nt, tt_nt, u_nt = train_model.full_WENO(my_problem, trainable=False, plot=False, vectorized=False)

# plt.figure(2)
# u_exact, V_exact = my_problem.exact(first_step=False)
# plt.plot(S_nt,V_nt[:,-1],S_nt,V_exact)

u_exact, exact = my_problem.exact(first_step=False)
plt.figure(2)
plt.plot(S_nt,V_nt[:,-1],S_t,V_t[:,-1],S_t, exact)

error_nt = my_problem.err(u_nt.detach().numpy()[:,-1],first_step=False)
error_t = my_problem.err(u_t.detach().numpy()[:,-1],first_step=False)

# plt.figure(3)
# for k in range(0,len(V_nt[0])):
#     plt.plot(S_nt,V_nt[:,k])
#
# plt.figure(3)
# for k in range(0,len(V_t[0])):
#     plt.plot(S_t,V_t[:,k])

# plt.plot(S_nt,V_nt[:,-1])
# plt.plot(S_t,V_t[:,-1])

#problem_ex = problem(space_steps=100*2*2*2*2*2*2*2, time_steps=50*4*4*4*4*4*4*4, params = params)
# problem_ex = problem(space_steps=100*2*2*2*2, time_steps=50*4*4*4*4, params = params)
# _, u_exact_adjusted = train_model.compute_exact(Buckley_Leverett, problem_ex, 100, 50, just_one_time_step = False, trainable= False)
#
# # error_nt = train_model.compute_error(V_nt[:,1], u_exact_adjusted)
# # error_t = train_model.compute_error(V_t[:,1], u_exact_adjusted)
# # plt.figure(3)
# # plt.plot(S_nt, V_nt[:, 1], S_t, V_t[:,1], S_t, u_exact_adjusted)
#
# error_nt = train_model.compute_error(V_nt[:,-1], u_exact_adjusted[:,-1])
# error_t = train_model.compute_error(V_t[:,-1], u_exact_adjusted[:,-1])
# plt.figure(3)
# plt.plot(S_nt, V_nt[:, -1], S_t, V_t[:,-1], S_t, u_exact_adjusted[:,-1])