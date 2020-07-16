import torch
import matplotlib.pyplot as plt
from define_problem_Digital import Digital_option
from define_WENO_Network import WENONetwork
from define_problem_heat_eq import heat_equation
from define_problem_Call import Call_option
from define_problem_transport_eq import transport_equation
from define_problem_PME import PME
from define_problem_Call_GS import Call_option_GS
from define_problem_Digital_GS import Digital_option_GS
from define_problem_Buckley_Leverett import Buckley_Leverett

train_model = WENONetwork()
#train_model = torch.load('model')

torch.set_default_dtype(torch.float64)

params=None
params =  {'T': 0.4, 'e': 1e-13, 'L': 1, 'R': 1, 'C': 1.0060472785249628}
#params = {'T': 0.4, 'e': 1e-13, 'L': 1, 'R': 1, 'C': 5.467189905555848}
#params = {'sigma': 0.3, 'rate': 0.02, 'E': 50, 'T': 1, 'e': 1e-13, 'xl': -6, 'xr': 1.5, 'psi':20}
#params = {'sigma': 0.3, 'rate': 0.25, 'E': 50, 'T': 1, 'e': 1e-13, 'xl': -1.5, 'xr': 2, 'psi':30}

#problem = PME
#problem = heat_equation
#problem = transport_equation
#problem= Digital_option
#problem= Call_option
#problem = Call_option_GS
#problem = Digital_option_GS
problem = Buckley_Leverett

problem_main = problem(space_steps=100, time_steps=40, params = params)
params = problem_main.get_params()
#problem_ex = problem(space_steps=100*2*2, time_steps=40*4*4, params = params)
problem_ex = problem(space_steps=100*2*2*2*2*2*2*2, time_steps=40*4*4*4*4*4*4*4, params = params)
u = train_model.run_weno(problem_main, vectorized=True, trainable = False, just_one_time_step = True)
uu=u.detach().numpy()
_,x,t = problem_main.transformation(u)
#plt.plot(x, uu[:, -1])
#params = problem_main.get_params()

u_exact, u_exact_adjusted = train_model.compute_exact(Buckley_Leverett, problem_ex, 100, 40, just_one_time_step = True, trainable= False)

error = train_model.compute_error(problem, u, u_exact_adjusted, trainable=False)

plt.plot(x, uu[:, -1], x, u_exact_adjusted[:,-1])

#plt.plot(x, uu[:, 1], x, u_exact_adjusted[:,1])

# for k in range(0,len(u_exact_adjusted[0])):
#     plt.plot(x,u_exact_adjusted[:,k])
#
# for k in range(0,len(u_exact[0])):
#     plt.plot(u_exact[:,k])

# u_exact = my_problem.exact()
# plt.plot(x,uu[:,-1],x,u_exact)
#
# u_last = u[:,-1]
# error = my_problem.err(u_last)
#
# u_whole_exact=my_problem.whole_exact()
# error_whole= uu-u_whole_exact

