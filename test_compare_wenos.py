import torch
import matplotlib.pyplot as plt
from define_problem_Digital import Digital_option
from define_WENO_Network import WENONetwork
from define_problem_heat_eq import heat_equation
from define_problem_Call import Call_option
from define_problem_PME import PME
from define_problem_Digital_GS import Digital_option_GS
from define_problem_Buckley_Leverett import Buckley_Leverett

torch.set_default_dtype(torch.float64)

train_model = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Digital_Option_Test/Models/Model_11/99.pt")

params=None
#params = {'T': 0.4, 'e': 1e-13, 'L': 1, 'R': 1, 'C': 0.5}
#params = {'T': 0.4, 'e': 1e-13, 'L': 1, 'R': 1, 'C': 0.7360486548170379}
#params = {'T': 2, 'e': 1e-13, 'L': 6, 'power' : 5}
params = {'sigma': 0.3, 'rate': 0.1, 'E': 50, 'T': 1, 'e': 1e-13, 'xl': -6, 'xr': 1.5}
#params = {'sigma': 0.3, 'rate': 0.25, 'E': 50, 'T': 1, 'e': 1e-13, 'xl': -1.5, 'xr': 2} #, 'psi': 30}
#problem = Call_option
problem= Digital_option
#problem = PME
#problem = Buckley_Leverett
my_problem = problem(space_steps=100, time_steps=None, params = params)
#u = train_model.run_weno( my_problem, traina0le=False, vectorized=False)
train_model.compare_wenos(my_problem)
params = my_problem.get_params()

# problem_ex = problem(space_steps=100*2*2, time_steps=50*4*4, params = params)
# u = train_model.run_weno(my_problem, vectorized=True, trainable = True, just_one_time_step = False)
# uu=u.detach().numpy()
# _,x,t = my_problem.transformation(u)
# #u_exact = torch.load('u_ex')
# #u_exact_adjusted= torch.load('u_ex_adj')
# u_exact, u_exact_adjusted = train_model.compute_exact(Buckley_Leverett, problem_ex, 200, 200, just_one_time_step = False, trainable= False)
# uue = u_exact_adjusted.detach().numpy()
# #uue = my_problem.exact()
# plt.figure(2)
# plt.plot(x, uu[:, -1], x, uue[:,-1])
# #plt.plot(x, uu[:, -1], x, uue)
#
# unt = train_model.run_weno(my_problem, vectorized=True, trainable = False, just_one_time_step = False)
# uunt=unt.detach().numpy()
# plt.figure(3)
# plt.plot(x, uunt[:, -1], x, uue[:,-1])
# #plt.plot(x, uunt[:, -1], x, uue)

# u_exact = torch.load('u_ex')
# u_exact_adjusted= torch.load('u_ex_adj')