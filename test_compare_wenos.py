import torch
from define_problem_Digital import Digital_option
from define_WENO_Network import WENONetwork
from define_problem_heat_eq import heat_equation
from define_problem_Call import Call_option
from define_problem_PME import PME
from define_problem_Digital_GS import Digital_option_GS

torch.set_default_dtype(torch.float64)

train_model = torch.load('model')

#params=None
#params = {'T': 2, 'e': 1e-13, 'L': 6, 'power' : 5}
#params = {'sigma': 0.3, 'rate': 0.1, 'E': 50, 'T': 1, 'e': 1e-13, 'xl': -6, 'xr': 1.5}
params = {'sigma': 0.3, 'rate': 0.25, 'E': 50, 'T': 1, 'e': 1e-13, 'xl': -1.5, 'xr': 2, 'psi': 30}
#problem = Call_option
problem= Digital_option_GS
#problem = PME
my_problem = problem(space_steps=160, time_steps=1, params = params)
#u = train_model.run_weno( my_problem, trainable=False, vectorized=False)
train_model.compare_wenos(my_problem)
my_problem.get_params()