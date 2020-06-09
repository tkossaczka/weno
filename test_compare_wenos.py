import torch
from define_problem_Digital import Digital_option
from define_WENO_Network import WENONetwork
from define_problem_heat_eq import heat_equation
from define_problem_Call import Call_option
from define_problem_PME import PME

torch.set_default_dtype(torch.float64)

train_model = torch.load('model')

#params=None
params = {'T': 2, 'e': 1e-13, 'L': 6, 'power' : 5}
#problem = Call_option
#problem= Digital_option
problem = PME
my_problem = problem(space_steps=160, time_steps=1, params = params)
train_model.compare_wenos(my_problem)
my_problem.get_params()