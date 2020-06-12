import torch
from define_problem_Digital import Digital_option
from define_WENO_Network import WENONetwork
from define_problem_heat_eq import heat_equation
from define_problem_Call import Call_option
from define_problem_transport_eq import transport_equation
from define_problem_PME import PME

torch.set_default_dtype(torch.float64)

train_model = torch.load('model')

params=None
#params = {'T': 2, 'e': 1e-13, 'L': 6, 'power' : 6}
problem = Call_option
my_problem = problem(space_steps=40, time_steps=None, params = params)
V, S, tt = train_model.full_WENO(my_problem, trainable=False, plot=True, vectorized=False)
