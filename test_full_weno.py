import torch
from define_problem_Digital import Digital_option
from define_WENO_Network import WENONetwork
from define_problem_heat_eq import heat_equation
from define_problem_Call import Call_option
from define_problem_transport_eq import transport_equation

torch.set_default_dtype(torch.float64)

train_model = torch.load('model')

params=None
problem = transport_equation
my_problem = problem(space_steps=40, time_steps=None, params = params)
V, S, tt = train_model.full_WENO(my_problem, trainable=True, plot=True, vectorized=False)
