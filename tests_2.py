import torch
from define_problem_Digital import Digital_option
from define_WENO_Network import WENONetwork
from define_problem_heat_eq import heat_equation
from define_problem_Call import Call_option

train_model = torch.load('model')

params=None
problem= Digital_option
my_problem = problem(space_steps=160, time_steps=1, params = params)
train_model.compare_wenos(my_problem)
my_problem.get_params()