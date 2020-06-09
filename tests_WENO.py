import torch
from define_problem_Digital import Digital_option
from define_WENO_Network import WENONetwork
from define_problem_heat_eq import heat_equation
from define_problem_Call import Call_option
from define_problem_transport_eq import transport_equation
from define_problem_PME import PME

#train_model = WENONetwork()
train_model = torch.load('model')

torch.set_default_dtype(torch.float64)

params=None
problem = PME
#problem = heat_equation
#problem = transport_equation
#problem= Digital_option
#problem= Call_option


