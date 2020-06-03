import torch
from define_problem_Digital import Digital_option
from define_WENO_Network import WENONetwork
from define_problem_heat_eq import heat_equation
from define_problem_Call import Call_option
from define_problem_transport_eq import transport_equation

train_model = WENONetwork()
train_model = torch.load('model')

torch.set_default_dtype(torch.float64)

params=None
problem = heat_equation
problem = transport_equation
problem= Digital_option
problem= Call_option

# FULL WENO
my_problem = problem(space_steps=160, time_steps=None, params = params)
V, S, tt = train_model.full_WENO(my_problem, trainable=True, plot=True, vectorized=False)
my_problem.get_params()

# COMPARE WENOS
params=None
problem= Call_option
my_problem = problem(space_steps=160, time_steps=1, params = params)
train_model.compare_wenos(my_problem)
my_problem.get_params()

# COMPUTE ORDERS
params={'sigma': 0.3, 'rate': 0.1, 'E': 50, 'T': 1, 'e': 1e-13, 'xl': -6, 'xr': 1.5}
params = {'T': 1, 'e': 1e-13, 'L': 3.141592653589793}
err_trained, order_trained = train_model.order_compute(20,  params, problem, trainable=True)
err_not_trained, order_not_trained = train_model.order_compute(20, params, problem, trainable=False)
