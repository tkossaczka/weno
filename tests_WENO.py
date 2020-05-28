import torch
from define_problem import Digital_option
from define_WENO_Network import WENONetwork

train_model = WENONetwork()


my_problem = Digital_option(space_steps=160, time_steps=None, params = {'sigma': 0.3, 'rate': 0.1, 'E': 50, 'T': 1, 'e': 1e-13, 'xl': -6, 'xr': 1.5, 'm': 160})
u = train_model.run_weno(my_problem, trainable=False)


my_problem = Digital_option(space_steps=160, time_steps=1, params = None)
train_model.compare_wenos(my_problem)
my_problem.get_params()


my_problem = Digital_option(space_steps=20, time_steps=None, params = {'sigma': 0.3, 'rate': 0.1, 'E': 50, 'T': 1, 'e': 1e-13, 'xl': -6, 'xr': 1.5, 'm': 160})
train_model.order_compute(my_problem....)