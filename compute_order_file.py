import torch
from define_problem_Digital import Digital_option
from define_WENO_Network import WENONetwork
from define_problem_heat_eq import heat_equation
from define_problem_Call import Call_option
from define_problem_transport_eq import transport_equation

with torch.no_grad():
    train_model = torch.load('model')
    # problem = heat_equation
    problem = Call_option
    torch.set_default_dtype(torch.float64)

    #params = None
    params = {'sigma': 0.3, 'rate': 0.1, 'E': 50, 'T': 1, 'e': 1e-13, 'xl': -6, 'xr': 1.5}
    #params = {'T': 1, 'e': 1e-13, 'L': 3.141592653589793}
    #err_trained, order_trained = train_model.order_compute(5, 20,  params, problem, trainable=True)
    err_not_trained, order_not_trained = train_model.order_compute(5, 20, params, problem, trainable=False)