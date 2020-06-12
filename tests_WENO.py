import torch
from define_problem_Digital import Digital_option
from define_WENO_Network import WENONetwork
from define_problem_heat_eq import heat_equation
from define_problem_Call import Call_option
from define_problem_transport_eq import transport_equation
from define_problem_PME import PME
from define_problem_Call_GS import Call_option_GS
from define_problem_Digital_GS import Digital_option_GS

#train_model = WENONetwork()
train_model = torch.load('model')

torch.set_default_dtype(torch.float64)

#params=None
#params = {'sigma': 0.3, 'rate': 0.02, 'E': 50, 'T': 1, 'e': 1e-13, 'xl': -6, 'xr': 1.5, 'psi':20}
params = {'sigma': 0.3, 'rate': 0.25, 'E': 50, 'T': 1, 'e': 1e-13, 'xl': -1.5, 'xr': 2, 'psi':30}

#problem = PME
#problem = heat_equation
#problem = transport_equation
#problem= Digital_option
#problem= Call_option
#problem = Call_option_GS
problem = Digital_option_GS

my_problem = problem(space_steps=20, time_steps=None, params = params)
u = train_model.run_weno(my_problem, vectorized=True, trainable = False)
uu=u.detach().numpy()



