import torch
import matplotlib.pyplot as plt
from define_problem_Digital import Digital_option
from define_WENO_Network import WENONetwork
from define_problem_heat_eq import heat_equation
from define_problem_Call import Call_option
from define_problem_transport_eq import transport_equation
from define_problem_PME import PME
from define_problem_Buckley_Leverett import Buckley_Leverett

torch.set_default_dtype(torch.float64)

train_model = torch.load('model')

#params=None
params = {'T': 0.4, 'e': 1e-13, 'L': 1, 'R': 1, 'C': 4.047660603259527}
#params = {'sigma': 0.3, 'rate': 0.1, 'E': 50, 'T': 1, 'e': 1e-13, 'xl': -6, 'xr': 1.5}
#params = {'T': 2, 'e': 1e-13, 'L': 6, 'power' : 6}
#problem = Call_option
#problem = transport_equation
problem = Buckley_Leverett
#problem = heat_equation
#problem = PME
my_problem = problem(space_steps=100*2*2, time_steps=50*4*4, params = params)
V, S, tt = train_model.full_WENO(my_problem, trainable=False, plot=True, vectorized=False)
# u_exact = my_problem.exact()
plt.figure(2)
plt.plot(S,V[:,1]) #,S,u_exact)
# V_last = V[:,-1]
# error = my_problem.err(V_last)

for k in range(0,len(V[0])):
    plt.plot(S,V[:,k])