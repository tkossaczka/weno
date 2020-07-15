from define_problem_Digital import Digital_option
from define_problem_heat_eq import heat_equation
from define_problem_transport_eq import transport_equation
from define_problem_Buckley_Leverett import Buckley_Leverett
from define_SFD2_Solver import SFD2_Solver

model = SFD2_Solver()

params = None
#params = {'sigma': 0.3, 'rate': 0.1, 'E': 50, 'T': 1, 'e': 1e-13, 'xl': -6, 'xr': 1.5}
#problem = Digital_option
#problem = heat_equation
problem = Buckley_Leverett
my_problem = problem(space_steps=1000, time_steps=None, params = params)

u = model.run_sfd(my_problem, vectorized=True)

model.compare(my_problem)
my_problem.get_params()

#err, order = model.order_compute(5, 20, params, problem)

