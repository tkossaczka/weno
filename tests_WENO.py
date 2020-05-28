from define_problem import Digital_option
from define_WENO_Network import WENONetwork

train_model = WENONetwork()

params=None

# FULL WENO
my_problem = Digital_option(space_steps=160, time_steps=None, params = params)
V, S, tt = train_model.full_WENO(my_problem, trainable=True, plot=True)
my_problem.get_params()

# COMPARE WENOS
my_problem = Digital_option(space_steps=160, time_steps=1, params = params)
train_model.compare_wenos(my_problem)
my_problem.get_params()

# COMPUTE ORDERS
params={'sigma': 0.3, 'rate': 0.1, 'E': 50, 'T': 1, 'e': 1e-13, 'xl': -6, 'xr': 1.5}
err_trained, order_trained = train_model.order_compute(20,  params, Digital_option, trainable=True)
err_not_trained, order_not_trained = train_model.order_compute(20, params, Digital_option, trainable=False)