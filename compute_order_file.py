import torch
import numpy as np
from define_problem_Digital import Digital_option
from define_WENO_Network import WENONetwork
from define_problem_heat_eq import heat_equation
from define_problem_Call import Call_option
from define_problem_transport_eq import transport_equation
from define_problem_PME import PME
from define_problem_Call_GS import Call_option_GS
from define_problem_Digital_GS import Digital_option_GS
from define_problem_Buckley_Leverett import Buckley_Leverett

with torch.no_grad():
    train_model = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/Models/Model_26/190.pt")
    #train_model = torch.load('C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Models/Model_18/46')
    #train_model = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Models/Model_16/37")
    # train_model = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Digital_Option_Test/Models/Model_21/3390.pt")
    #train_model = WENONetwork()
    problem = heat_equation
    # problem = transport_equation
    #problem = heat_equation
    #problem = Call_option
    #problem = Digital_option
    #problem = PME
    #problem = Call_option_GS
    #problem = Digital_option_GS
    #problem = Buckley_Leverett
    torch.set_default_dtype(torch.float64)

    nb = 6
    params = None
    #params = {'T': 0.4, 'e': 1e-13, 'L': 1, 'R': 1, 'C': 0.25}
    #params = {'T': 0.2, 'e': 1e-13, 'L': 1, 'R': 1, 'C': 0.3}
    #params = {'sigma': 0.3, 'rate': 0.2, 'E': 50, 'T': 1, 'e': 1e-13, 'xl': -6, 'xr': 1.5}
    #params = {'sigma': 0.3, 'rate': 0.02, 'E': 50, 'T': 1, 'e': 1e-13, 'xl': -1.5, 'xr': 2, 'psi': 30}
    # params = {'T': 1, 'e': 1e-13, 'L': 3.141592653589793}
    err_trained_max, order_trained_max, err_trained_l2, order_trained_l2 = train_model.order_compute(nb, 10, None,  params, problem, trainable=True)
    err_not_trained_max, order_not_trained_max, err_not_trained_l2, order_not_trained_l2 = train_model.order_compute(nb,10,None, params, problem, trainable=False)

order_mat_max = np.zeros((nb,4))
order_mat_max[:,0] = err_not_trained_max[:,0]
order_mat_max[1:,1] = order_not_trained_max[:,0]
order_mat_max[:,2] = err_trained_max[:,0]
order_mat_max[1:,3] = order_trained_max[:,0]

order_mat_l2 = np.zeros((nb,4))
order_mat_l2[:,0] = err_not_trained_l2[:,0]
order_mat_l2[1:,1] = order_not_trained_l2[:,0]
order_mat_l2[:,2] = err_trained_l2[:,0]
order_mat_l2[1:,3] = order_trained_l2[:,0]

order_mat_max_l2 = np.zeros((nb,4))
order_mat_max_l2[:,0] = err_trained_max[:,0]
order_mat_max_l2[1:,1] = order_trained_max[:,0]
order_mat_max_l2[:,2] = err_trained_l2[:,0]
order_mat_max_l2[1:,3] = order_trained_l2[:,0]


import pandas as pd
# pd.DataFrame(order_mat).to_csv("err_mat.csv")
pd.DataFrame(order_mat_max_l2).to_latex()