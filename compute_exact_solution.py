import torch
import pandas as pd
import numpy as np
from define_WENO_Network_2 import WENONetwork_2
from define_problem_PME import PME
from define_problem_Buckley_Leverett import Buckley_Leverett
import random
import os, sys, argparse
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

# problem = PME
problem = Buckley_Leverett

train_model = WENONetwork_2()
parameters = []

# type = "boxes"

# params["T"] = 0.5
# params["e"] = 10 ** (-13)
# params["L"] = 6
# params["power"] = random.uniform(2,5)
# params["d"] = 1

def save_problem_and_solution(save_path, sample_id):
    print("{},".format(sample_id))
    # problem_ex = problem(sample_id = None, example= "boxes", space_steps=64 * 2 * 2 * 2 * 2, time_steps=None, params=None)
    # power = problem_ex.params['power']
    # height = problem_ex.height
    problem_ex = problem(sample_id=None, example="gravity", space_steps=64 * 2 * 2 * 2 * 2, time_steps=None, params=None)
    C = problem_ex.params['C']
    G = problem_ex.params['G']
    # u_exact, u_exact_64 = train_model.compute_exact(problem, problem_ex, 64, 41, just_one_time_step=False, trainable=False)
    u_exact = train_model.compute_exact(problem, problem_ex, 64, 41, just_one_time_step=False, trainable=False)
    u_exact = u_exact.detach().numpy()
    # u_exact_64 = u_exact_64.detach().numpy()

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    np.save(os.path.join(save_path, "u_exact_{}".format(sample_id)), u_exact)
    # np.save(os.path.join(save_path, "u_exact64_{}".format(sample_id)), u_exact_64)

    # if not os.path.exists(os.path.join(save_path, "parameters.txt")):
    #     with open(os.path.join(save_path, "parameters.txt"), "a") as f:
    #         f.write("{},{}\n".format("sample_id","power"))
    # with open(os.path.join(save_path, "parameters.txt"), "a") as f:
    #     f.write("{},{}\n".format(sample_id, power))

    if not os.path.exists(os.path.join(save_path, "parameters.txt")):
        with open(os.path.join(save_path, "parameters.txt"), "a") as f:
            f.write("{},{},{}\n".format("sample_id","C","G"))
    with open(os.path.join(save_path, "parameters.txt"), "a") as f:
        f.write("{},{},{}\n".format(sample_id, C, G))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate exact solutions with given sample number for filename')
    parser.add_argument('save_path', default='', help='sample number for filename')
    parser.add_argument('sample_number', default='0', help='sample number for filename')

    args = parser.parse_args()
    save_problem_and_solution(args.save_path, args.sample_number)

# seq 174 400 | xargs -i{} -P6 python compute_exact_solution.py C:\Users\Tatiana\Desktop\Research\Research_ML_WENO\PME_Test\PME_Data_1024 {}
# seq 7 250 | xargs -i{} -P6 python compute_exact_solution.py C:\Users\Tatiana\Desktop\Research\Research_ML_WENO\Buckley_Leverett_CD_Test\Buckley_Leverett_CD_Data_1024 {}

# def validation_problems(j):
#     params_vld = []
#     params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': 2, 'd': 1})
#     params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': 3, 'd': 1})
#     params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': 4, 'd': 1})
#     params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': 5, 'd': 1})
#     return params_vld[j]
#
# params = {'T': 0.5, 'e': 1e-13, 'L': 6, 'power': 6, 'd': 1}
# problem_ex_test = problem(sample_id = None, example= "boxes", space_steps=64 * 2 * 2 * 2 * 2 , time_steps=None, params=params)
# u_ex = train_model.compute_exact(PME, problem_ex_test, 64, 214, just_one_time_step=False, trainable=False)
# torch.save(u_ex, "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Basic_test_set/u_ex_4")
#torch.save(u_ex64, "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/PME_Test/PME_Data_1024/Basic_test_set/u_ex64_3")

# def validation_problems(j):
#     params_vld = []
#     params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 1, 'G': 5})
#     params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 1, 'G': 0})
#     params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.5, 'G': 2})
#     params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.5, 'G': 1})
#     params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.25, 'G': 4})
#     return params_vld[j]
#
# params = {'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.25, 'G': 4}
# problem_ex_test = problem(sample_id=None, example= "gravity", space_steps=64 * 2 * 2 * 2 * 2 , time_steps=None, params=params)
# u_ex = train_model.compute_exact(Buckley_Leverett, problem_ex_test, 64, 41, just_one_time_step=False, trainable=False)
# torch.save(u_ex, "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Basic_test_set_1/u_ex_4")
# torch.save(u_ex64, "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Basic_test_set/u_ex64_4")

# def validation_problems(j):
#     params_vld = []
#     params_vld.append({'T': 0.2, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 1, 'G': 5})
#     params_vld.append({'T': 0.2, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 1, 'G': 0})
#     params_vld.append({'T': 0.2, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.5, 'G': 2})
#     params_vld.append({'T': 0.2, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.5, 'G': 1})
#     params_vld.append({'T': 0.2, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.25, 'G': 4})
#     return params_vld[j]
#
# params = {'T': 0.2, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 1, 'G': 0}
# problem_ex_test = problem(sample_id=None, example= "gravity", space_steps=64 * 2 * 2 * 2 * 2 , time_steps=None, params=params)
# u_ex, u_ex64 = train_model.compute_exact(Buckley_Leverett, problem_ex_test, 64, 82, just_one_time_step=False, trainable=False)
# torch.save(u_ex, "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Basic_test_set_3/u_ex_1")
# torch.save(u_ex64, "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Basic_test_set_3/u_ex64_1")

# def validation_problems(j):
#     params_vld = []
#     params_vld.append({'T': 0.05, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 1, 'G': 5})
#     params_vld.append({'T': 0.05, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 1, 'G': 0})
#     params_vld.append({'T': 0.05, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.5, 'G': 2})
#     params_vld.append({'T': 0.05, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.5, 'G': 1})
#     params_vld.append({'T': 0.05, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.25, 'G': 4})
#     return params_vld[j]
#
# params = {'T': 0.05, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.25, 'G': 4}
# problem_ex_test = problem(sample_id=None, example= "gravity", space_steps=64 * 2 * 2 * 2 * 2 , time_steps=None, params=params)
# u_ex, u_ex64 = train_model.compute_exact(Buckley_Leverett, problem_ex_test, 64, 21, just_one_time_step=False, trainable=False)
# torch.save(u_ex, "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Basic_test_set_4/u_ex_4")
# torch.save(u_ex64, "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Basic_test_set_4/u_ex64_4")

# def validation_problems(j):
#     params_vld = []
#     params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 1, 'G': 3})
#     params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.4, 'G': 0})
#     params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.5, 'G': 5})
#     params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.3, 'G': 3})
#     params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.25, 'G': 1})
#     return params_vld[j]
#
# params ={'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.25, 'G': 1}
# problem_ex_test = problem(sample_id=None, example= "gravity", space_steps=64 * 2 * 2 * 2 * 2 , time_steps=None, params=params)
# u_ex = train_model.compute_exact(Buckley_Leverett, problem_ex_test, 64, 41, just_one_time_step=False, trainable=False)
# torch.save(u_ex, "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Basic_test_set_2/u_ex_4")
# torch.save(u_ex64, "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Basic_test_set_2/u_ex64_4")

# params = {'T': 0.5, 'L': -2, 'R': 2, 'e': 1e-13}
# problem_ex_test = problem(sample_id=None, example= "degenerate", space_steps=64 * 2 * 2 * 2 * 2 , time_steps=None, params=params)
# u_ex, u_ex64 = train_model.compute_exact(Buckley_Leverett, problem_ex_test, 64, 52, just_one_time_step=False, trainable=False)
# torch.save(u_ex, "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Basic_test_set_1/u_ex_5")
# torch.save(u_ex64, "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Basic_test_set_1/u_ex64_5")