import torch
import pandas as pd
import numpy as np
from define_WENO_Network_2 import WENONetwork_2
from define_problem_PME import PME
import random
import os, sys, argparse

torch.set_default_dtype(torch.float64)

problem = PME

train_model = WENONetwork_2()
parameters = []
type = "boxes"

# params["T"] = 0.5
# params["e"] = 10 ** (-13)
# params["L"] = 6
# params["power"] = random.uniform(2,5)
# params["d"] = 1

def save_problem_and_solution(save_path, sample_id):
    print("{},".format(sample_id))
    problem_ex = problem(type=type, space_steps=64 * 2 * 2 * 2 * 2, time_steps=None, params=None)
    power = problem_ex.params['power']
    u_exact, u_exact_64 = train_model.compute_exact(problem, problem_ex, 64, 214, just_one_time_step=False, trainable=False)
    u_exact = u_exact.detach().numpy()
    u_exact_64 = u_exact_64.detach().numpy()

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    np.save(os.path.join(save_path, "u_exact_{}".format(sample_id)), u_exact)
    np.save(os.path.join(save_path, "u_exact64_{}".format(sample_id)), u_exact_64)

    if not os.path.exists(os.path.join(save_path, "parameters.txt")):
        with open(os.path.join(save_path, "parameters.txt"), "a") as f:
            f.write("{},{},{}\n".format("sample_id","power"))
    with open(os.path.join(save_path, "parameters.txt"), "a") as f:
        f.write("{},{},{}\n".format(sample_id, power))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate exact solutions with given sample number for filename')
    parser.add_argument('save_path', default='', help='sample number for filename')
    parser.add_argument('sample_number', default='0', help='sample number for filename')

    args = parser.parse_args()
    save_problem_and_solution(args.save_path, args.sample_number)

# seq 0 6 | xargs -i{} -P6 python compute_exact_solution.py C:\Users\Tatiana\Desktop\Research\Research_ML_WENO\PME_Test\PME_Data_1024 {}
