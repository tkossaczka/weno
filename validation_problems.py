import random
import torch
import pandas as pd
import numpy as np

class validation_problems():

    def validation_problems_barenblatt_default(j):
        params_vld = []
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 2, 'd': 1})
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 3, 'd': 1})
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 4, 'd': 1})
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 5, 'd': 1})
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 6, 'd': 1})
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 7, 'd': 1})
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 8, 'd': 1})
        rng = 7
        return params_vld[j], rng

    def validation_problems_barenblatt_default_2(j):
        params_vld = []
        params_vld.append({'T': 1.5, 'e': 1e-13, 'L': 6, 'power': 2, 'd': 1})
        params_vld.append({'T': 1.5, 'e': 1e-13, 'L': 6, 'power': 3, 'd': 1})
        params_vld.append({'T': 1.5, 'e': 1e-13, 'L': 6, 'power': 4, 'd': 1})
        params_vld.append({'T': 1.5, 'e': 1e-13, 'L': 6, 'power': 5, 'd': 1})
        params_vld.append({'T': 1.5, 'e': 1e-13, 'L': 6, 'power': 6, 'd': 1})
        params_vld.append({'T': 1.5, 'e': 1e-13, 'L': 6, 'power': 7, 'd': 1})
        params_vld.append({'T': 1.5, 'e': 1e-13, 'L': 6, 'power': 8, 'd': 1})
        rng = 7
        return params_vld[j], rng

    def validation_problems_barenblatt_default_3(j):
        params_vld = []
        params_vld.append({'T': 1.2, 'e': 1e-13, 'L': 6, 'power': 2, 'd': 1})
        params_vld.append({'T': 1.2, 'e': 1e-13, 'L': 6, 'power': 3, 'd': 1})
        params_vld.append({'T': 1.2, 'e': 1e-13, 'L': 6, 'power': 4, 'd': 1})
        params_vld.append({'T': 1.2, 'e': 1e-13, 'L': 6, 'power': 5, 'd': 1})
        params_vld.append({'T': 1.2, 'e': 1e-13, 'L': 6, 'power': 6, 'd': 1})
        params_vld.append({'T': 1.2, 'e': 1e-13, 'L': 6, 'power': 7, 'd': 1})
        params_vld.append({'T': 1.2, 'e': 1e-13, 'L': 6, 'power': 8, 'd': 1})
        rng = 7
        return params_vld[j], rng

    def validation_problems_barenblatt_fract(j):  # tieto boli dobre, model 62
        params_vld = []
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 2.2, 'd': 1})
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 2.8, 'd': 1})
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 3.9, 'd': 1})
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 4.4, 'd': 1})
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 5.1, 'd': 1})
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 6.2, 'd': 1})
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': 7.1, 'd': 1})
        rng = 7
        return params_vld[j], rng

    def validation_problems_barenblatt_fract_2(j):  # tieto boli dobre, model 62
        params_vld = []
        params_vld.append({'T': 1.2, 'e': 1e-13, 'L': 6, 'power': 2.2, 'd': 1})
        params_vld.append({'T': 1.2, 'e': 1e-13, 'L': 6, 'power': 2.8, 'd': 1})
        params_vld.append({'T': 1.2, 'e': 1e-13, 'L': 6, 'power': 3.9, 'd': 1})
        params_vld.append({'T': 1.2, 'e': 1e-13, 'L': 6, 'power': 4.4, 'd': 1})
        params_vld.append({'T': 1.2, 'e': 1e-13, 'L': 6, 'power': 5.1, 'd': 1})
        params_vld.append({'T': 1.2, 'e': 1e-13, 'L': 6, 'power': 6.2, 'd': 1})
        params_vld.append({'T': 1.2, 'e': 1e-13, 'L': 6, 'power': 7.1, 'd': 1})
        rng = 7
        return params_vld[j], rng

    def validation_problems_barenblatt(j):
        a0 = 2.0
        a = 2.157
        aa = 3.012
        b = 3.697
        bb = 3.987
        c = 4.158
        d = 4.723
        dd = 5.041
        e = 5.568
        ee = 6.087
        f = 6.284
        ff = 7.124
        g = 7.958
        params_vld = []
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': a0, 'd': 1})
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': a, 'd': 1})
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': aa, 'd': 1})
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': b, 'd': 1})
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': bb, 'd': 1})
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': c, 'd': 1})
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': d, 'd': 1})
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': dd, 'd': 1})
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': e, 'd': 1})
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': ee, 'd': 1})
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': f, 'd': 1})
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': ff, 'd': 1})
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 6, 'power': g, 'd': 1})
        rng = 13
        return params_vld[j], rng

    def validation_problems_barenblatt_2d_default(j):
        params_vld = []
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 10, 'power': 2, 'd': 2})
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 10, 'power': 3, 'd': 2})
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 10, 'power': 4, 'd': 2})
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 10, 'power': 5, 'd': 2})
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 10, 'power': 6, 'd': 2})
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 10, 'power': 7, 'd': 2})
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 10, 'power': 8, 'd': 2})
        rng = 7
        return params_vld[j], rng

    def validation_problems_barenblatt_2d_default_2(j):
        params_vld = []
        params_vld.append({'T': 1.5, 'e': 1e-13, 'L': 10, 'power': 2, 'd': 2})
        params_vld.append({'T': 1.5, 'e': 1e-13, 'L': 10, 'power': 3, 'd': 2})
        params_vld.append({'T': 1.5, 'e': 1e-13, 'L': 10, 'power': 4, 'd': 2})
        params_vld.append({'T': 1.5, 'e': 1e-13, 'L': 10, 'power': 5, 'd': 2})
        params_vld.append({'T': 1.5, 'e': 1e-13, 'L': 10, 'power': 6, 'd': 2})
        params_vld.append({'T': 1.5, 'e': 1e-13, 'L': 10, 'power': 7, 'd': 2})
        params_vld.append({'T': 1.5, 'e': 1e-13, 'L': 10, 'power': 8, 'd': 2})
        rng = 7
        return params_vld[j], rng

    def validation_problems_barenblatt_2d_default_3(j):
        params_vld = []
        params_vld.append({'T': 1.2, 'e': 1e-13, 'L': 10, 'power': 2, 'd': 2})
        params_vld.append({'T': 1.2, 'e': 1e-13, 'L': 10, 'power': 3, 'd': 2})
        params_vld.append({'T': 1.2, 'e': 1e-13, 'L': 10, 'power': 4, 'd': 2})
        params_vld.append({'T': 1.2, 'e': 1e-13, 'L': 10, 'power': 5, 'd': 2})
        params_vld.append({'T': 1.2, 'e': 1e-13, 'L': 10, 'power': 6, 'd': 2})
        params_vld.append({'T': 1.2, 'e': 1e-13, 'L': 10, 'power': 7, 'd': 2})
        params_vld.append({'T': 1.2, 'e': 1e-13, 'L': 10, 'power': 8, 'd': 2})
        rng = 7
        return params_vld[j], rng

    def validation_problems_barenblatt_2d_fract(j):  # tieto boli dobre, model 62
        params_vld = []
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 10, 'power': 2.2, 'd': 2})
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 10, 'power': 2.8, 'd': 2})
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 10, 'power': 3.9, 'd': 2})
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 10, 'power': 4.4, 'd': 2})
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 10, 'power': 5.1, 'd': 2})
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 10, 'power': 6.2, 'd': 2})
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 10, 'power': 7.1, 'd': 2})
        rng = 7
        return params_vld[j], rng

    def validation_problems_barenblatt_2d(j):
        a0 =2.012
        a = 2.157
        aa = 3.012
        b = 3.697
        bb = 3.987
        c = 4.158
        d = 4.723
        dd = 5.041
        e = 5.568
        ee = 6.087
        f = 6.284
        ff = 7.124
        g = 7.958
        params_vld = []
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 10, 'power': a0, 'd': 2})
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 10, 'power': a, 'd': 2})
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 10, 'power': aa, 'd': 2})
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 10, 'power': b, 'd': 2})
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 10, 'power': bb, 'd': 2})
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 10, 'power': c, 'd': 2})
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 10, 'power': d, 'd': 2})
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 10, 'power': dd, 'd': 2})
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 10, 'power': e, 'd': 2})
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 10, 'power': ee, 'd': 2})
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 10, 'power': f, 'd': 2})
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 10, 'power': ff, 'd': 2})
        params_vld.append({'T': 2, 'e': 1e-13, 'L': 10, 'power': g, 'd': 2})
        rng = 13
        return params_vld[j], rng

    def validation_problems_boxes(j):
        params_vld = []
        params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': 2, 'd': 1})
        params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': 3, 'd': 1})
        params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': 4, 'd': 1})
        params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': 5, 'd': 1})
        params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 6, 'power': 6, 'd': 1})
        rng = 5
        return params_vld[j], rng

    def validation_problems_BL(j):
        params_vld = []
        params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 1, 'G': 5})
        params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 1, 'G': 0})
        params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.5, 'G': 2})
        params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.5, 'G': 1})
        params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.25, 'G': 4})
        folder = 1
        rng = 5
        return params_vld[j], rng, folder

    def validation_problems_BL_2(j):
        params_vld = []
        params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 1, 'G': 3})
        params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.4, 'G': 0})
        params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.5, 'G': 5})
        params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.3, 'G': 3})
        params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.25, 'G': 1})
        folder = 2
        rng = 5
        return params_vld[j], rng, folder

    def validation_problems_BL_3(j):
        params_vld = []
        params_vld.append({'T': 0.2, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 1, 'G': 5})
        params_vld.append({'T': 0.2, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 1, 'G': 0})
        params_vld.append({'T': 0.2, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.5, 'G': 2})
        params_vld.append({'T': 0.2, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.5, 'G': 1})
        params_vld.append({'T': 0.2, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.25, 'G': 4})
        folder = 3
        rng = 5
        return params_vld[j], rng, folder

    def validation_problems_BL_4(j):
        params_vld = []
        params_vld.append({'T': 0.05, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 1, 'G': 5})
        params_vld.append({'T': 0.05, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 1, 'G': 0})
        params_vld.append({'T': 0.05, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.5, 'G': 2})
        params_vld.append({'T': 0.05, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.5, 'G': 1})
        params_vld.append({'T': 0.05, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.25, 'G': 4})
        folder = 4
        rng = 5
        return params_vld[j], rng, folder

    def validation_problems_BL_5(j):
        rng = 12
        folder = 1
        params_vld = []
        for k in range(rng):
            df = pd.read_csv("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Validation_set_{}/parameters.txt".format(folder))
            C = float(df[df.sample_id == j]["C"])
            G = float(df[df.sample_id == j]["G"])
            params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': C, 'G': G})
        return params_vld[j], rng, folder

    def validation_problems_BL_6(j):
        params_vld = []
        params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 1, 'G': 7})
        params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 1, 'G': 6.5})
        params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.25, 'G': 7})
        params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.1, 'G': 8})
        params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.25, 'G': 9})
        folder = 6
        rng = 5
        return params_vld[j], rng, folder

    def exacts_test_BL(folder):
        u_ex_0 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Basic_test_set_{}/u_ex128_0".format(folder))
        u_ex_1 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Basic_test_set_{}/u_ex128_1".format(folder))
        u_ex_2 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Basic_test_set_{}/u_ex128_2".format(folder))
        u_ex_3 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Basic_test_set_{}/u_ex128_3".format(folder))
        u_ex_4 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Basic_test_set_{}/u_ex128_4".format(folder))
        # u_ex_0 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Basic_test_set_{}/u_ex64_0".format(folder))
        # u_ex_1 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Basic_test_set_{}/u_ex64_1".format(folder))
        # u_ex_2 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Basic_test_set_{}/u_ex64_2".format(folder))
        # u_ex_3 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Basic_test_set_{}/u_ex64_3".format(folder))
        # u_ex_4 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Basic_test_set_{}/u_ex64_4".format(folder))
        # u_ex_w_0 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Basic_test_set_{}/u_ex_0".format(folder))
        # u_ex_w_1 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Basic_test_set_{}/u_ex_1".format(folder))
        # u_ex_w_2 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Basic_test_set_{}/u_ex_2".format(folder))
        # u_ex_w_3 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Basic_test_set_{}/u_ex_3".format(folder))
        # u_ex_w_4 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Basic_test_set_{}/u_ex_4".format(folder))
        u_exs = [u_ex_0, u_ex_1, u_ex_2, u_ex_3, u_ex_4]
        return u_exs

    def exacts_validation_BL(folder):
        u_ex_0 = torch.Tensor(np.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Validation_set_{}/u_exact128_0.npy".format(folder)))
        u_ex_1 = torch.Tensor(np.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Validation_set_{}/u_exact128_1.npy".format(folder)))
        u_ex_2 = torch.Tensor(np.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Validation_set_{}/u_exact128_2.npy".format(folder)))
        u_ex_3 = torch.Tensor(np.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Validation_set_{}/u_exact128_3.npy".format(folder)))
        u_ex_4 = torch.Tensor(np.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Validation_set_{}/u_exact128_4.npy".format(folder)))
        u_ex_5 = torch.Tensor(np.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Validation_set_{}/u_exact128_5.npy".format(folder)))
        u_ex_6 = torch.Tensor(np.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Validation_set_{}/u_exact128_6.npy".format(folder)))
        u_ex_7 = torch.Tensor(np.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Validation_set_{}/u_exact128_7.npy".format(folder)))
        u_ex_8 = torch.Tensor(np.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Validation_set_{}/u_exact128_8.npy".format(folder)))
        u_ex_9 = torch.Tensor(np.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Validation_set_{}/u_exact128_9.npy".format(folder)))
        u_ex_10 = torch.Tensor(np.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Validation_set_{}/u_exact128_10.npy".format(folder)))
        u_ex_11 = torch.Tensor(np.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_CD_Test/Buckley_Leverett_CD_Data_1024/Validation_set_{}/u_exact128_11.npy".format(folder)))
        u_exs = [u_ex_0, u_ex_1, u_ex_2, u_ex_3, u_ex_4, u_ex_5, u_ex_6, u_ex_7, u_ex_8, u_ex_9, u_ex_10, u_ex_11]
        return u_exs