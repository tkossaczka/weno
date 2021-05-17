import random

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
        rng = 5
        return params_vld[j], rng

    def validation_problems_BL_2(j):
        params_vld = []
        params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 1, 'G': 3})
        params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.4, 'G': 0})
        params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.5, 'G': 5})
        params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.3, 'G': 3})
        params_vld.append({'T': 0.1, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.25, 'G': 1})
        rng = 5
        return params_vld[j], rng

    def validation_problems_BL_3(j):
        params_vld = []
        params_vld.append({'T': 0.2, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 1, 'G': 5})
        params_vld.append({'T': 0.2, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 1, 'G': 0})
        params_vld.append({'T': 0.2, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.5, 'G': 2})
        params_vld.append({'T': 0.2, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.5, 'G': 1})
        params_vld.append({'T': 0.2, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.25, 'G': 4})
        rng = 5
        return params_vld[j], rng

    def validation_problems_BL_4(j):
        params_vld = []
        params_vld.append({'T': 0.05, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 1, 'G': 5})
        params_vld.append({'T': 0.05, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 1, 'G': 0})
        params_vld.append({'T': 0.05, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.5, 'G': 2})
        params_vld.append({'T': 0.05, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.5, 'G': 1})
        params_vld.append({'T': 0.05, 'e': 1e-13, 'L': 0, 'R': 1, 'C': 0.25, 'G': 4})
        rng = 5
        return params_vld[j], rng
