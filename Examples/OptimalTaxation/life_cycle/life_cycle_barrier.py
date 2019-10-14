import numpy as np
from casadi import *
import matplotlib.pyplot as plt
from multiprocessing import Pool, Process, cpu_count
import itertools
import time
from plot_efficient_frontier import plot_frontier
import argparse
from auxiliary_functions import *


def find_efficient_frontier(a, b, n, S, lb_x, ub_x, r, conf):

    wage = wage_function(conf)

    # empty lists for results
    f1 = []
    f2 = []
    tk = []
    tl = []
    tc = []

    x0 = 1
    for R in (np.linspace(a, b, n)):
        r_t = R #40.2266607101565

        x0 = 1

        # Solve optimization problem; Set barrier for revenue to r_t
        res = S(x0=x0, lbg=0, ubg=[0]*121 + [inf], lbx=lb_x, ubx=ub_x, p=r_t)
        # access optimizers
        x0 = res["x"]

        consumption, labour, assets, debts, borrowing, tax_rates = extract_solution(x0, conf)
        f1_n, f2_n = evaluate_objectives(assets, consumption, labour, wage, tax_rates, r, conf)

        plot_solutions(consumption, labour, assets, debts, borrowing, tax_rates)
        asset_constraint(assets, labour, wage, consumption, borrowing, tax_rates, r, conf)
        # Append solutions to list
        tk.append(tax_rates[0])
        tl.append(tax_rates[1])
        tc.append(tax_rates[2])
        f1.append(f1_n)
        f2.append(f2_n)

    plt.plot(f2, f1, '.')
    plt.show()

    return f2, f1, tk, tl, tc


def run_sim(a, b, n, r, conf):

    wage = wage_function(conf)

    S, lb_x, ub_x = init_optimization_problem(r, wage, conf, True)
    x0 = np.array([0.418255, 1.34045, 1.46791, 1.60299, 1.75051, 1.9116, 2.08751, 2.27962, 2.4894, 2.71848, 2.96865, 3.24184, 3.54017, 3.86596, 4.22172, 4.61023, 5.03448, 5.49778, 6.00372, 6.55621, 7.15954, 7.8184, 8.53789, 9.32359, 10.1816, 11.1186, 12.1417, 13.2591, 14.4792, 15.8117, 17.2668, 18.8557, 20.5909, 22.4858, 24.5551, 26.8148, 29.2824, 31.9771, 34.9198, 38.1333, 41.6425, 45.4747, 49.6595, 54.2294, 59.2198, 64.6695, 70.6208, 77.1196, 84.2166, 91.9666, 100.43, 109.672, 119.764, 130.786, 142.821, 155.964, 170.317, 185.991, 203.106, 221.797, 0.952728, 1.01641, 1.07272, 1.10724, 1.13052, 1.1469, 1.15857, 1.16682, 1.17247, 1.17609, 1.17808, 1.17872, 1.17825, 1.17685, 1.17466, 1.17179, 1.16834, 1.1644, 1.16002, 1.15526, 1.15017, 1.1448, 1.13917, 1.13332, 1.12727, 1.12105, 1.11466, 1.10812, 1.10145, 1.09465, 1.08772, 1.08066, 1.07347, 1.06614, 1.05867, 1.05104, 1.04322, 1.03521, 1.02698, 1.01848, 1.0097, 1.00057, 0.991043, 0.981056, 0.970525, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9.83166e-016, 0, 0.81095, 2.4559, 4.94208, 8.27172, 12.4453, 17.4631, 23.3266, 30.0389, 37.6049, 46.032, 55.3299, 65.5108, 76.5891, 88.5814, 101.507, 115.385, 130.239, 146.092, 162.967, 180.889, 199.88, 219.962, 241.156, 263.478, 286.939, 311.548, 337.303, 364.196, 392.205, 421.298, 451.426, 482.522, 514.496, 547.236, 580.597, 614.401, 648.43, 682.421, 716.059, 748.969, 780.705, 810.746, 838.479, 857.658, 872.803, 882.963, 887.043, 883.781, 871.729, 849.23, 814.388, 765.041, 698.724, 612.632, 503.578, 367.946, 201.634, 0, 5.26099e-015, 5.89231e-015, 6.79067e-015, 7.77156e-015, 0, 7.10543e-015, 1.77636e-015, 0, 7.10543e-015, 6.99441e-015, 8.88178e-016, 0, 0, 7.32747e-015, 1.11022e-015, 0, 1.77636e-015, 7.99361e-015, 8.88178e-016, 0, 8.88178e-016, 0, 0, 0, 0, 7.32747e-015, 7.10543e-015, 7.10543e-015, 0, 0, 7.99361e-015, 7.54952e-015, 0, 2.22045e-016, 0, 2.22045e-016, 7.77156e-015, 6.88338e-015, 0, 4.44089e-016, 3.55271e-015, 4.44089e-016, 1.11022e-015, 1.88738e-015, 0, 2.22045e-016, 2.22045e-016, 2.22045e-016, 4.44089e-016, 2.22045e-016, 6.66134e-016, 4.44089e-016, 2.22045e-016, 0, 0, 0, 1.77636e-015, 0, 0, 0, 0, 0, 0, 0, 0, -4.19803e-016, -4.996e-016, 0, 0, 0, 0, -3.22659e-016, -6.94583e-015, 0, 0, -2.84495e-016, 0, 0, 0, -7.63278e-017, -2.22045e-016, 0, -8.74301e-016, 0, -5.20417e-016, 0, 0, 0, 0, -8.88178e-016, 0, 0, 0, 0, -5.19029e-015, 0, 0, 0, 0, -3.41394e-015, 0, 0, -6.59195e-017, -8.32667e-017, -1.60982e-015, -3.66374e-015, 0, 0, 0, -9.15934e-016, 0, 0, 0, 0, 0, -2.22045e-016, 0, 0, 0, 0, 0, 0, 0.08, 0])

    #x0 = x0 + abs(np.random.randn((len(x0)))*0.00000001)

    res = S[0](x0=x0, lbg=0, ubg=[0] * 121 + [inf], lbx=lb_x, ubx=ub_x, p=5.15188)
    consumption, labour, assets, debts, borrowing, tax_rates = extract_solution(res["x"], conf)
    pv_rev_global, pv_util_global, tk_global, tl_global, tc_global = \
                find_efficient_frontier(a, b, n, S, lb_x, ub_x, r, conf)

    return pv_rev_global, pv_util_global, tk_global, tl_global, tc_global




if __name__ == "__main__":

    # init parameters
    r = 0.1
    beta = 0.95
    r_debt = 0.15
    T = 60
    T_ret = 45
    L_max = 10

    gamma = 0.5
    eta = 8

    # finite difference step size
    h = 0.00001

    # vector of intrest rates; currently constant
    r = np.full(T, r)
    costb = 0.02

    # create dictionary with all settings
    conf = {'beta': beta,
            'r_debt': r_debt,
            'T': T,
            'T_ret': T_ret,
            'L_max': L_max,
            'gamma': gamma,
            'eta': eta,
            'plot': False,
            'plot_steps': True,
            'h': h,
            'n_steps_max': 10,
            'k_max': 100,
            'tol': 0.0000001,
            'tax_setting': 5,
            'serial': False,
            'max_debt': 2,
            'costb': 0.02,
            "borrowing":inf
            }

    # tax settings:

    # 1: capital tax and income tax
    # 2: income tax // NOT possible
    # 3: capital tax = income tax // NOT possible
    # 4: capital tax = income tax, consumption tax
    # 5: capital tax, income tax, consumption tax

    p = []
    n_procs = cpu_count() - 1
    n_procs = 1

    # Try except for
#    try:
#        parser = argparse.ArgumentParser(description='Process some integers.')
#        parser.add_argument('gamma')
#        parser.add_argument('eta')
#        parser.add_argument('n_procs')
#        args = parser.parse_args()

#        gammas_etas = [(float(args.gamma), float(args.eta))]
#        n_procs = int(args.n_procs)
 #   except:
    gammas_etas = [(0.5, 1), (0.5, 8), (3, 1), (3, 8)]
  #      pass

    timestr = time.strftime("%m-%d-%H%M")
    start = time.time()
    for gamma, eta in gammas_etas:
        conf["gamma"] = gamma
        conf["eta"] = eta
        a = 5
        b = 40
        n = 5
        ## HERE the actual execution starts
        if conf['serial'] or n_procs == 1:
            # SERIAL EXECUTION
            n_procs = 1
            data = [run_sim(a,b, n, r, conf)]
        else:
            # PARALLEL EXECUTION
            p = Pool(n_procs)
 #           data = p.starmap(run_sim, [(tax_comb[i:i + len(tax_comb)//n_procs] , r, conf) for i in range(0, len(tax_comb), len(tax_comb)//n_procs)] )

        pv_rev_d = np.concatenate([np.array(data[i][0]) for i in range(n_procs)])
        pv_util_d = np.concatenate([np.array(data[i][1]) for i in range(n_procs)])
        tk_d = np.concatenate([np.array(data[i][2]) for i in range(n_procs)])
        tl_d = np.concatenate([np.array(data[i][3]) for i in range(n_procs)])
        tc_d = np.concatenate([np.array(data[i][4]) for i in range(n_procs)])

        full_array = np.array([pv_rev_d, pv_util_d, tk_d, tl_d, tc_d])
        np.savetxt(timestr + "_gamma_" + str(conf['gamma']) + "_eta_" + str(conf["eta"]) + "_res.csv", full_array.T, delimiter=',')
        plot_frontier(pv_rev_d, pv_util_d, tk_d, tl_d, tc_d, timestr)
    elapsed_time = time.time() - start
    print("Elapsed time", elapsed_time, "\n n_procs ", n_procs, "\n n_points ", n)

