import time
import itertools

from casadi import *
from multiprocessing import Pool, Process, cpu_count
from plot_efficient_frontier import plot_frontier_carlos as plot_frontier

from life_cycle_gradient_descent import run_sim as run_sim_gradient


def execute_moo_optimization(conf):
    """"
        Execute multi-objective optimization problem.
    """


    # extract parameters from configuration dictionary
    gamma = conf["gamma"]
    eta = conf["eta"]
    n_points = conf["n_starting_points"]
    r = conf["r"]
    T = conf["T"]

    # build interest rate vector of length
    r = np.full(T, r)

    # set n_procs
    n_procs = cpu_count() - 1
    #n_procs = 1

    # create gamma eta pairs from input
    gammas_etas = itertools.product(gamma, eta)

    # n points per process
    n = n_points // n_procs


    start = time.time()

    for gamma, eta in gammas_etas:
        conf["gamma"] = gamma
        conf["eta"] = eta

        if conf['serial'] or n_procs == 1:
            # SERIAL EXECUTION
            start_point = np.array([0.41, 0.41, 0.41])

            if conf["tax_setting"] == 1:
                start_point[2] = 0

            data = [run_sim_gradient(start_point, 0.005, n, r, conf)]
        else:
            # PARALLEL EXECUTION
            with Pool(n_procs) as p:
                start_points = np.tile(np.array([0.6 / (n_procs + 1) * i + 0.1 for i in range(n_procs)]), (3, 1))

                if conf["tax_setting"] == 1:
                    start_points[2, :] = 0

                d_tau = 1/n/n_procs
                arg = [(start_points[:, i], d_tau, n, r, conf) for i in range(n_procs)]
                data = p.starmap(run_sim_gradient, arg)

        pv_rev_d = np.concatenate([np.array(data[i][0]) for i in range(n_procs)])
        pv_util_d = np.concatenate([np.array(data[i][1]) for i in range(n_procs)])
        tk_d = np.concatenate([np.array(data[i][2]) for i in range(n_procs)])
        tl_d = np.concatenate([np.array(data[i][3]) for i in range(n_procs)])
        tc_d = np.concatenate([np.array(data[i][4]) for i in range(n_procs)])
        GD_steps = [data[i][5][j] for i in range(n_procs) for j in range(len(data[i][5]))]

        full_array = np.array([pv_rev_d, pv_util_d, tk_d, tl_d, tc_d])
        elapsed_time = time.time() - start
        print("Elapsed time", elapsed_time, "\n n_procs ", n_procs, "\n n_points ", len(pv_rev_d))


        return full_array, GD_steps


if __name__ == '__main__':
    conf = {
        'tax_setting': 5,  # tax setting: see below for detailed explanation
        'beta': 0.95,  # Utility discounting
        'r_debt': 0.1,  # debt interest rate
        'r': 0.1,   # interest rate
        'T': 60,  # agent's maturity
        'T_ret': 45,  # retirement age
        'L_max': 10,  # maximum labour
        'gamma': [3],  # Agent's utility from consumption
        'eta': [8],  # Agent's disutility from labour
        'plot': False,
        'plot_steps': False,
        'plot_GD_steps': False,
        'h': 1E-6,  # step size finite difference
        'n_steps_max': 30,  # max number of steps for 1 point on pareto frontier
        'k_max': 10,  # max number of steps for bi-linear search
        'tol': 1E-6,  # tolerance for gradient stopping
        'tol_line_search': 1E-8,  # tolerance to abort line search
        'serial': False,
        'max_debt': 1000,    # Max_debt -> maximum debt allowed # [0, 1000] never binding;
        'borrowing': 10000,  # np.inf -> allow unlimited borrowing // 0 -> allow no borrowing
        'n_starting_points': 50 # number of starting points for "frontier searching"
    }
    timestr = time.strftime("%m-%d-%H%M")

    for gamma_eta in [(0.5, 1)]:#, (3, 1), (0.5, 8), (0.5, 1)]: #[(0.5, 1), (0.5, 8), (3, 1), (3, 8)]:
        conf["gamma"] = [gamma_eta[0]]
        conf["eta"] = [gamma_eta[1]]

        result_array, temporary = execute_moo_optimization(conf)

        [pv_rev_d, pv_util_d, tk_d, tl_d, tc_d] = result_array.tolist()

        np.savetxt(timestr + "_gamma_" + str(conf['gamma']) + "_eta_" +
                   str(conf["eta"]) + "_res.csv", result_array.T, delimiter=',')

        plot_frontier(pv_rev_d, pv_util_d, tk_d, tl_d, tc_d, conf,
                      timestr + "_gamma_" + str(conf['gamma']) + "_eta_" +
                      str(conf["eta"]) + "_tax_setting_" + str(conf["tax_setting"]))