import time
import itertools

from casadi import *
from multiprocessing import Pool, Process, cpu_count
from plot_efficient_frontier import plot_frontier_carlos as plot_frontier

from life_cycle_gradient_descent import run_sim as run_sim_gradient
from life_cycle_gradient_descent import wage_function
from life_cycle_gradient_descent import init_optimization_problem
from life_cycle_gradient_descent import multi_gradient_descent_step

from homotopy_tracing import pHomotopyMap
import sys
sys.path.append("D:\git_repos\m-hompack\src_clean\python_interface")
import hompack

global_options = {}


def execute_moo_optimization(conf):
    """"
        Execute multi-objective optimization problem.
    """

    global global_options
    # extract parameters from configuration dictionary
    gamma = conf["gamma"]
    eta = conf["eta"]
    n_points = conf["n_starting_points"]
    r = conf["r"]
    T = conf["T"]

    # build interest rate vector of length
    r = np.full(T, r)

    # set n_procs
    n_procs = cpu_count() - 3
    n_procs = 1
    h = 1E-4

    # create gamma eta pairs from input
    gammas_etas = itertools.product(gamma, eta)

    # n points per process
    n = n_points // n_procs


    start = time.time()

    for gamma, eta in gammas_etas:
        conf["gamma"] = gamma
        conf["eta"] = eta

        # SERIAL EXECUTION
        start_point = np.array([0.1, 0.1, 0.1])

        if conf["tax_setting"] == 1:
            start_point[2] = 0

        def run_sim(start_point, d_tau, n, r, conf):

            # after finding a point on the efficient frontier, move by d_tau to find next
            d_tau_v = np.full(3, d_tau)
            if conf['tax_setting'] == 1:
                # consumption tax = 0
                start_point[2] = 0
                d_tau_v[2] = 0
            elif conf['tax_setting'] == 4:
                # capital tax equal to labour tax
                start_point[0] = start_point[1]
            elif conf['tax_setting'] == 5:
                # all taxes allowed
                pass

            # get wage vector
            wage = wage_function(conf)

            # initialize NLP problem - S = casadi optimization object - lb_x ub_x are the lower and upper bounds
            S, lb_x, ub_x = init_optimization_problem(r, wage, conf)
            return S, lb_x, ub_x

        S, lb_x, ub_x = run_sim(start_point, 1, n, r, conf)
#        data = [run_sim_gradient(start_point, 0.005, n, r, conf)]
        homotopy_problem = pHomotopyMap()

        NFE = 1


        # FIRST ONE ALWAYS FREE FOR lambda
        Y = np.zeros(4, dtype=np.float64)
        Y[1] = start_point[0]
        Y[2] = start_point[1]
        Y[3] = start_point[2]



        iflag = np.zeros(1, dtype=np.int32)
        iflag[0] = -2

        arctol = np.zeros(1, dtype=np.float64)
        anstol = np.zeros(1, dtype=np.float64)
        arctol[0] = 1E-6
        anstol[0] = 1E-6

        A = np.zeros(1, dtype=np.float64)
        A[0] = 1

        NFE = np.zeros(1, dtype=np.int32)
        arclen = np.zeros(1, dtype=np.float64)

        test = np.zeros((len(Y))* 1000)
        V = np.zeros((3,2))
        fwd_solver = S[0].forward(3)

        global_options["S"] = S
        global_options["h"] = h
        global_options["fwd_solver"] = fwd_solver
        global_options["conf"] = conf
        global_options["lb_x"] = lb_x
        global_options["ub_x"] = ub_x
        global_options["wage"] = wage_function(conf)
        global_options["r"] = r

        ## FIND INITIAL POINT ON FRONTIER
        #        pv_rev, pv_util, tax_rates, start_point, intermediate = multi_gradient_descent_step(start_point, lb_x, ub_x, S,
        #                                                                               wage_function(conf), r, fwd_solver, 1, conf)
        start_point = [0.0001, 0.0001, 0.0001]
        sol = S[1](x0=1, lbg=0, ubg=0, lbx=lb_x, ubx=ub_x, p=start_point)
        global_options["x_0"] = sol["x"].full()
        tax_rates = start_point
        Y[1] = tax_rates[0]
        Y[2] = tax_rates[1]
        Y[3] = tax_rates[2]
        global_options["x_0"] = sol["x"].full()

        homotopy_problem.global_options = global_options
        hompack.HomotopyTracing.fixpdf_c_numpy(Y, iflag, arctol, anstol, 0, A, NFE, arclen,
                                               homotopy_problem.__disown__(), test)


        test = test.reshape(((len(Y)), 1000))
        test = test.transpose()

#        pv_rev_d = np.concatenate([np.array(data[i][0]) for i in range(n_procs)])
#        pv_util_d = np.concatenate([np.array(data[i][1]) for i in range(n_procs)])
#        tk_d = np.concatenate([np.array(data[i][2]) for i in range(n_procs)])
#        tl_d = np.concatenate([np.array(data[i][3]) for i in range(n_procs)])
#        tc_d = np.concatenate([np.array(data[i][4]) for i in range(n_procs)])
#        GD_steps = [data[i][5] for i in range(n_procs)]

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
        'gamma': [0.5],  # Agent's utility from consumption
        'eta': [3],  # Agent's disutility from labour
        'plot': False,
        'plot_steps': False,
        'plot_GD_steps': False,
        'h': 1E-6,  # step size finite difference
        'n_steps_max': 1,  # max number of steps for 1 point on pareto frontier
        'k_max': 1,  # max number of steps for bi-linear search
        'tol': 1E-6,  # tolerance for gradient stopping
        'tol_line_search': 1E-8,  # tolerance to abort line search
        'serial': False,
        'max_debt': 1000,    # Max_debt -> maximum debt allowed # [0, 1000] never binding;
        'borrowing': 1000,  # np.inf -> allow unlimited borrowing // 0 -> allow no borrowing
        'n_starting_points': 10 # number of starting points for "frontier searching"
    }
    timestr = time.strftime("%m-%d-%H%M")

    for gamma_eta in [(0.5, 3)]:#, (3, 1), (0.5, 8), (0.5, 1)]: #[(0.5, 1), (0.5, 8), (3, 1), (3, 8)]:
        conf["gamma"] = [gamma_eta[0]]
        conf["eta"] = [gamma_eta[1]]

        result_array, temporary = execute_moo_optimization(conf)

        [pv_rev_d, pv_util_d, tk_d, tl_d, tc_d] = result_array.tolist()

        np.savetxt(timestr + "_gamma_" + str(conf['gamma']) + "_eta_" +
                   str(conf["eta"]) + "_res.csv", result_array.T, delimiter=',')

        plot_frontier(pv_rev_d, pv_util_d, tk_d, tl_d, tc_d, conf,
                      timestr + "_gamma_" + str(conf['gamma']) + "_eta_" +
                      str(conf["eta"]) + "_tax_setting_" + str(conf["tax_setting"]))