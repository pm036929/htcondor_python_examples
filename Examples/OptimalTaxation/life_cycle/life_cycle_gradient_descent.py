import time
from auxiliary_functions import *
import numpy as np
from lifecycle import wage_function

def optimize(lbx, ubx, S, tax_rates, wage, r, conf, x0=1, lam_x0=0, lam_g0=0):
            """
                Optimization problem.
            :param lbx: lower bound for x
            :param ubx:
            :param S: nlp casadi object
            :param tax_rates: list of tax rates : [tk, tl, tc]
            :param wage:
            :param conf:
            :param x0:
            :return:
            """

            # Actual casadi optimization happens here
            # S is a casadi nlp object
            success = False
            iteration = 0
            while not success and iteration < 5:
                try:
                    res = S[0](x0=x0, lbg=0, ubg=0, lbx=lbx, ubx=ubx, p=tax_rates, lam_x0=lam_x0, lam_g0=lam_g0)
                    stats = S[0].stats()
                    success = stats["success"]
                    if not success:
                        x0 = np.random.rand(302)
                except RuntimeError:
                    x0 = np.random.rand(302)
                iteration = iteration + 1

            stats = S[0].stats()
            if not stats["success"]:
                raise ValueError("No solution found -> no correct gradient descent direction." +
                                 str(tax_rates))
            solution = res["x"]

            x0 = res["x"].full().squeeze()
            lam_x0 = res["lam_x"].full().squeeze()
            lam_g0 = res["lam_g"].full().squeeze()

            stats = S[0].stats()
            if not stats["success"]:
                raise ValueError(str(tax_rates))


            # solution elements: (consumption, labour, assets, debts, borrowing)
            #                     0:T, T:2*T, 2*T:3*T

            consumption, labour, assets, debts, borrowing, tax_rates = extract_solution(solution, conf, tax_rates)
            #plot_solutions(consumption, labour, assets, debts, borrowing, tax_rates)
            pv_utility, pv_rev = evaluate_objectives(assets, consumption, labour, wage, tax_rates, r, conf)

            return pv_utility, pv_rev, solution, x0, lam_x0, lam_g0



def calc_sesitivity(tax_rates, h, lb_x, ub_x, S, wage, r, conf, fwd_solver, x_0=1, lam_x0=0, lam_g0=0):
    """
    :param tax_rates:
    :param h:
    :param lb_x:
    :param ub_x:
    :param S:
    :param wage:
    :param conf:
    :return:
    """


    # Forward mode AD for the NLP solver object

    nfwd = 3

    iteration = 0
    success = False
    while not success and iteration < 10:
        try:
            res = S[0](x0=x_0, lbg=0, ubg=0, lbx=lb_x, ubx=ub_x, p=tax_rates)#, lam_x0=lam_x0, lam_g0=lam_g0)
            stats = S[0].stats()
            success = stats["success"]
            if not success:
                x_0 = np.random.rand(302)
        except:
            x_0 = np.random.rand(302)
        iteration = iteration + 1

    stats = S[0].stats()
    if not stats["success"]:
        raise ValueError("No solution found -> no correct gradient descent direction."+
                         str(tax_rates))

    # extract solution as initial guess for next step
    sol = res
    x_0 = sol["x"].full().squeeze()
    lam_x0 = sol["lam_x"].full().squeeze()
    lam_g0 = sol["lam_g"].full().squeeze()


    P = MX.sym('P', 3)

    # Seeds, initalized to zero
    fwd_lbx = [DM.zeros(sol['x'].sparsity()) for i in range(nfwd)]
    fwd_ubx = [DM.zeros(sol['x'].sparsity()) for i in range(nfwd)]
    fwd_p = [DM.zeros(P.sparsity()) for i in range(nfwd)]
    fwd_lbg = [DM.zeros(sol['g'].sparsity()) for i in range(nfwd)]
    fwd_ubg = [DM.zeros(sol['g'].sparsity()) for i in range(nfwd)]

    # Let's preturb P
#    if tax_rates[0] > 1E-8 and tax_rates[0] < 1-1E-8:
    fwd_p[0][0] = 1  # first nonzero of P

#    if tax_rates[1] > 1E-8 and tax_rates[1] < 1-1E-8:
    fwd_p[1][1] = 1  # second nonzero of P

#    if tax_rates[2] > 1E-8 and tax_rates[2] < 1-1E-8:
    fwd_p[2][2] = 1  # second nonzero of P

    # Calculate sensitivities using AD
    sol_fwd = fwd_solver(out_x=sol['x'], out_lam_g=sol['lam_g'], out_lam_x=sol['lam_x'],
                         out_f=sol['f'], out_g=sol['g'], lbx=lb_x, ubx=ub_x, lbg=0, ubg=0,
                         fwd_lbx=horzcat(*fwd_lbx), fwd_ubx=horzcat(*fwd_ubx),
                         fwd_lbg=horzcat(*fwd_lbg), fwd_ubg=horzcat(*fwd_ubg),
                         p=tax_rates, fwd_p=horzcat(*fwd_p))
#    print(time.time() - tic)
    consumption, labour, assets, debts, borrowing, tax_rates = extract_solution(x_0, conf, tax_rates)

    dx_p = sol_fwd["fwd_x"].full()
    f2 = pv_tax(assets, labour, wage, consumption, tax_rates, r, conf)

    d_consumption, d_labour, d_assets, d_debts, d_borrowing, d_tax_rates = extract_solution(dx_p, conf, tax_rates)

    C = SX.sym('consumption', conf['T'], 1)
    L = SX.sym('labour', conf['T'], 1)
    A = SX.sym('assets', conf['T'] + 1, 1)
    T = SX.sym('tax_rates', 3, 1)

    pv_t = pv_tax(A, L, wage, C, T, r, conf)

    d_r_c = jacobian(pv_t , C)
    d_r_l = jacobian(pv_t , L)
    d_r_a = jacobian(pv_t , A)
    d_t = jacobian(pv_t, T)

#    d_r_l = jacobian(pv_tax(assets, L, wage, consumption, tax_rates, r, conf), L)
#    d_r_a = jacobian(pv_tax(A, labour, wage, consumption, tax_rates, r, conf), A)
#    d_t = jacobian(pv_tax(assets, labour, wage, consumption, T, r, conf), T)
    d_rev_cla = d_r_c @ d_consumption + d_r_l @ d_labour + d_r_a @ d_assets + d_t * (fwd_p[0] + fwd_p[1] + fwd_p[2]).T

    f_rev_cla = Function('d_rev',  [C, L, A, T], [d_rev_cla])
    rev_cla = f_rev_cla(consumption, labour, assets, tax_rates)

    d_util = sol_fwd["fwd_f"].full().squeeze()

    # normalize gradients
    d_util = d_util / np.linalg.norm(d_util)
    d_rev = rev_cla.full()[0] / np.linalg.norm(rev_cla.full()[0])

    return d_util, -d_rev, f2, -sol['f'][0], x_0, lam_x0, lam_g0




def find_efficient_frontier(start_point, n, d_tau, S, lb_x, ub_x, r, conf):

    # accepted revenues and utilities
    pv_rev_global = []
    pv_util_global = []

    # accepted tax rates
    tk_global = []
    tl_global = []
    tc_global = []

    # intermediate
    intermediate_global = []
    
    wage = wage_function(conf)
    x0 = 1

    # init forward solver
    fwd_solver = S[0].forward(3)


    for i_start in range(n):
        pv_rev, pv_util, tax_rates, start_point, intermediate = multi_gradient_descent_step(start_point, lb_x, ub_x, S,
                                                                               wage, r, fwd_solver, x0, conf)

        if pv_rev is not None:
            pv_rev_global.append(pv_rev)
            pv_util_global.append(pv_util)

            tk_global.append(tax_rates[0])
            tl_global.append(tax_rates[1])
            tc_global.append(tax_rates[2])
            intermediate_global.append(intermediate)
    
    #        start_point = np.array([tk, tl, tc]) + d_rev * min(0.001/d_rev)
#        start_point = np.clip(start_point, 0, 1)

        # "Moove along" Pareto Frontier
        #start_point = np.array([tk, tl, tc]) + d_tau #d_rev * min(0.001 / d_rev)
        #start_point = np.clip(start_point, 0, 1)


    return pv_rev_global, pv_util_global, tk_global, tl_global, tc_global, intermediate_global

def multi_gradient_descent_step(start_point, lb_x, ub_x, S, wage, r, fwd_solver, x0, conf):

    GD_steps = {'util':[], 'rev':[], 'tax_rates': []}
    lam_x0 = 0
    lam_g0 = 0

    for step_i in range(conf['n_steps_max']):
        tax_rates = start_point
        tax_rates[tax_rates < 1E-6] = 0
        if conf["tax_setting"] == 1:
            tax_rates[2] = 0

        #        tax_rates[tax_rates > 1-1E-6] = 1

        try:
            d_util, d_rev, pv_rev, pv_util, x0, lam_x0, lam_g0 = calc_sesitivity(tax_rates, conf['h'], lb_x, ub_x, S, wage, r, conf, fwd_solver,
                                                                 x0, lam_x0, lam_g0)
        except (RuntimeError, ValueError):
            start_point = np.random.rand(3) # tax_rates + d_rev * min(0.01/d_rev)
            continue

        if np.any(np.isnan(d_util)) or np.any(np.isnan(d_rev)):
            start_point = np.random.rand(3) # tax_rates + d_rev * min(0.01/d_rev)
            continue

        if conf['plot_GD_steps']:
            GD_steps['util'].append(pv_util)
            GD_steps['rev'].append(pv_rev)
            GD_steps['tax_rates'].append(tax_rates)

        if conf["tax_setting"] == 1:
            d_util[2] = 0
            d_rev[2] = 0


        # calculate alpha = scalar for linear combination of d_util and d_rev
        # such that d_GD is a improving direction for both u and rev
        if d_util @ d_rev < min(np.linalg.norm(d_util), np.linalg.norm(d_rev)) ** 2:
            alpha = d_rev @ (d_rev - d_util) / np.linalg.norm((d_rev - d_util)) ** 2
        elif np.linalg.norm(d_rev) == min(np.linalg.norm(d_util), np.linalg.norm(d_rev)):
            alpha = 0
        else:
            alpha = 1
        d_GD = (1 - alpha) * d_rev + alpha * d_util

        if np.any(np.logical_and(tax_rates < 1E-6 , d_GD > 0)) or np.any(np.logical_and(tax_rates > 1-1E-6 , d_GD < 0)):
            d_util[np.logical_and(tax_rates < 1E-6 , d_GD > 0)] = 0
            d_rev[np.logical_and(tax_rates < 1E-6 , d_GD > 0)] = 0

            d_util[np.logical_and(tax_rates > 1-1E-6 , d_GD < 0)] = 0
            d_rev[np.logical_and(tax_rates > 1-1E-6 , d_GD < 0)] = 0

            # calculate alpha = scalar for linear combination of d_util and d_rev
            # such that d_GD is a improving direction for both u and rev
            if d_util @ d_rev < min(np.linalg.norm(d_util), np.linalg.norm(d_rev)) ** 2:
                alpha = d_rev @ (d_rev - d_util) / np.linalg.norm((d_rev - d_util)) ** 2
            elif np.linalg.norm(d_rev) == min(np.linalg.norm(d_util), np.linalg.norm(d_rev)):
                alpha = 0
            else:
                alpha = 1
            d_GD = (1 - alpha) * d_rev + alpha * d_util



        # Abort criteria: Gradient approx = 0
        if np.linalg.norm(d_GD) < conf["tol"] or np.max(np.abs(d_GD)) < conf["tol"]:
            try:
                d_util, d_rev, pv_rev, pv_util, x0, lam_x0, lam_g0 = calc_sesitivity(tax_rates, conf['h'], lb_x, ub_x, S, wage, r, conf,
                                                                 fwd_solver, x0)
            except ValueError:
                break
            new_start_point = np.random.rand(1, 3)#np.minimum(np.maximum(0, tax_rates - d_rev/np.linalg.norm(d_rev)*0.1),1)

            return pv_rev, pv_util, tax_rates, new_start_point, GD_steps

        try:
            opt_step_length, x0, lam_x0, lam_g0 = bisection_linesearch(tax_rates, d_GD, pv_rev, pv_util, S, lb_x, ub_x, r, conf, x0, lam_x0, lam_g0)
        except ValueError:
            # Optimization did not succeed for some tax rate
            opt_step_length = None
            raise

        if opt_step_length is None:
            start_point = np.random.rand(3) # tax_rates + d_rev * min(0.01/d_rev)
            start_point = np.clip(start_point, 0, 1)
            return None, None, None, start_point, None
        elif opt_step_length == 0:
            # No progress could be made even though gradient criteria not met -> perturb
#            start_point = np.minimum(np.maximum(0, tax_rates + np.random.rand(3)/0.005), 1)
            start_point = np.minimum(np.maximum(0,  np.random.rand(3)), 1)

        else:
            start_point = np.minimum(np.maximum(0, tax_rates - opt_step_length * d_GD), 1) #+ 0.01

    start_point = np.random.rand(3)  # tax_rates + d_rev * min(0.01/d_rev)

    return None, None, None, start_point, []


def bisection_linesearch(tax_rates, d_GD, pv_rev, pv_util, S, lb_x, ub_x, r, conf, x0 = 1, lam_x0=0, lam_g0=0):

    diff = 100

    wage = wage_function(conf)
    tax_rates_g = tax_rates.copy()

    # Determine initial step_length
#    step_length = min((np.max(tax_rates) * 0.5 / np.max(np.abs(d_GD))), 0.1)
    step_length = 0.01
    step_lengths = []
    step_lengths.append(step_length)

    if conf["plot_steps"]:
        guesses_rev = []
        guesses_util = []

        # Add starting point to guess list
        guesses_rev.append(pv_rev)
        guesses_util.append(pv_util)

    if conf["plot_steps"]:
        plt.plot(guesses_rev, guesses_util, ".")
        plt.show()

    success_i = -1
    failed_i = -1

    k = 0
    while diff > conf["tol_line_search"] or success_i != max(k - 1, 0):
#    while True:

        if k > conf["k_max"]:
            break

        tax_rates_g = tax_rates - step_length * d_GD
        tax_rates_g = np.minimum(np.maximum(0, tax_rates_g), 1)

        try:
            [pv_util_g, pv_rev_g, x0,_, lam_x0, lam_g0] = optimize(lb_x, ub_x, S, tax_rates_g, wage, r, conf, x0, lam_x0, lam_g0)
        except ValueError:
            success_i = -1
        #    raise
            break

        if conf["plot_steps"]:
            # Add point to guess list
            guesses_rev.append(pv_rev_g)
            guesses_util.append(pv_util_g)
            plt.plot(guesses_rev, guesses_util, ".")
            plt.show()

        # Check if current guess improves both objectives
#        if pv_rev_g >= pv_rev - 1E-6 and pv_util_g >= pv_util- 1E-6:
        if pv_rev_g >= pv_rev  and pv_util_g >= pv_util :
            if failed_i == -1:
                # If never failed, double scaling factor
                step_length = step_length * 2
            else:
                # If already failed before, go in the middle between current and previously failed
                step_length = (step_lengths[failed_i] - step_length) / 2 + step_length
            success_i = k
        else:
            # Not improving
            if success_i > -1:
                step_length = (step_length - step_lengths[success_i]) / 2 + step_lengths[success_i]
            else:
                step_length = step_length / 4
            failed_i = k
        diff = np.max(abs((step_lengths[k] - step_length) * np.max(tax_rates)))

        step_lengths.append(step_length)
        k += 1
#        guesses_rev.append(pv_rev_g)
#        guesses_util.append(pv_util_g)
    if success_i != -1:
        return step_lengths[success_i], x0, lam_x0, lam_g0
    else:
        return 0, x0, lam_x0, lam_g0


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

    pv_rev_global, pv_util_global, tk_global, tl_global, tc_global, intermediate_values = \
        find_efficient_frontier(start_point, n, d_tau_v, S, lb_x, ub_x, r, conf)

    return pv_rev_global, pv_util_global, tk_global, tl_global, tc_global, intermediate_values




