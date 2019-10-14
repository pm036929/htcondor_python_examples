import numpy as np
from casadi import *
import matplotlib.pyplot as plt
from multiprocessing import Pool, Process, cpu_count
import itertools
import time
from plot_efficient_frontier import plot_frontier
import argparse
from auxiliary_functions import *



def optimize(lbx, ubx, S, tax_rates, wage, r, conf, x0=1):
            """
                Optimize problem.
            :param lbx: lower bound for x
            :param ubx:
            :param S: nlp casadi object
            :param tax_rates: list of tax rates : [tk, tl, tc]
            :param wage:
            :param conf:
            :param x0:
            :return:
            """
#
#            p = SX.sym('taxes', 3, 1)
#            dx = jacobian(S(x0=x0, lbg=0, ubg=0, lbx=lbx, ubx=ubx, p=p), p)

            # Actual casadi optimization happens here
            # S is a casadi nlp object
            res = S(x0=x0, lbg=0, ubg=0, lbx=lbx, ubx=ubx, p=tax_rates)

            solution = res["x"]
            # solution elements: (consumption, labour, assets, debts, borrowing)
            #                     0:T, T:2*T, 2*T:3*T

            T = conf["T"]
            R = 1 + r


            # TODO double check if correct timing

            tax_interest = r * solution[2*T+1:3*T+1] * tax_rates[0]
            tax_labour = wage * solution[T:2*T] * tax_rates[1]
            tax_consumption = solution[:T] * tax_rates[2]

            pv_rev_k = sum1(R**(np.arange(-1, -T - 1, -1)) * tax_interest)
            pv_rev_l = sum1(R**(np.arange(0, -T, -1)) * tax_labour)
            pv_rev_c = sum1(R**(np.arange(0, -T, -1)) * tax_consumption)

            pv_util = sum1(conf["beta"]**(np.arange(0, T)) *
                          utility(solution[:T], solution[T:2*T], conf))


            # Warm start: give solution as initial guess to function caller
            x0 = solution
            return pv_rev_k, pv_rev_l, pv_rev_c, pv_util, solution


def wage_function(conf):

    wage = np.zeros(conf["T"])
    T_ret = conf['T_ret']

    wage[:conf["T_ret"]] = -36.9994 + 3.52022 * (np.arange(T_ret) + 1 + 17) \
                           - 0.101878 * (np.arange(T_ret) + 1 + 17)**2 \
                           + .00134816 * (np.arange(T_ret) + 1 + 17)**3 \
                           - 7.06233*1E-6 * (np.arange(T_ret) + 1 + 17)**4

    return wage

def calc_sesitivity(tax_rates, h, lb_x, ub_x, S, wage, r, conf):
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
    rev_FD = np.zeros((3, 3))
    util_FD = np.zeros((3, 3))
    tk = tax_rates[0]
    tl = tax_rates[1]
    tc = tax_rates[2]

    x0 = 1

    res = S(x0=x0, lbg=0, ubg=0, lbx=lb_x, ubx=ub_x, p=tax_rates)

    hsolver = S.factory('h', S.name_in(), ['sym:jac:f:p1'])
    #hsol = hsolver(x0=res['x'], lam_x0=res['lam_x'], lam_g0=res['lam_g'],
    #               lbx=[], ubx=[], lbg=0, ubg=0, p=tax_rates)

    for i_tk_h, tk_h in enumerate([tk - h, tk, tk + h]):
        tax_rates = [tk_h, tl, tc]
        [pv_rev_k, pv_rev_l, pv_rev_c, pv_util, x0] = optimize(lb_x, ub_x, S, tax_rates, wage, r, conf, x0)
        pv_rev = pv_rev_k + pv_rev_l + pv_rev_c

        rev_FD[0, i_tk_h] = pv_rev
        util_FD[0, i_tk_h] = pv_util

    for i_tl_h, tl_h in enumerate([tl - h, tl, tl + h]):
        tax_rates = [tk, tl_h, tc]
        [pv_rev_k, pv_rev_l, pv_rev_c,pv_util, x0] = optimize(lb_x, ub_x, S, tax_rates, wage, r, conf, x0)
        pv_rev = pv_rev_k + pv_rev_l + pv_rev_c

        rev_FD[1, i_tl_h] = pv_rev
        util_FD[1, i_tl_h] = pv_util

    for i_tc_h, tc_h in enumerate([tc - h, tc, tc + h]):
        tax_rates = [tk, tl, tc_h]
        [pv_rev_k, pv_rev_l, pv_rev_c, pv_util, x0] = optimize(lb_x, ub_x, S, tax_rates, wage, r, conf, x0)
        pv_rev = pv_rev_k + pv_rev_l + pv_rev_c

        rev_FD[2, i_tc_h] = pv_rev
        util_FD[2, i_tc_h] = pv_util

    d_util = np.array([(util_FD[0, 2] - util_FD[0, 0]) / (2 * h),
                       (util_FD[1, 2] - util_FD[1, 0]) / (2 * h),
                       (util_FD[2, 2] - util_FD[2, 0]) / (2 * h),
                       ])

    d_rev = np.array([(rev_FD[0, 2] - rev_FD[0, 0]) / (2 * h),
                      (rev_FD[1, 2] - rev_FD[1, 0]) / (2 * h),
                      (rev_FD[2, 2] - rev_FD[2, 0]) / (2 * h)])

    if tk - h < 0:
        d_util[0] = (util_FD[0, 2] - util_FD[0, 1]) / h
        d_rev[0] = (rev_FD[0, 2] - rev_FD[0, 1]) / h

    if tl - h < 0:
        d_util[1] = (util_FD[1, 2] - util_FD[1, 1]) / h
        d_rev[1] = (rev_FD[1, 2] - rev_FD[1, 1]) / h

    if tc - h < 0:
        d_util[2] = (util_FD[2, 2] - util_FD[2, 1]) / h
        d_rev[2] = (rev_FD[2, 2] - rev_FD[2, 1]) / h

    if conf['tax_setting'] == 1:
        # No consumption tax
        d_util[2] = 0
        d_rev[2] = 0

    elif conf['tax_setting'] == 4:
        # flat capital = income, consumption
        d_util[0] += d_util[1]
        d_util[1] = d_util[0]

        d_rev[0] += d_rev[1]
        d_rev[1] = d_rev[0]

    elif conf['tax_setting'] == 5:
        # all taxes allowed
        pass


    return d_util, d_rev, rev_FD[0, 1], util_FD[0, 1], x0


def find_efficient_frontier(tax_comb, Pdf, Pdf_t0, newton, r, conf, S, lb_x, ub_x):

    pv_rev_global = []
    pv_util_global = []
    tk_global = []
    tl_global = []
    tc_global = []

    wage = wage_function(conf)

    res = S(x0=1, lbg=0, ubg=0, lbx=lb_x, ubx=ub_x, p=[0.1, 0.1, 0.1])
    x_0 = res["x"].full()
    x_0 = np.append(res["x"].full()[:, 0], [0.1, 0.1, 0.1])
    consumption, labour, assets, tax_rates = extract_solution(x_0)

    #    for tl, tk, tc in tax_comb:
    for step_i in range(conf['n_steps_max']):

            tk = 0.1
            tl = 0.1
            tc = 0.1
            tax_rates = [tk, tl, tc]

            consumption, labour, assets, tax_rates = extract_solution(x_0)
            print(sum1(asset_constraint(assets, labour, wage, consumption, tax_rates, r, conf)))

            dx = newton(assets, consumption, labour, tax_rates)
            x_0 = x_0 - dx
            consumption, labour, assets, tax_rates = extract_solution(x_0)

            dx = newton(assets, consumption, labour, tax_rates)
            x_0 = x_0 - dx
            consumption, labour, assets, tax_rates = extract_solution(x_0)

            #consumption, labour, assets, tax_rates = extract_solution(x_0)

#            dx = newton(assets, consumption, labour, tax_rates)
#            x_0 = x_0 - dx
#            consumption, labour, assets, tax_rates = extract_solution(x_0)

            print(sum1(asset_constraint(assets, labour, wage, consumption, tax_rates, r, conf)))

            if tax_rates[0] < 0.0001:
                exGrad = Pdf_t0(assets, consumption, labour, tax_rates)
                exGrad[-3]
            else:
                exGrad = Pdf(assets, consumption, labour, tax_rates)

            # extract gradients of util and rev w.r.t. parameters
            d_util = exGrad[:,0].full()
            d_rev = exGrad[:,1].full()


            d_util = -d_util
            d_rev = -d_rev



            if d_util.T @ d_rev < min(np.linalg.norm(d_util), np.linalg.norm(d_rev)) ** 2:
                alpha = d_rev.T @ (d_rev-d_util)/np.linalg.norm((d_rev-d_util)) ** 2
            elif np.linalg.norm(d_rev) == min(np.linalg.norm(d_util), np.linalg.norm(d_rev)):
                alpha = 0
            else:
                alpha = 1
            d_GD = (1 - alpha) * d_rev + alpha * d_util

            # bi-section step finding
            tol = conf["tol"]
            success_i = -1
            bt_i = -1

            #scaling_factor = (0.5 / np.max(np.abs(d_GD)))
            scaling_factor = 0.5
            scaling_factors = []
            failed_i = -1

            guesses_rev = []
            guesses_util = []

            diff = np.max(abs(scaling_factor * tax_rates[0]))
            k = 0
            scaling_factors.append(scaling_factor)
            f1, f2 = validate_objectives(assets, consumption, labour, wage, tax_rates, r, conf)

            guesses_rev.append(f2)
            guesses_util.append(f1)

            if conf["plot_steps"]:
                plt.plot(guesses_rev, guesses_util, ".")
                plt.show()
                a = 1

            print("____________________________")


            while diff > tol or success_i != max(k - 1, 0):


                x_0_t = x_0 - scaling_factor * d_GD[:, 0].T

                print("NORM: ", np.linalg.norm(d_GD))
                consumption, labour, assets, tax_rates = extract_solution(x_0_t)
                asset_cons_vio = sum1(asset_constraint(assets, labour, wage, consumption, tax_rates, r, conf))
#                print(asset_cons_vio)
                if abs(asset_cons_vio) > 0.001:
                    break
                print(tax_rates)
                f1_n, f2_n = validate_objectives(assets, consumption, labour, wage, tax_rates, r, conf)

                if tax_rates[0] < 0 or tax_rates[0] > 1 or tax_rates[1] < 0 or tax_rates[1] > 1 or \
                        tax_rates[2] < 0 or tax_rates[2] > 1:
                    failed_i = k
                    if success_i > -1:
                        break
                        scaling_factor = (scaling_factor - scaling_factors[success_i] ) / 2 + scaling_factors[success_i]
                    else:
                        scaling_factor = scaling_factor / 4
                elif f1_n > f1 and f2_n > f2:
                    if failed_i == -1:
                        scaling_factor = scaling_factor * 2
                    else:
                        scaling_factor = (scaling_factors[failed_i] - scaling_factor) / 2 + scaling_factor
                    success_i = k
                else:
                    if success_i > -1:
                        scaling_factor = (scaling_factor - scaling_factors[success_i] ) / 2 + scaling_factors[success_i]
                        break
                    else:
                        scaling_factor = scaling_factor / 4
                    failed_i = k
#                diff = np.max(abs((scaling_factors[k]-scaling_factor) * tax_rates[0]))

                scaling_factors.append(scaling_factor)
                k += 1
                guesses_rev.append(f2_n)
                guesses_util.append(f1_n)

                if conf["plot_steps"]:
                    plt.plot(guesses_rev, guesses_util, ".")
                    plt.show()
                    a = 1
                if k > conf["k_max"]:
                    break

            if success_i  == -1:
                tk = tax_rates[0]
                tl = tax_rates[1]

            else:
                x_0 = x_0 - scaling_factors[success_i] * d_GD[:, 0].T
                consumption, labour, assets, tax_rates = extract_solution(x_0)
                f1_n, f2_n = validate_objectives(assets, consumption, labour, wage, tax_rates, r, conf)

                pv_rev_global.append(f2_n)
                pv_util_global.append(f1_n)
                tk_global.append(tax_rates[0])
                tl_global.append(tax_rates[1])
                tc_global.append(tax_rates[2])


    return pv_rev_global, pv_util_global, tk_global, tl_global, tc_global

def init_newton(r, wage, conf):
        # define casadi type symbolic variables
        consumption = SX.sym('consumption', conf['T'], 1)
        labor = SX.sym('labour', conf['T'], 1)
        assets = SX.sym('assets', conf['T'] + 1, 1)
        tax_rates = SX.sym('tax_rates', 3, 1)

        # create dictionary containing casadi optimization settings
        # x: symbolic (!) variables to optimize for
        # f: objective function
        # g: constraints
        # p: parameters [tk, tl, tc] in this case

        # Define decent direction

        # order:    consumption, labour, assets, debts, borrowing, tax_rates
        dx = vertcat(consumption, labor, assets)

        g1_ = asset_constraint(assets, labor, wage, consumption, tax_rates, r, conf)

        dg1_ = jacobian(g1_, dx)

        g = vertcat(g1_)
        dg = vertcat(dg1_)


        dx_ = pinv(dg) @ g

        newton_step = Function('newton', [assets, consumption, labor, tax_rates], [vertcat(dx_, [0,0,0])])

        return newton_step


def init_deriv(r, wage,conf):

    # define casadi type symbolic variables
    consumption = SX.sym('consumption', conf['T'], 1)
    labor = SX.sym('labour', conf['T'], 1)
    assets = SX.sym('assets', conf['T'] + 1, 1)
    debts = SX.sym('debts', conf['T'] + 1, 1)
    borrowing = SX.sym('borrowing', conf['T'], 1)
    tax_rates = SX.sym('tax_rates', 3, 1)


    # create dictionary containing casadi optimization settings
    # x: symbolic (!) variables to optimize for
    # f: objective function
    # g: constraints
    # p: parameters [tk, tl, tc] in this case

    #Define decent direction

    # order:    consumption, labour, assets, debts, borrowing, tax_rates
    dx = vertcat(consumption, labor, assets, tax_rates)

    df1 = jacobian(pv_utility(consumption, labor, conf), dx)
    df2 = jacobian(pv_tax(assets, labor, wage, consumption, tax_rates,r, conf), dx)
    df = vertcat(df1, df2)

    dg1_ = jacobian(asset_constraint(assets, labor, wage, consumption, tax_rates,[0]*conf['t']), r, conf), dx)
    dg = vertcat(dg1_).T
    # q is matrix featuring orthonormal culmns
    q, _ = qr(dg)

    P = SX.eye(q.size1())
    for i in range(q.size2()):
        P = P - q[:, i] @ q[:, i].T    # project gradients

    Pdf_ = P @ df.T


    ## Restrict tax_rates[0]
    dg2_1 = jacobian(tax_rates[0], dx)
    dg_1_1 = vertcat(dg1_, dg2_1).T
    # q is matrix featuring orthonormal culmns
    q, _ = qr(dg_1_1)
    P = SX.eye(q.size1())
    for i in range(q.size2()):
        P = P - q[:, i] @ q[:, i].T    # project gradients

    Pdf_0 = P @ df.T


    Pdf = Function('Pdf_', [assets, consumption, labor, tax_rates], [Pdf_])
    Pdf_t0 = Function('Pdf_', [assets, consumption, labor, tax_rates], [Pdf_0])

    return Pdf, Pdf_t0


def init_optimization_problem(r, wage,conf):

    # Define lower bounds for optimization
    lb_consumption = np.full(conf['T'], 0.001)
    lb_labour = np.zeros(conf['T'])
    lb_assets = np.full(conf['T'] + 1, -100)

    # Define upper bounds for optimization
    ub_consumption = np.full(conf['T'], np.inf)
    ub_labour = np.full(conf['T'], conf["L_max"])
    ub_assets = np.concatenate((np.zeros(1), np.full(conf['T'], np.inf)))

    # concatenate lower bounds and concatenate upper bounds to lb_x and ub_x, respectively
    lb_x = np.concatenate((lb_consumption, lb_labour, lb_assets))
    ub_x = np.concatenate((ub_consumption, ub_labour, ub_assets))

    # define casadi type symbolic variables
    consumption = SX.sym('consumption', conf['T'], 1)
    labour = SX.sym('labour', conf['T'], 1)
    assets = SX.sym('assets', conf['T'] + 1, 1)
    debts = SX.sym('debts', conf['T'] + 1, 1)
    borrowing = SX.sym('borrowing', conf['T'], 1)

    # define parameter vector
    p_0 = SX.sym('p_0', 1, 1)
    p_1 = SX.sym('p_1', 1, 1)
    p_2 = SX.sym('p_2', 1, 1)

    p = vertcat(p_0,p_1, p_2)

    # create dictionary containing casadi optimization settings
    # x: symbolic (!) variables to optimize for
    # f: objective function
    # g: constraints
    # p: parameters [tk, tl, tc] in this case
    nlp = {'x': vertcat(consumption, labour, assets),
           'f': -objective(consumption, labour, conf),
           'g': vertcat(asset_constraint(assets, labour, wage, consumption,
                                            p, r, conf)),
           'p': p}

    solver_opt = {'verbose': False,'print_time': False, 'ipopt': {'print_level': 5,
                                              'tol': 1E-10, 'constr_viol_tol': 1E-10}}

    # creating casadi optimization object
    S = nlpsol("S", "ipopt", nlp, solver_opt)



#    hsolver = S.factory('h', S.name_in()+ ['p_0', 'p_1', 'p_2'], ['sym:grad:f:p_1'])
    #hsol = hsolver(x0=res['x'], lam_x0=res['lam_x'], lam_g0=res['lam_g'],
    #               lbx=[], ubx=[], lbg=0, ubg=0, p=tax_rates)


    #I_fwd = I.factory('I_fwd', ['x0', 'z0', 'p', 'fwd:p'], ['fwd:xf', 'fwd:qf']))
    return S, lb_x, ub_x

def extract_solution(x, tax_rates=None):
    consumption = x[:T]
    labour = x[T:2*T]
    assets = x[2*T:3*T+1]
    if tax_rates is None:
        tax_rates = x[3 * T + 1:]

    return consumption, labour, assets, tax_rates


def run_sim(tax_combos, r, conf):

    if conf['tax_setting'] == 1:
        tcs = [0]
    elif conf['tax_setting'] == 4:
        tks = tls
    elif conf['tax_setting'] == 5:
        pass
    T = conf["T"]
    wage = wage_function(conf)

    Pdf, Pdf_t0 = init_deriv(r, wage, conf)

    newton = init_newton(r, wage, conf)

    S, lb_x, ub_x = init_optimization_problem(r, wage, conf)


    # solution elements: (consumption, labour, assets, debts, borrowing)

    pv_rev_global, pv_util_global, tk_global, tl_global, tc_global = find_efficient_frontier(tax_combos, Pdf, Pdf_t0, newton, r, conf, S, lb_x, ub_x)

    return pv_rev_global, pv_util_global, tk_global, tl_global, tc_global




if __name__ == "__main__":

    # init parameters
    r = 0.1
    beta = 0.95
    r_debt = 0.15
    T = 60
    T_ret = 45
    L_max = 10

    gamma = 1.5
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
            'plot': True,
            'plot_steps': False,
            'h': h,
            'n_steps_max': 5000,
            'k_max': 30,
            'tol': 0.0000001,
            'tax_setting': 5,
            'serial': False,
            'max_debt':2,
            'costb': 0.02
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
    tk_max = 0.6
    tl_max = 0.9
    tc_max = 0.6


    tks = np.linspace(0, tk_max, 2)
    tls = np.linspace(0, tl_max, 2)
    tcs = np.linspace(0, tc_max, 2)

    if conf['tax_setting'] == 1:
        # No consumption tax
        tax_comb = list(itertools.product(*[tks, tls, [0]]))
    elif conf['tax_setting'] == 4:
        # flat capital tax and income tax; consumption tax
        assert(len(tks) == len(tls))
        tax_comb = ((tks[i], tls[i], tcs[c_i]) for i in range(len(tks))
                    for c_i in range(len(tcs)))

    elif conf['tax_setting'] == 5:
        # capital tax, income tax, consumption tax
        tax_comb = list(itertools.product(*[tks, tls, tcs]))

    # Try except for
    try:
        parser = argparse.ArgumentParser(description='Process some integers.')
        parser.add_argument('gamma')
        parser.add_argument('eta')
        parser.add_argument('n_procs')
        args = parser.parse_args()

        gammas_etas = [(float(args.gamma), float(args.eta))]
        n_procs = int(args.n_procs)
    except:
        gammas_etas = [(gamma, eta)]
        pass

    timestr = time.strftime("%m-%d-%H%M")
    start = time.time()
    for gamma, eta in gammas_etas:
        conf["gamma"] = gamma
        conf["eta"] = eta

        ## HERE the actual execution starts
        if conf['serial'] or n_procs == 1:
            # SERIAL EXECUTION
            n_procs = 1
            data = [run_sim(tax_comb, r, conf)]
        else:
            # PARALLEL EXECUTION
            p = Pool(n_procs)
            data = p.starmap(run_sim, [(tax_comb[i:i + len(tax_comb)//n_procs] , r, conf) for i in range(0, len(tax_comb), len(tax_comb)//n_procs)] )

        pv_rev_d = np.concatenate([np.array(data[i][0]) for i in range(n_procs)])
        pv_util_d = np.concatenate([np.array(data[i][1]) for i in range(n_procs)])
        tk_d = np.concatenate([np.array(data[i][2]) for i in range(n_procs)])
        tl_d = np.concatenate([np.array(data[i][3]) for i in range(n_procs)])
        tc_d = np.concatenate([np.array(data[i][4]) for i in range(n_procs)])

        full_array = np.array([pv_rev_d, pv_util_d, tk_d, tl_d, tc_d])
        np.savetxt(timestr + "_gamma_" + str(conf['gamma']) + "_eta_" + str(conf["eta"]) + "_res.csv", full_array.T, delimiter=',')
        plot_frontier(pv_rev_d, pv_util_d, tk_d, tl_d, tc_d, timestr)
    elapsed_time = time.time() - start
    print("Elapsed time", elapsed_time, "\n n_procs ", n_procs, "\n n_points ", len(tax_comb))

