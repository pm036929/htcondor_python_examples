from life_cycle import *

import numpy as np
import matplotlib.pyplot as plt


def wage_function_wo(conf):

    wage = np.zeros(conf["T"])

    wage = np.maximum(1 / 10, (4 * np.arange(1, T+1)/ T)*(1 - np.arange(1, T+1) / T))


    return wage


if __name__ == "__main__":


    # init parameters
    beta = 0.96
    costb = 2/100
    T = 40
    T_ret = 40
    L_max = 10
    max_debt = 2

    gamma = 3
    eta = 2

    h = 0.0000001

    # create dictionary for convenience
    conf = {
            'beta': beta,
            'T': T,
            'T_ret': T_ret,
            'L_max': L_max,
            'gamma': gamma,
            'eta': eta,
            'plot': False,
            'plot_steps': False,
            'h': h,
            'n_steps_max': 10,
            'k_max': 100,
            'tol': 0.0000001,
            'tax_setting': 1,
            'serial': False,
            'costb': costb,
            'max_debt': max_debt
            }

    alpha = 0.129275
    a = 1.0765

    tks = np.linspace(0, 0.9, 2)
    tls = np.linspace(0, 0.4, 2)
    tcs = np.linspace(0, 0.5, 2)

    # window we look at:
    window_size = T + 10
    assert(window_size > T)

    tax_rates = [0, 0, 0]#[tks[0], tls[0], tcs[0]]
    r_full = np.full(window_size, 0.15) + np.arange(0,window_size) * 0.002
    wage_factor = np.full(window_size, 1 * 1.002)

    global_capital = np.zeros(window_size)
    global_labour = np.zeros(window_size)
    r_debt = r_full[1] + costb

    theta = 0.1
    for i_it in range(50):
        if i_it > 10:
            theta = 1
        # 1. born before T = 0
        # i_gen starts at 0 - T
        for i_gen in range(T-1):
            r = np.full(T, r_full[0])
            wage = wage_factor[0] * wage_function_wo(conf)
            wage[T - i_gen - 1:] = wage_factor[0: i_gen + 1] * wage_function_wo(conf)[T - i_gen - 1:]
            r[T - i_gen - 1:] = r_full[0:i_gen + 1]
            # TODO correct overlapping part
            # r [] =
            # wage[] =

            S, lb_x, ub_x = init_optimization_problem(r, wage, conf)
            [_, _, _, _, solution] = optimize(lb_x, ub_x, S, tax_rates, wage, r, conf)
            # solution elements: (consumption, labour, assets, debts, borrowing)
            #                     0:T, T:2*T, 2*T:3*T+1
            global_capital[:1 + i_gen] += np.squeeze(solution[3 * T - i_gen: 3 * T + 1].full()) - np.squeeze(solution[4 * T + 1 - i_gen: 4 * T + 2].full())
            global_labour[:1 + i_gen] += np.squeeze(solution[2 * T-1-i_gen:2 * T].full()) * wage[len(wage)-1-i_gen:]

        # 2. Born within window
        for i_gen in range(window_size):
            if i_gen+T > window_size:
                # born within window, but life extends past window
                r = np.full(T, r_full[-1])
                wage = wage_factor[-1] * wage_function_wo(conf)

            else:
                # born within window, life does not extend past window
                r = r_full[i_gen:i_gen+T]
                wage = wage_factor[i_gen:T+i_gen] * wage_function_wo(conf)

            S, lb_x, ub_x = init_optimization_problem(r, wage, conf)
            [_, _, _, _, solution] = optimize(lb_x, ub_x, S, tax_rates, wage, r, conf)
            # solution elements: (consumption, labour, assets, debts, borrowing)
            #                     0:T, T:2*T, 2*T:3*T

            last_index = min(i_gen + T, window_size)

            last_index_sol_1 = 3 * T + 1 - (i_gen + T - last_index)
            last_index_sol_2 = 2 * T - (i_gen + T - last_index)
            last_index_sol_3 = 4 * T + 2 - (i_gen + T - last_index)

            global_capital[i_gen:last_index] += np.squeeze(solution[2 * T+1:last_index_sol_1].full()) - np.squeeze(solution[3*T+2:last_index_sol_3].full())
            global_labour[i_gen:last_index] += np.squeeze(solution[T:last_index_sol_2].full())* wage[:last_index_sol_2 - T]



        # compensate for overlapping generatiosn



        # TODO correct overlapping generations
        # r []
        # wage[]
#        plt.plot(global_capital)
#        plt.show()

#        plt.plot(global_labour)
#        plt.show()


        r_new = a * alpha * (global_labour / global_capital ) ** (1 - alpha)
        wage_new = a * (1 - alpha) * (global_capital/global_labour)**alpha

        plt.plot(r_full)
        plt.plot(r_new)
        r_full = r_full * (1 - theta) + theta * r_new
        plt.plot(r_full)
        plt.title("r")
        plt.show()


        plt.plot(wage_factor)
        plt.plot(wage_new)
        global_labour
        wage_factor = wage_factor * (1 - theta) + theta * wage_new
        plt.plot(wage_factor)
        plt.show()

    r_full