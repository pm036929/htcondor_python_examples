from casadi import *
import matplotlib.pyplot as plt
from lifecycle import *

def extract_solution(x, conf, tax_rates=None):
    """
        Extract assets consumption etc. from vector x.
        If lifecycle_gradient -> x does not contain tax_rates
    :param x:
    :param conf:
    :param tax_rates:
    :return:
    """
    T = conf["T"]

    consumption = x[:T]
    labour = x[T:2*T]
    assets = x[2*T:3*T+1]
    debts = x[3*T+1:4*T+2]
    borrowing = x[4*T+2:5*T+2]

    if tax_rates is None:
        tax_rates = x[5 * T + 2:]

    return consumption, labour, assets, debts, borrowing, tax_rates


def pv_utility(consumption, labour, conf):
    """

    """
    # Present value of utility. Note: sum1 is a casadi function
    return sum1(conf["beta"]**(np.arange(conf["T"])) * utility(consumption, labour, conf))


def pv_tax(assets, labour, wage, consumption, tax_rates, r, conf):
    """
        Present value of tax.
    """

    R = 1 + r
    T = conf["T"]

    # tax per time period
    tax_vect = tax(assets, labour, wage, consumption, tax_rates, r, conf)

    # discount tax
    pv_rev = sum1(R ** (np.arange(0, -T, -1)) * tax_vect)

    return pv_rev

def evaluate_objectives(assets, consumption, labour, wage, tax_rates, r, conf):
    """
        Evaluate objectives.
    """
    pv_u = pv_utility(consumption, labour, conf)
    pv_r = pv_tax(assets, labour, wage, consumption, tax_rates, r, conf)

    return pv_u, pv_r


def init_optimization_problem(r, wage, conf, revenue_constraint=False):
    """
        Initialize the optimization problem.
    """

    # Define lower bounds for optimization
    lb_consumption = np.full(conf['T'], 0.001)
    lb_labour = np.zeros(conf['T'])
    lb_assets = np.zeros(conf['T'] + 1)
    lb_debt = np.zeros(conf['T'] + 1)
    lb_borrowing = np.full(conf['T'], -conf["borrowing"])
    lb_tax = np.full(3, 0)

    # Define upper bounds for optimization
    ub_consumption = np.full(conf['T'], np.inf)
    ub_labour = np.concatenate((np.full(conf['T_ret'], conf["L_max"]-0.001), np.zeros(conf["T"]-conf["T_ret"])))
    ub_assets = np.concatenate((np.zeros(1), np.full(conf['T'], np.inf)))
    ub_debt = np.array(conf['T'] * [conf['max_debt']] + [0])
    ub_borrowing = np.full(conf['T'], conf["borrowing"])
    ub_tax = np.full(3, 1)


    # define casadi type symbolic variables
    consumption = SX.sym('consumption', conf['T'], 1)
    labour = SX.sym('labour', conf['T'], 1)
    assets = SX.sym('assets', conf['T'] + 1, 1)
    debts = SX.sym('debts', conf['T'] + 1, 1)
    borrowing = SX.sym('borrowing', conf['T'], 1)

    # revenue_constraint = True => lifecycle_barrier
    # revenue_constraint = False => lifecycle_gradient_descent
    #
    if revenue_constraint:
        # create dictionary containing casadi optimization settings
        # x: symbolic (!) variables to optimize for
        # f: objective function
        # g: constraints
        # p: parameters [tk, tl, tc] in this case

        # concatenate lower bounds and concatenate upper bounds to lb_x and ub_x, respectively
        lb_x = np.concatenate((lb_consumption, lb_labour, lb_assets, lb_debt, lb_borrowing, lb_tax))
        ub_x = np.concatenate((ub_consumption, ub_labour, ub_assets, ub_debt, ub_borrowing, ub_tax))

        # tax_rates
        tax_rates = SX.sym('tax_rates', 3, 1)

        # parameter for NLP = revenue constraint
        p = SX.sym('R', 1, 1)

        mu = SX.sym('mu', 5 * conf['T'] + 2, 1)


        # NLP definition = Our optimization problem
        nlp = {'x': vertcat(consumption, labour, assets, debts, borrowing, tax_rates, mu),
               'f': 1,
               'g': vertcat(debt_constraint(debts, borrowing, r, conf),
                            asset_constraint(assets, labour, wage, consumption,
                                             borrowing, tax_rates, r, conf), tax_rates[2],
                            pv_tax(assets, labour, wage, consumption, tax_rates, r, conf) - p,
                            pv_utility(consumption, labour, conf)
                            ),
               'p': p}
        #
    else:

        # concatenate lower bounds and concatenate upper bounds to lb_x and ub_x, respectively
        lb_x = np.concatenate((lb_consumption, lb_labour, lb_assets, lb_debt, lb_borrowing))
        ub_x = np.concatenate((ub_consumption, ub_labour, ub_assets, ub_debt, ub_borrowing))

        # parameter for NLP = tax_rates
        p = SX.sym('p', 3, 1)

        # NLP definition = Our optimization problem
        nlp = {'x': vertcat(consumption, labour, assets, debts, borrowing),
               'f': -pv_utility(consumption, labour, conf),
               'g': vertcat(debt_constraint(debts, borrowing, r, conf),
                            asset_constraint(assets, labour, wage, consumption,
                                             borrowing, p, r, conf)),
               'p': p}

    #  solver_opt = {'verbose': False, 'print_time': False, 'ipopt': {'print_level': 5,
    #                                                               'tol': 1E-12, 'constr_viol_tol': 1E-12}}
    #  solver_opt = dict(qpsol='qrqp', qpsol_options=dict(print_iter=False, tol=1e-16), print_time=False)

    # creating casadi optimization object
    solver_opt = dict(qpsol='qrqp', hessian_approximation='exact',
                      print_header=False, print_iteration= False, print_status=False,
                      qpsol_options=dict(print_iter=False, jit=True, max_iter = 100, error_on_fail=False), print_time=False, max_iter=20)
   # solver_opt = dict(qpsol="qpOASES",hessian_approximation='exact', tol_pr=1E-10, tol_du=1E-10)

    S_sqp = nlpsol("S", "sqpmethod", nlp, solver_opt)

#    solver_opt = {'verbose': False, 'print_time': False, 'knitro': {'algorithm': 4}}
#    S_knitro = nlpsol("S", "knitro", nlp, solver_opt)
#    solver_opt = {'verbose': False, 'print_time': False,'jit': True,
#                  'ipopt': {'print_level': 0,'tol': 1E-09, 'constr_viol_tol': 1E-09, 'sb':'yes'}}
    S_knitro = nlpsol("S", "ipopt", nlp)

    S = [S_sqp, S_sqp]

    return S, lb_x, ub_x



def plot_solutions(consumption, labour, assets, debts, borrowing, tax_rates):
    # create figure with 2x2 subplots
    plt.figure(1)

    plt.subplot(221)
    plt.plot(assets, '.')
    plt.title("Assets")

    plt.subplot(222)
    plt.plot(labour, '.')
    plt.title("Labour")

    plt.subplot(223)
    plt.plot(consumption, '.')
    plt.title('Consumption')

    plt.subplot(224)
    plt.plot(borrowing, '.')
    plt.title('Borrowing')

    plt.show()

    plt.plot(debts, '.')
    plt.title('Debt')

    plt.show()
