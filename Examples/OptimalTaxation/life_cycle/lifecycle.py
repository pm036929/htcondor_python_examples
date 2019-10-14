from casadi import *


def utility(consumption, labour, conf):
    """
        Agent's utility.
    :param consumption:
    :param labour:
    :param conf:
    :return:
    """
    B = 1
    gamma = conf["gamma"]
    eta = conf["eta"]

    util = consumption ** (1 - gamma) / (1 - gamma) - B * labour ** (1 + eta) / (1 + eta)

    return util

def tax(assets, labour, wage, consumption, tax_rates, r, conf):
    """
        Calculate the taxes given the capital, labour and consumption tax rate.
    :param assets: np array containing T+1 elements
    :param labour: T elements
    :param wage: T elements
    :param consumption: T elements
    :param tax_rates: list: [tk, tl, tc]
    :param conf: dictionary containing setting
    :return:
    """
    taxes = r * assets[0:conf['T']] * tax_rates[0] \
            + wage * labour * tax_rates[1] \
            + consumption * tax_rates[2]

    return taxes


def asset_constraint(assets, labour, wage, consumption, borrowing, tax_rates, r, conf):
    """
        Asset_constraint for optimization procedure - length T.

    :param assets:
    :param labour:
    :param wage:
    :param consumption:
    :param borrowing:
    :param tax_rates:
    :param conf:
    :return:
    """

    T = conf['T']

    constraint = (1 + r) * assets[0:T] + wage * labour - consumption \
                 - tax(assets, labour, wage, consumption, tax_rates, r, conf)\
                 + borrowing - assets[1:T+1]

    return constraint


def debt_constraint(debt, borrowing, r, conf):
    """
        Debt constraint for optimization
    :param debt:
    :param borrowing:
    :param r:
    :param conf:
    :return:
    """
    r_debt = conf['r_debt']
    T = conf['T']

    # Recall:
    # debt[0:T] = First T elements = indices: 0, 1, ... ,T-1
    # debt[1:T+1] = T elements = 1, ... ,T

    constraint = (1 + r_debt) * debt[0:T] + borrowing - debt[1:T+1]

    return constraint


def wage_function(conf):
    """
        Agent's wage
    :param conf:
    :return:
    """

    wage = np.zeros(conf["T"])
    T_ret = conf['T_ret']

    # Calculate wage earned for time until retirement
    # array initialized with zeros -> after retirement wage of 0
    wage[:conf["T_ret"]] = -36.9994 + 3.52022 * (np.arange(T_ret) + 1 + 17) \
                           - 0.101878 * (np.arange(T_ret) + 1 + 17)**2 \
                           + .00134816 * (np.arange(T_ret) + 1 + 17)**3 \
                           - 7.06233*1E-6 * (np.arange(T_ret) + 1 + 17)**4

    return wage
