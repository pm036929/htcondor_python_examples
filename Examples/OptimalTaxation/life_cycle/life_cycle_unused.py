import numpy as np
from scipy import optimize
import matplotlib.pyplot as py
from casadi import *


T = 100
R = 60

w = np.ones(T)
w[R:T] = 0

r = 0.2
beta = 0.9

x0 = np.ones(2*T+1)*10
consumption = MX.sym('consumption', T, 1)
savings = MX.sym('savings', T+1, 1)


def non_lin_cons(consumption, savings, w, r):
    eq = savings[1:] - ((1+r)*savings[:-1] + w - consumption)
    return eq

def utility(consumption, beta, T):
    util = (np.exp(-consumption.T) @(beta**(np.arange(0, T))))
    return util

non_lin_ = optimize.NonlinearConstraint(lambda x: non_lin_cons(x[:T], x[T:], w, r), 0, 0)

bounds = optimize.Bounds(0, [*np.repeat(np.inf, T), 0, *np.repeat(np.inf, T-1), 0])

#res = optimize.minimize(lambda x: utility(x[:T], beta, T), x0, constraints=non_lin_, bounds=bounds, tol=1E-12,
#            options={'maxiter': 10000, 'disp': True})


nlp = {'x': vertcat(consumption, savings),
       'f': utility(consumption, beta, T),
       'g': vertcat(non_lin_cons(consumption, savings, w, r))}

S = nlpsol("S", "knitro", nlp)
res = S(x0=1, lbg=0, ubg=0, lbx=0, ubx=[*np.repeat(np.inf, T), 0, *np.repeat(np.inf, T-1), 0])
print(res)
x_res = res['x'].full()
#x_res = res.x
plt = py.plot(range(T), x_res[:T])

plt = py.plot(range(T),  x_res[T+1:])
py.show()
