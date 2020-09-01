"""Python module for computing the optimal discriminator intervals
    l_1, r_1, l_2, r_2
    p : true distribution
    q : generator distribution
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy.stats import norm


def f(_x, *args):
    _params = args[0]
    p_x = 0.5 * (norm.pdf(_x, loc=_params["p_mu_1"]) + norm.pdf(_x, loc=_params["p_mu_2"]))
    q_x = 0.5 * (norm.pdf(_x, loc=_params["q_mu_1"]) + norm.pdf(_x, loc=_params["q_mu_2"]))
    return p_x - q_x


def solve_fx(_f, x0, _params):
    _res = root(_f, x0, _params)
    return float(_res.x)


def find_optimal_discriminator(params):
    sorted_param_names = sorted(params, key=params.get)

    x = np.linspace(params[sorted_param_names[0]], params[sorted_param_names[-1]], 1000)
    y = f(x, params)

    crosses = np.where(np.diff(np.sign(y[y != 0])))[0]

    x0s = x[crosses]
    res = root(f, x0s, params)

    vals = list(res.x)

    if len(vals) == 1:
        if y[crosses[0]] > 0:
            l_1 = -np.float("inf")
            r_1 = vals[0]
            l_2 = r_1
            r_2 = l_2
        else:
            l_1 = vals[0]
            r_1 = vals[0]
            l_2 = vals[0]
            r_2 = np.float("inf")
    elif len(vals) == 2:
        if y[crosses[0]] > 0:
            l_1 = -np.float("inf")
            r_1 = vals[0]
            l_2 = vals[1]
            r_2 = np.float("inf")
        else:
            l_1 = vals[0]
            r_1 = 0.5 * (vals[0] + vals[1])
            l_2 = r_1
            r_2 = vals[1]
    elif len(vals) == 3:
        if y[crosses[0]] > 0:
            l_1 = -np.float("inf")
            r_1 = vals[0]
            l_2 = vals[1]
            r_2 = vals[2]
        else:
            l_1 = vals[0]
            r_1 = vals[1]
            l_2 = vals[2]
            r_2 = np.float("inf")
    elif len(vals) == 0:
        # any arbitrary value
        l_1, r_1, l_2, r_2 = -10.0, 0.0, 0.0, 10.0
    else:
        raise Exception("There should be at most 3 crossings!")

    return l_1, r_1, l_2, r_2


if __name__ == "__main__":
    # visualization
    params = {"p_mu_1": -2, "p_mu_2": 2, "q_mu_1": -1, "q_mu_2": 2.5}
    l_1, r_1, l_2, r_2 = find_optimal_discriminator(params)
    x = np.linspace(-10, 10, 500)
    f_x = f(x, params)

    disc_bounds = (l_1, r_1, l_2, r_2)
    f_bounds = f(disc_bounds, params)

    plt.plot(x, f_x, disc_bounds, f_bounds, "r.")
    plt.fill_between(x, 0, f_x, where=f_x >= 0, facecolor="green", interpolate=True)

    x_1 = np.linspace(-10 if np.isinf(l_1) else l_1, 10 if np.isinf(r_1) else r_1, 10)
    plt.fill_between(x_1, -0.3, 0.3, facecolor="blue", alpha=0.5, label="$[l_1,r_1]$")

    x_2 = np.linspace(-10 if np.isinf(l_2) else l_2, 10 if np.isinf(r_2) else r_2, 10)
    plt.fill_between(x_2, -0.3, 0.3, facecolor="red", alpha=0.5, label="$[l_2,r_2]$")
    plt.xlabel("$x$")
    plt.ylabel("$p(x)-q(x)$")
    plt.title("Optimal Discriminator")
    plt.legend()
    plt.show()
