import numpy as np
import pandas as pd


def composite_simpson(f, a, b, n):
    """Composite Simpson 1/3 rule with n (even) sub-intervals (a single number)."""
    if n % 2 != 0:
        raise ValueError("n must be even")
    h = (b - a) / n
    total = f(a) + f(b)
    for k in range(1, n):
        if k % 2 == 0:
            total += 2 * f(a + k * h)
        else:
            total += 4 * f(a + k * h)
    return (h / 3) * total


def simpson(f, a, b, n=2, exact=None):
    """Simpson's 1/3 rule for the integral of f over [a, b].

    n     : number of sub-intervals (even), or a list of them (n = 2 is the basic rule)
    exact : optional true value, adds error and order

    Returns a DataFrame: n, h, value [, error, order]
    """
    n = np.atleast_1d(n)
    h = (b - a) / n
    table = pd.DataFrame({"n": n, "h": h, "value": [composite_simpson(f, a, b, int(k)) for k in n]})

    if exact is not None:
        error = np.abs(table["value"].to_numpy() - exact)
        with np.errstate(divide="ignore", invalid="ignore"):
            order = np.log(error[:-1] / error[1:]) / np.log(h[:-1] / h[1:])
        table["error"] = error
        table["order"] = np.append(order, np.nan)

    return table
