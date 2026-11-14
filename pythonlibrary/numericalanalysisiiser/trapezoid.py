import numpy as np
import pandas as pd


def composite_trapezoid(f, a, b, n):
    """Composite trapezoidal rule with n sub-intervals (a single number)."""
    h = (b - a) / n
    total = f(a) + f(b)
    for j in range(1, n):
        total += 2 * f(a + j * h)
    return (h / 2) * total


def trapezoid(f, a, b, n=1, exact=None):
    """Trapezoidal rule for the integral of f over [a, b].

    n     : number of sub-intervals, or a list of them (n = 1 is the basic rule)
    exact : optional true value, adds error and order

    Returns a DataFrame: n, h, value [, error, order]
    """
    n = np.atleast_1d(n)
    h = (b - a) / n
    table = pd.DataFrame({"n": n, "h": h, "value": [composite_trapezoid(f, a, b, int(k)) for k in n]})

    if exact is not None:
        error = np.abs(table["value"].to_numpy() - exact)
        with np.errstate(divide="ignore", invalid="ignore"):
            order = np.log(error[:-1] / error[1:]) / np.log(h[:-1] / h[1:])
        table["error"] = error
        table["order"] = np.append(order, np.nan)

    return table
