import numpy as np
import pandas as pd

FORMULAS = {
    "forward": lambda f, x, h: (f(x + h) - f(x)) / h,
    "backward": lambda f, x, h: (f(x) - f(x - h)) / h,
    "central": lambda f, x, h: (f(x + h) - f(x - h)) / (2 * h),
    "second": lambda f, x, h: (f(x + h) - 2 * f(x) + f(x - h)) / h**2,
}


def finite_difference(f, x, h, exact=None, methods=("forward", "backward", "central")):
    """Finite difference approximation of f'(x) ("second": f''(x)).

    h       : a step size or a list of step sizes
    exact   : optional true derivative, adds <method>_error and <method>_order
    methods : any of "forward", "backward", "central", "second"

    Returns a DataFrame: h, <method> [, <method>_error, <method>_order]
    """
    h = np.atleast_1d(np.asarray(h, dtype=float))
    table = pd.DataFrame({"h": h})

    for method in methods:
        table[method] = [FORMULAS[method](f, x, step) for step in h]
        if exact is not None:
            error = np.abs(table[method].to_numpy() - exact)
            with np.errstate(divide="ignore", invalid="ignore"):
                order = np.log(error[:-1] / error[1:]) / np.log(h[:-1] / h[1:])
            table[method + "_error"] = error
            table[method + "_order"] = np.append(order, np.nan)

    return table
