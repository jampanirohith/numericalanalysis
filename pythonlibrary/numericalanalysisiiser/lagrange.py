import numpy as np
import pandas as pd


def lagrange(x, y, alpha, exact=None):
    """Lagrange polynomial through the points (x, y), evaluated at alpha.

    alpha : a number or a list of points
    exact : optional function f(x), adds the columns exact and error

    Returns a DataFrame: alpha, p [, exact, error]
    """
    alpha = np.atleast_1d(np.asarray(alpha, dtype=float))
    n = len(x)
    p = np.zeros(len(alpha))

    for i in range(n):
        l_i = np.ones(len(alpha))
        for j in range(n):
            if j != i:
                l_i *= (alpha - x[j]) / (x[i] - x[j])
        p += y[i] * l_i

    table = pd.DataFrame({"alpha": alpha, "p": p})
    if exact is not None:
        table["exact"] = [exact(a) for a in alpha]
        table["error"] = (table["exact"] - table["p"]).abs()
    return table
