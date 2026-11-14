import numpy as np
import pandas as pd


def divided_differences(x, y):
    """Divided difference table of the points (x, y).

    Returns a DataFrame: x, f[0], f[1], ...  (the first row holds the Newton coefficients)
    """
    n = len(x)
    columns = [list(y)]

    for j in range(1, n):
        col = []
        for i in range(n - j):
            col.append((columns[j - 1][i + 1] - columns[j - 1][i]) / (x[i + j] - x[i]))
        columns.append(col + [np.nan] * j)

    table = pd.DataFrame({"x": list(x)})
    for j in range(n):
        table[f"f[{j}]"] = columns[j]
    return table


def newton_interpolation(x, y, alpha, exact=None):
    """Newton form of the interpolation polynomial through the points (x, y), evaluated at alpha.

    alpha : a number or a list of points
    exact : optional function f(x), adds the columns exact and error

    Returns a DataFrame: alpha, p [, exact, error]
    """
    alpha = np.atleast_1d(np.asarray(alpha, dtype=float))
    n = len(x)

    a = list(y)
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            a[i] = (a[i] - a[i - 1]) / (x[i] - x[i - j])

    p = a[0] * np.ones(len(alpha))
    product = np.ones(len(alpha))
    for i in range(1, n):
        product = product * (alpha - x[i - 1])
        p += a[i] * product

    table = pd.DataFrame({"alpha": alpha, "p": p})
    if exact is not None:
        table["exact"] = [exact(a) for a in alpha]
        table["error"] = (table["exact"] - table["p"]).abs()
    return table
