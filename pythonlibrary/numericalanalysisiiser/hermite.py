import numpy as np
import pandas as pd


def hermite(x, y, dy, alpha, exact=None):
    """Hermite polynomial matching the values y and slopes dy at the nodes x, evaluated at alpha.

    alpha : a number or a list of points
    exact : optional function f(x), adds the columns exact and error

    Returns a DataFrame: alpha, p [, exact, error]
    """
    alpha = np.atleast_1d(np.asarray(alpha, dtype=float))
    n = len(x)

    a = np.repeat(x, 2)
    M = np.zeros((2 * n, 2 * n))
    M[:, 0] = np.repeat(y, 2)

    for i in range(1, 2 * n):
        if a[i] == a[i - 1]:
            M[i, 1] = dy[i // 2]
        else:
            M[i, 1] = (M[i, 0] - M[i - 1, 0]) / (a[i] - a[i - 1])

    for j in range(2, 2 * n):
        for i in range(j, 2 * n):
            M[i, j] = (M[i, j - 1] - M[i - 1, j - 1]) / (a[i] - a[i - j])

    c = np.diag(M)
    p = np.zeros(len(alpha))
    for k in range(2 * n - 1, -1, -1):
        p = c[k] + (alpha - a[k]) * p

    table = pd.DataFrame({"alpha": alpha, "p": p})
    if exact is not None:
        table["exact"] = [exact(a) for a in alpha]
        table["error"] = (table["exact"] - table["p"]).abs()
    return table
